# P-Adic VAE V6.2 — Engineering Specifications

> Living document. Append findings from each session. Date each section.
> Last updated: 2026-03-23 (session 3)

---

## 1. Project Identity

**Goal**: Learn a continuous embedding of all 19,683 ternary operations (3^9) where 3-adic valuation determines radial position in a Poincaré ball. High valuation → near origin; low valuation → near boundary. The geometry should emerge structurally, not be memorized.

**Q metric** (primary training signal):
```
Q = dist_corr + 1.5 × hierarchy_A
```
- `dist_corr`: Spearman correlation between pairwise radial distances and pairwise 3-adic valuations
- `hierarchy_A`: Negated Spearman correlation between valuation and radius (positive = good)
- **Target**: Q ≥ 2.2. Empirically plateaued at ~2.14 over 84-epoch baseline run.

**Hardware constraint**: RTX 3050 6GB. Float64 required throughout (geoopt numerical stability near boundary). AMP disabled.

---

## 2. Dataset Geometry — Critical Facts

| Level | Count | % of total | Natural freq/batch (512) |
|-------|-------|------------|--------------------------|
| v=0   | 13122 | 66.67%     | 341                      |
| v=1   |  4374 | 22.22%     | 114                      |
| v=2   |  1458 |  7.41%     |  38                      |
| v=3   |   486 |  2.47%     |  13                      |
| v=4   |   162 |  0.82%     |   4                      |
| v=5   |    54 |  0.27%     |   1.4                    |
| v=6   |    18 |  0.09%     |   0.5                    |
| v=7   |     6 |  0.03%     |   0.15                   |
| v=8   |     2 |  0.01%     |   0.05                   |
| v=9   |     1 |  0.005%    |   0.03                   |

**The geometric series constraint**: Level counts follow exactly `count_v ≈ 2 × 3^(9-v)` for v=0..8 and 1 for v=9. Every level has exactly 3× fewer samples than the previous. This is not a coincidence — it is the algebraic structure of 3-adic valuations over Z/3^9Z.

**Implication for sampling**: Any frequency-based reweighting must account for this geometric structure. Standard log-frequency (used in NLP) collapses to `log(count_v) = (9-v)×log(3)` — i.e., linear in valuation — producing only ~13.5× ratio v=9:v=0, insufficient for batch presence. Full inverse (1/count) produces 13,122× ratio, causing ~1848 appearances of the single v=9 sample per epoch (overfitting). **The correct choice is `weight = 1/count^0.5`** = `3^(v/2)`, preserving the geometric series structure at half the exponent.

---

## 3. Sampling Specification (Fix 4)

**Implementation**: `WeightedRandomSampler` in `src/train.py` with `weight = 1 / count^0.5`.

**Empirical batch distribution** (batch_size=512, seed=42):
```
v=0: ~217/batch  v=1: ~125  v=2: ~72  v=3: ~42
v=4: ~24/batch   v=5: ~14   v=6: ~8   v=7: ~5  v=8: ~3  v=9: ~2
```

**Per-sample per-epoch appearances**:
- v=9 (1 sample): ~66 times/epoch [was 1848 with 1/count; 3-4 with 1/log]
- v=0 (each sample): ~0.57 times/epoch [was 0.15 with 1/count]

**Constraints and caveats**:
- `replacement=True` is required (v=9 must be oversampled beyond its 1 physical count)
- `num_samples=len(train_ds)` keeps epoch length consistent with natural training
- v=9 may land in val set (~10% probability). When it does, `level_counts[9]=0` → weight stays 0.0. A warning is printed. This is handled gracefully but v=9 will be absent from training in that run
- v=8 (2 samples) and v=9 (1 sample) will often contribute only 1 sample to a given batch. Losses that require `numel() > 1` per level (variance in `RichHierarchyLoss`, `MonotonicRadialLoss` ordering) will skip or degrade gracefully on those batches — this is already guarded in the existing loss code
- The sampler uses a separate `torch.Generator` seeded from `seed` to remain deterministic and independent of the shuffle generator

**Do not use**:
- `1/count`: v=9 seen 1848×/epoch from 1 unique sample → memorization, not geometry
- `1/log(count+1)`: ~13.5× ratio → v=9 appears 0-1 times per batch (nearly absent)
- Any `alpha < 0.5`: approaches log territory, too mild for lower levels
- Any `alpha > 0.5`: approaches 1/count territory, overfitting on rare samples

---

## 4. VAE-B Decoder Optimization (Fix 3)

**Problem**: `decode_b=True` (old default) runs `decoder_B(logmap0(z_B_hyp))` every forward pass. Since `loss_fn_b` has `coverage_weight=0.0`, the logits_B tensor was multiplied by 0.0 in `RichHierarchyLoss`, wasting:
1. The `decoder_B` forward pass (~15% of total forward compute)
2. The `F.cross_entropy(logits_B, targets)` inside `RichHierarchyLoss.forward` (~5% additional)
3. The corresponding backward graph nodes (zero gradient, but still traced)

**Solution** (three-layer fix):
1. `vae.py TernaryVAEV6.forward(decode_b=False)`: skips `decoder_B` forward; returns `logits_B=None`
2. `losses/combined.py CombinedLoss.forward`: passes `logits=None` to `rich_hierarchy` when `coverage_weight == 0.0`
3. `losses/padic_geodesic.py RichHierarchyLoss.forward`: guards `if logits is None → coverage_loss = 0.0`

**Invariants that must hold**:
- `decode_b=False` MUST only be used when the caller guarantees `coverage_weight=0.0` in the corresponding loss function. Violating this returns `logits_B=None` into a loss that expects logits, causing a runtime error at the `.shape[-1]` access. This is intentional (fail-loud).
- The val loop calls `model(batch_ops)` without `decode_b=False` — decoder_B IS run during validation (correct: val uses the full model for diagnostic purposes)
- The `ModelAuditor` gradient check also uses default `decode_b=True` — intentional

**Gradient safety**: With `logits=None`, `coverage_loss = tensor(0.0)` — a leaf with no grad_fn. No gradient path to decoder_A through `loss_fn_b`. No gradient path to decoder_B (it was never called). Verified: 280/280 tests pass.

---

## 5. Loss Weight Balance (Fix 1+2)

**Root cause of Q plateau at ~2.14** (confirmed via 84-epoch empirical run):
- `geodesic.weight=0.5` provides the only `dist_corr` signal
- Combined radial losses (rich_hierarchy×5 + radial×5 + monotonic×1 = effective weight ~15) are ~30× stronger
- `hierarchy_A` saturates at ~0.836 by epoch 16 (radial losses solved)
- `dist_corr` stalls at ~0.40 (geodesic signal too weak to compete)

**Current weights** (post-fix):
```yaml
radial.weight:            1.0   # was 5.0 (reduced 5×)
geodesic.weight:          2.0   # was 0.5 (increased 4×)
geodesic.phase_start_epoch: 10  # was 30 (starts 20 epochs earlier)
```

**Expected ratio** (post-fix): geodesic (~2.0) vs combined radial (~5+1+1=7) ≈ 2:7. Previously 0.5:15 = 1:30. The dist_corr signal can now compete.

**Caveat**: `rank.weight=0.5` (GlobalRankLoss) is currently enabled and provides overlapping signal with geodesic. May be worth disabling if geodesic at w=2.0 is sufficient. Monitor `losses['rank']` vs `losses['geodesic']` magnitude ratio in next run.

---

## 6. Architecture Constraints

### Numerical precision
- All geometry and loss code uses **float64** throughout. This is non-negotiable: geoopt operations near the Poincaré boundary (radius → 1.0) are numerically unstable in float32.
- `torch.set_default_dtype(torch.float64)` is called at startup in `set_determinism()`.
- AMP (`use_amp=False`) is disabled. Float64 + AMP = incompatible on RTX 3050.

### Curvature sharing
- `proj_A` has `learnable_curvature=True`; `proj_B` has `learnable_curvature=False`.
- **This is intentional**: both projections share the curvature learned by proj_A. Having independent curvatures would allow VAE-B to learn a different geometry incompatible with the shared Poincaré structure.

### tangent_scale parameter
- `HyperbolicProjection` has a learnable `tangent_scale` (init=0.1).
- Encoder outputs have norm ~4.0. Without scaling, `expmap0(4.0)` saturates near radius 0.95. With `tangent_scale=0.1`: input norm ~0.4 → radius ~0.38 at init.
- **Do not initialize to 0.0** (kills all gradients through the projection at step 0).
- **Do not initialize > 0.5** (causes expmap0 saturation from epoch 0, all radii = 0.95).

### Target radii
- Linear interpolation: `target_r(v) = outer*(1-t) + inner*t` where `t = v/9`
- `outer_radius=0.85` (v=0), `inner_radius=0.08` (v=9)
- These are shared centrally via `radius_defaults.py` → `auto_share_radius_config()`. Any change to these values in `rich_hierarchy` config automatically propagates to `radial` and `monotonic` losses.

### StateNet / LR controller
- `LRScales.__post_init__` validates all scales in `(0, 1]`. Setting any scale to 0.0 is done by the controller at runtime (it sets the optimizer group's `lr` to 0), not by mutating the config.
- `CoverageThresholds.fix_threshold=0.35, train_threshold=0.45` calibrated to per-digit accuracy (random baseline ≈ 0.33). Not perfect-sample coverage.

---

## 7. Known Open Issues (as of 2026-03-19)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `rank` loss (w=0.5) overlaps with `geodesic` — may be redundant at geodesic w=2.0 | Low | Monitor |
| 2 | v=8/v=9 often contribute only 1 sample/batch — variance term in `RichHierarchyLoss` skipped | Low | Handled gracefully |
| 3 | `ModelAuditor` gradient check runs `decoder_B` (decode_b=True) even though training skips it | Low | Intentional |
| 4 | `CombinedLoss._compute_coverage_loss` fallback (line 474) still computes CE from logits when `rich_hierarchy is None` — but `logits_B=None` in this path would crash | Medium | Not triggered in v6 config (rich_hierarchy always enabled) — add guard if ever disabling rich_hierarchy |
| 5 | `WeightedRandomSampler` with `num_workers>0` shares the generator across workers without explicit per-worker seeding for the sampler — minor non-determinism across runs with different worker counts | Low | Cosmetic |

---

## 8. File Responsibility Map

| File | Primary responsibility | Key constraints |
|------|----------------------|-----------------|
| `src/core/ternary.py` | 3-adic arithmetic, LUTs, level counts | Immutable singleton; all lookups O(1) |
| `src/geometry/poincare.py` | geoopt ops (expmap0, logmap0, distance) | Float64; manifold cache keyed by (c, device) |
| `src/models/vae.py` | Dual VAE forward; `decode_b` flag | `decode_b=False` requires coverage_weight=0.0 in caller |
| `src/models/hyperbolic_projection.py` | tangent_net + expmap0 | tangent_scale init must be in (0, 0.5] |
| `src/models/lr_controller.py` | MetricBasedLR; LR scale decisions | Operates on metrics, not model internals |
| `src/losses/padic_geodesic.py` | All hierarchy/radial losses | `logits=None` guard in RichHierarchyLoss |
| `src/losses/combined.py` | Config-driven loss composition | Passes `logits=None` when coverage_weight=0.0 |
| `src/losses/radius_defaults.py` | Centralized radius propagation | Source of truth for inner/outer radius |
| `src/config/statenet_config.py` | StateNet dataclass; validation | LRScales validates (0, 1]; CoverageThresholds validates order |
| `src/train.py` | Training loop; sampler; loss routing | `decode_b=False` always; sqrt-weighted sampler |
| `src/presets/v6.yaml` | Active training config | See §5 for current loss weights |

---

## 9. KL Divergence — Geometry and Constraints

**Implementation**: `HyperbolicKLDivergence` in `src/losses/hyperbolic_kl.py`.

The standard Gaussian KL `0.5 * (σ² + μ² - log σ² - 1)` is incorrect for hyperbolic VAEs because it ignores the metric distortion of the Poincaré ball. The corrected formula scales the variance term by the conformal factor `λ(x) = 2/(1 - c||x||²)`:

```
KL_hyp = 0.5 * (λ(z_hyp)² × σ² + μ² - log σ² - d)
```

`z_hyp` (the manifold point) is used for the conformal factor, not `μ` (which lives in tangent space). If `z_hyp=None` is passed, `μ` is used as fallback — geometrically incorrect but safe for debugging.

**Current config** (`v6.yaml`):
```yaml
hyperbolic_kl:
  beta: 0.1      # Scales variance inside the KL formula
  weight: 0.01   # Outer multiplier in CombinedLoss
  free_bits: 0.5 # Minimum KL per dim; prevents posterior collapse
```

**Caveat**: `beta` and `weight` are multiplicative and their interaction is non-obvious. The effective KL contribution to total loss is `weight × beta × KL_raw`. At `beta=0.1, weight=0.01`: the KL enters at `0.001×KL_raw` — very conservative. If posterior collapse occurs (all `logvar → large negative`), increase `free_bits` before increasing `weight`; free_bits is a gentler intervention.

**VAE-B KL**: `loss_fn_b` includes KL loss (it is not disabled like coverage). VAE-B's `mu_B/logvar_B` are passed to `loss_fn_b`. This is correct — VAE-B should be a true VAE, not a deterministic encoder.

---

## 10. LR Controller — Decision Logic

**MetricBasedLR** makes three independent decisions per epoch:

| Component | Gate | Freeze condition | Unfreeze condition |
|-----------|------|------------------|--------------------|
| encoder_a | coverage | `coverage < fix_threshold` (0.35) | `coverage ≥ train_threshold` (0.45) OR hierarchy stalled |
| encoder_b | hierarchy plateau | `improvement < plateau_threshold` for `plateau_patience` epochs | `improvement > plateau_threshold` in window |
| projections | grad norm | `grad_norm < grad_threshold` for `grad_patience` epochs | `grad_norm > avg × spike_multiplier` |

**Hysteresis**: `hysteresis_epochs=5` prevents rapid flip-flopping. Each component tracks `_last_change[component]`; a change can only occur if `epoch - last_change ≥ hysteresis_epochs`.

**Critical invariant**: The controller outputs LR **scales** (0.0–1.0), not absolute LRs. The actual LR applied is `scale × base_lr × cosine_factor`. Setting scale=0.0 freezes the component without touching `requires_grad`. The `train.py` loop additionally calls `model.set_encoder_{a,b}_trainable()` to sync `requires_grad` — this is an optimization (stops gradient computation at source) but is separate from the LR=0 freeze.

**Stale momentum zero-out**: When a component unfreezes, `optimizer.state[p]['exp_avg']` and `exp_avg_sq` are zeroed for that group. This prevents the momentum spike that would occur from accumulated stale estimates during the freeze period. This is implemented in `train.py:1263–1275`.

**`q_history` window differs from other histories**: `_q_history` uses `window × 2` (line 199 in `lr_controller.py`). This is intentional — Q tracking needs longer context to detect genuine improvement vs noise.

---

## 11. Geometry Backend — Constraints and Caveats

**Manifold cache**: `poincare.py` caches `geoopt.PoincareBall` instances keyed by `(curvature, device_str)`. Since `proj_A` has `learnable_curvature=True`, its curvature changes each optimizer step. **The cache is keyed on the initial curvature float** — it does NOT update when curvature is learned. After training starts, `proj_A.manifold.c` diverges from the cached manifold's `c`. This means `get_manifold(c)` calls from loss functions (using the initial `c=1.0`) use a different manifold than `proj_A` internally. In practice this is a small error (curvature rarely moves far from 1.0) but could be material in long runs.

**`hyperbolic_radius` creates a zero tensor per call**: `origin = torch.zeros_like(z)` allocates a fresh tensor every call. In the training loop this is called once per batch per loss that computes radii (RichHierarchyLoss, RadialHierarchyLoss, MonotonicRadialLoss = 3 allocations per batch). With float64 and batch_size=512, latent_dim=16: `512 × 16 × 8 bytes = 65KB` × 3 = ~200KB/batch. Minor on 6GB GPU but worth noting.

**`log_map_zero` max_norm clamp**: The `max_norm` parameter in `log_map_zero` (used in VAE decoder) clamps points before applying logmap. This is a safety net against the rare case where expmap produces a point very close to the boundary (norm > max_radius). Without this, logmap at norm → 1.0 returns `arctanh(1) → ∞`. The clamp is applied AFTER expmap in `HyperbolicProjection.forward` (Euclidean clamp on `z_hyp`), and BEFORE logmap in the decoder. Both guards are necessary.

**Float64 manifold + float64 model**: The `manifold.to(device_str)` call in `get_manifold` moves geoopt's internal `k` buffer (curvature) to the device. Without this, a CPU `k` mixed with CUDA `z_hyp` causes device mismatch errors. The `HyperbolicProjection.to(torch.float64)` call at the end of `__init__` ensures all parameters, including `tangent_scale`, are float64 at init.

---

## 13. Test Suite Contracts

**280 tests across 8 files.** Tests are contract-driven, not implementation-mirroring.

Key contracts tested:
- `compute_Q(dist_corr, hierarchy)`: `hierarchy` is **pre-negated** Spearman before calling. Positive = good ordering. Contract: `Q(h=+0.5) > Q(h=-0.5)`.
- `RichHierarchyLoss.forward` returns `(Dict[str, Tensor], MetricsDict)` — not a scalar
- `CombinedLoss` with all losses disabled raises `ValueError` (not silent zero loss)
- `WeightedRandomSampler` weight formula: verified empirically that rare levels appear in batches

**What is NOT tested** (by design):
- Shape assertions (PyTorch guarantees these)
- That `logits_B=None` propagates correctly through `loss_fn_b` — this is a runtime invariant enforced by the `decode_b=False` + `coverage_weight=0.0` contract, not a unit test

---

## 14. Appendix: Empirical Baseline (84-epoch CPU run, 2026-03-11)

| Metric | Value | Notes |
|--------|-------|-------|
| Best Q | 2.141 | Plateau from epoch ~40 |
| hierarchy_A at plateau | 0.836 | Saturated early (epoch 16) |
| dist_corr at plateau | 0.404 | Never improved after epoch 20 |
| Coverage (per-digit acc) | ~0.43 | StateNet threshold 0.45 — barely active |
| StateNet A | 0.05 (active) | Never frozen; working as designed |
| StateNet B | 0.10 (active) | Never frozen |
| Geodesic phase start | epoch 30 | 30 epochs of zero dist_corr signal |
| geodesic:radial ratio | 0.5:15 ≈ 1:30 | Root cause of Q plateau |

---

## 15. Utility Modules — Specifications (2026-03-19)

### 15.1 `src/utils/checkpoint.py`

**Purpose**: Safe checkpoint loading with backwards-compatible format detection.

**Public API**:
| Function | Signature | Behavior |
|----------|-----------|----------|
| `load_checkpoint_compat` | `(path, map_location="cpu") → Dict` | Loads `.pt` file; raises `FileNotFoundError` if absent; uses `weights_only=False` (required for dicts with non-tensor metadata) |
| `get_model_state_dict` | `(checkpoint: Dict) → Dict` | Handles 3 checkpoint shapes: `{model_state_dict: ...}`, `{model: ...}`, or bare state dict |

**Security note**: `weights_only=False` uses Python `pickle` — only load checkpoints from trusted sources.

**Checkpoint format written by `train.py`**:
```python
{
    "epoch": int,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss_fn_state_dict": loss_fn.state_dict(),  # includes learned log_sigmas
    "q_value": float,
    "metrics": dict,
}
```

---

### 15.2 `src/utils/hardware_monitor.py`

**Purpose**: GPU/RAM memory tracking during training. Used in train.py for terminal status lines and OOM diagnostics.

**Dependencies**: `psutil` (optional, RAM only — gracefully degrades to `RAM: N/A`). `torch.cuda` (GPU).

**Public API**:
| Method | Returns | Notes |
|--------|---------|-------|
| `get_gpu_memory_mb()` | `Dict[str, float]` | keys: `allocated`, `reserved`, `peak`, `total` |
| `get_gpu_memory_gb()` | `Dict[str, float]` | Same ÷ 1024 |
| `get_gpu_utilization_pct()` | `float` | 0–100; `0.0` on CPU |
| `get_status_string()` | `str` | `"GPU: 2.1/6.0GB (35%) \| RAM: 8.2/32.0GB (26%)"` |
| `get_peak_status_string()` | `str` | Adds `peak=Xgb` field |
| `check_memory_warning()` | `Optional[str]` | `None` if below `warn_threshold` (default 0.90) |
| `reset_peak_stats()` | `None` | Resets `max_memory_allocated` counter |
| `get_oom_diagnostic(batch_size)` | `str` | Multi-line debug string; suggests `batch_size // 2` |

**Constraint**: Only `is_cuda=True` paths actually call CUDA APIs. All methods are safe to call on CPU-only runs.

---

### 15.3 `src/utils/tensorboard_logger.py`

**Purpose**: All TensorBoard I/O in one place (Single Responsibility). Optional — gracefully disabled if `tensorboard` not installed.

**Initialization**: Pass `tensorboard_dir=None` to disable; log path becomes `{tensorboard_dir}/ternary_vae_{experiment_name}`.

**Key method: `log_manifold_embedding`**
- Encodes up to `n_samples` (default 5000) operations through the full model
- Logs 4 embedding tags per call: `VAE_A_TangentSpace`, `VAE_A_Poincare`, `VAE_B_TangentSpace`, `VAE_B_Poincare`
- Metadata columns: `index`, `prefix_1` (op%3), `prefix_2` (op%9), `prefix_3` (op%27), `tree_depth` (3-adic valuation), `radius_A`, `radius_B`
- Sampling is seeded (`seed=42`) → reproducible across epochs
- Calls `model(x)` under `torch.no_grad()` → does NOT use `decode_b=False`, so both decoders run

**Caveat**: `log_manifold_embedding` is expensive (encodes 5000 samples). Call only every N epochs.

**mypy exemption**: `ignore_errors = true` in `pyproject.toml` because `SummaryWriter` is assigned conditionally via try/except import and mypy cannot resolve the union.

---

## 16. Config Layer — Specifications (2026-03-19)

### 16.1 `src/config/constants.py`

```python
N_TERNARY_OPERATIONS = 19683  # 3^9 — canonical count, imported everywhere
```

This is the **single source of truth** for dataset size. Never hardcode `19683`.

### 16.2 `src/config/paths.py`

Canonical filesystem paths derived from `__file__` (portable, no hardcoded roots):

| Variable | Path | Notes |
|----------|------|-------|
| `PROJECT_ROOT` | `…/3-adic-ml/` | Resolved from `paths.py` location |
| `RUNS_DIR` | `PROJECT_ROOT/runs/` | TensorBoard + experiment logs |
| `CHECKPOINTS_DIR` | `PROJECT_ROOT/models/checkpoints/` | Best/periodic checkpoints |
| `MODELS_DIR` | `PROJECT_ROOT/models/` | Model artefacts |
| `SRC_PRESETS_DIR` | `PROJECT_ROOT/src/presets/` | YAML config presets |

**Constraint**: `train.py` derives its checkpoint output path from these constants. If you restructure the repo, update `paths.py` first.

### 16.3 `src/config/statenet_config.py`

**StateNetConfig dataclass hierarchy**:

```
StateNetConfig
├── enabled: bool
├── coverage: CoverageThresholds    [fix_threshold, train_threshold, floor]
├── hierarchy: HierarchyThresholds  [plateau_threshold, plateau_patience, patience_ceiling, stall_patience]
├── controller: ControllerThresholds [grad_threshold, grad_patience, patience_ceiling, spike_multiplier]
├── timing: TimingConfig            [warmup_epochs, hysteresis_epochs, window_size]
├── lr_scales: LRScales             [encoder_a, encoder_b, projections, decoders]
└── initial: InitialStates          [encoder_a_trainable, encoder_b_trainable, projections_trainable]
```

**Invariants enforced at construction**:
- `CoverageThresholds`: `0 ≤ fix_threshold < train_threshold ≤ 1.0` (raises `ValueError`)
- `LRScales`: all values must be in `(0, 1]` — zero LR is illegal here; use `statenet.enabled=false`

**Unknown key detection**: `from_dict()` raises `ValueError` on unrecognized YAML keys — prevents silent misconfiguration.

**Round-trip**: `from_dict(config.to_dict()) == config` (both directions lossless).

---

## 17. Tooling — Specifications (2026-03-19)

### 17.1 Ruff (linter + formatter)

Configured in `pyproject.toml [tool.ruff]`. Replaces: flake8, isort, pyupgrade, pep8, pyflakes.

**Active rule sets**: `E` (pycodestyle), `W` (warnings), `F` (pyflakes), `B` (bugbear), `UP` (pyupgrade), `C4` (comprehensions), `I` (isort), `RUF` (ruff-native).

**Key ignores**:
| Rule | Reason |
|------|--------|
| `E501` | Line length: float64 tensor expressions make long lines — formatter handles |
| `UP006/UP007/UP045/UP035` | Python 3.10 `Optional[X]` modernization deferred |
| `B905` | zip strict: already addressed in train.py |
| `TRY003` | Long exception messages: intentional for diagnostics |
| `RUF002` | Ambiguous unicode: math notation in docstrings |
| `RUF022` | Unsorted `__all__`: grouped by category intentionally |

**Per-file ignores**:
- `src/train.py`: `E402` — `sys.path.insert` before internal imports is required
- `tests/*.py`: `F841` (unused variable), `RUF059` (unused unpacked variable)

**Current status**: 0 errors as of 2026-03-19.

### 17.2 Mypy (static type checker)

**Key settings**:
- `ignore_missing_imports = true` — torch/geoopt have no bundled stubs
- `explicit_package_bases = true` — prevents "module found twice" error from `src/` layout
- `check_untyped_defs = true` — validates unannotated functions
- `warn_return_any = false` — too noisy with torch tensor ops

**Per-module exemptions** (`ignore_errors = true`):
- `src.train` — complex runtime unions (scheduler/tqdm), `sys.path` manipulation
- `src.losses.padic_geodesic` — `_target_radii` registered as buffer → `Tensor|Module` union
- `src.utils.tensorboard_logger` — `SummaryWriter` conditional import creates unresolvable union

**Current status**: 0 errors across 23 files as of 2026-03-19.

**Annotation coverage** (from AST analysis):
| File | Return-annotated % | Gap |
|------|--------------------|-----|
| `hyperbolic_projection.py` | 62.5% | Low; cast fixes applied |
| `hyperbolic_kl.py` | 66.7% | Medium |
| `padic_geodesic.py` | 69.2% | Medium |
| `vae.py` | ~85% | Acceptable |
| Core/geometry modules | 90–100% | Well-covered |

---

## 18. Static Analysis Summary (2026-03-19)

**Codebase metrics** (23 files, AST-measured):

| Metric | Value |
|--------|-------|
| Total LOC | ~7,393 |
| Classes | 33 |
| Functions | 195 |
| Return-annotated | 88.7% |
| Highest cyclomatic complexity | `train()` function, c=66 |
| Clean ruff | 0 errors |
| Clean mypy | 0 errors (23 files) |

**Complexity hotspot**: `train()` in `src/train.py` at c=66. This is intentional — the training loop integrates 8+ subsystems. If it needs to be split, break at natural phase boundaries (data prep, model init, epoch loop, metrics, checkpointing).

**Type debt**: Three files below 70% annotation coverage. Not blocking — mypy errors in those files are suppressed at module level. Address before any public API release.

---

## 19. dist_corr Root Cause Analysis (2026-03-19) — Critical Finding

### What dist_corr Actually Measures

From `train.py:573–577`:
```python
r_dists = np.abs(r_sample[:, None] - r_sample[None, :])   # pairwise RADIUS differences
v_dists = np.abs(v_sample[:, None] - v_sample[None, :])   # pairwise INDIVIDUAL valuation diffs
dist_corr = spearmanr(r_dists[triu_idx], v_dists[triu_idx]).correlation
```

**`dist_corr = Spearman(|r_i − r_j|, |v₃(i) − v₃(j)|)`** where `r_i = poincare_distance(z_i, 0)` and `v₃(i)` is the individual 3-adic valuation of operation i. This is purely about individual radii — NOT about pairwise geodesic distances.

### What PAdicGeodesicLoss Actually Does

```python
diff = |batch_indices[i] − batch_indices[j]|
valuation = v₃(diff)                            # valuation of INDEX DIFFERENCE
d_target = 3.0 * exp(−valuation / 3.0)
loss = smooth_l1(poincare_distance(z_i, z_j), d_target)
```

It targets `poincare_distance(z_i, z_j) ∼ v₃(|i−j|)` — the **pairwise geodesic** as a function of the **index-difference valuation**. This encodes the 3-adic metric on ℤ (the theoretical goal of the project), but it is a fundamentally different quantity from what dist_corr measures.

**Concrete example of the mismatch**:

| Pair | v₃(i), v₃(j) | `|v_i − v_j|` (dist_corr cares) | v₃(\|i−j\|) (geodesic targets) |
|------|--------------|----------------------------------|---------------------------------|
| (3, 6) | 1, 1 | **0** — same radius expected | 1 — d≈2.15 demanded |
| (1, 2) | 0, 0 | **0** — same radius expected | 0 — d=3.0 demanded |
| (0, 9) | 9, 2 | **7** — large radius gap expected | 2 — d≈1.56 demanded |

For same-valuation pairs (3,6) and (1,2): dist_corr expects identical radii. Geodesic loss demands large separation. In a 16-dim Poincaré ball, this separation is overwhelmingly angular — same radius but different directions. The geodesic loss's gradient budget is spent on angular structure, which is invisible to dist_corr.

### Loss-to-Metric Attribution Map

| Loss | Weight | What it drives | Drives dist_corr? |
|------|--------|----------------|-------------------|
| RichHierarchyLoss — hierarchy | 5.0 | Per-level mean radius MSE + 0.1×variance | Weakly via variance_loss |
| RichHierarchyLoss — separation | 3.0 | Adjacent level mean gaps | Partially |
| RadialHierarchyLoss | 1.0 | Per-point MSE to target + margin pairs | Partially |
| **PAdicGeodesicLoss** | **2.0** | `poincare_dist ~ v₃(\|i−j\|)` — angular structure | **No — wrong pairwise metric** |
| GlobalRankLoss | 0.5 | Ordinal radius ordering (same as hierarchy_A) | No (masks same-v pairs) |
| MonotonicRadialLoss | 1.0 | Level mean ordering | Partially |

**The highest-weighted single loss (geodesic, 2.0) does not drive Q at all.** It is theoretically valid for p-adic metric embedding but targets a metric that does not appear in the Q formula.

### The Actual dist_corr Bottleneck

For dist_corr to be high, two conditions must hold simultaneously:
1. **Between-level ordering**: r(v=0) > r(v=1) > … > r(v=9) — hierarchy_A measures this, currently saturated at 0.836
2. **Within-level tightness**: points at the same valuation level must cluster at their target radius with low scatter

Condition 2 is only addressed by `variance_loss` in `RichHierarchyLoss.forward()`:
```python
# padic_geodesic.py:855 — HARDCODED, not configurable from YAML
hierarchy_loss = hierarchy_loss + 0.1 * variance_loss
```

### Quantitative Analysis

**Simulation (empirical, n=500 samples from real valuation distribution)**:

| Scenario | dist_corr | Notes |
|----------|-----------|-------|
| All points at exact targets | **0.9912** | Theoretical upper bound |
| Within-level std = 0.05 euclid | 0.8784 | Achievable |
| Within-level std = 0.10 euclid | 0.6467 | Marginal |
| Within-level std = 0.15 euclid | **0.4461** | ≈ Observed plateau of 0.404 |
| Within-level std = 0.20 euclid | 0.3119 | Getting worse |
| Radii compressed to 0.23–0.57 (observed) | **0.9912** | Compression alone does NOT hurt dist_corr |

**Critical insight**: Radii compression is NOT the bottleneck. Within-level scatter std ≈ 0.15 euclid is the diagnosis of the current plateau at dist_corr ≈ 0.404.

**To reach Q = 2.2 with hierarchy_A = 0.836**:
```
dist_corr_needed = 2.2 − 1.5 × 0.836 = 0.946
→ Requires within-level std ≤ 0.05 euclid (3× reduction from current ~0.15)
```

**variance_loss gradient signal analysis**:

| variance_weight | Effective contribution vs mean_MSE |
|-----------------|------------------------------------|
| 0.1 (current)   | ~0.9× (roughly equal — but mean_MSE gradient is already near-zero at plateau) |
| 0.5             | ~4.5× (variance gradient dominates once mean is near target) |
| 1.0             | ~9.0× (strong tightening, potential instability risk) |

At the plateau, mean_MSE ≈ 0 (points are near their target means), so variance_loss is the ONLY remaining gradient signal for dist_corr — and it's only weighted 0.1. Increasing it is the highest-leverage change available.

### Target Radius Reference Table

Computed from `_exponential_target_radii(9, inner=0.08, outer=0.85, scale=3.0)`:

| v | Euclid target | Hyperbolic target | Gap to next level (euclid) |
|---|--------------|-------------------|---------------------------|
| 0 | 0.8500 | 2.5123 | — |
| 1 | 0.6203 | 1.4510 | 0.2297 |
| 2 | 0.4557 | 0.9837 | 0.1646 |
| 3 | 0.3378 | 0.7031 | 0.1179 |
| 4 | 0.2533 | 0.5178 | 0.0845 |
| 5 | 0.1927 | 0.3903 | 0.0606 |
| 6 | 0.1493 | 0.3009 | 0.0434 |
| 7 | 0.1182 | 0.2376 | 0.0311 |
| 8 | 0.0960 | 0.1925 | 0.0223 |
| 9 | 0.0800 | 0.1603 | 0.0160 |

Total euclid span: 0.77. The inter-level gaps shrink geometrically (factor ~0.72 per level). Within-level std must be <<0.016 at v=8,9 for high-valuation levels to contribute to dist_corr.

### LR Controller — No Interference

`MetricBasedLR` gates encoder_a on coverage and encoder_b on hierarchy_A plateau. It does NOT gate any component on dist_corr. This is not a problem — the controller is not suppressing dist_corr, it's simply unaware of it. No change needed here.

---

## 20. Action Plan — Ranked by Expected ROI (2026-03-19)

### Fix 5A — Expose `variance_weight` as YAML config (HIGHEST PRIORITY)

**Files**: `src/losses/padic_geodesic.py`, `src/losses/combined.py`, `src/presets/v6.yaml`

**Change**:
```python
# padic_geodesic.py — RichHierarchyLoss.__init__
def __init__(self, ..., variance_weight: float = 0.1):
    self.variance_weight = variance_weight
    ...

# forward() line 855
hierarchy_loss = hierarchy_loss + self.variance_weight * variance_loss
```

```python
# combined.py — _init_losses()
self.rich_hierarchy = RichHierarchyLoss(
    ...,
    variance_weight=rich_cfg.get('variance_weight', 0.1),
)
```

```yaml
# v6.yaml
rich_hierarchy:
  variance_weight: 0.5   # was hardcoded 0.1 — 5x increase, targets dist_corr directly
```

**Why 0.5**: At plateau, mean_MSE gradient ≈ 0, so variance_loss becomes the dominant radial signal. 0.5 gives 4.5× the current gradient without overwhelming the mean_MSE during early training when mean placement matters more.

**Expected impact**: Within-level std should reduce from ~0.15 → ~0.07, pushing dist_corr from 0.404 → ~0.75. Q estimate: 0.75 + 1.5 × 0.836 = **2.00**. Not enough alone.

**Risk**: Low. The change is continuous and still subordinate to mean_MSE during early epochs.

---

### Fix 5B — Reduce `geodesic.weight` from 2.0 to 0.5 (HIGH PRIORITY)

**File**: `src/presets/v6.yaml`

```yaml
geodesic:
  weight: 0.5   # was 2.0 — loss drives angular structure, not Q
```

**Rationale**: PAdicGeodesicLoss is the highest individual weight loss and does not drive dist_corr. Keeping it at a low weight preserves the theoretically correct p-adic embedding signal while freeing gradient budget. The freed 1.5× weight budget can be reallocated.

**Budget reallocation** (also in v6.yaml):
```yaml
rich_hierarchy:
  hierarchy_weight: 7.0   # was 5.0 — more gradient for tightening radial placement
  variance_weight: 0.5    # new config key (Fix 5A)
```

**Expected impact**: Reduces gradient interference from angular structure; more budget for radial precision.

---

### Fix 5C — Add within-level scatter loss to GlobalRankLoss (MEDIUM PRIORITY)

`GlobalRankLoss.forward()` currently masks out same-valuation pairs entirely:
```python
diff_mask = v_diff != 0   # line 524 — same-v pairs contribute ZERO gradient
```

Adding a within-level scatter penalty closes this gap:
```python
# For same-valuation pairs: push them to the same radius
same_v_mask = v_diff == 0
if same_v_mask.any():
    r_diff_same = (r_i[same_v_mask] - r_j[same_v_mask]) ** 2
    scatter_loss = r_diff_same.mean()
    loss = loss + scatter_weight * scatter_loss
```

This is the most direct differentiable proxy for dist_corr. To be exposed as `rank.scatter_weight: 0.3` in YAML.

**Expected impact**: Combined with 5A+5B, should bring within-level std to ~0.05 and dist_corr to ~0.88. Q estimate: 0.88 + 1.5 × 0.90 = **2.23** (also requires hierarchy_A to reach 0.90).

---

### Priority order for implementation

| Fix | Files changed | Complexity | Expected dist_corr gain |
|-----|--------------|------------|-------------------------|
| 5A: variance_weight=0.5 | 3 files, ~8 lines | Low | +0.35 (0.40 → 0.75) |
| 5B: geodesic.weight=0.5 | 1 file (yaml only) | Trivial | Indirect (frees budget) |
| 5C: same-v scatter loss | 2 files, ~15 lines | Medium | +0.13 (0.75 → 0.88) |
| Training run to validate | — | — | Empirical confirmation |

**Do 5B first** (yaml-only, zero risk), then 5A (small code change), then run training for 50 epochs to see if dist_corr responds before implementing 5C.

### What NOT to change

- **Compression** (radii 0.23–0.57 vs target 0.08–0.85): Simulation proved compression alone does not affect dist_corr. The between-level ordering is preserved under linear rescaling, so rank correlation is identical. Do not spend effort on pushing tangent_scale higher.
- **geodesic.phase_start_epoch**: Already reduced from 30 → 10. Starting even earlier won't help since the loss isn't driving dist_corr.
- **LR controller thresholds**: The controller is not interfering with dist_corr and does not need gating on dist_corr — the radial losses handle it if properly weighted.
- **PAdicGeodesicLoss formula**: It is theoretically correct for p-adic metric embedding (v₃(|i−j|) IS the correct 3-adic distance). The issue is that Q doesn't reward it. Keep it at low weight as a structural regularizer.

---

## 21. Level Prefix & Soft Margin — Specification (2026-03-23)

### Background: ARI Ceiling at 0.844

Four V7 training runs established a structural ARI ceiling:

| Run | Config | ARI (K-means@15 vs prefix3) | Q |
|-----|--------|---------------------------|---|
| V7.0 baseline | No AC loss | 0.721 | 2.163 |
| V7.1 AC light | weight=0.3, prefix_k=3, phase_start=50 | 0.810 | 2.163 |
| V7.1 AC aggressive | weight=1.0, prefix_k=3, phase_start=10, n_pairs=2000 | 0.820 | 2.163 |
| V7.2 large | latent_dim=64, hidden_dim=128, 60 direction dims | 0.844 | 2.163 |

Root cause: AC leverage is concentrated at v=0. Deeper levels have too few prefix classes (v=1: 2 classes, v=2: 1 class at k=2) for AC to act on.

### level_prefix_k — Per-Level Prefix Depth

**Config key**: `loss.angular_coherence.level_prefix_k`

```yaml
level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]
```

- v=0 → k=3 (27 classes of ~486 ops)
- v=1 → k=4 (18 classes of ~243 ops, digit0 fixed to -1)
- v=2 → k=5 (27 classes of ~54 ops, digit0,digit1 fixed to -1)
- v=3+ → 0 = skip (already converged, within-sim ≥ 0.99)

When `level_prefix_k` is set, `AngularCoherenceLoss.forward()` processes levels independently with `n_pairs // n_active_levels` pairs per level. When `level_prefix_k=None`, falls back to global `prefix_k` (backward compatible).

### target_sim — Soft Margin

**Config key**: `loss.angular_coherence.target_sim`

```yaml
target_sim: [1.0, 0.85, 0.70, 0, 0, 0, 0, 0, 0, 0]
```

Per-level soft margin: `F.relu(target_sim_v - cos_sim).mean()` instead of `(1 - cos_sim).mean()`.

**Critical constraint**: `target_sim[0]` MUST be 1.0 (not < 1.0). At v=0, within-class cosine similarity is already ~0.981. Setting `target_sim=0.90` makes `F.relu(0.90 - 0.981) = 0` — the loss becomes identically zero, destroying the primary ARI driver. This was empirically confirmed: `target_sim[0]=0.90` caused ARI regression from 0.844 → 0.716.

With `target_sim[0]=1.0`: `F.relu(1.0 - cos_sim)` is equivalent to `(1.0 - cos_sim)` since cos_sim ∈ [-1, 1], preserving the original hard-pull behavior.

### n_pairs Budget

`n_pairs: 3000` provides ~1000 pairs per active level (3 active levels). Same-class pair yield per level:
- v=0: ~1000 pairs from ~1737 batch ops, k=3 → ~37 same-class pairs
- v=1: ~1000 pairs from ~1003 batch ops, k=4 → ~37 same-class pairs
- v=2: ~1000 pairs from ~578 batch ops, k=5 → ~37 same-class pairs

### Expected Outcome

- v=0 ARI should recover to ≥ 0.844 baseline (hard pull restored)
- v=1/v=2 may show improvement from deeper prefix splits (k=4, k=5)
- Net ARI target: ≥ 0.85, ideally 0.90+
- Q metric expected to remain at 2.163 ceiling (AC loss is direction-only)

---

## 22. Live ARI Integration — Specification (2026-03-23)

### Problem

ARI was only computed offline via `diagnose_direction_geometry.py`. The AQ metric (intra_sim - inter_sim) is a proxy but doesn't measure how well K-means clusters align with digit prefix classes.

### Implementation

**File**: `src/train.py` — eval block (runs every `eval_every` epochs)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

v0_mask = (vals == 0)
dir_v0 = dir_A[v0_mask].detach().cpu().numpy()
idx_v0 = idx_cat[v0_mask]
if n_v0 > 5000:
    sub = np.random.choice(n_v0, 5000, replace=False)
    dir_v0, idx_v0 = dir_v0[sub], idx_v0[sub]
labels = KMeans(n_clusters=15, n_init=3, random_state=42).fit_predict(dir_v0)
pfx3 = TERNARY.digit_prefix_class(idx_v0, 3).cpu().numpy()
ari_prefix3 = adjusted_rand_score(pfx3, labels)
```

**TensorBoard scalar**: `Direction/ARI_prefix3`

### Performance

- K-means(k=15, n_init=3) on 5000 × 60 matrix: ~50ms per eval
- Runs only every `eval_every` epochs (default 5), adding ~10ms/epoch amortized
- Zero GPU memory impact (CPU-only on detached tensors)

### Why K-means(k=15) and prefix_k=3

- v=0 has 18 distinct digit_prefix_class(k=3) values (out of 27 possible; 9 impossible since digit0 ∈ {-1, +1} at v=0)
- K-means(k=15) is close to the true number of clusters without overfitting
- ARI is invariant to label permutation
- prefix_k=3 empirically validated (ARI=0.72 at k=3 vs 0.57 at k=2)

---

## 23. Metrics Blind Spots — Audit Finding (2026-03-23)

### Training-Time Metrics Gap

| Metric | Source | Real-time? |
|--------|--------|-----------|
| Direction/AQ (intra-inter sim) | train.py | Yes |
| Direction/intra_level_sim | train.py | Yes |
| Direction/inter_level_sim | train.py | Yes |
| **Direction/ARI_prefix3** | **train.py** | **Yes (new, 2026-03-23)** |
| Per-level within-sim | diagnose_direction_geometry.py | No (offline only) |
| kNN digit overlap | diagnose_direction_geometry.py | No (offline only) |

### Per-Level Loss Metrics Not Logged

All loss classes return detailed per-level metrics in their `metrics_dict`, but `train.py` only logs aggregate loss values to TensorBoard. Per-level details (e.g., `r_v0..r_v9` from MonotonicRadialLoss, `angular_coherence_pairs` from AngularCoherenceLoss) are computed but silently discarded.

### Dataset Quality (Verified 2026-03-23)

- All 19,683 operations verified unique, correctly generated
- Valuation distribution follows geometric series: count_v = 2·3^(8-v) for v<9
- `digit_prefix_class(k)` produces perfectly uniform distributions for all k
- WeightedRandomSampler uses sqrt-inverse valuation weighting for batch balance

### Codebase Health (Verified 2026-03-23)

- Zero import errors across 9 core modules
- All 11 loss classes functional
- Full V7 factored mode support confirmed
- Architecture flow: z_r→radius, z_θ→direction, z_hyp=r*dir
- Gradient isolation (d(r)/d(z_θ)=0) confirmed by F.normalize Jacobian
