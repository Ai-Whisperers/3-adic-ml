# P-Adic VAE Architecture (V6.2)

## Architecture Summary

**Dual VAE + True Hyperbolic Geometry + LR Controller**

### Core Components

| Component | Structure | Purpose |
|-----------|-----------|---------|
| **VAE-A** | Encoder 9→128→64, Decoder 16→64→27 | Coverage (reconstruction) |
| **VAE-B** | Same structure, independent weights | Hierarchy learning |
| **Hyperbolic Projection** | Tangent net + expmap0 → Poincaré ball | True hyperbolic mapping |
| **LR Controller** | MetricBasedLR with Q-gated thresholds | Dynamic LR scale control |

### What Makes It "P-Adic"

1. **Data**: All 19,683 ternary operations (3^9) with values {-1, 0, 1}
2. **3-adic valuation**: v_3(n) measures divisibility of the **index integer n** by powers of 3
3. **Geometric encoding**: High valuation → near origin, low valuation → near boundary

> **Important**: `v_3(n)` is a property of the **index n as an integer**, not of the algebraic
> content of the 9-digit operation it represents. All hierarchy losses are indexing-derived.
> Only `AngularCoherenceLoss` uses intrinsic digit content (via `digit_prefix_class`).
> See `docs/DATA-SEMANTICS.md` for full analysis including the v=9 singleton convention
> and dataset expansion options.
4. **Loss aligns**: Poincaré distances to 3-adic valuations (ultrametric → hyperbolic)

### Architecture Flow (V6.0 - True Hyperbolic)

```
Input (9 values, {-1,0,1})
    |
+-- Encoder A --+    +-- Encoder B --+
|  9->128->64   |    |  9->128->64   |
|  mu_A, sig_A  |    |  mu_B, sig_A  |
+------+--------+    +------+--------+
       |                    |
   z_tangent (16-dim)   z_tangent        <- Tangent space at origin (Euclidean)
       |                    |
   +----------------------------+
   |  DualHyperbolicProjection  |
   |  tangent_net + expmap0     |
   +----------------------------+
       |                    |
   z_A_hyp              z_B_hyp          <- Poincaré manifold points
       |                    |
   logmap0              logmap0          <- Back to tangent space
       |                    |
   Decoder A            Decoder B
       |
   Reconstruction logits
```

---

## Loss System (Config-Driven)

### Primary Losses

| Loss | Purpose | Implementation |
|------|---------|----------------|
| **RichHierarchyLoss** | Unified hierarchy + coverage + separation | Per-level mean radii with margins |
| **PAdicGeodesicLoss** | Poincaré distance alignment | Random pairs within batch |
| **RadialHierarchyLoss** | Direct radius enforcement | Weighted MSE to target radii |
| **GlobalRankLoss** | Soft ranking violations | Sigmoid-based differentiable ranking |
| **MonotonicRadialLoss** | Per-level ordering | Groups by valuation, enforces r[v] > r[v+1] |

### Design Decisions

**Pair Sampling**: Uses random within-batch sampling (not synthetic stratified). This is intentional:
- Uses **real embeddings** from actual batch data
- `MonotonicRadialLoss` handles per-valuation-level structure explicitly
- Avoids fabricating artificial index pairs

**Per-Level Metrics**: Tracked via `MonotonicRadialLoss`:
- Logs `r_v0`, `r_v1`, ..., `r_v9` (mean radius per level)
- Tracks `margin_violations` and `mean_violation_magnitude`
- In hyperbolic geometry, radial ordering implies distance ordering

---

## LR Controller (Option C)

Training uses **continuous LR scales** via `MetricBasedLR` (not boolean freeze/unfreeze):

| Component | LR Scale | Role |
|-----------|----------|------|
| `encoder_a` | 0.05× base | Coverage encoder (slowest learner) |
| `encoder_b` | 0.1× base | Hierarchy encoder (medium learner) |
| `projections` | 1.0× base | Hyperbolic projection (fastest adapter) |

**Q-Metric**: `Q = dist_corr + 1.5 × |hierarchy|` guides threshold annealing.

**How it works**: Each epoch, `MetricBasedLR.update(metrics)` returns LR scales, then `update_optimizer_lr_scales()` applies them to optimizer param groups. Setting LR=0 effectively freezes a component.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | `TernaryVAEV6`, `TernaryVAEV6Controllable`, `EncoderHead` |
| `src/models/lr_controller.py` | `MetricBasedLR`, `TrainingMetrics`, LR scale control |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/config/statenet_config.py` | `StateNetConfig` dataclass (configuration) |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/core/ternary.py` | Immutable 3-adic field logic |
| `src/losses/padic_geodesic.py` | All hierarchy/geodesic losses |
| `src/losses/combined.py` | Config-driven loss composition |
| `src/train.py` | Unified training entry point |

## Training

```bash
python src/train.py --config src/presets/v6.yaml
```

### Config Keys (V6.0)

| Key | Purpose |
|-----|---------|
| `anchor_checkpoint.path` | Pre-trained weights to start from |
| `option_c.encoder_a_lr_scale` | LR multiplier for coverage encoder (default: 0.05) |
| `option_c.encoder_b_lr_scale` | LR multiplier for hierarchy encoder (default: 0.1) |
| `option_c.projections_lr_scale` | LR multiplier for projections (default: 1.0) |
| `statenet.initial.encoder_a_trainable` | Initial trainability for encoder A |
| `model.name` | `TernaryVAEV6Controllable` |

### Training Loop Algorithm (V6.2)

```
For each epoch:
    1. Training phase (BOTH VAEs contribute to loss):
       - Forward: out = model(batch_ops)
       - Loss A: losses_A = loss_fn(z_A_hyp, batch_idx, logits_A, batch_ops, epoch,
                                     mu=mu_A, logvar=logvar_A)
       - Loss B: losses_B = loss_fn(z_B_hyp, batch_idx, logits_B, batch_ops, epoch,
                                     mu=mu_B, logvar=logvar_B)
       - Total: loss = losses_A["total"] + losses_B["total"]
       - Backward + gradient clipping
       - Optimizer step

    2. Validation phase:
       - Collect z_A_hyp, z_B_hyp from both VAEs
       - Compute hierarchy metrics (Spearman correlation, Q metric)

    3. LR Controller update:
       - metrics = TrainingMetrics(coverage, hierarchy_a/b, dist_corr, q_value, grad_norm)
       - controller_state = lr_controller.update(metrics)
       - update_optimizer_lr_scales(optimizer, base_lr, lr_scales)

    4. Checkpointing:
       - Save best_Q.pt when Q improves
       - Periodic checkpoints every save_every epochs
```

---

## P-Adic VAEs

- **Core idea**: Dual VAE + Controller where latents live in **ultrametric p-adic space (p=3)**, inducing hierarchy by construction
- **Geometry**: Discrete → continuous bridge via **p-adic → hyperbolic projections** (Poincaré ball with expmap0/logmap0)
- **Dynamics**: Dual-VAE (coverage/hierarchy) with LR controller; ELBO stability via geometry-aware optimization
- **Evidence**: Empirical correlations between ultrametric distance and semantic/functional similarity
- **Applications**: Hierarchical AI, neurosymbolic AI, semantic compression, protein/codon pipelines
- **Constraints**: RTX 3050 6GB compatible, aggressive memory discipline
- **Philosophy**: Meaning = geometry; hierarchy **emerges structurally**, not memorized

---

## Configuration Details

### Config Structure (Nested)

The YAML config uses **nested structure** (not flat keys):

```yaml
statenet:
  enabled: true
  initial:
    encoder_a_trainable: false  # Set to true for both VAEs trainable
    encoder_b_trainable: true
    projections_trainable: true
  coverage:
    fix_threshold: 0.995
    train_threshold: 1.0
  hierarchy:
    plateau_patience: 10
  timing:
    warmup_epochs: 10

option_c:
  enabled: true
  encoder_a_lr_scale: 0.05
  encoder_b_lr_scale: 0.1
  projections_lr_scale: 1.0
```

### Making Both VAEs Trainable

To train with BOTH encoders trainable from the start:

1. Set `statenet.initial.encoder_a_trainable: true` in YAML
2. Or programmatically:

```python
from src.config import StateNetConfig

config = StateNetConfig.from_dict(yaml_cfg.get('statenet', {}))
config.initial.encoder_a_trainable = True
config.initial.encoder_b_trainable = True
```

### Integration Pattern (train.py)

```python
from src.config import StateNetConfig
from src.models import (
    TernaryVAEV6Controllable,
    MetricBasedLR,
    TrainingMetrics,
    update_optimizer_lr_scales,
)

# 1. Load config
sn_config = StateNetConfig.from_dict(config.get('statenet', {}))

# 2. Create controller
lr_controller = MetricBasedLR(sn_config)

# 3. Create model
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    projections_trainable=sn_config.initial.projections_trainable,
)

# 4. In training loop
metrics = TrainingMetrics(epoch=epoch, coverage=cov, hierarchy_a=h_a, ...)
state = lr_controller.update(metrics)
update_optimizer_lr_scales(optimizer, base_lr, state['lr_scales'])
```

### StateNetConfig Dataclass Structure

```python
@dataclass
class StateNetConfig:
    enabled: bool = True
    coverage: CoverageThresholds      # fix_threshold, train_threshold, floor
    hierarchy: HierarchyThresholds    # plateau_threshold, plateau_patience, ...
    controller: ControllerThresholds  # grad_threshold, grad_patience, ...
    annealing: AnnealingConfig        # enabled, step, q_decrease_threshold
    timing: TimingConfig              # warmup_epochs, hysteresis_epochs, window_size
    lr_scales: LRScales               # encoder_a, encoder_b, projections, decoders
    initial: InitialStates            # encoder_a_trainable, encoder_b_trainable, ...
```

### See Also

- `src/README.md` - Full integration guide with code examples
- `src/presets/v6.yaml` - Reference V6.0 configuration

---

## Critical Fixes (V6.2 - 2026-03-11)

### Adversarial Audit Results & Fixes Applied

Full adversarial audit revealed 3 critical bugs + 2 design flaws. All fixed:

| Bug | Severity | Fix |
|-----|----------|-----|
| **VAE-B dead**: z_B_hyp never passed to loss | CRITICAL | Both z_A_hyp and z_B_hyp now go through CombinedLoss in train.py |
| **max_radius saturation**: all points clamped to 0.95 | CRITICAL | Added learnable `tangent_scale` (init=0.1) in HyperbolicProjection |
| **Config key mismatch**: `radial_weight` vs `weight` | CRITICAL | Fixed v6.yaml to use `weight: 5.0` |
| **No KL divergence**: model was a deterministic AE | HIGH | Wired HyperbolicKLDivergence into CombinedLoss.forward() |
| **Dead config keys**: 11+ unused YAML keys | MEDIUM | Removed zero_structure, richness_weight, annealing, encoder/decoder_dropout, logvar_min/max |

### Post-Fix Training Results (20 epochs, CPU)

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Initial radii | All 0.95 (saturated) | 0.23-0.57 (spread) |
| VAE-B gradients | 0/14 params | 14/14 params |
| Hierarchy corr | N/A (flat) | 0.83 from epoch 0 |
| Accuracy | N/A | 55% at epoch 20 |
| Loss curve | N/A | 15.6 → 2.1 (converging) |

### tangent_scale Parameter

`HyperbolicProjection` now has a learnable `tangent_scale` parameter:
- Initialized to 0.1 (prevents expmap0 saturation)
- Encoder outputs (~4.0 norm) are scaled to ~0.4, giving expmap0 radii ~0.38
- Learned during training; adapts to optimal magnitude for hierarchy

---

## Codebase Review Summary (2026-01-26)

### Architecture Verification

The entire `src/` codebase (~5000 lines, 22 files) has been reviewed. Key findings:

**Confirmed Working:**
- Option C (LR-based trainability) is fully implemented
- True hyperbolic geometry via geoopt expmap0/logmap0
- StateNetConfig → MetricBasedLR → train.py integration is correct
- All losses use proper hyperbolic distances
- TernarySpace singleton is immutable and thread-safe

**No statenet.py**: The "StateNet" system is distributed across:
- `src/config/statenet_config.py` - Configuration dataclass
- `src/models/lr_controller.py` - Decision logic (MetricBasedLR)
- `src/models/vae.py` - Component trainability
- `src/train.py` - Integration point

### Dead Code Removed (2026-01-26)

| Item | Reason |
|------|--------|
| `CheckpointCompatibilityError` | Never raised, validation not called |
| `AnnealingConfig` | Heuristic meta-control, logic never implemented |

### Design Decisions (Not Issues)

| Item | Location | Rationale |
|------|----------|-----------|
| `proj_B.learnable_curvature=False` | `hyperbolic_projection.py:238` | Intentional - both projections share A's curvature |

### New: Learnable Loss Weights (V6.1)

Loss weights can now be **trainable** using homoscedastic uncertainty weighting (Kendall et al. 2018):

```yaml
loss:
  learnable_weights: true  # Enable
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0   # Initial weight
    coverage_weight: 1.0
    separation_weight: 3.0
```

**How it works:**
- Each loss gets `nn.Parameter` log_sigma (initialized from config weights)
- Effective weight = `0.5 * exp(-2 * log_sigma)`
- Regularization `+log_sigma` (Kendall & Gal 2018) — stable equilibrium at `log_sigma = 0.5*log(loss)`
- Gradients flow through → network learns optimal balance

**Key difference from removed AnnealingConfig:**
- AnnealingConfig was heuristic (adjusted thresholds based on Q metric)
- Learnable weights are **trainable** (gradients flow, network learns)

**Usage:**
```python
loss_fn = CombinedLoss(config['loss'], curvature=1.0)

# Include loss_fn parameters in optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=base_lr
)

# Monitor learned weights
print(loss_fn.get_learned_weights())  # {'hierarchy': 4.2, 'coverage': 1.8, ...}
```

**⚠ Do not use in practice.** The formula has a fundamental instability: when any loss approaches 0 (which happens at 100% accuracy), `log_sigma → -∞ → weight → ∞`. Observed in V21.0 final: `coverage=1797, hierarchy=192`. Use fixed weights (`learnable_weights: false`) for all runs. The sign bug (`-log_sigma`) was fixed in 2026-07, but the underlying instability with near-zero losses remains. See [[v21-training-results]].

### Quick Reference

```python
# Core imports
from src.core import TERNARY, valuation, distance, target_radius
from src.geometry import hyperbolic_radius, poincare_distance, exp_map_zero
from src.losses import CombinedLoss
from src.models import TernaryVAEV6Controllable, MetricBasedLR, TrainingMetrics
from src.config import StateNetConfig, N_TERNARY_OPERATIONS

# Make both VAEs trainable
sn_config = StateNetConfig.from_dict(yaml_cfg.get('statenet', {}))
sn_config.initial.encoder_a_trainable = True  # Default is False
sn_config.initial.encoder_b_trainable = True  # Default is True

# Integration pattern
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    ...
)
controller = MetricBasedLR(sn_config)

# Training loop
state = controller.update(TrainingMetrics(...))
update_optimizer_lr_scales(optimizer, base_lr, state['lr_scales'])
```

### File Map

| Category | Files |
|----------|-------|
| **Entry** | `train.py` |
| **Models** | `vae.py`, `hyperbolic_projection.py`, `lr_controller.py` |
| **Config** | `config/statenet_config.py`, `config/constants.py` |
| **Core** | `core/ternary.py` (TernarySpace singleton) |
| **Geometry** | `geometry/poincare.py` (geoopt backend) |
| **Losses** | `losses/combined.py`, `losses/padic_geodesic.py` |
| **Utils** | `utils/checkpoint.py`, `utils/tensorboard_logger.py` |
| **Tests** | `tests/` (280 tests across 8 files) |

---

## Test Suite (280 tests)

Comprehensive test coverage following `docs/plans/TESTS_CRITICAL_TARGETS.md`.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run by tier
pytest tests/test_core_ternary.py tests/test_geometry_poincare.py -v  # Tier 1
pytest tests/test_losses.py tests/test_losses_combined.py -v          # Tier 2

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

| File | Tests | Coverage |
|------|-------|----------|
| `test_core_ternary.py` | 28 | Valuation formula, ultrametric inequality, round-trip |
| `test_core_ternary_extended.py` | 29 | Distance formula, tree structure, level consistency |
| `test_geometry_poincare.py` | 29 | exp/log composition, ball containment, triangle inequality |
| `test_geometry_poincare_extended.py` | 27 | Möbius add, geodesics, distance matrix, formula verification |
| `test_losses.py` | 64 | Gradient flow, non-negativity, monotonicity, edge cases |
| `test_losses_combined.py` | 37 | CombinedLoss, phase gating, learnable weight formula |
| `test_gradient_flow.py` | 11 | Full pipeline gradient flow, radius spread, KL integration |

### What's Tested

**Tier 1 - Mathematical Invariants (113 tests):**
- 3-adic valuation: `v_3(0)=9, v_3(1)=0, v_3(3)=1, v_3(9)=2`
- Ultrametric inequality: `d(a,c) ≤ max(d(a,b), d(b,c))`
- exp/log composition: `log_map_zero(exp_map_zero(v)) ≈ v`
- Ball containment: `||exp_map_zero(v)|| < 1`

**Tier 2 - Loss Correctness (101 tests):**
- Gradient flow through all 6 loss classes
- Loss non-negativity (all losses ≥ 0)
- Target distance monotonicity (`d(v) > d(v+1)`)
- Learnable weight formula: `w = 0.5 * exp(-2 * log_sigma)`
- Phase gating for geodesic loss

### Test Philosophy

Tests validate **mathematical invariants** and **actual computation**, not:
- Shape assertions (PyTorch guarantees these)
- Import success (Python handles this)
- Constructor return values (can't be None)

See `docs/plans/TESTS_CRITICAL_TARGETS.md` for full testing strategy.

---

## Direction Geometry — Level Prefix & Soft Margin (V7.2+ - 2026-03-23)

### ARI Ceiling & Root Cause

Four V7 runs established an ARI ceiling at 0.844. Root cause: `AngularCoherenceLoss` with global `prefix_k=3` has leverage only at v=0 (18 classes). At v=1 (2 classes) and v=2 (1 class), AC has minimal/zero leverage.

### level_prefix_k — Per-Level Prefix Depth

```yaml
angular_coherence:
  level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]  # v=0→k=3, v=1→k=4, v=2→k=5, v=3+→skip
  target_sim: [1.0, 0.85, 0.70, 0, 0, 0, 0, 0, 0, 0]  # Soft-margin targets
  n_pairs: 3000  # ~1000 per active level
```

When `level_prefix_k` is set, `AngularCoherenceLoss.forward()` processes levels independently. When `None`, falls back to global `prefix_k` (backward compatible).

### target_sim Constraint

**`target_sim[0]` MUST be 1.0** — setting it to 0.90 caused ARI regression from 0.844 → 0.716 because `F.relu(0.90 - 0.981) = 0` (v=0 within-class sim is already 0.981, so the loss becomes identically zero). With `target_sim[0]=1.0`, `F.relu(1.0 - cos_sim)` is equivalent to the original `(1.0 - cos_sim)` formula.

### Live ARI in Training Loop

ARI is now computed during training (every `eval_every` epochs) in `src/train.py`:
- Extracts v=0 direction vectors, subsamples to 5000
- K-means(k=15, n_init=3) → compared to `digit_prefix_class(k=3)` via `adjusted_rand_score`
- Logged as `Direction/ARI_prefix3` in TensorBoard
- ~50ms per eval, zero GPU impact (CPU-only on detached tensors)

### Metrics Blind Spots (Audit Finding)

Per-level loss details (`r_v0..r_v9`, `angular_coherence_pairs`) are computed by loss classes but not logged to TensorBoard — only aggregate values are logged. Per-level within-sim and kNN digit overlap remain offline-only via `diagnose_direction_geometry.py`.

### Implementation Files

| File | Change |
|------|--------|
| `src/losses/padic_geodesic.py` | `AngularCoherenceLoss`: `level_prefix_k`, `target_sim` params, per-level forward |
| `src/losses/combined.py` | Passes `level_prefix_k` and `target_sim` from YAML |
| `src/presets/v7_large.yaml` | Config for level_prefix_k, target_sim, n_pairs |
| `src/train.py` | Live ARI computation in eval block |

---

## V21.0 Training Results & V22 Plan (2026-07-06)

### V21.0 Results (run: v21.0_clean_run_20260706_070224)

Best checkpoint: **epoch 230** (`best_Q.pt`), trained for 1000 epochs (~4.25h on RTX 3050).

| Metric | V7 baseline | V21 smoke test (ep50) | V21 best (ep230) |
|--------|-------------|----------------------|-----------------|
| ARI (prefix-3, v=0) | 0.844 | 0.887 | **1.000** |
| Within-class cosine sim | — | 0.9998 | 0.9996 |
| Spearman hierarchy | — | 0.8335 | 0.8335 (ceiling) |
| Val accuracy | — | 100% | 100% |
| Best Q | — | 1.90 | 1.9755 |

**Radial hierarchy (ep230):**

| Level | Mean radius | Count |
|-------|-------------|-------|
| v=0 | 0.9039 | 13122 |
| v=1 | 0.6539 | 4374 |
| v=2 | 0.4810 | 1458 |
| v=3 | 0.3509 | 486 |
| v=4 | 0.2583 | 162 |
| v=5 | 0.1864 | 54 |
| v=6 | 0.1393 | 18 |
| v=7 | 0.0913 | 6 |
| v=8 | 0.0405 | 2 |
| v=9 | 0.0005 | 1 |

Perfect monotonic ordering across all 10 levels. v=9 is essentially at the origin.

### V21.0 Bugs Fixed

| Bug | Fix |
|-----|-----|
| `learnable_weights` sign: `-log_sigma` → all weights collapse to 0 | Changed to `+log_sigma` in `src/losses/combined.py` |
| `phase_start_epoch: 500` → Stage 2 never activated | Changed to `20` in `v21.0_clean_run.yaml` |

### Spearman 0.833 Ceiling

The Spearman correlation between valuation and embedding radius plateaus at **0.8335** within the first 20 epochs and never improves regardless of Stage 2 losses, loss weights, or training duration. This is not a bug — it reflects the limits of the Spearman proxy on this dataset. ARI=1.0 confirms that the model's angular geometry is perfect; the 0.833 ceiling is intrinsic to how Spearman measures radial ordering on the discrete v_3 distribution.

### V22 Plan

**Priority 1 — Stable baseline (disable learnable_weights):**
```yaml
loss:
  learnable_weights: false  # was true; formula unstable when loss→0
```
Run 1000 epochs with fixed weights to get a clean comparison without weight divergence noise.

**Priority 2 — Algebraic structure verification:**
Now that ARI=1.0 and radial hierarchy is perfect, the next question is whether the model captured algebraic structure beyond index-based hierarchy. Use `FiniteTernaryGroupEngine` (or equivalent) to measure:
- Do operations with the same algebraic identity map to nearby embeddings?
- Are distributive/associative orbits geometrically clustered?
- Can we decode algebraic properties from latent directions?

**Priority 3 — KL enforcement (optional):**
The model currently operates near-deterministically (KL ≈ 0.024). If a proper VAE is desired, increase `hyperbolic_kl.weight` from 0.05 and reduce `free_bits` from 0.5. Trade-off: higher KL may hurt ARI.

**Do NOT chase Spearman.** The 0.833 ceiling is architectural/data-intrinsic. Any attempt to raise it via loss tuning is futile based on V21 evidence.

---

## V23.0 Training Results (2026-07-11)

### V23.0 Results (run: v23.0_algebraic_20260711_121743)

Best checkpoint: **epoch 210** (`best_Q.pt`), trained for 1000 epochs (~4.3h on RTX 3050).

V23 adds algebraic structure signal: 4-bit signature (comm+assoc+id+abs), 12 non-trivial classes, global lookup for rare classes via `model.get_hyperbolic_representations()`.

| Metric | V21.0 (baseline) | V23.0 |
|--------|-----------------|-------|
| Val accuracy | 100% | **100%** |
| Best Q | 1.9755 (ep230) | **1.9698 (ep210)** |
| Spearman hierarchy A | 0.8335 | **0.8335** |
| Spearman hierarchy B | 0.8335 | **0.8335** |
| Monotonic ordering | PERFECT | **PERFECT** |
| Grokking events | 0 | **0** |

**Radial hierarchy (ep210):**

| Level | Mean radius | Count |
|-------|-------------|-------|
| v=0 | 0.9083 | 13122 |
| v=1 | 0.6566 | 4374 |
| v=2 | 0.4761 | 1458 |
| v=3 | 0.3465 | 486 |
| v=4 | 0.2501 | 162 |
| v=5 | 0.1858 | 54 |
| v=6 | 0.1382 | 18 |
| v=7 | 0.0978 | 6 |
| v=8 | 0.0429 | 2 |
| v=9 | 0.0009 | 1 |

Perfect monotonic ordering across all 10 levels. v=9 essentially at the origin.

### V23.0 Notes

**Q slightly lower than V21 (1.9698 vs 1.9755):** Expected. Algebraic losses compete with hierarchy signal — trade-off between algebraic structure clustering and radial ordering. Both runs hit the 0.8335 Spearman ceiling identically.

**Lagrangian dual ascent confirmed working:** 17/29 lambdas active at best checkpoint (ep 210). `lambda_margin` and `lambda_scatter` received gentle pressure (max ~0.003–0.010); `lambda_prior` stayed at zero (model satisfies μ-norm targets naturally). This was a critical fix — the dual ascent was silently broken in all prior runs due to `dual_state.update()` never being called.

**Curriculum activated correctly:** Stage 2 losses (radial, geodesic, rank, monotonic) activated at the first evaluation (val_acc=0.9998 ≥ threshold=0.80). Coverage reaches 100% and stays there.

**Best checkpoint early (ep 210):** Model converges geometrically within first ~200 epochs; subsequent training refines algebraic directions without improving Q further.

### V23 Bugs Fixed (2026-07-11 audit)

| Bug | Severity | Fix |
|-----|----------|-----|
| `min_class_size` schema key vs `min_global_size` in loss + YAML | HIGH | Renamed to `min_global_size` in `AlgebraicCoherenceLossConfig` |
| `dual_state.update()` never called | CRITICAL | Fixed in `engine.py`: violations accumulated in `train_epoch()` and routed to dual ascent every epoch |
| Fallback coverage loss crash when `logits=None` | CRITICAL | Added `and logits is not None` guard in `combined.py` |
| `active` list missing `hyperbolic_contrastive` / `surrogate_loss` | HIGH | Added via `getattr` in `combined.py` |
| `n_layers=0` silent shape mismatch in `HyperbolicProjection` | MEDIUM | Raises `ValueError` at `__init__` |

### V24 Directions

- **Measure algebraic clustering:** Now that the algebraic losses are active, evaluate whether same-signature operations are geometrically clustered in direction space. Use `AlgebraicCoherenceLoss._global_indices` to extract per-class embeddings and compute within-class vs between-class angular distances.
- **ARI on algebraic classes:** Compute ARI using `algebraic_signature` as labels (instead of `digit_prefix_class`) to quantify whether the algebraic structure is encoded in the latent directions.
- **KL enforcement (optional):** Model operates near-deterministically (KL ≈ 0.024). Increase `hyperbolic_kl.weight` and reduce `free_bits` if a proper VAE posterior is desired. Trade-off: higher KL may hurt Q.

---

## V24.0 — tangent_scale Log-Reparameterization Fix (2026-07-11)

### Root Cause: VAE-B Directional Collapse

V23.0 deep analysis revealed `tangent_scale_B` collapsed from 0.042 → ~6e-8 between epochs 25-75, causing all z_B_hyp to point in the same direction (pairwise cosine sim = 1.000000). Mechanism:

1. `tangent_scale → 0` makes `tangent_net(0 * z_θ) = tangent_net(0)` = constant for all samples
2. All directions normalize to the same vector → angular coherence loss on B trivially satisfies → zero gradient
3. Stable degenerate fixed point: once collapsed, cannot recover

VAE-A escaped by chance of initialization. Radii (via independent `linear_r`) were unaffected, preserving Spearman=0.8335 despite directional collapse.

### Fix: Log-Space Reparameterization

**File:** `src/models/hyperbolic_projection.py`

```python
# Before (collapses to 0 — degenerate fixed point):
self.tangent_scale = nn.Parameter(torch.tensor(tangent_scale_init))
z_theta_scaled = self.tangent_scale * z_theta

# After (always > 0 — no fixed point at 0):
self.log_tangent_scale = nn.Parameter(torch.tensor(math.log(tangent_scale_init)))
z_theta_scaled = self.log_tangent_scale.exp() * z_theta
```

Changes applied:
- `__init__`: store `log(tangent_scale_init)` as `log_tangent_scale` parameter
- Validation: `tangent_scale_init <= 0` raises error (was `< 0`, allowing 0)
- All 3 forward paths: `self.tangent_scale` → `self.log_tangent_scale.exp()`
- `import math` added at top of file

**Checkpoint key change:** `tangent_scale` → `log_tangent_scale` (V24 is a fresh run; no migration needed)

**Monitoring:** TensorBoard logs `log_tangent_scale` in log-space. Effective scale = `exp(logged_value)`. Expect both A and B to stay in range (-5, +1) throughout training.

### Config: `src/presets/v24.0_tangent_fix.yaml`

Identical to V23.0 in all other respects. The fix isolates the `tangent_scale` change.

Expected improvement: VAE-B develops diverse directions (ARI_B should be > 0), confirming both VAEs contribute genuinely to the hierarchy signal.
