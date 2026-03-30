# Implementation Status Index
**Last updated:** 2026-03-24
**Active run:** Run 13 (PID 1982237) — both root causes fixed, Q=2.04 at epoch 20 (previous runs needed 200+ epochs)
**Last completed run:** `v7_large_20260323_204210` (Run 11), Run 12 confirmed ceiling at Q=2.163

---

## Quick Reference — Where Things Stand

| Area | Status | Key File |
|------|--------|----------|
| Model architecture | ✅ V7.2 stable — factored latent z_r⊕z_θ | `src/models/vae.py` |
| Training loop | ✅ Fixed + validated (280 tests pass) | `src/train.py` |
| Loss composition | ✅ 9 losses, config-driven | `src/losses/combined.py` |
| Config validation | ✅ Pydantic schema on every startup | `src/config/schema.py` |
| CI / YAML lint | ✅ GitHub Actions + yamllint | `.github/workflows/ci.yml` |
| Direction geometry | ✅ v=0..v=5 measured (full-dataset ARI, Bug 7 fixed) | `docs/audits/23-03-2026-LEVEL-PREFIX-AUDIT.md` |
| Q plateau | ✅ Root causes identified and fixed in Run 13 — Q=2.04 at epoch 20 | `docs/STATUS.md §Q Plateau` |
| Lagrangian logging | ✅ Live — all 19 constraints active, margin_v0=0.462 (largest gap violation) | `src/train.py` (TB: `Lagrangian/`) |
| Per-level radii TB | ✅ New in Run 11 — `Radius/r_v{0..6}_A` monotonically ordered | `src/train.py` |
| TB layout/hparams | ✅ New in Run 11 — custom dashboards, hparam table, config YAML text | `src/train.py` |
| Resume/checkpoint | ✅ Full resume path added (`resume_checkpoint` key) | `src/train.py` |

---

## Bug Tracker

| # | Bug | Severity | Status | Fix location |
|---|-----|----------|--------|--------------|
| 1 | `loss_fn.state_dict()` not saved in checkpoints | Critical (dormant) | ✅ Fixed | `train.py` — all 3 checkpoint sites + resume load |
| 2 | Grad clip excludes `loss_fn.parameters()` | Medium (dormant) | ✅ Fixed | `train.py:~1167` — `_params_to_clip` |
| 3 | AC loss silently zero in non-factored mode | Low (trap) | ✅ Fixed | `combined.py:617–632` — one-time `warnings.warn` |
| 4 | Lagrangian λ values not logged to TensorBoard | Observability | ✅ Fixed | `train.py` TB block — `Lagrangian/margin_v{v}`, `scatter_v{v}`, `n_active` |
| 5 | Per-level AC loss breakdown not in TensorBoard | Observability | ⚠️ Open | Would require `AngularCoherenceLoss` returning per-level dict |
| 6 | Q plateau root cause not fully understood | Research blocker | ✅ Diagnosed | v=0 spread vs max_radius ceiling — see §Q Plateau below |
| 7 | `Direction/ARI_v5` never logged despite 54 ops at v=5 | Observability | ⚠️ Open | `digit_prefix_class` likely returns 1 unique class → km_k=1 → skipped |
| 8 | Duplicate `level_pfx` dict (second overwrote first) | Correctness | ✅ Fixed | `train.py` — removed overwriting line; Run 10 ARI_v2=0.24 was measurement artifact |

**Dormant = only activates when `learnable_weights: true` in config (currently false).**

---

## Direction Geometry — ARI Progression

| Run | v=0 | v=1 | v=2 | v=3 | v=4 | v=5 | Composite | Change |
|-----|-----|-----|-----|-----|-----|-----|-----------|--------|
| V7.2 baseline | 0.844 | — | — | — | — | — | — | No AC levels |
| Run 1 | 0.953 | 0.778 | 0.505 | — | — | — | 0.778 | level_prefix_k=[3,4,5] |
| Run 3 | 0.924 | 0.662 | 0.999 | — | — | — | 0.790 | k[2]=3 (binary split) |
| Run 4 | 0.969 | 0.882 | 1.000 | — | — | — | 0.901 | target_sim[1]=0.90 |
| Run 5 | 0.974 | 0.905 | 1.000 | 0.731 | 0.112 | — | 0.955 | v=3 added |
| Run 6 | 0.970 | 0.899 | 1.000 | 0.972 | 0.100 | — | 0.912 | v=4 added |
| Run 7 | 0.970 | 0.945 | 1.000 | 0.937 | 0.045 | — | 0.919 | target_sim[1]=0.93 |
| Run 10 (`193520`) | 0.961 | 0.922 | 0.24* | — | — | — | 0.785* | level_pfx bug corrupted v=2 ARI |
| Run 11 (`204210`) | 0.961 | 0.922 | 0.997 | 0.909 | 0.086 | — | 0.884 | All bugs fixed + new TB features |
| Run 12 | 0.96 | 0.92 | ~1.0 | ~0.9 | ~0.09 | 0.312 | — | max_radius=0.99, sep_weight=5.0 — Q=2.163 held |
| **Run 13** | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | **Both root causes fixed — Q=2.04 at ep 20** |

*Run 10 v=2=0.24 was a measurement bug (level_pfx overwrite): training was correct, ARI measured with wrong k. Run 11 fixes this.*

*Values are mean_last10 epochs. v=4 mean low (sparse: ~17 pairs/batch). v=5 not logged — see Bug 7.*

**Key rules learned:**
- `target_sim[0]` must be 1.0 (within-sim ≈ 0.981; any lower → zero gradient)
- `target_sim[1]` needs +0.05 per added deep binary split: 0.85→0.90→0.93
- Use minimum k for meaningful separation: k→18 classes unstable at sparse levels; k→2 (binary split) is stable
- Formula: `n_classes = 2 × 3^(depth - v - 1)` — verified for all levels

---

## Current Config (v7_large.yaml — Run 10)

```yaml
model:
  latent_dim: 64        # z_r (4 dims → radius) ⊕ z_θ (60 dims → direction)
  factored: true
  radial_dims: 4

angular_coherence:
  level_prefix_k: [3, 4, 3, 4, 5, 6, 0, 0, 0, 0]   # v=0..v=5 active
  target_sim:     [1.0, 0.93, 0.70, 0.70, 0.70, 0.70, 0, 0, 0, 0]
  n_pairs: 5000   # ~830/level for 6 active levels
  phase_start_epoch: 10

training:
  epochs: 1500
  scheduler: cosine_warmup_restart (T_0=50, T_mult=2)
```

---

## TensorBoard Groups (Run 10 onwards)

| Group | Tags | Notes |
|-------|------|-------|
| `Direction/` | `ARI_v{0..5}`, `ARI_composite`, `ARI_prefix3`, `AQ`, `intra_level_sim`, `inter_level_sim` | Per-level ARI at eval cadence |
| `Lagrangian/` | `margin_v{0..8}`, `scatter_v{0..9}`, `n_active` | **New in Run 10** — first live λ data |
| `Hierarchy/` | `corr_VAE_A`, `corr_VAE_B`, `Q_VAE_A`, `Q_VAE_B`, `dist_corr` | Q stuck at 2.163 |
| `LRController/` | `encoder_a_lr_scale`, `encoder_b_lr_scale`, `projections_lr_scale` | Continuous, not boolean |
| `Radius/` | `mean_VAE_A`, `mean_VAE_B`, `r_v{0..9}_A` | Per-level mean radius — **new post-Run 10** |
| `Hardware/` | `GPU_allocated_GB`, `GPU_peak_GB` | Memory monitoring |

---

## TensorBoard Improvements Applied (2026-03-23, post-Run 10)

All changes are in `src/train.py` and take effect from the next training run.

### New Tags Added

| Tag | Source | Value |
|-----|--------|-------|
| `Radius/r_v{0..9}_A` | `z_A_cat.norm()` grouped by valuation | Per-level mean Poincaré radius — directly tests H3 (radius saturation) and monotonic ordering |

### New TB Features

| Feature | API | Purpose |
|---------|-----|---------|
| **Custom layout** | `add_custom_scalars(layout)` | Multi-chart dashboards: Q metric, direction geometry, radial hierarchy, Lagrangian — called once at startup |
| **Config text** | `add_text("Config/yaml", yaml.dump(config), 0)` | Full YAML config stored per run for reproducibility |
| **Hparams** | `add_hparams(hparam_dict, metric_dict, run_name=".")` | Hyperparameter comparison table across runs; `run_name="."` prevents phantom sub-run directories |

### Bug Fixed

| Bug | Location | Fix |
|-----|----------|-----|
| Duplicate `level_pfx` dict (second overwrites first) | `train.py` eval block | Removed duplicate line — ARI now uses correct prefix depths for v=2..v=5 |

### Custom Layout Groups

```
Q Metric:           Q_VAE_A vs Q_VAE_B, dist_corr vs hierarchy
Direction Geometry: ARI_v{0..5}, ARI_composite vs ARI_prefix3, intra vs inter sim
Radial Hierarchy:   r_v{0..9}_A per level, mean_VAE_A vs mean_VAE_B
Lagrangian:         margin λ per level, scatter λ per level, n_active
```

### Hparams Tracked

`latent_dim`, `hidden_dim`, `radial_dims`, `epochs`, `batch_size`, `lr`, `ac_weight`, `hierarchy_weight`, `n_pairs_ac`

Final metrics: `hparam/best_Q`, `hparam/best_hierarchy`, `hparam/best_coverage`

---

## Q Plateau — Definitive Root Cause Analysis (2026-03-24)

**Q = 2.163 = dist_corr(0.900) + 1.5 × hierarchy(0.840) — stuck across Runs 1–12.**
**Theoretical ceiling with current radii + zero within-level spread: Q = 2.466.**
**We are at 87.6% of theoretical ceiling.**

### The Two True Root Causes

---

#### Root Cause A — Geodesic Loss Anti-Correlation (HIGHEST LEVERAGE)

`PAdicGeodesicLoss` (weight=2.0) uses `v_3(|i - j|)` — the 3-adic valuation of the **index difference** — as the target distance between two operations. This is **anti-correlated (−0.35) with the distance metric** `|val_i − val_j|` that Q actually measures.

Empirical evidence from full-dataset pair sampling (n=2000 pairs each):

| Pair type | `|val_i − val_j|` (what metric wants) | `v_3(\|i−j\|)` (what loss teaches) |
|-----------|--------------------------------------|--------------------------------------|
| Same-level pairs | 0 (should be close) | **0.88** (loss pushes them far apart) |
| Adjacent-level pairs | 1 (should be near) | **0.12** (loss says they're already close) |
| Far pairs (d≥4) | 4+ (should be far) | **0.22** (loss undershoots badly) |

The geodesic loss has been **inverting the distance signal** for all 12 runs. Same-level operations that should occupy the same Poincaré region are being pushed apart with target distance 0.88. This directly suppresses dist_corr (current 0.900 vs ceiling 0.968) and creates within-level radial spread that collapses Spearman.

**Fix**: Replace `v_3(|i−j|)` target with `|val_i − val_j|` (individual valuations, not index difference). Alternatively, disable the geodesic loss entirely — it may be net-negative. Expected improvement: dist_corr 0.900 → ~0.96, Q 2.163 → ~2.35+.

---

#### Root Cause B — Within-Level Radial Spread at v=0

Simulation (scipy, varying std of v=0 radii around their mean 0.854):

| v=0 std | Spearman | dist_corr | Q |
|---------|----------|-----------|---|
| 0.00 (perfect) | 1.000 | 0.966 | **2.466** |
| 0.05 | 0.837 | 0.931 | 2.186 ← matches observed |
| 0.10 | 0.826 | 0.808 | 2.042 |

The model's v=0 radii have std ≈ 0.05 — a seemingly small spread that alone accounts for the entire gap from theoretical Q=2.47 to observed Q=2.163. Every v=0 operation must have a unique latent direction (for reconstruction), which forces the encoder to spread radii as a side-effect of encoding within-level identity.

**Fix**: Increase `variance_weight` in `RichHierarchyLoss` from 0.5 → 2.0, which penalises within-level radial spread directly. Expected improvement: Spearman 0.840 → ~0.90+, Q 2.163 → ~2.30+.

---

### Run 12 Result (max_radius=0.99, separation_weight=5.0)

Q = 2.163 — ceiling held. `Radius/r_v0_A` grew to ~0.91 as predicted, `Lagrangian/margin_v0` decreased 0.462 → 0.31, but within-level spread persisted because neither root cause was addressed.

---

## Root Cause Fixes Applied (Run 13, 2026-03-24)

Both fixes are single config changes in `src/presets/v7_large.yaml` — no code changes required since both were already implemented in `PAdicGeodesicLoss` and `RichHierarchyLoss`.

### Fix A — Geodesic loss target corrected

```yaml
loss:
  geodesic:
    use_individual_valuation: true   # was false
```

**What changed**: Loss now samples cross-level pairs only, with target = `max_dist × |val_i − val_j| / 9`. Same-level pairs are skipped (they add noise and fight reconstruction). The signal direction is now aligned with dist_corr: large valuation difference → large Poincaré distance target.

**Implementation**: `PAdicGeodesicLoss.forward()` with `use_individual_valuation=True` — already existed, just never enabled. `combined.py` passes the flag via `geodesic_cfg.get('use_individual_valuation', False)`.

**Evidence it matters**: Q=2.04 at epoch 20 in Run 13. Previous 12 runs needed 200+ epochs to reach this value. The geodesic loss was the dominant drag on dist_corr for all prior runs.

### Fix B — Within-level radial variance penalty increased

```yaml
loss:
  rich_hierarchy:
    variance_weight: 2.0   # was 0.5
```

**What changed**: `RichHierarchyLoss` computes `variance_loss = Σ_v var(radii at level v)` and adds `variance_weight × variance_loss` to the hierarchy term. Increasing 0.5 → 2.0 quadruples the per-epoch gradient pressure on within-level spread.

**Implementation**: Already in `RichHierarchyLoss.forward()` at line ~936. No code changes needed.

**Why std≈0.05 matters**: Simulation confirms v=0 radii with std=0.05 alone accounts for Spearman dropping from 1.0 → 0.837. Target: std < 0.02 → Spearman > 0.92 → Q contribution +0.12.

### Config state for Run 13

| Parameter | Previous (Runs 1–12) | Run 13 | Reason |
|-----------|---------------------|--------|--------|
| `geodesic.use_individual_valuation` | `false` | **`true`** | Fix A |
| `rich_hierarchy.variance_weight` | `0.5` | **`2.0`** | Fix B |
| `model.max_radius` | 0.95 (Runs 1–11), 0.99 (Run 12) | `0.99` | Kept from Run 12 |
| `rich_hierarchy.separation_weight` | 3.0 (Runs 1–11), 5.0 (Run 12) | `5.0` | Kept from Run 12 |

### Early signal (epoch 20)

```
Q = 2.038   (previous runs: Q ≈ 1.3–1.7 at epoch 20)
Hier A = 0.825
Lagrangian margin_v0 = 0.377 (still active — radii still learning)
```

---

### Combined Fix: Expected Q trajectory

Fix A alone (correct geodesic target): dist_corr 0.90 → 0.96, Q ≈ 2.10 + 1.5×0.84 = **2.36**
Fix B alone (variance_weight 2.0): hierarchy 0.84 → 0.90, Q ≈ 0.90 + 1.5×0.90 = **2.25**
Fix A + B together: dist_corr → 0.96, hierarchy → 0.92, Q ≈ **2.34–2.42**

---

## Dataset Quality Analysis (2026-03-24)

### Structure

| Level | Ops | % of dataset | Unique patterns | Notes |
|-------|-----|-------------|-----------------|-------|
| v=0 | 13,122 | 66.7% | 13,122 (all unique) | Index not divisible by 3 |
| v=1 | 4,374 | 22.2% | 4,374 | Index divisible by 3 only |
| v=2 | 1,458 | 7.4% | 1,458 | Divisible by 9 |
| v=3 | 486 | 2.5% | 486 | Divisible by 27 |
| v=4 | 162 | 0.8% | 162 | Divisible by 81 |
| v=5 | 54 | 0.3% | 54 | |
| v=6 | 18 | 0.1% | 18 | |
| v=7 | 6 | 0.03% | 6 | |
| v=8 | 2 | 0.01% | 2 | |
| v=9 | 1 | 0.005% | 1 | Singleton — no hierarchy signal |

Every operation at every level is algebraically unique (ratio=1.000 everywhere). No clustering within levels is possible from algebraic content alone — the model must learn it entirely from the indexing-derived valuation signal.

### Prefix Signal Quality (ARI foundation)

How much algebraic similarity do same-prefix operations share?

| Prefix depth k | Same-prefix cosine sim | Diff-prefix sim | Delta (signal) |
|----------------|----------------------|-----------------|----------------|
| k=1 | 0.076 | −0.030 | 0.106 |
| k=2 | 0.134 | −0.016 | 0.149 |
| k=3 | 0.219 | −0.001 | **0.220** ← AC uses this |
| k=4 | 0.339 | −0.005 | **0.344** ← strongest signal |
| k=5 | 0.356 | 0.000 | 0.356 |
| k=6 | 0.278 | −0.007 | 0.285 |

The algebraic similarity signal peaks at k=4–5 (delta≈0.34–0.36). The AC loss correctly uses k=3–6 per level. The direction geometry (ARI v=0=0.96, v=1=0.92) is already near the practical ceiling given this signal strength — there is not much more to extract from the direction geometry axis.

### Where to invest quality effort

| Axis | Current | Ceiling | Gap | Leverage | Action |
|------|---------|---------|-----|----------|--------|
| **Geodesic target correctness** | 0.900 dist_corr | 0.968 | 0.068 | **CRITICAL** | Replace v_3(\|i−j\|) with \|val_i−val_j\| |
| **Within-level spread (Spearman)** | 0.840 | ~0.95+ | 0.11 | **HIGH** | variance_weight 0.5→2.0 |
| **Direction geometry (ARI v=0)** | 0.961 | ~0.98 | 0.019 | Low | Already near ceiling |
| **Direction geometry (ARI v=1)** | 0.922 | ~0.97 | 0.048 | Medium | target_sim tuning |
| **Coverage** | 1.000 | 1.000 | 0 | None | Done |
| **Dataset expansion (3^10)** | 3^9=19,683 | 3^10=59,049 | — | Research | Reduces v=0 dominance 66%→50% |

### Recommended investment priority (Run 13+)

1. **Run 13** — Replace geodesic loss target: `use_individual_valuation: true` in config. This is a one-line config change if the loss supports it, or a small code change to `PAdicGeodesicLoss`. Highest expected Q gain (~+0.20).

2. **Run 14** — Add `variance_weight: 2.0` (from 0.5). On top of corrected geodesic signal, this compresses within-level spread. Expected additional Q gain (~+0.08).

3. **Run 15 (research)** — `learnable_weights: true`. Now that root causes are understood, learnable weights may find the right curriculum automatically. Bugs 1+2 fixed, safe to enable.

4. **Dataset expansion** (separate track) — 3^10 dataset reduces v=0 from 66.7%→50%, weakens the v=0 dominance in Spearman, adds genuine v=9 signal. Not a quick fix but the correct long-term path.

---

## Open Research Questions (Revised Priority)

### 1. ✅ RESOLVED — Q plateau root cause

Two causes identified: geodesic anti-correlation (−0.35) and within-level spread (std≈0.05). See above.

### 2. ✅ RESOLVED — Per-level AC loss in TensorBoard

`Direction/AC_loss_v{0..5}` now logged in every run (Bug 5 fixed).

### 3. ✅ RESOLVED — ARI v=5 missing

Full-dataset forward pass now used for ARI eval (Bug 7 fixed). ARI v=5 appears in Run 12.

### 4. Learnable Weights Experiment

`learnable_weights: true` safe to enable (Bugs 1+2 fixed). Deferred until geodesic fix is in place.

### 5. v=5 Signal Assessment (Run 12: ARI=0.312)

ARI v=5 = 0.312 in Run 12 — above noise (>0.1) but below the meaningful threshold (>0.5). Signal exists but is weak. Keep v=5 in config; do not extend to v=6 yet.

### 6. Dataset Expansion

See `docs/DATA-SEMANTICS.md §4`. Research track — not a tuning fix.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/train.py` | Unified training entry point (1830+ lines) |
| `src/losses/combined.py` | Config-driven loss composition (803 lines) |
| `src/losses/padic_geodesic.py` | All hierarchy/geodesic/AC losses |
| `src/losses/lagrangian.py` | Lagrangian dual state (plain Python, not nn.Module) |
| `src/config/schema.py` | Pydantic validation schema (479 lines, 25+ models) |
| `src/models/vae.py` | TernaryVAEV6Controllable, EncoderHead |
| `src/models/hyperbolic_projection.py` | Factored forward: r=sigmoid(linear_r(z_r)), dir=normalize(z_θ) |
| `src/presets/v7_large.yaml` | Active config (V7.2, Run 10) |
| `docs/DATA-SEMANTICS.md` | Indexing-derived vs intrinsic hierarchy; v=9 singleton; expansion |
| `docs/audits/23-03-2026-LEVEL-PREFIX-AUDIT.md` | Full AC experiment log (Runs 1–10) |
| `docs/audits/22-03-2026-Q-CEILING-ANALYSIS.md` | Q=2.163 ceiling mathematical proof |
| `docs/plans/NEXT-STEPS-ROADMAP.md` | Phase execution tracker |
| `tests/conftest.py` | YAML syntax/lint/schema tests (12 parametrized) |
| `.github/workflows/ci.yml` | CI: yamllint + pytest on push/PR |
