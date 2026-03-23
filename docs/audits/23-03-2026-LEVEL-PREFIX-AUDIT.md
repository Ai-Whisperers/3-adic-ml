# Level Prefix & Soft Margin — Implementation Audit (2026-03-23)

## Summary

This audit documents the implementation of `level_prefix_k` and soft-margin cosine
similarity in `AngularCoherenceLoss`, the regression caused by the initial configuration,
the root cause analysis, and the corrective fix.

## Background

### ARI Ceiling at 0.844

Four V7 training runs established a structural ARI ceiling:

| Run | Config | ARI (K-means@15 vs prefix3) | Q |
|-----|--------|---------------------------|---|
| V7.0 baseline | No AC loss | 0.721 | 2.163 |
| V7.1 AC light | weight=0.3, prefix_k=3, phase_start=50 | 0.810 | 2.163 |
| V7.1 AC aggressive | weight=1.0, prefix_k=3, phase_start=10, n_pairs=2000 | 0.820 | 2.163 |
| V7.2 large | latent_dim=64, hidden_dim=128, 60 direction dims | 0.844 | 2.163 |

Root cause identified in `22-03-2026-IDENTITY-GEOMETRY-AUDIT.md`:
- v=0: 6 prefix_k=2 classes → already clustered well (within-sim=0.981)
- v=1: Only 2 prefix_k=2 classes of 2187 ops each → AC has minimal leverage
- v=2: Only 1 prefix_k=2 class → AC has zero leverage (all ops share same prefix)
- v=3+: Already near-perfect clustering (within-sim≥0.99)

### Proposed Solution

1. **`level_prefix_k`**: Per-level prefix depth `[3, 4, 5, 0, 0, 0, 0, 0, 0, 0]`
   - v=0 → k=3 (27 classes of ~486 ops)
   - v=1 → k=4 (18 classes of ~243 ops, since digit0 is fixed to -1)
   - v=2 → k=5 (27 classes of ~54 ops, since digit0,digit1 fixed to -1)
   - v=3+ → skip (already converged)

2. **Soft margin**: `F.relu(target_sim - cos_sim)` instead of `(1 - cos_sim)`
   - Stops gradient once pair similarity exceeds target
   - Preserves reconstruction diversity at direction-diverse levels

## Implementation (Commit af4847e)

### Files Modified

1. **`src/losses/padic_geodesic.py`** — `AngularCoherenceLoss` class:
   - Added `level_prefix_k: Optional[List[int]]` parameter
   - Added `target_sim: Union[float, List[float]]` parameter
   - `forward()` now processes levels independently when `level_prefix_k` is set
   - Per-level pair sampling: `n_pairs // n_active_levels` pairs per level
   - Soft margin: `F.relu(target_sim_v - cos_sim).mean()` per level
   - Fallback path: when `level_prefix_k=None`, uses global `prefix_k` (backward compat)

2. **`src/losses/combined.py`** — Wiring:
   - Passes `level_prefix_k` and `target_sim` from YAML to `AngularCoherenceLoss`

3. **`src/presets/v7_large.yaml`** — Config:
   - `level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]`
   - `target_sim: [0.90, 0.80, 0.70, 0, 0, 0, 0, 0, 0, 0]` ← **THIS CAUSED REGRESSION**

## First Run Results — REGRESSION

| Metric | V7.2 (before) | V7.2+level_prefix (after) | Delta |
|--------|---------------|--------------------------|-------|
| Q | 2.163 | 2.163 | 0 |
| Coverage | 0.997 | 0.997 | 0 |
| ARI (prefix3) | 0.844 | 0.716 | **-0.128** |
| Epoch (best Q) | 340 | 435 | +95 |

### Root Cause of Regression

**`target_sim[0] = 0.90` turned off the AC signal at v=0.**

The v=0 within-class cosine similarity was already 0.981 at baseline. With
`target_sim=0.90`, `F.relu(0.90 - 0.981) = F.relu(-0.081) = 0` — the loss
was **identically zero** for nearly all v=0 pairs throughout training.

This is the opposite of what we wanted. The v=0 hard pull `(1 - cos_sim)` was
the primary driver of the 0.721→0.844 ARI improvement across runs. Setting
`target_sim=0.90` removed that driver entirely.

### Corrective Fix

```yaml
# BEFORE (broken):
target_sim: [0.90, 0.80, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# AFTER (fixed):
target_sim: [1.0, 0.85, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Why `target_sim[0] = 1.0` restores the original behavior:**
- `F.relu(1.0 - cos_sim)` where cos_sim ∈ [-1, 1]
- `1.0 - cos_sim ∈ [0, 2]`, always positive
- `F.relu` is a no-op → equivalent to `(1.0 - cos_sim).mean()` (original formula)

Additionally, `n_pairs` bumped from 2000→3000 to give ~1000 pairs per active
level (with 3 active levels). This improves the same-class pair yield:
- v=0: ~1000 pairs from ~1737 batch ops, k=3 → ~37 same-class pairs
- v=1: ~1000 pairs from ~1003 batch ops, k=4 → ~37 same-class pairs
- v=2: ~1000 pairs from ~578 batch ops, k=5 → ~37 same-class pairs

## Metrics System Audit

### Blind Spots Identified

ARI and per-level direction quality are computed **only** in the offline diagnostic
(`diagnose_direction_geometry.py`), not during training. During training, only three
direction metrics are logged to TensorBoard:

| Metric | Source | Real-time? |
|--------|--------|-----------|
| Direction/AQ (intra-inter sim) | train.py | Yes |
| Direction/intra_level_sim | train.py | Yes |
| Direction/inter_level_sim | train.py | Yes |
| ARI (K-means vs prefix) | diagnose_direction_geometry.py | **No** |
| Per-level within-sim | diagnose_direction_geometry.py | **No** |
| kNN digit overlap | diagnose_direction_geometry.py | **No** |

### Loss Metrics Not Logged to TensorBoard

All loss classes return detailed per-level metrics in their `metrics_dict`, but
`train.py` only logs the aggregate loss values to TensorBoard. Per-level details
(e.g., `r_v0..r_v9` from MonotonicRadialLoss, `angular_coherence_pairs` from
AngularCoherenceLoss) are computed but silently discarded.

## Dataset Quality Verification

- All 19,683 operations verified unique, correctly generated
- Valuation distribution follows geometric series: count_v = 2·3^(8-v) for v<9
- `digit_prefix_class(k)` produces perfectly uniform distributions for all k
- WeightedRandomSampler uses sqrt-inverse valuation weighting for batch balance
- No data quality issues found

## Codebase Health

All 9 core modules verified at class level:
- Zero import errors
- All 11 loss classes functional
- Full V7 factored mode support confirmed
- Architecture flow correct: z_r→radius, z_θ→direction, z_hyp=r*dir
- Gradient isolation (d(r)/d(z_θ)=0) confirmed by F.normalize Jacobian

## Expected Outcome (Post-Fix)

With `target_sim[0]=1.0`:
- v=0 ARI should recover to 0.844 baseline (hard pull restored)
- v=1/v=2 may show improvement from deeper prefix splits (k=4, k=5)
- Net ARI target: ≥ 0.85, ideally 0.90+
- Q metric expected to remain at 2.163 ceiling (loss change is direction-only)
