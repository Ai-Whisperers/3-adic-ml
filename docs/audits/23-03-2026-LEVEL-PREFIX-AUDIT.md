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

## Live ARI Integration (Closing the Blind Spot)

### Problem

ARI was only computed offline via `diagnose_direction_geometry.py`, meaning we couldn't
track direction clustering quality during training. The AQ metric (intra_sim - inter_sim)
is a proxy but doesn't measure how well K-means clusters align with digit prefix classes.

### Implementation

**File:** `src/train.py`

Added lightweight ARI computation in the eval block (runs every `eval_every` epochs):

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# After AQ metric computation (uses existing dir_A, vals from AQ block)
v0_mask = (vals == 0)
dir_v0 = dir_A[v0_mask].detach().cpu().numpy()
idx_v0 = idx_cat[v0_mask]
if n_v0 > 5000:  # Subsample for speed
    sub = np.random.choice(n_v0, 5000, replace=False)
    dir_v0, idx_v0 = dir_v0[sub], idx_v0[sub]
labels = KMeans(n_clusters=15, n_init=3, random_state=42).fit_predict(dir_v0)
pfx3 = TERNARY.digit_prefix_class(idx_v0, 3).cpu().numpy()
ari_prefix3 = adjusted_rand_score(pfx3, labels)
```

**TensorBoard scalar:** `Direction/ARI_prefix3`

### Performance Impact

- K-means(k=15, n_init=3) on 5000 × 60 matrix: ~50ms per eval
- Runs only every `eval_every` epochs (default 5), adding ~10ms/epoch amortized
- Zero GPU memory impact (computation is CPU-only on detached tensors)

### Why K-means(k=15) and prefix_k=3

- v=0 has 18 distinct digit_prefix_class(k=3) values (out of 27 possible; 9 are impossible
  since digit0 ∈ {-1, +1} at v=0, never 0)
- K-means(k=15) is close to the true number of clusters without overfitting
- ARI is invariant to label permutation, so K-means labels don't need to match prefix IDs
- prefix_k=3 was empirically validated as the right granularity (ARI=0.72 at k=3 vs 0.57 at k=2)

## Post-Fix Training Results (Run v7_large_20260323_072059)

### Configuration

- `target_sim: [1.0, 0.85, 0.70, 0, 0, 0, 0, 0, 0, 0]` (v=0 hard pull restored)
- `level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]`
- `n_pairs: 3000`, 800 epochs, batch_size=4096
- TensorBoard fully operational (tensorboard 2.20.0 installed, 89 scalar tags)
- Live ARI logged as `Direction/ARI_prefix3` every 5 epochs (161 data points)

### Final Metrics

| Metric | Value |
|--------|-------|
| Q (VAE-A) | 2.156 |
| Coverage | 0.979 |
| ARI (prefix3, final) | 0.822 |
| ARI (prefix3, peak) | **0.859** (epoch 700) |
| ARI (last 20 evals, mean±std) | 0.837 ± 0.009 |
| AQ | 0.677 |

### ARI Trajectory Analysis

The live ARI data reveals four distinct phases:

1. **Cold start (epoch 0–50):** ARI near zero, rising slowly to ~0.15.
   AC loss is active from epoch 10 but the model is still learning basic reconstruction.

2. **Rapid climb (epoch 55–80):** ARI jumps from 0.23 to 0.67 in 25 epochs.
   This coincides with reconstruction stabilizing (coverage crosses 0.95).

3. **Noisy plateau (epoch 80–200):** ARI oscillates between 0.61 and 0.83.
   K-means is sensitive to initialization at this stage; clusters are forming but unstable.
   Notable dips to 0.61 (epoch 85) and 0.64 (epoch 95) before settling.

4. **Stable plateau (epoch 200–800):** ARI converges to 0.837 ± 0.009.
   The peak of 0.859 at epoch 700 is within noise of the plateau.
   Occasional dips to ~0.72 (epochs 350, 390, 405, 625) are K-means stochasticity,
   not model regressions — the AQ metric remains stable through these dips.

### Comparison to Previous Runs

| Run | ARI (diagnostic) | ARI (live final) | ARI (live peak) | Q |
|-----|-------------------|-------------------|------------------|---|
| V7.0 baseline | 0.721 | — | — | 2.163 |
| V7.1 AC light | 0.810 | — | — | 2.163 |
| V7.1 AC aggressive | 0.820 | — | — | 2.163 |
| V7.2 large | 0.844 | — | — | 2.163 |
| V7.2+level_prefix (broken) | 0.716 | — | — | 2.163 |
| V7.2+level_prefix (fixed, no TB) | 0.845 | — | — | 2.163 |
| **V7.2+level_prefix (fixed, TB)** | — | **0.822** | **0.859** | **2.156** |

### Verdict

**The `level_prefix_k` + soft-margin implementation did NOT improve ARI beyond the V7.2
baseline of 0.844.** The stable plateau of 0.837 ± 0.009 is statistically indistinguishable
from (or slightly below) the previous best. The deeper prefix splits at v=1 (k=4) and
v=2 (k=5) did not provide measurable benefit because:

1. **v=0 dominates ARI:** 66% of all data is v=0. The K-means ARI metric primarily
   reflects v=0 clustering quality. v=1/v=2 improvements are invisible in this metric.

2. **The live ARI measures v=0 only:** The K-means(k=15) is run on v=0 ops exclusively.
   Even if v=1/v=2 direction clustering improved, the current metric cannot detect it.

3. **ARI ceiling is structural:** With 18 prefix_k=3 classes at v=0 and K-means(k=15),
   the maximum achievable ARI is bounded by the mismatch between 15 clusters and 18 classes.
   Perfect clustering would give ARI ≈ 0.90 (not 1.0) due to this k mismatch.

## Next Steps

### Option A: Fix the ARI Metric (Measure What We Changed)

The current ARI metric is blind to v=1/v=2 improvements. To evaluate level_prefix_k properly:

1. **Add per-level ARI to the training loop:** Run K-means separately on v=0, v=1, v=2
   direction vectors and log `Direction/ARI_v0`, `Direction/ARI_v1`, `Direction/ARI_v2`.
   Use `digit_prefix_class(k=level_prefix_k[v])` as ground truth for each level.

2. **Increase K-means k to 18 for v=0:** Match the true number of prefix_k=3 classes
   at v=0 (there are 18, not 15). This removes the k mismatch ceiling.

### Option B: Increase AC Strength at v=1/v=2

The soft margin may be too permissive at v=1 (target_sim=0.85) and v=2 (target_sim=0.70).
Try `target_sim: [1.0, 0.95, 0.85, 0, ...]` to push harder.

### Option C: Weighted Composite ARI

Compute ARI per level and combine: `ARI_composite = 0.6*ARI_v0 + 0.3*ARI_v1 + 0.1*ARI_v2`.
This makes v=1/v=2 improvements visible in a single number.

### Recommendation

**Do Option A first** (fix the metric), then re-evaluate whether level_prefix_k is helping.
It's possible the implementation is working correctly but we can't see it because the
metric only measures v=0. If per-level ARI shows v=1/v=2 improvement, the feature is
validated. If not, try Option B.
