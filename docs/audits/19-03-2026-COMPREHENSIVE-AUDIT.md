# Comprehensive Training Audit — 2026-03-19

**Run**: 84 epochs observed, V6.2 architecture, config `src/presets/v6.yaml`
**Auditor**: Claude Sonnet 4.6 (automated audit)
**Scope**: Training log analysis + source code review for 5 high-leverage threads

---

## 1. Executive Summary

### Key Numbers at Epoch 84

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Loss | 2.79 | — | Converging (from 46.16) |
| Train Acc | 0.432 | — | Above chance (0.333) |
| Val Acc | 0.436 | — | Slight generalization |
| Hier A | 0.836 | — | Strong |
| Hier B | 0.839 | — | Strong |
| Q | 2.141 | ≥ 2.2 | **Stalled below target** |
| Cov | 0.001 | — | Effectively zero |
| Encoder A LR scale | 0.05 | Dynamic | **Never changed** |

### What Is Working

1. **Hierarchy structure is good.** Hier A/B both above 0.83 from epoch 10 onward. VAE-B's hierarchy (0.839) slightly leads VAE-A's (0.836), consistent with VAE-B being hierarchy-specialized.
2. **Loss trajectory is healthy.** 46.16 → 2.79 over 84 epochs, no collapse, no divergence.
3. **Generalization is marginal but real.** Val acc (0.436) consistently tracks or leads train acc (0.432) from epoch 50 onward, suggesting no overfitting.
4. **Dual VAE is functional.** Both VAEs receive gradients (V6.2 fix confirmed working).

### What Is Broken or Stalled

1. **Q is 0.06 below target (2.141 vs 2.2) and has effectively plateaued since epoch 28** (Q=2.067 at ep 28, Q=2.141 at ep 84 — only +0.074 improvement over 56 epochs).
2. **Coverage (Cov) is 0.000–0.001 throughout 84 epochs** — effectively zero, meaning the hyperbolic radial spread is maximally concentrated.
3. **Encoder A LR scale never changes from 0.05** — the StateNet controller is not transitioning states despite 84 epochs of training.
4. **Accuracy is plateauing** around 0.42–0.44, far below what a well-structured VAE should achieve at epoch 84.

### Q Plateau Analysis

Q is defined as `dist_corr + 1.5 × hierarchy_a`. With hierarchy_a ≈ 0.836 (essentially fixed from ep 16 onward), Q is almost entirely determined by dist_corr. The Q ceiling is approximately `dist_corr + 1.5 × 0.839 ≈ dist_corr + 1.259`. To reach Q=2.2, dist_corr must reach ~0.94. The geodesic loss that drives dist_corr only activates at phase_start_epoch=30 — yet even post-epoch 30, Q improvement is only +0.07 over 54 epochs. The geodesic signal is weak relative to the hierarchy losses that already dominate.

---

## 2. Training Dynamics Analysis

### Phase 1: Rapid Initialization (Epochs 0–10)

- Loss drops from 46.16 to 6.85 — 85% of total loss reduction happens here.
- Hierarchy explodes from 0.076/0.104 to 0.803/0.831 — the hyperbolic geometry and radial losses are immediately effective at establishing structure.
- Q rises from 0.099 to 1.965 — also driven almost entirely by hierarchy_a in this phase.
- Cov=0.000 throughout. The radial entropy is minimal from the start.
- **Interpretation**: The tangent_scale (0.1 init) is working — points are not saturating at 0.95. The hierarchy losses dominate and rapidly organize radial structure.

### Phase 2: Fine-Tuning Plateau (Epochs 10–30)

- Loss slows: 6.85 → 5.17. Rate of improvement drops by ~10x.
- Hierarchy stabilizes: A goes 0.803 → 0.828, B goes 0.831 → 0.835. Marginal gains.
- Q stabilizes: 1.965 → 2.075. Barely moving.
- Accuracy slowly climbs: 0.385 → 0.393. Reconstruction improving but slowly.
- **Interpretation**: The dominant radial hierarchy losses have found a local optimum. The geodesic loss has not yet activated (phase_start_epoch=30). The model is grinding on fine-grained radius adjustments.

### Phase 3: Geodesic Phase (Epochs 30–60)

- Geodesic loss activates at epoch 30. Expected to drive dist_corr and push Q toward 2.2.
- Actual Q movement: 2.075 (ep30) → 2.127 (ep60) = +0.052 over 30 epochs.
- Loss continues declining: 5.17 → 3.17.
- Accuracy improving: 0.393 → 0.421.
- **Interpretation**: The geodesic loss is contributing to reconstruction quality (loss curve continues declining) but is not substantially moving Q. Hierarchy metrics are already near their ceiling; geodesic cannot push them further because the radial hierarchy losses are numerically dominant (weight 5.0 vs geodesic 0.5).

### Phase 4: Long Plateau (Epochs 60–84)

- Q barely moves: 2.127 (ep60) → 2.141 (ep84) = +0.014 over 24 epochs.
- Loss still declining: 3.17 → 2.79, but at diminishing returns.
- Accuracy still climbing very slowly: 0.421 → 0.432.
- Cov remains 0.000–0.001.
- **Interpretation**: The model has reached a stable local optimum defined by the radial hierarchy losses. Q is stuck because dist_corr cannot improve further under the current loss regime. The 0.1 weight on geodesic vs 5.0+5.0 on radial hierarchy means the geodesic signal is ~50x weaker. This is the dominant bottleneck.

---

## 3. The 5 High-Leverage Findings

### Finding 1: PAdicGeodesicLoss Formula — `v_3(|i-j|)` vs `min(v(i), v(j))`

**Observed evidence**: Q plateaued at 2.141, well below the 2.2 target. The `distance_correlation` metric (which drives dist_corr in Q) is not improving post-epoch 60. The geodesic loss uses `v_3(|i-j|)` computed via `TERNARY.valuation(|i-j|)`.

**Root cause analysis**: The formula `v_3(|i-j|)` computes the 3-adic valuation of the *difference* between operation indices. This is the correct p-adic metric — in the 3-adic integers, `|a - b|_3 = 3^{-v_3(a-b)}`, so `v_3(|a-b|)` directly measures 3-adic closeness. The alternative `min(v(i), v(j))` would measure closeness in the 3-adic tree by looking at which subtree both elements belong to, which is a different (and arguably coarser) signal.

**The actual problem**: The formula is mathematically defensible, but there is a structural issue: the pair indices `i, j` are *batch indices* (positions in the batch, 0 to batch_size-1), not the actual operation indices from the dataset. The code uses `batch_indices[i_idx]` and `batch_indices[j_idx]` to get operation indices before computing the difference (line 188: `diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())`). This is correct — it does use actual operation indices. However, `v_3(|i-j|)` for two random operations from the full dataset of 19,683 operations will predominantly produce v=0 (since most integers have v_3=0), meaning most pairs get the same large target distance. This creates a degenerate signal where the loss is dominated by pairs that all want to be far apart, giving weak gradient signal for structure.

**Severity**: MEDIUM. The formula is not wrong, but the signal quality is poor due to the natural frequency distribution of v_3 values over random pairs.

**Recommended fix**: Stratified pair sampling in `PAdicGeodesicLoss.forward()`. Sample pairs with a target distribution: 20% same-valuation pairs, 40% adjacent-valuation pairs, 40% cross-valuation pairs. This ensures gradient signal at all levels of the hierarchy.
- **File**: `/d1/VAEs/3-adic-ml/src/losses/padic_geodesic.py`, `PAdicGeodesicLoss.forward()`, lines 170–183.
- **Alternative**: Weight pairs by rarity of valuation level — high-v pairs are exponentially rarer and should be upweighted.

---

### Finding 2: VAE-B Decoder Necessity

**Observed evidence**: VAE-B uses `coverage_weight=0.0` (fix RC4). This means `RichHierarchyLoss` contributes zero reconstruction loss for VAE-B. The `coverage_loss` branch inside `RichHierarchyLoss.forward()` computes cross-entropy but it gets multiplied by 0.0 when CombinedLoss applies weights.

**Root cause analysis**: With `coverage_weight=0.0`, decoder_B receives no direct reconstruction gradient. It receives gradient only indirectly via the hierarchy losses operating on `z_B_hyp`, which comes from encoder_B through the hyperbolic projection. The decoder_B output (`logits_B`) is passed to `RichHierarchyLoss` via `kwargs["logits"]`, but that coverage term is zeroed out. Decoder_B computes a full forward pass (64→27 linear, activations, etc.) for zero loss contribution.

**Severity**: LOW-MEDIUM (performance cost, not correctness). Decoder_B performs a full forward/backward pass contributing no loss signal. This is wasted compute. On CPU, this adds ~15% overhead to the training step.

**Recommended fix**: Either (a) completely disable decoder_B when `coverage_weight=0.0` and detach the `logits_B` path before it enters loss computation, or (b) add a small reconstruction weight to VAE-B (e.g., `coverage_weight=0.2`) to make decoder_B useful again as a complementary reconstruction pathway.
- **File**: `/d1/VAEs/3-adic-ml/src/train.py`, training loop where `losses_B` is computed.
- **File**: `/d1/VAEs/3-adic-ml/src/presets/v6.yaml`, VAE-B loss config section.

---

### Finding 3: Within-Level Variance 0.1 Coefficient — Helping or Hurting?

**Observed evidence**: Hier A plateaued from epoch 16 onward. From epoch 16 (0.814) to epoch 84 (0.836), the improvement is only +0.022 over 68 epochs. Hier B shows the same pattern (0.832 at ep16, 0.839 at ep84).

**Root cause analysis**: The variance term in `RichHierarchyLoss` (line 855: `hierarchy_loss = hierarchy_loss + 0.1 * variance_loss`) penalizes within-level scatter. This is the correct direction — tighter clusters per valuation level should improve Spearman correlation. However, the 0.1 coefficient means the variance term contributes at most ~10% of the hierarchy gradient.

The deeper issue is that the **Hier A/B metric (Spearman correlation between valuation and radius) is limited by between-level separation, not within-level variance**. Spearman rank correlation is determined by whether the *ordering* of mean radii across levels is correct, not by within-level compactness. From epoch 16 onward, the mean radii are already correctly ordered — the hierarchy metric has hit its ceiling for this loss configuration.

To push Hier A beyond 0.836, the system needs either: (a) larger between-level separation margins, or (b) better coverage of all valuation levels in each batch (the v=9 level has only 1 operation out of 19,683, so it almost never appears).

**The 0.1 coefficient**: Neither clearly helping nor hurting. It is contributing non-zero gradient to tighten clusters but has negligible effect on the plateau because Spearman correlation is already well-ordered. The variance penalty may actually be harmful at high Hier values because it pushes all same-level points toward their mean, reducing diversity within a valuation level and potentially confusing other loss terms.

**Severity**: LOW. The coefficient is architecturally sound but insufficient to break the plateau.

**Recommended fix**: Do not change the 0.1 coefficient. Instead, address the root cause: ensure all valuation levels appear in each batch. The current batch_size=512 over 19,683 operations means the single v=9 operation appears with frequency 512/19683 ≈ 2.6%, and v=8 (3 operations) appears with 7.7%. These levels are chronically underrepresented, causing poor gradient signal for the top of the hierarchy. Implement stratified sampling by valuation level.

---

### Finding 4: MetricBasedLR Stall — Encoder A Never Unfreezes

**Observed evidence**: The State column shows `A:0.05` for all 84 epochs. The config sets `encoder_a_trainable: true` initially and `fix_threshold: 0.35`, `train_threshold: 0.45`. Accuracy starts at 0.334 and ends at 0.432.

**Root cause analysis — the control logic**:

```
_compute_coverage_gate():
    if self._active['encoder_a']:
        if metrics.coverage < cfg.fix_threshold:  # 0.35
            freeze encoder_a
```

Note: the `coverage` metric here is **not** reconstruction accuracy — it is `compute_hyperbolic_coverage(z_A_hyp)`, the radial entropy metric logged as `Cov` in the training output. This is the entropy of the histogram of normalized hyperbolic radii.

Cov=0.000–0.001 throughout 84 epochs. This is far below `fix_threshold=0.35`. Therefore:
- Encoder A starts `_active=True` (config: `encoder_a_trainable: true`)
- At every epoch: `metrics.coverage (≈0.001) < fix_threshold (0.35)` → encoder A is immediately frozen
- Frozen → scale returns 0.0... but the log shows 0.05, not 0.0

Wait — the scale shown in the log is `A:0.05`, not `A:0.00`. This reveals a second issue: the LR scale shown in the log equals `config.lr_scales.encoder_a = 0.05`, meaning encoder A is being returned as `_active=True` returning 0.05, not being frozen.

Re-examining the logic: `_compute_coverage_gate` returns `0.0` only if `_active['encoder_a']` transitions to False. If coverage is always below fix_threshold, encoder A should freeze at epoch 10 (after warmup). The fact that the scale stays at 0.05 (not 0.0) means one of:
1. The `_can_change('encoder_a', epoch)` hysteresis is preventing the state change, OR
2. The `coverage` metric passed to `TrainingMetrics` is not `Cov` (the hyperbolic coverage from `compute_hyperbolic_coverage`) but instead reconstruction accuracy (0.334–0.436).

If `metrics.coverage` is reconstruction accuracy (≥0.334 from epoch 0), then:
- `coverage (0.334) < fix_threshold (0.35)` → freeze at epoch 0 would happen... but the scale shows 0.05 (active), not 0.0 (frozen).

**Most likely explanation**: The `warmup_epochs: 10` is preventing any state changes before epoch 10. After epoch 10, the initial state (`_active['encoder_a'] = True`) meets the freeze condition (coverage < 0.35), but then immediately tries to unfreeze via the hierarchy stall path. `_is_hierarchy_a_stalled()` checks if `_hierarchy_a_stall_count >= stall_patience (5)`. With hierarchy_a improving rapidly in epochs 10–16, stall_patience would not be met initially, causing oscillation between freeze/unfreeze events — but the net effect from the logged output is that encoder A is ALWAYS returning scale=0.05, which is the "active" scale.

**Alternative explanation**: There is a subtle bug in the coverage gate. After freezing encoder A (setting `_active['encoder_a'] = False`), the unfreeze condition is `coverage_ok OR hierarchy_stalled`. The hierarchy is improving through epoch 28, so hierarchy_stalled is False. But `coverage_ok = metrics.coverage >= train_threshold (0.45)` — if coverage is accuracy 0.43 at epoch 84, it never reaches 0.45, so it never unfreezes via coverage. If `Cov` is always 0.001, it would never reach 0.45 either. The perpetual oscillation: freeze → check → still low coverage → check stall → not stalled → freeze again, but then each freeze call resets `_last_change` preventing immediate unfreeze, causing encoder A to stay frozen. But the logged scale is 0.05 (active scale), not 0.0 (frozen).

**The actual bug**: The `_compute_coverage_gate` returns `(self.config.lr_scales.encoder_a, None)` when `not self._can_change(...)`. If encoder A was initialized as active (`_active['encoder_a']=True`) and hysteresis prevents a state change at each epoch, the gate returns the active scale (0.05) every time — despite coverage always being below the fix threshold. The hysteresis check `_can_change` compares `epoch - self._last_change['encoder_a']` against `hysteresis_epochs=5`. The initial `_last_change['encoder_a']` is set to `-hysteresis_epochs` = -5. So at epoch 10 (first post-warmup epoch), `10 - (-5) = 15 >= 5`, so change IS allowed. But if the state changes (freeze), `_last_change` becomes 10, and at epoch 12, `12 - 10 = 2 < 5`, so changes are blocked. The scale would be 0.0 (frozen) for epochs 12–14, then at epoch 15 the unfreeze check fires. If hierarchy is not stalled and coverage is low, it stays frozen. This creates a freeze/unfreeze oscillation invisible in the log if only even epochs are printed.

**Bottom line on Severity**: HIGH. The logged `A:0.05` every 2 epochs is suspicious and likely indicates encoder A is oscillating between 0.0 and 0.05 within consecutive epochs (the log only shows even epochs), OR encoder A is genuinely never freezing due to the hysteresis preventing the first freeze. The controller is not providing the expected dynamic adaptation.

**Recommended fix**:
1. Log the actual LR applied to encoder_a every epoch, not just even epochs, to diagnose the oscillation.
2. The `fix_threshold: 0.35` must match the units of `metrics.coverage`. Verify in `train.py` that `TrainingMetrics(coverage=...)` receives `compute_hyperbolic_coverage()` (returns 0.001) and not reconstruction accuracy (returns 0.43). If it receives accuracy, the threshold should be changed to match.
3. Add a log line showing `active_states` from `controller.update()` return value to make state transitions visible.
- **File**: `/d1/VAEs/3-adic-ml/src/train.py`, where `TrainingMetrics` is constructed.
- **File**: `/d1/VAEs/3-adic-ml/src/models/lr_controller.py`, `_compute_coverage_gate()`.

---

### Finding 5: Triple Loss Redundancy — Which Dominates?

**Observed evidence**: Three losses enforce radial ordering simultaneously:
- `rich_hierarchy`: hierarchy_weight=5.0, separation_weight=3.0 (+ coverage_weight=1.0)
- `radial`: weight=5.0, margin_weight=0.5
- `monotonic`: weight=1.0, target_loss_weight=0.5

Plus two softer ordering signals:
- `rank`: weight=0.5 (global sigmoid ranking)
- `geodesic`: weight=0.5 (pairwise distance alignment, phase_start=30)

**Root cause analysis**: All three primary losses enforce the same geometric invariant: high-valuation points near origin, low-valuation points near boundary. They differ in mechanism:

- `RichHierarchyLoss`: MSE on per-level mean radius + within-level variance + separation margins (operates on mean per level).
- `RadialHierarchyLoss`: Weighted MSE on per-point radius to target + pairwise margin loss (operates per-point with sampling).
- `MonotonicRadialLoss`: Per-level ordering enforcement via violations (operates on level-ordered means).

The effective weight ratio for hierarchy enforcement is approximately `rich_hierarchy.hierarchy(5.0) + radial(5.0) + monotonic(1.0) = 11.0` vs `geodesic(0.5) + rank(0.5) = 1.0`. The radial hierarchy signals are **11× stronger** than the distance/rank signals. This explains why Hier A quickly reaches 0.83 (driven by radial losses) but dist_corr (driven by geodesic) is insufficient to push Q above 2.14.

**Gradient competition**: `RichHierarchyLoss` and `RadialHierarchyLoss` both use precomputed target radii from the same `_exponential_target_radii` function (inner_radius=0.08, outer_radius=0.85, scale=3.0). They are pushing each point toward the same target. The only difference is that `RadialHierarchyLoss` applies per-point weights (valuation-based, exponent 0.25) while `RichHierarchyLoss` computes per-level means. These two losses are nearly duplicative.

**Which dominates**: `RichHierarchyLoss` (weight 5.0) and `RadialHierarchyLoss` (weight 5.0) jointly dominate. `MonotonicRadialLoss` (weight 1.0) provides marginal additional signal. `GlobalRankLoss` (weight 0.5) is probably redundant given the above. `PAdicGeodesicLoss` is the weakest signal at 0.5 and only activates post-epoch 30.

**Severity**: MEDIUM. The redundancy is not causing training failure but is creating an imbalanced gradient budget that caps Q below 2.2.

**Recommended fix**: To break the Q plateau, rebalance the loss weights:
- Reduce `radial.weight` from 5.0 to 1.0 (it duplicates `rich_hierarchy`)
- Increase `geodesic.weight` from 0.5 to 2.0
- Reduce `rank.weight` from 0.5 to 0.0 (fully redundant with geodesic + monotonic)
- These changes shift the hierarchy:distance ratio from 11:1 to approximately 6:2, giving geodesic signal 3× more relative influence.
- **File**: `/d1/VAEs/3-adic-ml/src/presets/v6.yaml`, loss section.

---

## 4. Loss Redundancy Analysis

### Loss Contribution Assessment

| Loss | Weight | Mechanism | Redundancy | Verdict |
|------|--------|-----------|------------|---------|
| `rich_hierarchy` (hierarchy) | 5.0 | Per-level mean MSE + variance | Overlaps radial | **Keep, reduce** |
| `rich_hierarchy` (separation) | 3.0 | Per-level margin enforcement | Overlaps monotonic | Keep |
| `rich_hierarchy` (coverage) | 1.0 | Cross-entropy reconstruction | Unique signal | Keep |
| `radial` | 5.0 | Per-point weighted MSE | Near-duplicate of rich_hierarchy | **Reduce to 1.0** |
| `radial` (margin) | 0.5 | Pairwise margin sampling | Overlaps monotonic | Marginal |
| `geodesic` | 0.5 | Poincaré distance alignment | Unique dist_corr signal | **Increase to 2.0** |
| `rank` | 0.5 | Sigmoid ranking violations | Subset of geodesic signal | Remove |
| `monotonic` | 1.0 | Per-level ordering violations | Overlaps separation | Keep (different formulation) |
| `hyperbolic_kl` | 0.01 | True VAE regularization | Unique | Keep |

### Why Both Rich Hierarchy and Radial Coexist

`RichHierarchyLoss` operates on per-level *means* — it pulls each level's centroid to target. `RadialHierarchyLoss` operates on per-point *individual* radii — it pulls each point to target. Together they should create both correct level means AND tight individual placement. However, at weight 5.0 + 5.0, they produce gradient saturation in the radial direction, leaving insufficient optimization budget for geodesic/ranking losses.

### The Missing Signal

There is no loss term that directly enforces **within-level geodesic clustering** (i.e., that two points with the same valuation should be close to each other on the manifold, not just at similar radii). High Hier A/B only requires that mean radii are correctly ordered — two points at v=3 could be at opposite poles of their radius shell. This latent space is not truly hierarchical in the 3-adic sense; it is only radially organized.

---

## 5. StateNet Controller Failure

### Summary

The controller is configured with:
- `fix_threshold: 0.35` — freeze encoder A if coverage drops below this
- `train_threshold: 0.45` — unfreeze encoder A if coverage recovers above this
- `encoder_a_trainable: true` (initial state: active)
- `warmup_epochs: 10`

The metric labeled `Cov` in the training log (from `compute_hyperbolic_coverage`) returns values in [0.000, 0.001]. This represents **radial entropy** — how uniformly distributed the embeddings are across hyperbolic radius shells.

### The Core Problem: Metric Mismatch

The `fix_threshold: 0.35` is almost certainly calibrated for **reconstruction accuracy** (which ranges 0.33–0.44 in this run), not hyperbolic coverage entropy (which ranges 0.000–0.001). If `TrainingMetrics.coverage` is set to the hyperbolic coverage value (≈0.001), then:

- `coverage (0.001) < fix_threshold (0.35)` is always True
- Encoder A would perpetually try to freeze

But the log shows scale=0.05 (active scale), not 0.0 (frozen). This means either:
1. `metrics.coverage` is reconstruction accuracy (0.334 → 0.432), which starts below 0.35 and ends below 0.45 — causing oscillation between freeze/unfreeze attempts
2. Hysteresis is masking the oscillation in even-epoch logs

In either case, the controller is not providing meaningful dynamic adaptation. The encoder A LR scale is stuck at 0.05 through 84 epochs, meaning the initial differential (A:0.05, B:0.10, P:1.00) is locked in permanently. This defeats the purpose of the controller.

### Expected Behavior vs Actual Behavior

| Epoch Range | Expected State | Actual Logged State | Discrepancy |
|-------------|---------------|---------------------|-------------|
| 0–10 | Warmup: active at 0.05 | A:0.05 | Matches |
| 10–20 | Coverage check: acc~0.385 < 0.45 train_threshold. If coverage is acc: try to freeze (0.0) | A:0.05 | **Discrepancy** |
| 20–84 | Should adapt based on hierarchy stall detection | A:0.05 unchanged | **Stuck** |

### Fix

1. Determine what metric is passed to `TrainingMetrics.coverage` in `train.py`. If it is `compute_hyperbolic_coverage` (returns ~0.001), recalibrate `fix_threshold: 0.001` and `train_threshold: 0.01`. If it is reconstruction accuracy, calibrate to `fix_threshold: 0.38` and `train_threshold: 0.45`.
2. Add explicit logging of `controller.update()` return value, including `active_states` and any events, every epoch.
3. Consider lowering `hysteresis_epochs` from 5 to 2 to allow faster state transitions.

---

## 6. Cov=0.000 Mystery

### What Is Cov Measuring?

`compute_hyperbolic_coverage` in `train.py` (lines 419–431) computes:
1. Hyperbolic radius of each point in z_A_hyp
2. Normalize: `r_norm = 1 - exp(-r_raw)` — maps [0, ∞) → [0, 1)
3. Histogram with 10 bins over [0, 1]
4. Entropy of histogram / log(10) — normalized entropy in [0, 1]

Maximum coverage (=1.0) means embeddings are uniformly spread across all radii. Minimum (=0.0) means all embeddings are in one bin.

### Why Is It Always ~0.001?

Cov ≈ 0.001 means the histogram is almost entirely concentrated in a single radial bin, equivalent to approximately zero entropy. This occurs when:

- Nearly all embeddings have very similar hyperbolic radii (the radial shell they are at has very low variance), OR
- The normalized radii are all near zero (meaning raw radii are small), putting everything in the first bin.

The second explanation is more likely given the tangent_scale=0.1 init. With small tangent scale, `expmap0` outputs are clustered near the origin. Even at epoch 84, if mean radius ≈ 0.3 (typical for tangent_scale=0.1 with 4.0 norm inputs), the normalized value is `1 - exp(-0.3) ≈ 0.26`, putting all points in bin 3 (of 10). The histogram has all mass in bins 2–4, giving low entropy.

The radial hierarchy losses do spread points across the radius range [0.08, 0.85] (inner to outer), but the *hyperbolic* (geodesic) radius can be much larger than the Euclidean radius shown in the Poincaré ball visualization. Points at Euclidean radius 0.85 have hyperbolic radius `2 * atanh(0.85) ≈ 2.38` in the curvature=1 ball. After normalization: `1 - exp(-2.38) ≈ 0.907`.

**The actual issue**: The coverage metric requires embeddings to be spread across *all* of [0, 1) in normalized radius space, not just [0.26, 0.91]. The lower bins (normalized radius < 0.2, corresponding to points very near origin) are always empty because even v=9 points only get pushed to inner_radius=0.08 Euclidean (hyperbolic radius ≈ 0.16). This creates a systematic hole in the low-radius bins, ensuring coverage entropy never approaches 1.0.

### Implication for StateNet

If `fix_threshold: 0.35` is compared to Cov ≈ 0.001, encoder A would always freeze (Cov < 0.35). The controller would be permanently stuck. **This confirms the metric mismatch hypothesis from Section 5.** The Cov metric as currently computed is not meaningful as a coverage gate signal — it will never reach 0.35 under any realistic training configuration.

### Fix

Either: (a) Remove `compute_hyperbolic_coverage` as the coverage signal for `TrainingMetrics` and use reconstruction accuracy instead, OR (b) Recalibrate `fix_threshold` to a value appropriate for radial entropy (e.g., 0.05 to freeze when entropy is very low, 0.15 to unfreeze when entropy recovers).

---

## 7. Recommended Architecture Changes

Ordered by expected Q improvement impact:

### Priority 1: Fix Coverage Metric for StateNet (Expected: Unlock controller, potential Q +0.1)

**Change**: In `train.py`, pass reconstruction accuracy (not hyperbolic coverage entropy) as `TrainingMetrics.coverage`. The fix_threshold=0.35 and train_threshold=0.45 were calibrated for accuracy, not entropy. The hyperbolic coverage entropy is a useful metric to log but should not gate the LR controller.

### Priority 2: Rebalance Loss Weights (Expected: Q +0.05–0.10 from improved dist_corr)

Reduce radial redundancy, increase geodesic signal:
- `radial.weight`: 5.0 → 1.0
- `geodesic.weight`: 0.5 → 2.0
- `rank.weight`: 0.5 → 0.0 (disable)

The hierarchy losses will still dominate (rich_hierarchy 5.0 + radial 1.0 + monotonic 1.0 = 7.0 vs geodesic 2.0), but geodesic now has meaningful budget.

### Priority 3: Stratified Batch Sampling by Valuation Level (Expected: Q +0.05)

The v=9 level (1 operation) and v=8 level (3 operations) are chronically underrepresented in random batches of 512 from 19,683. Add stratified sampling guaranteeing at minimum N samples per valuation level per batch (e.g., N=5). This would give the hierarchy losses stable gradient signal at all levels.

### Priority 4: Reduce Geodesic Phase Start (Expected: Q +0.03)

Change `phase_start_epoch: 30` to `phase_start_epoch: 10`. Hierarchy is already well-established by epoch 10 (Hier A=0.803). The geodesic loss can safely start providing dist_corr signal earlier, giving it 20 more epochs of training time.

### Priority 5: Add Within-Level Geodesic Clustering Loss (Expected: Latent quality improvement, Q potentially +0.05)

Add a loss that explicitly pushes same-valuation points together in hyperbolic space (not just to the same radius shell). This enforces true 3-adic tree structure: `v(i) == v(j)` → small pairwise geodesic distance. This would lift the ceiling on Hier A/B beyond 0.84.

### Priority 6: Freeze Decoder_B or Give It Purpose (Expected: 10–15% compute savings)

With `coverage_weight=0.0` for VAE-B, decoder_B is a pure dead weight. Either disable it or restore a small coverage weight (0.2).

---

## 8. Type Enforcement and Quality Gates

### Current State

The codebase uses:
- Type annotations throughout (CLAUDE.md confirms)
- `float64` precision enforcement for geometry/loss
- `_validate_positive`, `_validate_weight`, `_validate_radii` guard functions in loss classes

### Missing Type Contracts

**mypy rules to enforce**:
```
# mypy.ini additions:
strict = true
warn_return_any = true
disallow_untyped_calls = true
```

Key type gaps observed:
1. `MetricBasedLR.update()` returns `Dict[str, Any]` — the `Any` type erases information. Define a `ControllerState` TypedDict with explicit fields: `lr_scales`, `events`, `type`, `best_q`, `active_states`.
2. `CombinedLoss.forward()` accepts `**kwargs` — callers can silently pass wrong keyword arguments. Define an explicit `LossInputs` TypedDict.
3. `TrainingMetrics` is a dataclass — confirm all fields have non-Optional types with default values to prevent `None` propagation into the controller.

**Ruff rules to add**:
```toml
# pyproject.toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "RUF"]
extend-select = ["ANN"]  # Annotation enforcement
```

### Test Contracts to Add

1. **Coverage metric range test**: Assert `compute_hyperbolic_coverage` returns values in [0, 1]. Assert it returns > 0.1 for uniformly distributed embeddings. Prevents future recalibration confusion.

2. **StateNet transition test**: Unit test that `MetricBasedLR` with coverage=0.001 and fix_threshold=0.35 correctly freezes encoder A within 1–2 epochs post-warmup. Currently this behavior is untested and appears incorrect.

3. **Loss weight ratio test**: Assert that `geodesic.weight / (sum of radial hierarchy weights)` is within some minimum ratio (e.g., > 0.1) to prevent accidental reintroduction of the 1:11 imbalance.

4. **Valuation frequency test**: Assert that a random batch of size 512 from the full dataset has zero samples from v=9 with probability `(1 - 1/19683)^512 ≈ 0.974`. Document this in a test to motivate the stratified sampling fix.

5. **Controller metric units test**: Assert that the `coverage` field passed to `TrainingMetrics` is on the same scale as `fix_threshold` and `train_threshold`. This is the root cause of the StateNet failure.

---

## 9. Next Steps

Concrete ordered list:

1. **[Day 1] Diagnose coverage metric vs threshold mismatch.** Add a debug log line printing `metrics.coverage`, `fix_threshold`, `train_threshold`, and `active_states` every epoch. Determine whether `TrainingMetrics.coverage` receives hyperbolic entropy or reconstruction accuracy. Fix the units mismatch. This is the highest-leverage single-line fix.

2. **[Day 1] Fix the `A:0.05` logging to show actual LR.** The even-epoch logging may be masking oscillation. Log actual optimizer param group LRs every epoch, not just the scale.

3. **[Day 2] Rebalance loss weights in v6.yaml.** Set `radial.weight: 1.0`, `geodesic.weight: 2.0`, `rank.weight: 0.0`. Run 50 epochs and compare Q trajectory against this run's baseline.

4. **[Day 2] Reduce geodesic phase_start_epoch from 30 to 10.** Minor config change, low risk.

5. **[Day 3] Implement stratified valuation sampling.** Add a `StratifiedValuationSampler` that guarantees minimum N samples per valuation level per batch. Plug into the DataLoader as a custom sampler. This affects both loss signal quality and StateNet coverage gating.

6. **[Day 3–4] Disable decoder_B or restore coverage_weight for VAE-B.** Decision: if the dual-VAE hypothesis is that VAE-A learns coverage and VAE-B learns hierarchy, then VAE-B should not have a decoder at all — only an encoder + projection. Alternatively, give VAE-B a small reconstruction weight (0.1) to keep decoder_B useful.

7. **[Day 4–5] Add within-level geodesic clustering loss.** New loss class `SameValuationClusteringLoss` that samples pairs with `v(i) == v(j)` and minimizes their pairwise Poincaré distance. Weight: 1.0. This breaks the current Hier ceiling at 0.84.

8. **[Day 5] Add the quality gate tests listed in Section 8.** Particularly the StateNet transition test and coverage metric range test. These prevent regression of the bugs identified in this audit.

9. **[Day 6–7] Full 200-epoch run with all fixes applied.** Target: Q ≥ 2.2, Cov > 0.05, Acc > 0.50.

10. **[Ongoing] Version the config.** The current `v6.yaml` has `version.date: "2026-01-24"` — update to reflect the actual config state after each change. Consider naming the patched config `v6.2.yaml` to distinguish from the run analyzed in this audit.

---

## Appendix: Training Log Summary Statistics

| Phase | Epoch Range | Q Start | Q End | ΔQ | Loss Start | Loss End |
|-------|-------------|---------|-------|----|------------|----------|
| Init | 0–10 | 0.099 | 1.965 | +1.866 | 46.16 | 6.85 |
| Fine-tune | 10–30 | 1.965 | 2.075 | +0.110 | 6.85 | 5.17 |
| Geodesic | 30–60 | 2.075 | 2.127 | +0.052 | 5.17 | 3.17 |
| Plateau | 60–84 | 2.127 | 2.141 | +0.014 | 3.17 | 2.79 |

Q improvement rate:
- Phase 1 (10 epochs): +0.187/epoch
- Phase 2 (20 epochs): +0.006/epoch
- Phase 3 (30 epochs): +0.002/epoch
- Phase 4 (24 epochs): +0.001/epoch

The 186× slowdown in Q improvement rate from Phase 1 to Phase 4 confirms a structural plateau, not a learning rate issue. The optimizer is still making progress on reconstruction loss (Phase 4 loss still declining), but the hierarchy/dist_corr metrics that drive Q are saturated under the current loss configuration.

---

*Audit generated from: 84 epoch training log (epochs 0–84, every 2 epochs), source code review of `src/losses/padic_geodesic.py`, `src/losses/combined.py`, `src/models/lr_controller.py`, `src/presets/v6.yaml`, `src/train.py` (coverage function). Architecture V6.2 as of 2026-03-11 critical fixes.*
