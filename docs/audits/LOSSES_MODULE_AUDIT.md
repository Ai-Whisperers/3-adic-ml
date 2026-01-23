# Losses Module Audit: src/losses/

**Date**: 2025-01-23
**Scope**: `src/losses/__init__.py`, `combined.py`, `padic_geodesic.py`
**Lines of Code**: 297 (combined.py) + 763 (padic_geodesic.py) + 10 (__init__.py) = ~1070 total

---

## Executive Summary

The losses module provides a **comprehensive suite of p-adic/hyperbolic loss functions** that correctly use hyperbolic distance (via `poincare_distance`) for radius computation. The module is well-designed with config-driven composition, seeded generators for reproducibility, and proper ultrametric structure enforcement.

**Key Strength**: All losses correctly use `poincare_distance(z, origin)` instead of Euclidean norm for radius computation (V5.12.2 fix).

**Verdict**: The losses module is **mathematically sound and production-ready**. It correctly implements the hyperbolic supervision signal, even though the VAE architecture (per the models audit) doesn't fully utilize it.

---

## File Structure

```
src/losses/
├── __init__.py          # Clean re-exports (7 symbols)
├── combined.py          # Config-driven loss composition (297 lines)
└── padic_geodesic.py    # Individual loss implementations (763 lines)
```

---

## Module Exports (__init__.py)

| Export | Type | Purpose |
|--------|------|---------|
| `PAdicGeodesicLoss` | Class | Unified geodesic alignment |
| `RadialHierarchyLoss` | Class | Direct radius enforcement |
| `CombinedGeodesicLoss` | Class | Curriculum-blended geodesic+radial |
| `GlobalRankLoss` | Class | Soft ranking violation |
| `MonotonicRadialLoss` | Class | Level-wise ordering |
| `RichHierarchyLoss` | Class | Hierarchy + coverage + separation |
| `CombinedLoss` | Class | Config-driven composition |

**Assessment**: Clean, complete exports.

---

## Detailed Analysis

### 1. combined.py - CombinedLoss

#### 1.1 Architecture

```python
class CombinedLoss(nn.Module):
    """Config-driven combined loss function."""

    def __init__(self, loss_config: Dict, curvature: float = 1.0):
        # Instantiate enabled losses based on config
        if config.get('rich_hierarchy', {}).get('enabled'):
            self.rich_hierarchy = RichHierarchyLoss(...)
        if config.get('radial', {}).get('enabled'):
            self.radial_loss = RadialHierarchyLoss(...)
        # ... etc
```

#### 1.2 Config Schema

```yaml
loss:
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
    coverage_weight: 1.0
    separation_weight: 3.0
  radial:
    enabled: true
    weight: 1.0
  geodesic:
    enabled: true
    phase_start_epoch: 50  # Phase-gated
    weight: 0.3
  rank:
    enabled: true
    weight: 0.5
  monotonic:
    enabled: true
    weight: 1.0
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Config-driven | ✅ Good | Flexible composition from YAML |
| Phase-gating | ✅ Good | geodesic_phase_start_epoch support |
| Weight exposure | ✅ Good | Per-component weights configurable |
| Default fallback | ✅ Good | Basic coverage loss if no rich_hierarchy |

#### 1.3 Forward Pass (Lines 164-238)

```python
def forward(self, z_hyp, indices, logits, targets, epoch=0):
    losses = {}
    total = torch.tensor(0.0, device=device)

    # 1. RichHierarchyLoss
    if self.rich_hierarchy is not None:
        rich_out = self.rich_hierarchy(z_hyp, indices, logits, targets)
        weighted_rich = (
            self.rich_hierarchy_weights['hierarchy'] * rich_out['hierarchy'] +
            self.rich_hierarchy_weights['coverage'] * rich_out['coverage'] +
            self.rich_hierarchy_weights['separation'] * rich_out['separation']
        )
        total = total + weighted_rich

    # 2. RadialHierarchyLoss (if enabled)
    # 3. PAdicGeodesicLoss (phase-gated)
    # 4. GlobalRankLoss
    # 5. MonotonicRadialLoss
    # 6. Fallback coverage loss

    losses['total'] = total
    return losses
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Additive composition | ✅ Standard | Losses simply sum |
| Phase gating | ✅ Good | epoch >= geodesic_phase_start |
| Metrics preservation | ✅ Good | Each loss returns detail dict |
| Fallback coverage | ✅ Good | Ensures reconstruction signal exists |

#### 1.4 Coverage Loss Fallback (Lines 240-274)

```python
def _compute_coverage_loss(self, logits, targets):
    if logits.shape[-1] == 3:  # (B, 9, 3) format
        targets_shifted = (targets + 1).long().clamp(0, 2)
        return F.cross_entropy(logits.view(-1, 3), targets_shifted.view(-1))
    elif logits.shape[-1] == 27:  # (B, 27) format
        logits_reshaped = logits.view(-1, 9, 3)
        targets_shifted = (targets + 1).long().clamp(0, 2)
        return F.cross_entropy(logits_reshaped.permute(0, 2, 1), targets_shifted)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Shape handling | ✅ Good | Handles both (B, 27) and (B, 9, 3) |
| Target shift | ✅ Correct | {-1,0,1} → {0,1,2} for cross_entropy |
| Clamp | ✅ Safe | Prevents index errors |

---

### 2. padic_geodesic.py - PAdicGeodesicLoss

#### 2.1 Core Insight

```
P-adic valuation v_3(|i-j|) → target hyperbolic distance

High valuation (divisible by 3^k) → small geodesic distance → both near origin
Low valuation (not divisible by 3) → large geodesic distance → apart
```

#### 2.2 Target Distance Mapping (Lines 83-93)

```python
def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
    """Map 3-adic valuation to target hyperbolic distance.

    d_target = max_dist * exp(-valuation / scale)
    """
    return self.max_target * torch.exp(-valuation / self.valuation_scale)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Exponential decay | ✅ Correct | Matches ultrametric structure |
| Configurable scale | ✅ Good | `valuation_scale` parameter |
| Max distance cap | ✅ Good | `max_target_distance` bounds output |

#### 2.3 Forward Pass (Lines 95-155)

```python
def forward(self, z_hyp, batch_indices):
    # Sample random pairs (reproducible via generator)
    i_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator)
    j_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator)

    # Compute actual Poincaré distance
    d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], self.curvature)

    # Compute target distance from 3-adic valuation
    diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())
    valuation = TERNARY.valuation(diff).float()
    d_target = self.target_distance(valuation)

    # Loss: align actual with target
    if self.use_smooth_l1:
        loss = F.smooth_l1_loss(d_actual, d_target)
    else:
        loss = F.mse_loss(d_actual, d_target)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Seeded generator | ✅ Good | Reproducible pair sampling |
| Poincaré distance | ✅ Correct | Uses src/geometry |
| Self-pair avoidance | ✅ Good | j_idx shifted when equal |
| Smooth L1 option | ✅ Good | Robust to outliers |

#### 2.4 Metrics (Lines 136-153)

```python
with torch.no_grad():
    corr = torch.corrcoef(torch.stack([d_actual, d_target]))[0, 1]
    mean_d_low_v = d_actual[valuation < 2].mean()
    mean_d_high_v = d_actual[valuation >= 4].mean()
```

| Metric | Purpose |
|--------|---------|
| `distance_correlation` | How well actual matches target ordering |
| `mean_d_low_valuation` | Distance for unrelated operations |
| `mean_d_high_valuation` | Distance for closely related operations |

---

### 3. padic_geodesic.py - RadialHierarchyLoss

#### 3.1 Core Insight

```
Direct radius enforcement:
  v_3(n) high → radius small (near origin)
  v_3(n) low  → radius large (near boundary)
```

#### 3.2 Hyperbolic Radius (Lines 223-226)

```python
# V5.12.2: Compute actual radius using hyperbolic distance, not Euclidean norm
origin = torch.zeros_like(z_hyp)
actual_radius = poincare_distance(z_hyp, origin, c=self.curvature)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Hyperbolic distance | ✅ Correct | Not Euclidean norm |
| Origin creation | ✅ Correct | Same shape as z_hyp |
| Curvature passed | ✅ Correct | Uses configured c |

#### 3.3 Valuation Weighting (Lines 233-240)

```python
if self.valuation_weighting:
    # V5.12.3: Clamped exponential weighting for gradient stability
    raw_weights = 1.0 + torch.exp(valuations * 0.25)  # Reduced from 0.4
    weights = torch.clamp(raw_weights, min=1.0, max=10.0)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| High-valuation emphasis | ✅ Good | Rare points weighted more |
| Clamping | ✅ Good | Prevents gradient explosion |
| Softened exponent | ✅ Good | 0.25 instead of 0.4 for stability |

#### 3.4 Margin Loss (Lines 245-276)

```python
# For pairs where v_i > v_j (i has higher valuation),
# we want r_i < r_j (i should be closer to origin)
higher_v_mask = v_i > v_j

if higher_v_mask.any():
    expected_margin = v_diff * self.radius_step * 0.5
    actual_diff = r_j[higher_v_mask] - r_i[higher_v_mask]
    violations = F.relu(expected_margin - actual_diff)
    margin_loss = violations.mean()
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Pairwise margin | ✅ Good | Enforces separation |
| Adaptive margin | ✅ Good | Proportional to valuation gap |
| ReLU hinge | ✅ Good | Only penalizes violations |

---

### 4. padic_geodesic.py - GlobalRankLoss

#### 4.1 Core Insight

```
Soft differentiable ranking:
  If v_i > v_j, then r_i should be < r_j
  Uses sigmoid for differentiable violation scoring
```

#### 4.2 Implementation (Lines 440-485)

```python
# Signed radius difference: should be positive if ordering is correct
signed_r_diff = expected_sign * r_diff

# Soft violation: sigmoid(-signed_r_diff / temperature)
violations = torch.sigmoid(-signed_r_diff / self.temperature)

# Weight by valuation difference magnitude
weights = torch.abs(v_diff)
weighted_violations = violations * weights

loss = weighted_violations.mean()
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Soft ranking | ✅ Good | Differentiable surrogate |
| Temperature control | ✅ Good | Adjustable sharpness |
| Magnitude weighting | ✅ Good | Large gaps more important |
| Hyperbolic radius | ✅ Correct | Uses poincare_distance |

---

### 5. padic_geodesic.py - MonotonicRadialLoss

#### 5.1 Core Insight

```
Level-wise means instead of random pairs:
  1. Group points by valuation level (0-9)
  2. Compute mean radius per level
  3. Enforce: mean_r[v] > mean_r[v+1] + margin
```

#### 5.2 Level Aggregation (Lines 578-598)

```python
for v in range(self.max_valuation + 1):
    mask = valuations == v
    if mask.any():
        level_means.append(radii[mask].mean())
        level_counts.append(mask.sum().item())
        levels_present.append(v)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Per-level means | ✅ Good | Aggregates noisy samples |
| Present-level tracking | ✅ Good | Handles sparse batches |
| Graceful degradation | ✅ Good | Returns 0 if < 2 levels |

#### 5.3 Margin Enforcement (Lines 614-623)

```python
radius_diffs = level_means[:-1] - level_means[1:]  # r[v] - r[v+1]
violations = margins - radius_diffs  # positive when violated

if self.use_soft_margin:
    loss = F.softplus(violations / self.temperature).mean() * self.temperature
else:
    loss = F.relu(violations).mean()
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Consecutive level comparison | ✅ Correct | Adjacent levels |
| Soft margin (softplus) | ✅ Good | Smooth gradient |
| Temperature scaling | ✅ Good | Maintains gradient magnitude |

---

### 6. padic_geodesic.py - RichHierarchyLoss

#### 6.1 Combined Objective

```
RichHierarchyLoss = Hierarchy + Coverage + Separation

1. Hierarchy: MSE(mean_radius[v], target_radius[v])
2. Coverage: CrossEntropy(logits, targets)
3. Separation: Margin violations between levels
```

#### 6.2 Hierarchy Loss (Lines 691-703)

```python
for v in present_levels:
    mask = valuations == v
    if mask.sum() > 0:
        mean_r = radii[mask].mean()
        target_r = self.target_radii[v]
        hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2

hierarchy_loss = hierarchy_loss / len(present_levels)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Per-level MSE | ✅ Correct | Mean radius to target |
| Normalization | ✅ Good | Divides by n_levels |
| Precomputed targets | ✅ Good | Registered buffer |

#### 6.3 Coverage Loss (Lines 705-721)

```python
targets_shifted = (targets + 1).long()

if logits.shape[-1] == 3:  # (B, 9, 3)
    coverage_loss = F.cross_entropy(logits.view(-1, 3), targets_shifted.view(-1))
elif logits.shape[-1] == 27:  # (B, 27)
    coverage_loss = F.cross_entropy(
        logits.view(-1, 9, 3).permute(0, 2, 1),
        targets_shifted,
    )
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Shape handling | ✅ Good | Both formats supported |
| Target shift | ✅ Correct | {-1,0,1} → {0,1,2} |
| No clamp | ⚠️ Note | Assumes valid data (comment says "data must be valid") |

#### 6.4 Separation Loss (Lines 723-746)

```python
sorted_levels = sorted(present_levels.tolist())

for v in sorted_levels:
    mask = valuations == v
    if mask.sum() > 0:
        mean_radii.append(radii[mask].mean())

# Enforce r[v] > r[v+1] + margin
for i in range(len(mean_radii) - 1):
    violation = F.relu(mean_radii[i + 1] - mean_radii[i] + 0.01)
    separation_loss = separation_loss + violation
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Sorted levels | ✅ Correct | Outer (v=0) to inner (v=9) |
| Fixed margin | ⚠️ Hardcoded | 0.01 margin is magic number |
| ReLU hinge | ✅ Good | Only penalizes violations |

#### 6.5 Return Format (Lines 748-752)

```python
# Return raw components - CombinedLoss applies weights from config
total = hierarchy_loss + coverage_loss + separation_loss

return {
    "total": total,
    "hierarchy": hierarchy_loss,
    "coverage": coverage_loss,
    "separation": separation_loss
}
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Component dict | ✅ Good | Allows external weighting |
| Raw values | ✅ Good | CombinedLoss applies weights |

---

### 7. padic_geodesic.py - CombinedGeodesicLoss

#### 7.1 Curriculum Blending

```python
def forward(self, z_hyp, batch_indices, tau=0.5):
    geo_loss, geo_metrics = self.geodesic_loss(z_hyp, batch_indices)
    rad_loss, rad_metrics = self.radial_loss(z_hyp, batch_indices)

    # Curriculum blend
    total_loss = (1 - tau) * rad_loss + tau * geo_loss
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Tau parameter | ✅ Good | External curriculum control |
| Early training | ✅ Good | tau=0 → pure radial |
| Late training | ✅ Good | tau=1 → pure geodesic |

---

## Issues Summary

### Critical (0)

None. The losses are mathematically correct and properly use hyperbolic geometry.

### High (0)

None. All losses correctly use `poincare_distance` for radius computation.

### Medium (3)

| Issue | Location | Description |
|-------|----------|-------------|
| M1 | `RichHierarchyLoss:745` | Hardcoded margin 0.01 |
| M2 | `PAdicGeodesicLoss:78` | `valuation_scale=3.0` is somewhat arbitrary |
| M3 | `CombinedLoss:186` | Tensor initialized on device but could use `device` parameter |

### Low (4)

| Issue | Location | Description |
|-------|----------|-------------|
| L1 | `combined.py:274` | Returns `tensor(0.0)` silently for unsupported logit shape |
| L2 | `MonotonicRadialLoss:635` | Magic weight 0.5 for target_loss |
| L3 | `GlobalRankLoss:462` | Gradient spike weight of 2.0 is arbitrary |
| L4 | Multiple | Device handling creates tensors on wrong device initially |

---

## Reproducibility Analysis

| Loss | Seeded Generator | Status |
|------|-----------------|--------|
| PAdicGeodesicLoss | ✅ Yes (seed=42) | Reproducible |
| RadialHierarchyLoss | ✅ Yes (seed=42) | Reproducible |
| GlobalRankLoss | ✅ Yes (seed=42) | Reproducible |
| MonotonicRadialLoss | N/A (deterministic) | Reproducible |
| RichHierarchyLoss | N/A (deterministic) | Reproducible |
| CombinedGeodesicLoss | ✅ Via sub-losses | Reproducible |

**Assessment**: All losses with random sampling use seeded generators.

---

## Geometry Integration Analysis

| Loss | Uses poincare_distance | Uses Euclidean norm | Status |
|------|----------------------|---------------------|--------|
| PAdicGeodesicLoss | ✅ For pair distances | ❌ | Correct |
| RadialHierarchyLoss | ✅ For radius | ❌ | Correct |
| GlobalRankLoss | ✅ For radius | ❌ | Correct |
| MonotonicRadialLoss | ✅ For radius | ❌ | Correct |
| RichHierarchyLoss | ✅ For radius | ❌ | Correct |

**V5.12.2 Fix Applied**: All losses correctly compute hyperbolic radius as `poincare_distance(z, origin)` instead of `torch.norm(z, dim=-1)`.

---

## Loss Objectives Summary

| Loss | Primary Objective | Supervision Signal |
|------|------------------|-------------------|
| PAdicGeodesicLoss | Align pairwise distances with p-adic valuation | Geodesic distance correlation |
| RadialHierarchyLoss | Push points to target radii by valuation | Direct radius control |
| GlobalRankLoss | Enforce monotonic radius ordering | Soft ranking violation |
| MonotonicRadialLoss | Enforce level-wise mean radius ordering | Adjacent level margins |
| RichHierarchyLoss | Combined hierarchy + reconstruction + separation | Unified training |
| CombinedLoss | Config-driven composition | Weighted sum |

---

## Potential Loss Conflicts

| Loss Pair | Conflict Risk | Notes |
|-----------|---------------|-------|
| Radial + Geodesic | Low | Both encourage correct radial structure |
| Radial + Rank | Low | Complementary (pointwise vs pairwise) |
| Radial + Monotonic | Low | Both target same radial ordering |
| Coverage + Hierarchy | Medium | May compete for encoder capacity |
| Geodesic + Monotonic | Low | Both enforce ultrametric structure |

**Recommendation**: Using all losses simultaneously is redundant. Prefer:
- **Option A**: RichHierarchyLoss alone (unified)
- **Option B**: Radial + Geodesic (complementary)
- **Option C**: Monotonic + Coverage (level-focused)

---

## Code Quality Assessment

| Metric | Score | Notes |
|--------|-------|-------|
| Correctness | 9/10 | All losses mathematically sound |
| Geometry Integration | 10/10 | Proper poincare_distance usage |
| Reproducibility | 10/10 | All random losses seeded |
| Documentation | 8/10 | Good docstrings, version notes |
| Configuration | 9/10 | Flexible config-driven composition |
| Magic Numbers | 6/10 | Several hardcoded constants |

---

## Recommendations

### Should Fix

1. **Extract magic numbers to config**:
   ```python
   # RichHierarchyLoss
   margin = config.get('separation_margin', 0.01)

   # MonotonicRadialLoss
   target_loss_weight = config.get('target_loss_weight', 0.5)
   ```

2. **Warn on unsupported logit shape**:
   ```python
   else:
       warnings.warn(f"Unsupported logit shape: {logits.shape}")
       return torch.tensor(0.0, device=device)
   ```

### Could Improve

1. **Unify device handling**: Use `device` parameter consistently
2. **Add loss sanity checks**: Warn if loss is NaN/Inf
3. **Provide recommended presets**: Document which loss combinations work best

---

## Verdict

**The losses module is the strongest component of the codebase.** It correctly implements hyperbolic geometry supervision with proper use of `poincare_distance`, seeded generators for reproducibility, and flexible config-driven composition.

The losses correctly supervise the hyperbolic structure, but as noted in the models audit, this supervision signal flows only through the loss—the decoder doesn't use `z_hyp` for generation. The losses are doing their job; the architecture isn't fully utilizing them.

**Rating**: 9/10 (Excellent, minor improvements possible)

---

**Audit completed**: 2025-01-23
**Auditor**: Claude Opus 4.5
