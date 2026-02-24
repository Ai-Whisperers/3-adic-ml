# Comprehensive Codebase Audit Report

**Date**: 2026-02-24
**Scope**: All source files in `src/` (22 files) and all test files in `tests/` (8 files)
**Auditor**: Claude Opus 4.6 (automated code review)
**Codebase**: P-Adic VAE V6.0/V6.1 at `/d1/VAEs/3-adic-ml`

---

## Executive Summary

The codebase is architecturally sound with well-structured mathematical foundations. The core components (ternary.py, poincare.py, loss functions, and the training loop) are functional and consistent. However, the audit identified **2 CRITICAL**, **5 MODERATE**, **9 LOW**, and **6 INFO** issues. The most significant problems are in the TensorBoard logger, which contains stale V5.x API calls that would crash if invoked, and an unreachable checkpoint validator that contradicts the training loop's from-scratch support.

### Previously Fixed Bugs (Excluded)

The following bugs were confirmed as already fixed and are NOT reported below:
- encoder_A/encoder_a case mismatch
- Double gate computation in MetricBasedLR.update()
- loss_fn.parameters() not added to optimizer
- cudnn.benchmark overriding determinism
- weight_decay not passed to RiemannianAdam
- log_map_zero ignoring max_norm parameter
- Scheduler vs LR controller using base_lr
- log_sigma float32 in float64 codebase

---

## CRITICAL (2 issues)

### C-1: TensorBoardLogger.log_manifold_embedding uses V5.x model API (will crash)

**File**: `src/utils/tensorboard_logger.py`, lines 425-547
**Type**: Stale reference / API mismatch

The `log_manifold_embedding` method calls the model with the old V5.x 5-argument signature:

```python
# Line 467 - V5.x API (5 args)
outputs = model(x, DEFAULT_TEMP_A, DEFAULT_TEMP_B, DEFAULT_BETA_A, DEFAULT_BETA_B)
```

The V6 model (`TernaryVAEV6.forward`) only accepts a single argument `x`. This call will raise a `TypeError` at runtime.

Additionally, lines 468-469 access output keys that do not exist in V6:
```python
z_A = outputs["z_A"]   # V6 key is "z_A_hyp"
z_B = outputs["z_B"]   # V6 key is "z_B_hyp"
```

Lines 471-476 also perform a manual Poincare projection (`z / (1 + norm) * 0.95`) which is redundant in V6 since the model already returns points on the Poincare ball via `expmap0`.

The stale constants at lines 29-34 are also V5.x artifacts:
```python
DEFAULT_TEMP_A = 1.0
DEFAULT_TEMP_B = 1.0
DEFAULT_BETA_A = 0.5
DEFAULT_BETA_B = 0.5
```

**Impact**: Calling `log_manifold_embedding` will crash the training run. Currently this method is not called from `train.py`, so it is latent -- but it is a public API that could be invoked by users.

**Recommendation**: Rewrite to use V6 model API: `outputs = model(x)`, access `outputs["z_A_hyp"]` and `outputs["z_B_hyp"]`, and remove the manual Poincare projection. Remove the stale `DEFAULT_TEMP_*` and `DEFAULT_BETA_*` constants.

---

### C-2: TensorBoardLogger.log_epoch expects V5.x loss dict keys (will crash)

**File**: `src/utils/tensorboard_logger.py`, lines 247-360
**Type**: Stale reference / API mismatch

The `log_epoch` method accesses numerous loss dictionary keys from the V5.x era that are not produced by the V6 training loop:

| Key accessed (V5.x) | Lines | V6 equivalent |
|---------------------|-------|---------------|
| `ce_A`, `ce_B` | 279, 286 | `cross_entropy` (single) |
| `kl_A`, `kl_B` | 280, 287 | Not produced (KL is inline in train.py) |
| `H_A`, `H_B` | 281, 288, 294-295 | Not produced |
| `phase` | 305 | Not produced |
| `rho` | 306 | Not produced |
| `grad_ratio` | 307 | Not produced |
| `ema_momentum` | 309 | Not produced |
| `lambda1`, `lambda2`, `lambda3` | 316-318 | Replaced by learnable weights |
| `temp_A`, `temp_B` | 326-327 | Not produced (no temperature scheduling in V6) |
| `beta_A`, `beta_B` | 333-334 | Not produced |
| `lr_scheduled` | 338 | Not produced |
| `lr_corrected` | 339 | Not produced |

Lines 218-245 (`log_hyperbolic_metrics`) also reference `v5.10`-tagged keys (`v5.10/HyperbolicKL`, `v5.10/StateNetSigma`, `v5.10/StateNetCurvature`).

**Impact**: Calling `log_epoch` with V6 loss dictionaries will raise `KeyError` on the first missing key. The `log_hyperbolic_metrics` method uses `.get()` with defaults so it would silently log zeros rather than crash.

**Recommendation**: Rewrite `log_epoch` to match the V6 loss dictionary structure produced by `CombinedLoss`. The V6 training loop in `train.py` uses a custom TensorBoard logging approach (lines 1050-1150) that bypasses `log_epoch` entirely, which is why this hasn't crashed in practice. Consider removing `log_epoch` or rewriting it to accept V6 keys.

---

## MODERATE (5 issues)

### M-1: checkpoint_validator contradicts train.py's from-scratch support

**File**: `src/utils/checkpoint_validator.py`, lines 44-53
**Type**: Inconsistency / misleading validation

`validate_training_config` treats a missing anchor checkpoint as an error for V6 models:

```python
if checkpoint_path is None or str(checkpoint_path).lower() == "null":
    errors.append(
        f"Model '{model_name}' requires an anchor checkpoint for proper initialization. "
        f"Set 'anchor_checkpoint.path' to a valid checkpoint."
    )
```

However, `train.py` explicitly supports training from scratch (line 292):
```python
print("[AUDIT] No anchor checkpoint specified (training from scratch)")
```

**Mitigating factor**: `validate_training_config` is never called from `train.py` (confirmed via grep). It is dead code in the training path.

**Impact**: If a user or CI script ever calls this validator before training, it will incorrectly reject valid from-scratch configurations.

**Recommendation**: Either remove the anchor checkpoint requirement from the validator, or add a `from_scratch: true` config key that the validator respects. Also consider either integrating the validator into `train.py` or removing it entirely.

---

### M-2: ScheduleBasedLR._get_scale_at_epoch division by zero with duplicate epochs

**File**: `src/models/lr_controller.py`, line 160
**Type**: Potential runtime error

```python
t = (epoch - e1) / (e2 - e1)  # Division by zero if e1 == e2
```

`_validate_schedules` (line 139) sorts schedule points by epoch but does not check for or deduplicate entries with the same epoch. If a user provides `[(10, 0.5), (10, 1.0)]`, this produces a `ZeroDivisionError`.

**Impact**: Only affects users of `ScheduleBasedLR` who provide duplicate epoch entries. `MetricBasedLR` (the default) is not affected.

**Recommendation**: Add duplicate epoch detection in `_validate_schedules`:
```python
epochs = [e for e, _ in sorted_schedule]
if len(epochs) != len(set(epochs)):
    raise ValueError(f"Duplicate epochs in schedule for {name}")
```

---

### M-3: LearnableLRController creates float32 tensor in float64 codebase

**File**: `src/models/lr_controller.py`, lines 532-539
**Type**: Dtype inconsistency

```python
metrics_tensor = torch.tensor([
    metrics.coverage,
    metrics.hierarchy_a,
    metrics.hierarchy_b,
    metrics.dist_corr_a,
    metrics.q_value,
    metrics.grad_norm_projections,
], dtype=torch.float32)
```

The rest of the codebase uses `float64` for numerical stability with geoopt. While this class is labeled "experimental", mixing dtypes can cause subtle precision issues or dtype mismatch errors if the MLP parameters are float64.

**Impact**: Low in practice since `LearnableLRController` is not used in the default training configuration. Could cause issues if someone enables it.

**Recommendation**: Change to `dtype=torch.float64` for consistency, or document the intentional dtype choice.

---

### M-4: compute_tree_coherence uses Python loops for index mapping

**File**: `src/train.py`, lines 425-437
**Type**: Performance

```python
index_to_pos = {idx.item(): pos for pos, idx in enumerate(indices)}

for pos, (idx, parent_idx) in enumerate(zip(indices, parent_indices)):
    parent_val = parent_idx.item()
    if parent_val < 0:
        continue
    if parent_val in index_to_pos:
        child_positions.append(pos)
        parent_positions.append(index_to_pos[parent_val])
```

This iterates over all indices in Python, calling `.item()` on each tensor element. For the full dataset (19,683 operations), this loop runs ~20K iterations in Python during every validation epoch.

**Impact**: Slows validation by several seconds per epoch. Not a correctness issue.

**Recommendation**: Vectorize using `torch.searchsorted` or a tensor-based hash map. Example:
```python
sorted_indices, sort_perm = indices.sort()
parent_pos = torch.searchsorted(sorted_indices, parent_indices)
valid = (parent_pos < len(indices)) & (sorted_indices[parent_pos.clamp(max=len(indices)-1)] == parent_indices)
```

---

### M-5: CombinedGeodesicLoss class is dead code

**File**: `src/losses/padic_geodesic.py`, lines 312-387
**Type**: Dead code

`CombinedGeodesicLoss` (described as "for V5.11") wraps `PAdicGeodesicLoss` and `RadialHierarchyLoss` with curriculum-based blending. It is:
- Never instantiated by `CombinedLoss` in `combined.py`
- Never referenced in `train.py`
- Not exported in `losses/__init__.py`

The V6 architecture uses `CombinedLoss` which instantiates these losses individually with its own weight management.

**Impact**: ~75 lines of unreachable code.

**Recommendation**: Remove or move to an archive. If kept, add a deprecation notice.

---

## LOW (9 issues)

### L-1: hyperbolic_kl.py is entirely unused

**File**: `src/losses/hyperbolic_kl.py` (193 lines)
**Type**: Dead code

`HyperbolicKLDivergence` and `StandardKLDivergence` are defined and exported via `__init__.py` but:
- Never imported by `combined.py` or `train.py`
- No test coverage
- The V6 training loop computes KL divergence inline (standard Gaussian KL)

**Impact**: 193 lines of unused code. The `HyperbolicKLDivergence` class implements a curvature-corrected KL which could be mathematically valuable if the architecture evolves to use it.

**Recommendation**: Either integrate into the loss system (as an option in `CombinedLoss`) or move to an `experimental/` directory. Add tests if keeping.

---

### L-2: Unused import: torch.nn.functional as F in train.py

**File**: `src/train.py`, line 48
**Type**: Unused import

```python
import torch.nn.functional as F
```

`F` is never referenced anywhere in train.py (confirmed via grep for `\bF\.`).

**Recommendation**: Remove the import.

---

### L-3: Unused import: poincare_distance in combined.py

**File**: `src/losses/combined.py`, line 40
**Type**: Unused import

```python
from src.geometry import poincare_distance
```

`poincare_distance` is not used directly in `combined.py`. The individual loss classes import it themselves.

**Recommendation**: Remove the import.

---

### L-4: PRESETS_DIR defined but not exported

**File**: `src/config/paths.py`, line 7 and `src/config/__init__.py`, line 12
**Type**: Dead code / inconsistency

`paths.py` defines both `PRESETS_DIR` (project root `presets/`) and `SRC_PRESETS_DIR` (under `src/presets/`). Only `SRC_PRESETS_DIR` is exported via `__init__.py`.

```python
# paths.py
PRESETS_DIR = PROJECT_ROOT / "presets"       # Not exported
SRC_PRESETS_DIR = PROJECT_ROOT / "src" / "presets"  # Exported
```

**Impact**: `PRESETS_DIR` is inaccessible to code importing from `src.config`.

**Recommendation**: Either export `PRESETS_DIR` or remove it if unneeded.

---

### L-5: patience_ceiling config fields are never consumed

**File**: `src/config/statenet_config.py`, lines 40 and 49
**Type**: Dead code (config fields)

```python
# HierarchyThresholds
patience_ceiling: int = 25  # "Max patience (annealing limit)"

# ControllerThresholds
patience_ceiling: int = 20  # "Max patience (annealing limit)"
```

These fields are defined in the config dataclasses but `MetricBasedLR` never reads them. The `patience_ceiling` was part of the removed `AnnealingConfig` system.

**Impact**: Users can set these in YAML but they have no effect.

**Recommendation**: Remove the fields or implement the ceiling logic in `MetricBasedLR`.

---

### L-6: checkpoint.py uses weights_only=False (security concern)

**File**: `src/utils/checkpoint.py`, line 36
**Type**: Security / best practice

```python
return torch.load(path, map_location=map_location, weights_only=False)
```

`weights_only=False` enables arbitrary code execution via pickle deserialization. This is a known security risk when loading untrusted checkpoints.

**Impact**: Low for a research codebase where checkpoints are self-generated. Becomes a concern if checkpoints are ever shared or downloaded from external sources.

**Recommendation**: Use `weights_only=True` as default, with a fallback to `False` only for legacy checkpoints:
```python
try:
    return torch.load(path, map_location=map_location, weights_only=True)
except Exception:
    return torch.load(path, map_location=map_location, weights_only=False)
```

---

### L-7: DataAuditor.prepare_data accepts but ignores device parameter

**File**: `src/train.py`, `DataAuditor.prepare_data` method
**Type**: Misleading API

The `DataAuditor` class accepts a `device` in `__init__` (stored as `self.device`) but `prepare_data` creates `TensorDataset` objects on CPU. Data is moved to GPU in the training loop via the DataLoader.

This is actually correct behavior (keeping data on CPU for DataLoader is standard practice), but the `device` parameter on `DataAuditor.__init__` is misleading -- it suggests data will be placed on the specified device.

**Impact**: Cosmetic / API clarity issue only. No functional bug.

**Recommendation**: Either remove the `device` parameter from `DataAuditor` or add a docstring clarifying that data stays on CPU for DataLoader compatibility.

---

### L-8: checkpoint_validator.py is dead code in the training path

**File**: `src/utils/checkpoint_validator.py` (94 lines)
**Type**: Dead code

`validate_training_config` is:
- Defined and exported via `utils/__init__.py`
- Never called from `train.py` or any other production code
- Contains the contradictory anchor checkpoint check (see M-1)

**Impact**: The entire module provides no value in the current training pipeline.

**Recommendation**: Either integrate into `train.py`'s startup sequence (with corrected logic) or remove.

---

### L-9: log_hyperbolic_metrics references v5.10-specific tags

**File**: `src/utils/tensorboard_logger.py`, lines 218-245
**Type**: Stale references

```python
self.writer.add_scalars("v5.10/HyperbolicKL", ...)
self.writer.add_scalar("v5.10/CentroidLoss", ...)
self.writer.add_scalars("v5.10/StateNetSigma", ...)
self.writer.add_scalars("v5.10/StateNetCurvature", ...)
```

These are V5.10-specific metrics that no longer exist in V6. The method uses `.get()` with defaults so it won't crash, but it silently logs meaningless zeros.

**Recommendation**: Update or remove the v5.10-specific logging paths.

---

## INFO (6 issues)

### I-1: Test coverage gaps

**Current coverage**: 214 tests across 6 test files covering 3 modules (core, geometry, losses).

**Untested modules** (no test files at all):

| Module | File | Lines | Complexity |
|--------|------|-------|------------|
| Models | `src/models/vae.py` | 528 | High - encoder/decoder, forward pass, reparameterization |
| Models | `src/models/hyperbolic_projection.py` | 267 | High - expmap0/logmap0 integration, learnable curvature |
| Models | `src/models/lr_controller.py` | 643 | High - 3 controller classes, gate logic, state machines |
| Utils | `src/utils/tensorboard_logger.py` | 577 | Medium - TensorBoard integration |
| Utils | `src/utils/hardware_monitor.py` | 263 | Low - monitoring only |
| Utils | `src/utils/checkpoint.py` | 57 | Low - thin wrapper |
| Utils | `src/utils/checkpoint_validator.py` | 94 | Low - validation logic |
| Losses | `src/losses/hyperbolic_kl.py` | 193 | Medium - curvature-corrected KL |

**Priority recommendations for new tests**:

1. **`vae.py`** (HIGH): Forward pass shape correctness, reparameterization trick, Poincare ball containment of outputs, gradient flow through full encoder-projection-decoder pipeline.

2. **`hyperbolic_projection.py`** (HIGH): `expmap0` output stays in ball, `logmap0(expmap0(v)) ~ v` roundtrip, learnable curvature stays positive, `DualHyperbolicProjection` produces valid manifold points.

3. **`lr_controller.py`** (MEDIUM): `MetricBasedLR` gate transitions (warmup -> active), hysteresis enforcement, `ScheduleBasedLR` interpolation, `update_optimizer_lr_scales` correctly modifies param groups.

---

### I-2: GrokkingDetector uses basic linear regression

**File**: `src/train.py`, `GrokkingDetector` class
**Type**: Design observation

The grokking detector fits a simple linear regression to the train/val loss gap to detect late generalization. This is a reasonable heuristic but may produce false positives if the gap naturally narrows due to learning rate decay rather than actual grokking.

**No action needed** -- this is a design choice, not a bug.

---

### I-3: torch.Generator state and reproducibility across config changes

**File**: `src/losses/padic_geodesic.py`
**Type**: Reproducibility observation

`PAdicGeodesicLoss`, `RadialHierarchyLoss`, and `GlobalRankLoss` each maintain their own `torch.Generator` for pair sampling. These generators advance independently. If the loss configuration changes between runs (e.g., disabling geodesic loss via phase gating), the other generators will see different sequences because the total number of generator advances per epoch changes.

This means two runs with identical seeds but different loss configurations will produce different pair samples for the shared losses, even if those losses are identically configured.

**No action needed** -- this is inherent to per-loss generators and is generally acceptable for training. Perfect cross-config reproducibility would require a single shared generator (which has its own downsides).

---

### I-4: CombinedLoss.compute_reconstruction_loss uses F.cross_entropy

**File**: `src/losses/combined.py`, lines 392, 400
**Type**: Design observation

The reconstruction loss uses `F.cross_entropy` with 3 classes (for ternary values {-1, 0, 1}). The targets are shifted from {-1, 0, 1} to {0, 1, 2} for class indices. This is correct and well-implemented.

**No action needed** -- noting for completeness.

---

### I-5: _manifold_cache in poincare.py is a module-level global

**File**: `src/geometry/poincare.py`, line 45
**Type**: Design observation

```python
_manifold_cache = {}
```

The manifold cache is a module-level dictionary keyed by `(curvature, device_str)`. This is efficient but:
- Not thread-safe (concurrent `get_manifold` calls could race)
- Grows without bounds (one entry per unique curvature/device pair)
- Not clearable without direct access to the private variable

**Impact**: Negligible for single-GPU training. Could matter for multi-threaded inference or long-running processes with dynamic curvatures.

**No action needed** for current use case.

---

### I-6: Copyright year range

**File**: All source files
**Type**: Administrative

Copyright headers read "Copyright 2024-2025" but the current date is 2026-02-24. This is cosmetic.

**No action needed** unless preparing for release.

---

## Summary Table

| Severity | Count | Key areas |
|----------|-------|-----------|
| CRITICAL | 2 | TensorBoard logger V5.x API (crash if called) |
| MODERATE | 5 | Validator contradiction, division-by-zero, dtype, performance, dead code |
| LOW | 9 | Unused imports, dead code modules, security, API clarity |
| INFO | 6 | Test coverage gaps, design observations, reproducibility |

## Recommended Priority Actions

1. **Immediate**: Fix or remove `TensorBoardLogger.log_manifold_embedding` and `log_epoch` (C-1, C-2)
2. **Short-term**: Remove dead code: `CombinedGeodesicLoss` (M-5), unused imports (L-2, L-3), `checkpoint_validator.py` or fix it (M-1, L-8)
3. **Medium-term**: Add tests for `vae.py` and `hyperbolic_projection.py` (I-1)
4. **Low priority**: Fix `ScheduleBasedLR` duplicate epoch handling (M-2), update `weights_only` (L-6), remove unused config fields (L-5)

---

*Generated by automated codebase audit. All line numbers reference the codebase state as of 2026-02-24.*
