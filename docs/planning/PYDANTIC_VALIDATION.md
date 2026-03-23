# Pydantic Validation — Implementation Plan

**Date:** 2026-03-23
**Status:** Analysis

## Problem Statement

Current validation in `src/config/statenet_config.py` uses dataclasses with `__post_init__`. The broader config (loss weights, model hyperparameters, angular_coherence) relies on `yaml.safe_load` + dict access with no schema enforcement.

**Consequence:** Typoes like `target_sin` instead of `target_sim`, or `level_prefix_k: [3, 4, 5]` instead of length-10 list, fail silently or at runtime with confusing errors.

## Current Validation State

| Config Section | Current Approach | Validation |
|---------------|------------------|------------|
| `statenet` | `StateNetConfig` dataclass | ✅ `__post_init__` checks |
| `loss` | Direct dict access | ❌ None |
| `model` | Direct dict access | ❌ None |
| `training` | Direct dict access | ❌ None |
| `angular_coherence` | `CombinedLoss` constructor | ⚠️ Minimal |

## Key Validation Gaps

### 1. `angular_coherence` — Most Critical

**Current Issues:**
```python
# No validation that level_prefix_k is length 10
level_prefix_k=ac_cfg.get('level_prefix_k', None)

# No validation that target_sim aligns with level_prefix_k
target_sim=ac_cfg.get('target_sim', 1.0)

# No validation of value ranges (prefix_k ∈ [2,9], target_sim ∈ [0,1])
```

**Example bugs caught by Pydantic:**
- `target_sim[0] = 0.90` caused ARI regression (should be 1.0)
- `level_prefix_k` wrong length → silent wrong indexing
- `prefix_k = 6` (too deep for v=0, only 18 classes at k=3)

### 2. Loss Weights

**Current Issues:**
```python
# No validation that weights are positive
hierarchy_weight = cfg.get('hierarchy_weight', 5.0)
coverage_weight = cfg.get('coverage_weight', 1.0)

# No validation that enabled flags are bool
enabled = cfg.get('enabled', True)
```

### 3. Model Architecture

**Current Issues:**
```python
# No validation that latent_dim is reasonable
latent_dim = model_cfg.get('latent_dim', 32)

# No cross-field validation (e.g., radial_dims < latent_dim)
radial_dims = model_cfg.get('radial_dims', 4)
```

## Proposed Implementation

### Option A: Minimal — Pydantic for `angular_coherence` Only

Add a single Pydantic model for the AC config, used in `CombinedLoss`:

```python
from pydantic import BaseModel, Field, field_validator

class AngularCoherenceConfig(BaseModel):
    """Validated config for AngularCoherenceLoss."""
    
    enabled: bool = True
    weight: float = Field(gt=0, description="Loss weight")
    n_pairs: int = Field(ge=100, le=20000, description="Pairs per batch")
    prefix_k: int = Field(ge=2, le=9, default=3)
    level_prefix_k: list[int] | None = Field(default=None)
    target_sim: float | list[float] = 1.0
    phase_start_epoch: int = Field(ge=0)
    
    @field_validator('level_prefix_k')
    @classmethod
    def validate_level_prefix_k_length(cls, v):
        if v is not None and len(v) != 10:
            raise ValueError(f"level_prefix_k must be length 10, got {len(v)}")
        return v
    
    @field_validator('target_sim')
    @classmethod
    def validate_target_sim(cls, v):
        if isinstance(v, float) and not (0.0 <= v <= 1.0):
            raise ValueError(f"target_sim must be in [0, 1], got {v}")
        return v
```

**Pros:** Focused, low-risk, catches the most common errors
**Cons:** Doesn't cover loss weights or model config

### Option B: Full Schema — All Config Sections

Create Pydantic models for every config section:

```python
from pydantic import BaseModel, Field, field_validator

class LossSection(BaseModel):
    rich_hierarchy: RichHierarchyConfig
    angular_coherence: AngularCoherenceConfig
    # ... other loss configs

class ModelConfig(BaseModel):
    name: Literal["TernaryVAEV6Controllable"]
    latent_dim: int = Field(ge=16, le=256)
    hidden_dim: int = Field(ge=64, le=512)
    radial_dims: int = Field(ge=1, le=16)
    
    @field_validator('radial_dims')
    @classmethod
    def validate_radial_dims(cls, v, info):
        # Cross-field: radial_dims < latent_dim
        latent = info.data.get('latent_dim')
        if latent and v >= latent:
            raise ValueError(f"radial_dims={v} must be < latent_dim={latent}")
        return v
```

**Pros:** Complete coverage, cross-field validation
**Cons:** Large refactor, breaking changes to `StateNetConfig`

### Option C: Hybrid — Pydantic for New Features, Dataclass for Existing

Keep `StateNetConfig` as dataclass (already validated). Add new Pydantic models for:
1. `AngularCoherenceConfig`
2. `LossWeightConfig` (validates all weight ranges)
3. `TrainingConfig` (epochs, batch_size ranges)

**Recommended for this codebase.** Minimal disruption, targeted improvement.

## Validation Rules for `angular_coherence`

### Required

| Field | Validation | Reason |
|-------|-----------|--------|
| `level_prefix_k` | Length 10, all values ∈ [0, 9] | Must match 10 valuation levels |
| `target_sim` | Length 10 (if list), all ∈ [0, 1] | Must align with level_prefix_k |
| `target_sim[0]` | **Must be 1.0** | AC signal depends on this |
| `prefix_k` | ∈ [2, 9] | k=2 minimum for prefix, k>9 exceeds digits |
| `n_pairs` | ∈ [100, 20000] | Too few = no signal, too many = OOM |
| `phase_start_epoch` | ∈ [0, epochs] | Must start after warmup |

### Recommended

| Field | Validation |
|-------|-----------|
| `level_prefix_k[v]` | If > 0, must be ≥ v+1 (enough free positions) |
| `target_sim[v]` | If > 0, should match `level_prefix_k[v]` semantics |

## Implementation Priority

1. **Phase 1 (Now):** Add `AngularCoherenceConfig` Pydantic model
   - Catches the most common config errors
   - Validates `target_sim[0]=1.0` invariant
   - Backward compatible (accepts None, falls back to defaults)

2. **Phase 2:** Add `LossWeightConfig` for all loss weights
   - Validates all weights are positive
   - Cross-loss validation (e.g., hierarchy_weight >> kl_weight)

3. **Phase 3 (Future):** Full `ModelConfig` schema
   - Cross-field validation (radial_dims < latent_dim)
   - Architecture compatibility checks

## Example Error Messages with Pydantic

**Before (current):**
```
KeyError: 'target_sim'
# or worse: silent wrong behavior
```

**After (Pydantic):**
```
ValidationError: 1 validation error for AngularCoherenceConfig
target_sim.0
  Input should be 1.0, got 0.90. The v=0 soft-margin target MUST be 1.0 
  for AC loss to function — within-sim at v=0 is already 0.98, so any 
  target < 1.0 makes the loss identically zero.
```

## Compatibility Notes

- `pydantic` v2 recommended (faster, better error messages)
- Existing `StateNetConfig` dataclass can coexist with Pydantic models
- YAML loading unchanged — Pydantic validates after `yaml.safe_load`
- Can add validation gradually without breaking existing configs

## Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add `pydantic>=2.0` |
| `src/config/__init__.py` | Export new Pydantic models |
| `src/losses/combined.py` | Use `AngularCoherenceConfig` in `CombinedLoss` |
| `src/losses/padic_geodesic.py` | Add type hints, accept validated config |

## Next Steps

1. **Decide:** Option A (minimal) vs Option C (hybrid) vs Option B (full)
2. **Draft:** Pydantic model for `AngularCoherenceConfig`
3. **Test:** Add validation to `CombinedLoss`, verify error messages
4. **Iterate:** Extend to other config sections as needed
