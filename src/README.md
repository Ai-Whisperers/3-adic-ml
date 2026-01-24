# src/ - P-Adic VAE Source Code

**Last Updated**: 2025-01-24

## Architecture Overview

```
src/
├── core/           # Foundation: 3-adic algebra (TernarySpace singleton)
├── geometry/       # Hyperbolic operations (Poincare ball)
├── losses/         # Training objectives (config-driven composition)
├── models/         # VAE architectures (encoder/decoder/projection)
├── config/         # Constants and paths
├── presets/        # YAML experiment configurations
└── utils/          # Checkpoints, coverage, TensorBoard
```

### Data Flow (True Hyperbolic - V6.0)

```
Input (ternary ops)
       │
       ▼
   ┌───────────┐
   │  Encoder  │ → μ, logvar (tangent space T₀M)
   └───────────┘
       │
       ▼ Reparameterization (in tangent space)
   ┌───────────┐
   │ z_tangent │ = μ + ε * σ (Euclidean - tangent space IS Euclidean)
   └───────────┘
       │
       ▼ expmap0 (true hyperbolic projection)
   ┌───────────┐
   │  z_hyp    │ (on Poincaré manifold)
   └───────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
   ┌───────────┐            ┌───────────┐
   │  logmap0  │            │  Losses   │ (hyperbolic distances)
   └───────────┘            └───────────┘
       │
       ▼
   ┌───────────┐
   │  Decoder  │ ← tangent space (Euclidean-compatible)
   └───────────┘
       │
       ▼
   Output (logits)
```

Key insight: The tangent space at origin T₀M IS Euclidean ℝⁿ. Standard MLPs work in tangent space.
The manifold operations (expmap0, logmap0) provide the non-Euclidean structure.

---

## Module Details

### src/core/ - Ternary Algebra Foundation

**Key File:** `ternary.py`

The `TernarySpace` singleton (`TERNARY`) is the single source of truth for all 3-adic operations:

| Operation | Method | Complexity | Description |
|-----------|--------|------------|-------------|
| Valuation | `TERNARY.valuation(indices)` | O(1) | 3-adic valuation v_3(n) via LUT |
| Distance | `TERNARY.distance(i, j)` | O(1) | 3-adic metric d_3 = 3^(-v_3(\|i-j\|)) |
| To ternary | `TERNARY.to_ternary(indices)` | O(1) | Index → 9-trit representation |
| From ternary | `TERNARY.from_ternary(ternary)` | O(n) | 9-trit → index |

**Constants:**
- `N_DIGITS = 9` (trits per operation)
- `N_OPERATIONS = 19683` (3^9 total operations)
- `MAX_VALUATION = 9`

**Memory:** ~865 KB per device (precomputed LUTs)

---

### src/geometry/ - Hyperbolic Operations

**Key File:** `hyperbolic.py`

Poincare ball operations with curvature c:

| Function | Formula | Usage |
|----------|---------|-------|
| `poincare_distance(x, y, c)` | 2/√c · artanh(√c · \|\|(-x)⊕y\|\|) | All radius computations |
| `mobius_add(x, y, c)` | Mobius addition on manifold | Hyperbolic translation |
| `expmap0(v, c)` | Projects tangent vector to manifold | Origin → ball |
| `logmap0(y, c)` | Projects manifold point to tangent | Ball → origin |

**Critical:** All losses use `poincare_distance` for radius computation (not Euclidean norm).

---

### src/losses/ - Training Objectives

**Key Files:** `combined.py`, `padic_geodesic.py`

#### CombinedLoss (Config-Driven Factory)

Reads YAML `loss:` section and instantiates enabled losses:

```yaml
loss:
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
    separation_margin: 0.01      # configurable
  radial:
    enabled: true
    valuation_weight_exponent: 0.25  # configurable
    margin_step_factor: 0.5          # configurable
  monotonic:
    enabled: true
    target_loss_weight: 0.5          # configurable
```

#### Loss Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `RichHierarchyLoss` | Unified hierarchy + coverage + separation | `separation_margin` |
| `RadialHierarchyLoss` | Direct radius enforcement per valuation | `valuation_weight_exponent`, `margin_step_factor` |
| `PAdicGeodesicLoss` | Poincare distance alignment | `max_target_distance`, `n_pairs` |
| `GlobalRankLoss` | Soft ranking violation | `temperature` |
| `MonotonicRadialLoss` | Level-wise ordering constraints | `target_loss_weight` |

**Hierarchy Principle:** High valuation (v_3) → small radius (near origin)

---

### src/models/ - VAE Architectures

**Key Files:** `vae.py`, `hyperbolic_projection.py`

#### Model Variants

| Model | Description | Frozen Checkpoint |
|-------|-------------|-------------------|
| `TernaryVAEV5_11` | Base hyperbolic VAE | Required |
| `TernaryVAEV5_11_PartialFreeze` | Option C: frozen encoder | Required |

#### Architecture Components

1. **Encoder**: Input → hidden → (μ, log_σ)
2. **Reparameterization**: z_euc = μ + σ * ε (Euclidean)
3. **Hyperbolic Projection**: z_hyp = project(z_euc) → Poincare ball
4. **Decoder**: z → hidden → logits (9×3 for ternary)

**Known Issue:** Decoder currently uses z_euc, not z_hyp. See Architecture Issues below.

---

### src/config/ - Configuration

**Files:** `constants.py`, `paths.py`

| Constant | Value | Usage |
|----------|-------|-------|
| `N_TERNARY_OPERATIONS` | 19683 | Coverage calculation |
| `STATENET_COVERAGE_FREEZE_THRESHOLD` | 0.99 | StateNet controller |
| `PROJECT_ROOT` | Auto-detected | Path resolution |

---

### src/presets/ - YAML Configurations

**File:** `research_extended_grokking.yaml`

Sections: device, model, loss, training, scheduler, targets, logging, checkpoints

Example loss config flow:
```
YAML loss.radial.valuation_weight_exponent: 0.3
  → train.py loads config
  → CombinedLoss(loss_cfg)
  → RadialHierarchyLoss(valuation_weight_exponent=0.3)
```

---

### src/utils/ - Utilities

| File | Purpose |
|------|---------|
| `checkpoint.py` | Safe checkpoint loading |
| `checkpoint_validator.py` | Config/checkpoint validation |
| `coverage_evaluator.py` | VAE coverage evaluation |
| `tensorboard_logger.py` | Training visualization |

---

## Key Concepts

### 3-Adic Valuation

v_3(n) = largest k such that 3^k divides n

| n | v_3(n) | Interpretation |
|---|--------|----------------|
| 1, 2, 4, 5, 7, 8 | 0 | Not divisible by 3 |
| 3, 6, 12, 15 | 1 | Divisible by 3 |
| 9, 18, 36 | 2 | Divisible by 9 |
| 27, 54 | 3 | Divisible by 27 |
| 0 | 9 | Convention (infinity) |

### Hierarchy Target

Operations with high valuation should map to small radii (near Poincare ball origin):
- v=0 → radius ≈ 0.85 (boundary)
- v=9 → radius ≈ 0.10 (origin)

### Reproducibility

All stochastic operations use seeded `torch.Generator`:
- Loss pair sampling
- TensorBoard embedding sampling
- `TERNARY.sample_indices(n, generator=gen)`

---

## Architecture (V6.0 - True Hyperbolic)

All architectural issues have been resolved with proper geoopt integration:

| Issue | Status | Solution |
|-------|--------|----------|
| **Decoder uses z_euc** | ✅ Fixed | Decoder receives `logmap0(z_hyp)` |
| **Euclidean reparameterization** | ✅ Fixed | Sample in tangent space (which IS Euclidean) |
| **Euclidean projection math** | ✅ Fixed | Use `expmap0` instead of direction × radius |

### How It Works

1. **Encoder** outputs μ, logvar in tangent space T₀M at origin
2. **Reparameterization**: `z_tangent = μ + ε * σ` (valid - tangent space is Euclidean)
3. **Projection**: `z_hyp = expmap0(transform(z_tangent))` (true hyperbolic)
4. **Losses**: Operate on `z_hyp` using `poincare_distance`
5. **Decoder**: Receives `logmap0(z_hyp)` (back to tangent space)

### Key Insight

The tangent space at the origin T₀M **IS** Euclidean ℝⁿ. This means:
- Standard MLPs work in tangent space
- Gaussian sampling is valid in tangent space
- `expmap0`/`logmap0` provide the bridge to/from the hyperbolic manifold

### Implementation Files

- `src/models/hyperbolic_projection.py`: Uses `expmap0` for projection
- `src/models/vae.py`: Uses `logmap0` for decoder input
- `src/geometry/poincare.py`: Provides `exp_map_zero`, `log_map_zero` via geoopt

---

## Quick Reference

### Import Patterns

```python
# Core
from src.core import TERNARY
valuations = TERNARY.valuation(indices)

# Geometry
from src.geometry import poincare_distance
radii = poincare_distance(z, origin, c=1.0)

# Losses
from src.losses import CombinedLoss
loss_fn = CombinedLoss(config['loss'], curvature=1.0)

# Config
from src.config import N_TERNARY_OPERATIONS, PROJECT_ROOT
```

### Running Training

```bash
python src/train.py --config src/presets/research_extended_grokking.yaml
```

---

**Maintainer:** Claude Opus 4.5
