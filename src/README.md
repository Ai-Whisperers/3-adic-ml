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

### Data Flow

```
Input (ternary ops)
       │
       ▼
   ┌───────────┐
   │  Encoder  │ → μ, σ (Euclidean)
   └───────────┘
       │
       ▼ Reparameterization
   ┌───────────┐
   │  z_euc    │ (Euclidean latent)
   └───────────┘
       │
       ▼ Hyperbolic Projection
   ┌───────────┐
   │  z_hyp    │ (Poincare ball)
   └───────────┘
       │
       ├──────────────────┐
       ▼                  ▼
   ┌───────────┐    ┌───────────┐
   │  Decoder  │    │  Losses   │ (uses z_hyp for hierarchy)
   └───────────┘    └───────────┘
       │
       ▼
   Output (logits)
```

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

## Architecture Issues (Active)

| Issue | Severity | Description |
|-------|----------|-------------|
| **Decoder uses z_euc** | 🔴 Critical | Decoder ignores z_hyp, making architecture Euclidean with hyperbolic supervision |
| **Euclidean reparameterization** | 🟠 High | Should use wrapped normal on manifold |
| **Euclidean projection math** | 🟠 High | Should use expmap0, not direction × radius |

See `docs/audits/MODELS_MODULE_AUDIT.md` for details.

### Critical Issue: Decoder Uses z_euc (Detailed Analysis)

**Current Data Flow (vae.py:226-261):**
```
Encoder → μ, σ
    ↓
Reparameterize: z_euc = μ + σ*ε  (Euclidean sampling)
    ↓
    ├── Projection: z_hyp = project(z_euc)  → Poincaré ball
    │       ↓
    │   Losses (hierarchy, radial, geodesic) ← uses z_hyp ✓
    │
    └── Decoder: logits = decoder(z_euc)    ← uses z_euc ✗
            ↓
        Reconstruction loss
```

**The Asymmetry:**
- **Decoder receives:** `z_euc` (line 246-247) - Euclidean latent
- **Losses receive:** `z_hyp` (train.py:718) - Hyperbolic projection
- They operate on **different representations** of the same sample

**Code Location:**
```python
# src/models/vae.py lines 246-247
logits_A = self.decoder_A(z_A_euc)  # ← ISSUE: uses Euclidean
logits_B = self.decoder_B(z_B_euc)
```

**Why This Matters:**
- Reconstruction optimizes Euclidean geometry
- Hierarchy losses optimize hyperbolic geometry
- These objectives are **partially decoupled** - not fully coherent

**Historical Context:**
- V5.12.1 config proposed using `log_map_zero(z_hyp)` for decoder input
- This was **documented but never implemented**
- Current design is intentional compatibility layer for v5.5 frozen checkpoint

**Fix Options:**

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **A** | `log_map_zero(z_hyp)` → decoder | Geometrically coherent | Needs decoder retraining |
| **B** | Direct `z_hyp` → decoder | Simple | Breaks norm assumptions |
| **C** | Learnable mapping layer | Gradual transition | Adds parameters |

**Recommended (Option A):**
```python
from src.geometry import log_map_zero
z_A_tangent = log_map_zero(z_A_hyp, c=curvature)
logits_A = self.decoder_A(z_A_tangent)
```

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
