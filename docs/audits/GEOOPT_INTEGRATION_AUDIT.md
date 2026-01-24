# Geoopt Full Integration Audit: True 3-Adic Learning (V6.0)

**Date**: 2025-01-24 (Updated from 2025-01-23)
**Version**: V6.0 (True Hyperbolic Architecture)
**Scope**: Complete architectural overhaul for genuine hyperbolic/p-adic learning

---

## Executive Summary

### V6.0 Status: Core Requirements IMPLEMENTED

The codebase now implements **true hyperbolic learning** via geoopt's `expmap0`/`logmap0`:

| Component | V5.11 Implementation | V6.0 Implementation | Status |
|-----------|---------------------|---------------------|--------|
| Encoder output | Euclidean μ, logvar | Tangent space T₀M vectors | ✅ Fixed |
| Sampling | `mu + eps * std` | Same (tangent space IS Euclidean) | ✅ Correct |
| Projection | `direction * radius` | `expmap0(z_tangent)` | ✅ Fixed |
| Decoder input | `z_euc` (ignores z_hyp!) | `logmap0(z_hyp)` | ✅ Fixed |
| Latent type | Regular Tensor | Regular Tensor | ❌ Not using ManifoldParameter |
| Optimizer | AdamW | RiemannianAdam (ineffective) | ❌ No ManifoldParams to optimize |
| KL divergence | None | HyperbolicKLDivergence | ✅ Implemented |
| Interpolation | Linear | Linear | ❌ No geodesic wrapper |

### The Problem (V5.11 - RESOLVED)

~~The "hyperbolic" geometry only exists in the loss function's distance computation. The model learns Euclidean representations that are penalized to have hyperbolic-like distances—it never truly operates on the manifold.~~

**V6.0**: The model now truly operates on the Poincaré manifold via expmap0/logmap0.

### The Solution (IMPLEMENTED)

Use geoopt's `expmap0`/`logmap0` as the **bridge** between Euclidean MLPs and the hyperbolic manifold:

```
Encoder (Euclidean) → Tangent Space T₀M → expmap0 → Manifold → logmap0 → Decoder (Euclidean)
                      ↑                              ↓
                      └──── EUCLIDEAN ────┘    └──── HYPERBOLIC ────┘
```

**Key Insight**: Tangent space at origin T₀M IS Euclidean ℝⁿ, so standard MLPs work there. The manifold operations (expmap0, logmap0, geodesic, dist) provide the non-Euclidean structure.

---

## Part 1: Current Codebase Assessment

### 1.1 Files Audited (V6.0 Status)

```
src/
├── config/
│   ├── __init__.py          ✅ Clean exports
│   ├── constants.py          ✅ StateNet constants (trainable terminology)
│   └── paths.py              ✅ Project paths
├── core/
│   └── ternary.py            ✅ TERNARY singleton, O(1) LUTs, correct valuations
├── geometry/
│   └── poincare.py           ✅ expmap0/logmap0 via geoopt
├── losses/
│   ├── combined.py           ✅ Config-driven, weights fixed
│   ├── hyperbolic_kl.py      ✅ HyperbolicKLDivergence (NEW)
│   └── padic_geodesic.py     ✅ Seeded generators, clamped weights
├── models/
│   ├── hyperbolic_projection.py  ✅ Uses expmap0 (V6.0 FIX)
│   ├── statenet.py           ✅ Trainable terminology
│   └── vae.py                ✅ Decodes from logmap0(z_hyp) (V6.0 FIX)
├── utils/
│   ├── checkpoint.py         ✅ Clean
│   ├── checkpoint_validator.py ✅ Clean
│   ├── coverage_evaluator.py ✅ Clean
│   └── tensorboard_logger.py ✅ Clean
├── presets/                  ✅ V6.0 configs
└── train.py                  ✅ RiemannianAdam (verifying stabilize)
```

### 1.2 Critical Architectural Flaws (V6.0 RESOLVED)

#### ~~Flaw 1: Decoder Ignores Hyperbolic Embeddings~~ ✅ FIXED

**V6.0 Implementation** (`src/models/vae.py`):

```python
# V6.0 (CORRECT):
z_A_tangent = self.reparameterize(mu_A, logvar_A)
z_A_hyp = self.projections.proj_A(z_A_tangent)  # Uses expmap0

# Decoder uses logmap0 (back to tangent space)
z_A_decoded = log_map_zero(z_A_hyp, c=self.curvature)
logits_A = self.decoder_A(z_A_decoded)  # ✅ Uses hyperbolic embedding!
```

#### ~~Flaw 2: Projection is Euclidean Scaling~~ ✅ FIXED

**V6.0 Implementation** (`src/models/hyperbolic_projection.py`):

```python
# V6.0 (CORRECT):
def forward(self, z_tangent: torch.Tensor) -> torch.Tensor:
    z_transformed = self.tangent_net(z_tangent)
    z_hyp = exp_map_zero(z_transformed, c=self.curvature)  # ✅ True expmap0!
    return z_hyp
```

#### Flaw 3: RiemannianAdam Effectiveness - Verifying

**Status**: RiemannianAdam is configured but effectiveness depends on ManifoldParameter usage. Needs verification.

#### ~~Flaw 4: No KL Divergence~~ ✅ FIXED

**V6.0 Implementation**: `HyperbolicKLDivergence` class added in `src/losses/hyperbolic_kl.py` for curvature-corrected KL divergence.

---

## Part 2: Geoopt Integration Requirements (V6.0 Status)

### 2.1 Core Changes - Implementation Status

#### Change 1: Wrapped Normal Sampling ✅ IMPLEMENTED

**File**: `src/models/vae.py`

**Key Insight**: Tangent space at origin T₀M IS Euclidean ℝⁿ. Standard Gaussian sampling in tangent space is mathematically correct.

```python
# V6.0 Implementation:
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # ✅ Samples in TANGENT SPACE (which IS Euclidean at origin)
```

The projection to manifold happens separately via `expmap0` in the projection layer.

#### Change 2: Decoder Uses logmap0 ✅ IMPLEMENTED

**File**: `src/models/vae.py`

```python
# V6.0 Implementation:
z_A_tangent = self.reparameterize(mu_A, logvar_A)
z_A_hyp = self.projections.proj_A(z_A_tangent)  # expmap0 inside

# Decoder uses logmap0 (back to tangent space)
z_A_decoded = log_map_zero(z_A_hyp, c=self.curvature)
logits_A = self.decoder_A(z_A_decoded)  # ✅ Uses hyperbolic embedding!
```

#### Change 3: Projection Uses expmap0 ✅ IMPLEMENTED

**File**: `src/models/hyperbolic_projection.py`

```python
# V6.0 Implementation:
def forward(self, z_tangent: torch.Tensor) -> torch.Tensor:
    z_transformed = self.tangent_net(z_tangent)
    z_hyp = exp_map_zero(z_transformed, c=self.curvature)  # ✅ True expmap0

    # Clamp to max_radius for numerical stability
    norm = z_hyp.norm(dim=-1, keepdim=True)
    z_hyp = torch.where(norm > self.max_radius, z_hyp * self.max_radius / norm, z_hyp)
    return z_hyp
```

#### Change 4: ManifoldParameter for Latents - ❌ NOT IMPLEMENTED

**File**: `src/models/vae.py`, `src/models/hyperbolic_projection.py`

**Status**: NOT IMPLEMENTED. The `as_manifold=True` option exists in `HyperbolicProjection.forward()` but is never used. In `vae.py:215`:
```python
z_A_hyp, z_B_hyp = self.projections(z_A_tangent, z_B_tangent)  # as_manifold=False by default
```

**Impact**: RiemannianAdam falls back to standard Adam behavior since there are no ManifoldParameters.

#### Change 5: Hyperbolic KL Divergence ✅ IMPLEMENTED

**File**: `src/losses/hyperbolic_kl.py`

```python
# V6.0 Implementation:
class HyperbolicKLDivergence(nn.Module):
    """Curvature-corrected KL divergence for hyperbolic VAEs.

    Based on Mathieu et al. 2019 "Continuous Hierarchical Representations
    with Poincaré Variational Auto-Encoders"
    """

    def __init__(self, curvature: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor,
                z_hyp: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Computes curvature-corrected KL using conformal factor
        ...
```

**Integration**: Added to `src/losses/__init__.py` exports.

#### Change 6: RiemannianAdam with Stabilize - ❌ NOT EFFECTIVE

**File**: `src/train.py:624`

```python
optimizer = get_riemannian_optimizer(param_groups, lr=base_lr)  # No stabilize passed
```

**Status**:
1. `stabilize` parameter is NOT passed
2. Even if passed, RiemannianAdam only applies Riemannian updates to ManifoldParameter objects
3. Since latents are regular Tensors (see Change 4), RiemannianAdam acts like standard Adam

**Impact**: Optimization is effectively Euclidean despite using RiemannianAdam.

#### Change 7: Geodesic Interpolation - ❌ NOT IMPLEMENTED

**File**: `src/geometry/poincare.py`

**Status**: NOT IMPLEMENTED. No `geodesic()` or `geodesic_interpolation()` wrapper exists.

The geoopt manifold has `manifold.geodesic(t, x, y)` but it's not exposed in `poincare.py`. The `PoincareModule` class provides dist, proj, expmap0, logmap0, add, conformal, transport — but no geodesic.

**Impact**: Interpolation between points uses linear (Euclidean) paths instead of geodesic (hyperbolic) paths.

#### Change 8: Config Schema ✅ IMPLEMENTED

**File**: `src/presets/*.yaml`

V6.0 configs use:

```yaml
model:
  name: TernaryVAEV6Controllable
  curvature: 1.0
  learnable_curvature: true

riemannian:
  enabled: true
  optimizer: adam
```

---

### 2.2 Additional Integration (Should Implement)

#### Change 9: Numerical Precision

Geoopt recommends float64 near the boundary:

```python
if self.geometry_mode == "fully_hyperbolic":
    z_tangent = z_tangent.double()
    z_hyp = manifold.expmap0(z_tangent)
```

#### Change 10: Möbius Operations

For operations IN hyperbolic space:

```python
# Hyperbolic addition (NOT Euclidean)
z_sum = manifold.mobius_add(z1, z2)

# Hyperbolic matrix-vector (for hyperbolic MLPs)
z_transformed = manifold.mobius_matvec(W, z)
```

#### Change 11: Fréchet Mean

Replace Euclidean mean with hyperbolic centroid:

```python
# WRONG:
z_mean = z_batch.mean(dim=0)

# CORRECT:
z_mean = manifold.frechet_mean(z_batch)
```

#### Change 12: Parallel Transport

For moving vectors between tangent spaces:

```python
v_transported = manifold.transp(x, y, v)
```

---

### 2.3 New Abstractions to Create

#### ManifoldBridge Class

**New File**: `src/geometry/manifold_bridge.py`

```python
from enum import Enum
import torch
import torch.nn as nn
import geoopt

class GeometryMode(Enum):
    EUCLIDEAN_PROJECTED = "euclidean_projected"
    FULLY_HYPERBOLIC = "fully_hyperbolic"

class ManifoldBridge(nn.Module):
    """Geometry-agnostic bridge between tangent space and manifold."""

    def __init__(
        self,
        mode: str = "euclidean_projected",
        curvature: float = 1.0,
        learnable: bool = False,
        max_radius: float = 0.95,
    ):
        super().__init__()
        self.mode = GeometryMode(mode)
        self.max_radius = max_radius

        if self.mode == GeometryMode.FULLY_HYPERBOLIC:
            self.manifold = geoopt.PoincareBall(c=curvature, learnable=learnable)
        else:
            self.manifold = None
            self._init_euclidean_projection()

    def _init_euclidean_projection(self):
        """Legacy direction*radius projection."""
        self.direction_net = None  # Use input directly
        self.radius_net = None

    def to_manifold(self, z_tangent: torch.Tensor) -> torch.Tensor:
        """Map tangent space → manifold."""
        if self.mode == GeometryMode.FULLY_HYPERBOLIC:
            return self.manifold.expmap0(z_tangent)
        else:
            # Legacy: direction * radius
            norm = z_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = z_tangent / norm
            radius = torch.sigmoid(norm) * self.max_radius
            return direction * radius

    def to_tangent(self, z_manifold: torch.Tensor) -> torch.Tensor:
        """Map manifold → tangent space (Euclidean-compatible)."""
        if self.mode == GeometryMode.FULLY_HYPERBOLIC:
            return self.manifold.logmap0(z_manifold)
        else:
            return z_manifold  # Already Euclidean

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterized sampling in appropriate geometry."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_tangent = mu + eps * std
        return self.to_manifold(z_tangent)

    def distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute distance in appropriate geometry."""
        if self.mode == GeometryMode.FULLY_HYPERBOLIC:
            return self.manifold.dist(z1, z2)
        else:
            return torch.norm(z1 - z2, dim=-1)

    def geodesic(self, t: float, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Interpolate along geodesic."""
        if self.mode == GeometryMode.FULLY_HYPERBOLIC:
            return self.manifold.geodesic(t, z1, z2)
        else:
            return z1 + t * (z2 - z1)

    def get_curvature(self) -> float:
        """Get current curvature value."""
        if self.manifold is not None:
            c = self.manifold.c
            return c.item() if hasattr(c, 'item') else float(c)
        return 0.0  # Euclidean = zero curvature
```

---

## Part 3: Implementation Plan (V6.0 Status)

### Phase 1: Foundation - Partial

| Task | File | Description | Status |
|------|------|-------------|--------|
| 1.1 | `src/geometry/manifold_bridge.py` | ManifoldBridge abstraction | ❌ Not created |
| 1.2 | `src/losses/hyperbolic_kl.py` | HyperbolicKLDivergence | ✅ Done |
| 1.3 | `src/geometry/poincare.py` | expmap0, logmap0 wrappers | ✅ Done |
| 1.4 | `src/geometry/poincare.py` | geodesic wrapper | ❌ Not done |
| 1.5 | `src/config/constants.py` | Trainable terminology | ✅ Done |

### Phase 2: VAE Integration ✅ COMPLETE

| Task | File | Description | Status |
|------|------|-------------|--------|
| 2.1 | `src/models/vae.py` | V6.0 architecture | ✅ Done |
| 2.2 | `src/models/vae.py` | Tangent space sampling | ✅ Done |
| 2.3 | `src/models/vae.py` | Decoder uses logmap0(z_hyp) | ✅ Done |
| 2.4 | `src/models/hyperbolic_projection.py` | Uses expmap0 | ✅ Done |

### Phase 3: Training Integration - Partial

| Task | File | Description | Status |
|------|------|-------------|--------|
| 3.1 | `src/losses/combined.py` | Add hyperbolic KL | ❌ Not integrated |
| 3.2 | `src/train.py` | RiemannianAdam stabilize | ❌ Not passed |
| 3.3 | `src/train.py` | Geometry config | ✅ Done |
| 3.4 | `src/train.py` | Use ManifoldParameter | ❌ Not done |
| 3.5 | `src/utils/checkpoint.py` | Save/load manifold state | ❌ Not verified |

### Phase 4: Presets & Validation - Partial

| Task | File | Description | Status |
|------|------|-------------|--------|
| 4.1 | `src/presets/5.12.4.yaml` | V6.0 config | ✅ Done |
| 4.2 | `tests/test_manifold_bridge.py` | Unit tests | Pending |
| 4.3 | `tests/test_hyperbolic_vae.py` | Integration tests | Pending |

---

## Part 4: Validation Criteria

### 4.1 Geometric Correctness

```python
def test_expmap_logmap_inverse():
    """expmap0 and logmap0 should be inverses."""
    z_tangent = torch.randn(100, 16)
    z_manifold = manifold.expmap0(z_tangent)
    z_recovered = manifold.logmap0(z_manifold)
    assert torch.allclose(z_tangent, z_recovered, atol=1e-5)

def test_points_on_manifold():
    """All points should satisfy manifold constraints."""
    z_hyp = model.encode(x)
    assert manifold.check_point_on_manifold(z_hyp).all()

def test_geodesic_endpoints():
    """Geodesic at t=0 and t=1 should match endpoints."""
    z_interp_0 = manifold.geodesic(0.0, z1, z2)
    z_interp_1 = manifold.geodesic(1.0, z1, z2)
    assert torch.allclose(z_interp_0, z1)
    assert torch.allclose(z_interp_1, z2)
```

### 4.2 P-Adic Structure Preservation

```python
def test_hierarchy_in_hyperbolic():
    """High valuation pairs should have smaller geodesic distance."""
    z_hyp = model.encode(all_operations)

    for v in range(1, 10):
        high_v_mask = valuations >= v
        low_v_mask = valuations < v

        high_v_dist = manifold.dist(z_hyp[high_v_mask], origin).mean()
        low_v_dist = manifold.dist(z_hyp[low_v_mask], origin).mean()

        # Higher valuation → closer to origin
        assert high_v_dist < low_v_dist
```

### 4.3 Euclidean Bridge Works

```python
def test_tangent_output_is_euclidean():
    """logmap0 output should work with standard Euclidean operations."""
    z_hyp = model.encode_to_manifold(x)
    z_tangent = manifold.logmap0(z_hyp)

    # Should work with linear layer
    output = nn.Linear(16, 27)(z_tangent)
    assert output.shape == (batch_size, 27)
```

---

## Part 5: Expected Improvements

### Metrics (Target vs Current)

| Metric | Current (Euclidean) | Target (Hyperbolic) |
|--------|---------------------|---------------------|
| Coverage | 40% | >95% |
| Hierarchy | -0.77 | <-0.90 |
| Q metric | 1.81 | >3.0 |
| Valuation-radius correlation | Weak | Strong (Spearman > 0.9) |

### Qualitative Improvements

1. **True geodesic interpolation**: Paths between points follow manifold curvature
2. **Hierarchical clustering**: Points naturally cluster by valuation level
3. **Boundary utilization**: High-valuation points near origin, low-valuation near boundary
4. **Gradient stability**: RiemannianAdam prevents drift off manifold

---

## Part 6: File Change Summary (V6.0 Status)

| File | Status | V6.0 Status |
|------|--------|-------------|
| `src/geometry/manifold_bridge.py` | NEW | ❌ Not created |
| `src/geometry/poincare.py` | MODIFY | ✅ expmap0/logmap0, ❌ geodesic |
| `src/losses/hyperbolic_kl.py` | NEW | ✅ Done |
| `src/losses/combined.py` | MODIFY | ❌ HyperbolicKL not integrated |
| `src/models/vae.py` | MODIFY | ✅ logmap0 decoder, ❌ ManifoldParameter |
| `src/models/hyperbolic_projection.py` | MODIFY | ✅ expmap0, ⚠️ as_manifold unused |
| `src/train.py` | MODIFY | ✅ V6.0 classes, ❌ stabilize not passed |
| `src/utils/checkpoint.py` | MODIFY | ❌ Not verified |
| `src/config/constants.py` | MODIFY | ✅ Done (trainable terminology) |
| `src/presets/5.12.4.yaml` | MODIFY | ✅ Done (V6.0 config) |
| `tests/test_manifold_bridge.py` | NEW | ❌ Pending |
| `tests/test_hyperbolic_vae.py` | NEW | ❌ Pending |

---

## References

1. **Nickel & Kiela (2017)** - "Poincaré Embeddings for Learning Hierarchical Representations"
2. **Mathieu et al. (2019)** - "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
3. **Ganea et al. (2018)** - "Hyperbolic Neural Networks"
4. **Geoopt Documentation** - https://geoopt.readthedocs.io/

---

**Audit updated**: 2025-01-24
**V6.0 Implementation**: Core geometry correct (expmap0/logmap0), optimization still Euclidean
**Not Implemented**:
- ManifoldParameter for latents (`as_manifold=True` exists but unused)
- RiemannianAdam stabilize parameter
- Geodesic interpolation wrapper
- HyperbolicKL integration in combined loss
**Breaking changes**: Class names updated to V6 (TernaryVAEV6, TernaryVAEV6Controllable)
**Terminology**: "trainable" replaces "frozen" (positive logic)
