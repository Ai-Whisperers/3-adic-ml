# Geoopt Full Integration Audit: True 3-Adic Learning

**Date**: 2025-01-23
**Version**: v5.12.8-reproducible → v6.0 (proposed)
**Scope**: Complete architectural overhaul for genuine hyperbolic/p-adic learning

---

## Executive Summary

### Current State

The codebase is a **Euclidean VAE with p-adic-inspired loss supervision**:

| Component | Current Implementation | True Hyperbolic Requirement |
|-----------|----------------------|----------------------------|
| Encoder output | Euclidean μ, logvar | Tangent space T₀M vectors |
| Sampling | `mu + eps * std` (Euclidean) | Wrapped normal via `expmap0` |
| Projection | `direction * radius` | `manifold.expmap0(z_tangent)` |
| Decoder input | `z_euc` (ignores z_hyp!) | `manifold.logmap0(z_hyp)` |
| Latent type | Regular Tensor | ManifoldParameter |
| Optimizer | AdamW (or fake RiemannianAdam) | RiemannianAdam with stabilize |
| KL divergence | None | Hyperbolic wrapped normal KL |
| Interpolation | Linear (wrong) | `manifold.geodesic()` |

### The Problem

The "hyperbolic" geometry only exists in the loss function's distance computation. The model learns Euclidean representations that are penalized to have hyperbolic-like distances—it never truly operates on the manifold.

### The Solution

Use geoopt's `expmap0`/`logmap0` as the **bridge** between Euclidean MLPs and the hyperbolic manifold:

```
Encoder (Euclidean) → Tangent Space T₀M → expmap0 → Manifold → logmap0 → Decoder (Euclidean)
                      ↑                              ↓
                      └──── EUCLIDEAN ────┘    └──── HYPERBOLIC ────┘
```

**Key Insight**: Tangent space at origin T₀M IS Euclidean ℝⁿ, so standard MLPs work there. The manifold operations (expmap0, logmap0, geodesic, dist) provide the non-Euclidean structure.

---

## Part 1: Current Codebase Assessment

### 1.1 Files Audited

```
src/
├── config/
│   ├── __init__.py          ✅ Clean exports
│   ├── constants.py          ✅ StateNet constants
│   └── paths.py              ✅ Project paths
├── core/
│   └── ternary.py            ✅ TERNARY singleton, O(1) LUTs, correct valuations
├── geometry/
│   └── poincare.py           ⚠️ Wrapper only—no expmap0/logmap0 usage
├── losses/
│   ├── combined.py           ✅ Config-driven, weights fixed
│   └── padic_geodesic.py     ✅ Seeded generators, clamped weights
├── models/
│   ├── hyperbolic_projection.py  ❌ Uses direction*radius, not expmap
│   ├── statenet.py           ✅ Fixed initialization
│   └── vae.py                ❌ Decodes from z_euc, ignores z_hyp
├── utils/
│   ├── checkpoint.py         ✅ Clean
│   ├── checkpoint_validator.py ✅ Clean
│   ├── coverage_evaluator.py ✅ Clean
│   └── tensorboard_logger.py ✅ Clean
├── presets/                  ✅ 19 YAML configs
└── train.py                  ⚠️ RiemannianAdam exists but ineffective
```

### 1.2 Critical Architectural Flaws

#### Flaw 1: Decoder Ignores Hyperbolic Embeddings

**Location**: `src/models/vae.py:245-247`

```python
# Current (WRONG):
z_A_hyp, z_B_hyp = self.projections(z_A_euc, z_B_euc)
logits_A = self.decoder_A(z_A_euc)  # ← Uses Euclidean, ignores z_hyp!
logits_B = self.decoder_B(z_B_euc)
```

The hyperbolic projection is computed but never used for decoding. The model is purely Euclidean.

#### Flaw 2: Projection is Euclidean Scaling

**Location**: `src/models/hyperbolic_projection.py:182-188`

```python
# Current (WRONG):
direction = F.normalize(z_euclidean + direction_residual, dim=-1)
radius = self.radius_net(z_euclidean) * self.max_radius
z_hyp = direction * radius  # Just Euclidean vector scaling!
```

This is NOT hyperbolic projection—it's Euclidean normalization followed by scaling. True hyperbolic projection requires `expmap0`.

#### Flaw 3: RiemannianAdam is Ineffective

**Location**: `src/train.py:650-651`

```python
optimizer = get_riemannian_optimizer(param_groups, lr=base_lr)
```

RiemannianAdam only works on `ManifoldParameter` objects. Since `z_hyp` is a regular Tensor, the optimizer falls back to Euclidean updates.

#### Flaw 4: No KL Divergence

The VAE has no KL term. For hyperbolic VAEs, this should be the wrapped normal KL divergence (Mathieu et al. 2019).

---

## Part 2: Geoopt Integration Requirements

### 2.1 Core Changes (Must Implement)

#### Change 1: Wrapped Normal Sampling

**File**: `src/models/vae.py`

```python
# BEFORE:
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# AFTER:
def reparameterize(self, mu, logvar, manifold=None):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z_tangent = mu + eps * std  # Sample in tangent space

    if manifold is not None:
        return manifold.expmap0(z_tangent)  # Wrapped normal → manifold
    return z_tangent
```

#### Change 2: Decoder Uses logmap0

**File**: `src/models/vae.py`

```python
# BEFORE:
logits_A = self.decoder_A(z_A_euc)

# AFTER:
if self.geometry_mode == "fully_hyperbolic":
    z_A_tangent = self.manifold.logmap0(z_A_hyp)
    logits_A = self.decoder_A(z_A_tangent)
else:
    logits_A = self.decoder_A(z_A_euc)
```

#### Change 3: Projection Uses expmap0

**File**: `src/models/hyperbolic_projection.py`

```python
# BEFORE:
z_hyp = direction * radius

# AFTER:
if self.geometry_mode == "fully_hyperbolic":
    z_hyp = self.manifold.expmap0(z_euclidean)
else:
    z_hyp = direction * radius  # Legacy mode
```

#### Change 4: ManifoldParameter for Latents

**File**: `src/models/vae.py`

```python
from geoopt import ManifoldParameter

# In forward():
z_hyp = self.manifold.expmap0(z_tangent)
z_hyp = ManifoldParameter(z_hyp, manifold=self.manifold)
```

#### Change 5: Hyperbolic KL Divergence

**New File**: `src/losses/hyperbolic_kl.py`

```python
import torch
import torch.nn as nn

class HyperbolicKLDivergence(nn.Module):
    """KL divergence for wrapped normal on Poincaré ball.

    Reference: Mathieu et al. 2019 "Continuous Hierarchical Representations
    with Poincaré Variational Auto-Encoders"
    """

    def __init__(self, manifold, beta: float = 1.0):
        super().__init__()
        self.manifold = manifold
        self.beta = beta

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(logvar)
        lambda_mu = self.manifold.lambda_x(mu)  # Conformal factor

        kl = 0.5 * (
            (var * lambda_mu.unsqueeze(-1).pow(2)).sum(-1) +
            mu.pow(2).sum(-1) -
            logvar.sum(-1) -
            mu.size(-1)
        )
        return self.beta * kl.mean()
```

#### Change 6: RiemannianAdam with Stabilize

**File**: `src/train.py`

```python
from geoopt.optim import RiemannianAdam

optimizer = RiemannianAdam(
    param_groups,
    lr=base_lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=weight_decay,
    stabilize=10,  # Re-project to manifold every 10 steps
)
```

#### Change 7: Geodesic Interpolation

**File**: `src/geometry/poincare.py`

```python
def geodesic_interpolation(z1, z2, t, manifold):
    """Interpolate along geodesic (not linear!)."""
    return manifold.geodesic(t, z1, z2)

# Usage for visualization:
z_path = [geodesic_interpolation(z1, z2, t, manifold)
          for t in torch.linspace(0, 1, steps=10)]
```

#### Change 8: Config Schema

**File**: `src/presets/*.yaml`

```yaml
geometry:
  mode: "fully_hyperbolic"  # or "euclidean_projected"
  curvature: 1.0
  learnable_curvature: false
  precision: "float64"  # Recommended for boundary stability

riemannian:
  enabled: true
  stabilize: 10

loss:
  hyperbolic_kl:
    enabled: true
    beta: 1.0
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

## Part 3: Implementation Plan

### Phase 1: Foundation (No Breaking Changes)

| Task | File | Description |
|------|------|-------------|
| 1.1 | `src/geometry/manifold_bridge.py` | Create ManifoldBridge abstraction |
| 1.2 | `src/losses/hyperbolic_kl.py` | Create HyperbolicKLDivergence |
| 1.3 | `src/geometry/poincare.py` | Add expmap0, logmap0, geodesic wrappers |
| 1.4 | `src/config/constants.py` | Add GeometryMode enum |

### Phase 2: VAE Integration

| Task | File | Description |
|------|------|-------------|
| 2.1 | `src/models/vae.py` | Add geometry_mode parameter |
| 2.2 | `src/models/vae.py` | Use ManifoldBridge for sampling |
| 2.3 | `src/models/vae.py` | Decoder uses logmap0(z_hyp) |
| 2.4 | `src/models/hyperbolic_projection.py` | Use expmap0 in fully_hyperbolic |

### Phase 3: Training Integration

| Task | File | Description |
|------|------|-------------|
| 3.1 | `src/losses/combined.py` | Add hyperbolic KL to loss |
| 3.2 | `src/train.py` | RiemannianAdam with stabilize |
| 3.3 | `src/train.py` | Validate geometry config |
| 3.4 | `src/utils/checkpoint.py` | Save/load manifold state |

### Phase 4: Presets & Validation

| Task | File | Description |
|------|------|-------------|
| 4.1 | `src/presets/production_hyperbolic.yaml` | New fully-hyperbolic preset |
| 4.2 | `tests/test_manifold_bridge.py` | Unit tests for ManifoldBridge |
| 4.3 | `tests/test_hyperbolic_vae.py` | Integration tests |

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

## Part 6: File Change Summary

| File | Status | Changes Required |
|------|--------|------------------|
| `src/geometry/manifold_bridge.py` | NEW | Create ManifoldBridge class |
| `src/geometry/poincare.py` | MODIFY | Add expmap0, logmap0, geodesic exports |
| `src/losses/hyperbolic_kl.py` | NEW | Create HyperbolicKLDivergence |
| `src/losses/combined.py` | MODIFY | Integrate hyperbolic KL |
| `src/models/vae.py` | MODIFY | geometry_mode, ManifoldBridge, logmap0 decoder |
| `src/models/hyperbolic_projection.py` | MODIFY | Use expmap0 in fully_hyperbolic |
| `src/train.py` | MODIFY | RiemannianAdam stabilize, geometry validation |
| `src/utils/checkpoint.py` | MODIFY | Save/load manifold curvature |
| `src/config/constants.py` | MODIFY | Add GeometryMode enum |
| `src/presets/production_hyperbolic.yaml` | NEW | Fully hyperbolic config |
| `tests/test_manifold_bridge.py` | NEW | Unit tests |
| `tests/test_hyperbolic_vae.py` | NEW | Integration tests |

---

## References

1. **Nickel & Kiela (2017)** - "Poincaré Embeddings for Learning Hierarchical Representations"
2. **Mathieu et al. (2019)** - "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
3. **Ganea et al. (2018)** - "Hyperbolic Neural Networks"
4. **Geoopt Documentation** - https://geoopt.readthedocs.io/

---

**Audit completed**: 2025-01-23
**Total changes**: 12 files (4 new, 8 modified)
**Estimated complexity**: High (architectural overhaul)
**Breaking changes**: Yes (new geometry.mode config required)
**Backward compatible**: Yes (euclidean_projected mode preserves current behavior)
