# Geometry Module Audit: src/geometry/

**Date**: 2025-01-23
**Scope**: `src/geometry/__init__.py`, `src/geometry/poincare.py`
**Lines of Code**: 357 (poincare.py) + 20 (__init__.py)

---

## Executive Summary

The geometry module is a **well-structured geoopt wrapper** that provides the foundational hyperbolic operations needed for true manifold learning. Unlike the rest of the codebase (which doesn't use these operations properly), this module correctly implements expmap, logmap, Möbius addition, parallel transport, and Riemannian optimizers.

**Verdict**: The module is **ready for true hyperbolic learning**—the issue is that the rest of the codebase doesn't use it properly.

---

## File Structure

```
src/geometry/
├── __init__.py      # Clean re-exports (19 symbols)
└── poincare.py      # Geoopt wrapper (357 lines)
```

---

## Detailed Analysis

### 1. Module Exports (__init__.py)

| Export | Type | Purpose |
|--------|------|---------|
| `get_manifold` | Function | Cached manifold factory |
| `poincare_distance` | Function | Geodesic distance |
| `poincare_distance_matrix` | Function | Pairwise distances |
| `project_to_poincare` | Function | Ball projection |
| `exp_map_zero` | Function | T₀M → Manifold |
| `log_map_zero` | Function | Manifold → T₀M |
| `mobius_add` | Function | Hyperbolic addition |
| `lambda_x` | Function | Conformal factor |
| `parallel_transport` | Function | Vector transport |
| `PoincareModule` | Class | Base module for hyperbolic layers |
| `create_manifold_parameter` | Function | Learnable manifold point |
| `create_manifold_tensor` | Function | Non-learnable manifold point |
| `get_riemannian_optimizer` | Function | Optimizer factory |
| `ManifoldParameter` | Class | Geoopt re-export |
| `ManifoldTensor` | Class | Geoopt re-export |
| `RiemannianAdam` | Class | Geoopt re-export |
| `RiemannianSGD` | Class | Geoopt re-export |

**Assessment**: ✅ Complete and well-organized exports.

---

### 2. Manifold Cache (Lines 39-70)

```python
_manifold_cache = {}

def get_manifold(c: float = 1.0, device: torch.device | str | None = None) -> GeooptPoincareBall:
    cache_key = (c, device_str)
    if cache_key not in _manifold_cache:
        manifold = geoopt.PoincareBall(c=c)
        if device_str != "cpu":
            manifold = manifold.to(device_str)
        _manifold_cache[cache_key] = manifold
    return _manifold_cache[cache_key]
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Caching | ✅ Good | Avoids recreating manifolds |
| Device handling | ✅ Good | Properly moves to GPU |
| Curvature keying | ✅ Good | Separate manifold per curvature |
| Memory | ⚠️ Minor | Cache grows unbounded (unlikely issue in practice) |
| Thread safety | ⚠️ Minor | Not thread-safe for multi-process (rare issue) |

---

### 3. Core Operations

#### 3.1 poincare_distance (Lines 73-88)

```python
def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, keepdim: bool = False) -> torch.Tensor:
    manifold = get_manifold(c, device=x.device)
    return manifold.dist(x, y, keepdim=keepdim)
```

**Assessment**: ✅ **Correct**. Properly delegates to geoopt's numerically stable implementation.

---

#### 3.2 project_to_poincare (Lines 91-111)

```python
def project_to_poincare(z: torch.Tensor, max_norm: float = 0.95, c: float = 1.0) -> torch.Tensor:
    manifold = get_manifold(c, device=z.device)
    z_proj = manifold.projx(z)  # geoopt projection

    # Additional max_norm constraint
    norm = torch.norm(z_proj, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / (norm + 1e-10), torch.ones_like(norm))
    return z_proj * scale
```

| Aspect | Status | Notes |
|--------|--------|-------|
| geoopt projx | ✅ Correct | Uses stable boundary projection |
| max_norm clipping | ⚠️ Redundant | projx already ensures ||z|| < 1/√c |
| Gradient continuity | ⚠️ Concern | torch.where may cause gradient discontinuity at max_norm threshold |

**Issue**: The double projection is technically correct but redundant. If `max_norm < 1/√c` (which 0.95 < 1.0 is), the additional clipping provides a safety margin but could cause gradient issues.

---

#### 3.3 exp_map_zero (Lines 114-128)

```python
def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    manifold = get_manifold(c, device=v.device)
    origin = torch.zeros_like(v)
    return manifold.expmap(origin, v)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Mathematical correctness | ✅ Correct | exp₀(v) = tanh(√c||v||) · v/(√c||v||) |
| Implementation | ⚠️ Suboptimal | Creates zero tensor; geoopt has direct `expmap0` |

**Suggestion**: Use `manifold.expmap0(v)` directly instead of `manifold.expmap(zeros, v)`.

---

#### 3.4 log_map_zero (Lines 131-146)

```python
def log_map_zero(z: torch.Tensor, c: float = 1.0, max_norm: float = 0.95) -> torch.Tensor:
    manifold = get_manifold(c, device=z.device)
    origin = torch.zeros_like(z)
    return manifold.logmap(origin, z)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Mathematical correctness | ✅ Correct | log₀(z) = arctanh(√c||z||) · z/(√c||z||) |
| Implementation | ⚠️ Suboptimal | Creates zero tensor; geoopt has direct `logmap0` |
| max_norm parameter | ❌ Unused | Parameter accepted but never used (dead code) |

**Issues**:
1. `max_norm` parameter is documented but never used
2. Should use `manifold.logmap0(z)` directly

---

#### 3.5 mobius_add (Lines 149-164)

```python
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    manifold = get_manifold(c, device=x.device)
    return manifold.mobius_add(x, y)
```

**Assessment**: ✅ **Correct**. Proper delegation to geoopt's Möbius addition.

---

#### 3.6 lambda_x (Lines 167-181)

```python
def lambda_x(x: torch.Tensor, c: float = 1.0, keepdim: bool = True) -> torch.Tensor:
    manifold = get_manifold(c, device=x.device)
    return manifold.lambda_x(x, keepdim=keepdim)
```

**Assessment**: ✅ **Correct**. Conformal factor λₓ = 2/(1 - c||x||²) properly delegated.

---

#### 3.7 parallel_transport (Lines 184-197)

```python
def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    manifold = get_manifold(c, device=x.device)
    return manifold.transp(x, y, v)
```

**Assessment**: ✅ **Correct**. Proper delegation to geoopt's parallel transport.

---

### 4. PoincareModule Class (Lines 200-252)

```python
class PoincareModule(nn.Module):
    def __init__(self, c: float = 1.0, max_norm: float = 0.95):
        super().__init__()
        self.c = c
        self.max_norm = max_norm
        self._manifold = get_manifold(c)  # ← Created on CPU!
```

| Method | Status | Notes |
|--------|--------|-------|
| `__init__` | ⚠️ Issue | Manifold created on CPU, may cause device mismatch |
| `dist` | ✅ Good | Delegates to poincare_distance |
| `proj` | ✅ Good | Delegates to project_to_poincare |
| `expmap0` | ✅ Good | Delegates to exp_map_zero |
| `logmap0` | ✅ Good | Delegates to log_map_zero |
| `add` | ✅ Good | Delegates to mobius_add |
| `conformal` | ✅ Good | Delegates to lambda_x |
| `transport` | ✅ Good | Delegates to parallel_transport |

**Issue**: Manifold is created on CPU at `__init__` time. If the module is later moved to GPU via `.to(device)`, the internal `_manifold` stays on CPU, potentially causing device mismatches.

**Fix**: Override `to()` method or lazily get manifold in each operation.

---

### 5. Helper Functions

#### 5.1 create_manifold_parameter (Lines 255-274)

```python
def create_manifold_parameter(data: torch.Tensor, c: float = 1.0, requires_grad: bool = True) -> ManifoldParameter:
    manifold = get_manifold(c, device=data.device)
    data_proj = manifold.projx(data)
    return ManifoldParameter(data_proj, manifold=manifold, requires_grad=requires_grad)
```

**Assessment**: ✅ **Correct**. Properly projects data and wraps as ManifoldParameter.

---

#### 5.2 create_manifold_tensor (Lines 277-292)

```python
def create_manifold_tensor(data: torch.Tensor, c: float = 1.0) -> ManifoldTensor:
    manifold = get_manifold(c, device=data.device)
    data_proj = manifold.projx(data)
    return ManifoldTensor(data_proj, manifold=manifold)
```

**Assessment**: ✅ **Correct**. Proper non-learnable manifold tensor creation.

---

#### 5.3 get_riemannian_optimizer (Lines 295-312)

```python
def get_riemannian_optimizer(params, lr: float = 1e-3, optimizer_type: str = "adam", **kwargs):
    if optimizer_type == "adam":
        return RiemannianAdam(params, lr=lr, **kwargs)
    elif optimizer_type == "sgd":
        return RiemannianSGD(params, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Adam support | ✅ Good | Delegates to RiemannianAdam |
| SGD support | ✅ Good | Delegates to RiemannianSGD |
| stabilize parameter | ⚠️ Missing | Not exposed; should be passed via kwargs |
| Default betas | ⚠️ Missing | Relies on geoopt defaults |

**Suggestion**: Consider exposing `stabilize` parameter explicitly for re-projection frequency.

---

#### 5.4 poincare_distance_matrix (Lines 315-335)

```python
def poincare_distance_matrix(z: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    manifold = get_manifold(c, device=z.device)
    z_i = z.unsqueeze(1)  # (n, 1, dim)
    z_j = z.unsqueeze(0)  # (1, n, dim)
    return manifold.dist(z_i, z_j, keepdim=False)
```

**Assessment**: ✅ **Correct**. Efficient vectorized pairwise distance computation.

---

## Missing Operations

| Operation | Status | Use Case |
|-----------|--------|----------|
| `geodesic(t, x, y)` | ❌ Missing | Interpolation along geodesic |
| `frechet_mean(z_batch)` | ❌ Missing | Hyperbolic centroid for batch stats |
| `mobius_matvec(M, x)` | ❌ Missing | Hyperbolic linear layer |
| `mobius_scalar_mul(r, x)` | ❌ Missing | Scalar multiplication on manifold |
| `gyration(u, v, w)` | ❌ Missing | Gyrovector space operation |
| `dist0(x)` | ❌ Missing | Distance from origin (radius) |

**Recommendation**: Add these for complete hyperbolic functionality:

```python
def geodesic(t: float, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Interpolate along geodesic from x to y."""
    manifold = get_manifold(c, device=x.device)
    return manifold.geodesic(t, x, y)

def frechet_mean(z: torch.Tensor, c: float = 1.0, max_iter: int = 100) -> torch.Tensor:
    """Compute Fréchet mean (hyperbolic centroid)."""
    # geoopt doesn't have built-in frechet_mean; implement iterative algorithm
    ...

def dist_to_origin(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Distance from origin (hyperbolic radius)."""
    manifold = get_manifold(c, device=x.device)
    origin = torch.zeros_like(x)
    return manifold.dist(origin, x)
```

---

## Issues Summary

### Critical (0)

None. The module is mathematically correct.

### High (0)

None. Core operations work properly.

### Medium (4)

| Issue | Location | Description |
|-------|----------|-------------|
| M1 | `log_map_zero:131` | `max_norm` parameter is unused (dead code) |
| M2 | `exp_map_zero:127-128` | Creates unnecessary zero tensor; use `expmap0` directly |
| M3 | `log_map_zero:145-146` | Creates unnecessary zero tensor; use `logmap0` directly |
| M4 | `PoincareModule:219` | Manifold created on CPU at init; device mismatch risk |

### Low (3)

| Issue | Location | Description |
|-------|----------|-------------|
| L1 | `project_to_poincare:106-111` | Redundant double projection |
| L2 | `_manifold_cache:40` | Unbounded cache growth (unlikely issue) |
| L3 | `get_riemannian_optimizer` | `stabilize` parameter not explicitly exposed |

---

## Code Quality Assessment

| Metric | Score | Notes |
|--------|-------|-------|
| Correctness | 9/10 | All operations mathematically correct |
| Completeness | 7/10 | Missing geodesic, frechet_mean, mobius_matvec |
| Documentation | 8/10 | Good docstrings with formulas |
| API Design | 8/10 | Clean, consistent function signatures |
| Error Handling | 6/10 | Missing input validation |
| Numerical Stability | 9/10 | Proper geoopt delegation |

---

## Recommendations

### Must Fix (Before True Hyperbolic Mode)

1. **Remove unused `max_norm` from `log_map_zero`** or use it:
   ```python
   def log_map_zero(z: torch.Tensor, c: float = 1.0) -> torch.Tensor:
       manifold = get_manifold(c, device=z.device)
       return manifold.logmap0(z)  # Use direct method
   ```

2. **Use direct expmap0/logmap0**:
   ```python
   def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
       manifold = get_manifold(c, device=v.device)
       return manifold.expmap0(v)  # Direct, no zero tensor
   ```

3. **Fix PoincareModule device handling**:
   ```python
   @property
   def manifold(self):
       # Lazily get manifold on correct device
       return get_manifold(self.c, device=next(self.parameters()).device)
   ```

### Should Add (For Complete Functionality)

1. **geodesic** - For interpolation/visualization
2. **dist_to_origin** - For radius computation (used everywhere in losses)
3. **frechet_mean** - For batch statistics

### Consider Adding (For Advanced Use)

1. **mobius_matvec** - For hyperbolic MLPs (Ganea et al.)
2. **Learnable curvature wrapper** - For adaptive geometry

---

## Verdict

**The geometry module is well-implemented and ready for true hyperbolic learning.** The issues are minor (dead parameter, suboptimal but correct implementations). The real problem is that the rest of the codebase (vae.py, hyperbolic_projection.py) doesn't use these functions properly—the expmap0/logmap0 bridge exists here but isn't utilized in the VAE forward pass.

**Rating**: 8/10 (Good, with minor improvements needed)

---

**Audit completed**: 2025-01-23
**Auditor**: Claude Opus 4.5
