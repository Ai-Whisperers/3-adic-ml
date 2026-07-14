# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Poincare Ball geometry with geoopt backend.

Numerically stable hyperbolic geometry operations. All functions automatically
use the device of their input tensors. The manifold cache is keyed by
(curvature, device) to prevent device mismatches.

Actively used: poincare_distance, hyperbolic_radius, exp_map_zero, log_map_zero,
               lambda_x, get_riemannian_optimizer, ManifoldParameter.
Available:     project_to_poincare, mobius_add, parallel_transport, geodesic,
               geodesic_interpolation, poincare_distance_matrix.
"""

from typing import Any, Union

import geoopt
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt import PoincareBall as GeooptPoincareBall
from geoopt.optim import RiemannianAdam, RiemannianSGD
import torch

# Global manifold cache keyed by (curvature, device)
_manifold_cache: dict[tuple[float, str], GeooptPoincareBall] = {}
RiemannianOptimizer = Union[RiemannianAdam, RiemannianSGD]


def get_manifold(c: Union[float, torch.Tensor] = 1.0, device: torch.device | str | None = None) -> GeooptPoincareBall:
    """Return a cached PoincareBall manifold for (curvature, device).

    Always pass device=tensor.device explicitly to avoid CPU/GPU mismatches.
    Tensor curvature is keyed by id() — not c.item() — to prevent memory
    leaks and gradient breaks when the parameter value drifts across batches.
    """
    c_clamped: Union[torch.Tensor, float]
    if isinstance(c, torch.Tensor):
        c_clamped = c.clamp(min=1e-6)
        c_key: Union[float, int] = id(c)
    else:
        c_val = max(float(c), 1e-6)
        c_clamped = c_val
        c_key = c_val

    if device is None:
        device_str = "cpu"
    elif isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    cache_key = (c_key, device_str)
    if cache_key not in _manifold_cache:
        manifold = geoopt.PoincareBall(c=c_clamped)
        if device_str != "cpu":
            manifold = manifold.to(device_str)
        _manifold_cache[cache_key] = manifold

    return _manifold_cache[cache_key]


def clamp_to_max_norm(x: torch.Tensor, max_norm: Union[float, torch.Tensor]) -> torch.Tensor:
    """Scale x down (never up) so its last-dim norm does not exceed max_norm."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
    return x * (max_norm / norm).clamp(max=1.0)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: Union[float, torch.Tensor] = 1.0, keepdim: bool = False) -> torch.Tensor:
    """Poincaré geodesic distance between x and y."""
    return get_manifold(c, device=x.device).dist(x, y, keepdim=keepdim)


def hyperbolic_radius(z: torch.Tensor, c: Union[float, torch.Tensor] = 1.0, keepdim: bool = False) -> torch.Tensor:
    """Hyperbolic distance from the origin (canonical radius for hierarchy losses)."""
    return get_manifold(c, device=z.device).dist0(z, keepdim=keepdim)


def project_to_poincare(z: torch.Tensor, max_norm: float = 0.95, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Project points onto the Poincaré ball with optional max_norm constraint."""
    manifold = get_manifold(c, device=z.device)
    z_proj = manifold.projx(z)
    return clamp_to_max_norm(z_proj, max_norm)


def exp_map_zero(v: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Exponential map at the origin: tangent space → Poincaré ball."""
    manifold = get_manifold(c, device=v.device)
    return manifold.expmap(torch.zeros_like(v), v)


def log_map_zero(z: torch.Tensor, c: Union[float, torch.Tensor] = 1.0, max_norm: float | None = None) -> torch.Tensor:
    """Logarithmic map at the origin: Poincaré ball → tangent space.

    Clamps z to max_norm before logmap to avoid arctanh divergence near the
    boundary. Defaults to ball_radius - 1e-5 = 1/sqrt(c) - 1e-5.
    """
    # Clamp c the same way get_manifold() does before using it in 1/sqrt(c) —
    # c<=0 would otherwise give a complex/inf ball_radius (e.g. c**0.5 on a
    # negative float returns a complex number in Python) ahead of get_manifold's
    # own defensive clamp below.
    c_safe = c.clamp(min=1e-6) if isinstance(c, torch.Tensor) else max(float(c), 1e-6)
    ball_radius = 1.0 / (c_safe ** 0.5)
    effective_max_norm = min(
        max_norm if max_norm is not None else ball_radius - 1e-5,
        ball_radius - 1e-5,
    )
    z_clamped = clamp_to_max_norm(z, effective_max_norm)
    manifold = get_manifold(c, device=z.device)
    return manifold.logmap(torch.zeros_like(z_clamped), z_clamped)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Möbius addition on the Poincaré ball."""
    return get_manifold(c, device=x.device).mobius_add(x, y)


def lambda_x(x: torch.Tensor, c: Union[float, torch.Tensor] = 1.0, keepdim: bool = True) -> torch.Tensor:
    """Conformal factor λ_x = 2 / (1 - c‖x‖²)."""
    return get_manifold(c, device=x.device).lambda_x(x, keepdim=keepdim)


def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Parallel transport tangent vector v from x to y."""
    return get_manifold(c, device=x.device).transp(x, y, v)


def geodesic(x: torch.Tensor, y: torch.Tensor, t: float, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Interpolate along the geodesic from x to y at parameter t ∈ [0, 1]."""
    return get_manifold(c, device=x.device).geodesic(t, x, y)


def geodesic_interpolation(x: torch.Tensor, y: torch.Tensor, steps: int = 10, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Generate `steps` evenly spaced points along the geodesic from x to y."""
    manifold = get_manifold(c, device=x.device)
    t_values = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
    return torch.stack([manifold.geodesic(t.item(), x, y) for t in t_values])


def create_manifold_parameter(data: torch.Tensor, c: Union[float, torch.Tensor] = 1.0, requires_grad: bool = True) -> ManifoldParameter:
    """Learnable ManifoldParameter on the Poincaré ball (projected on creation)."""
    manifold = get_manifold(c, device=data.device)
    return ManifoldParameter(manifold.projx(data), manifold=manifold, requires_grad=requires_grad)


def create_manifold_tensor(data: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> ManifoldTensor:
    """Non-learnable ManifoldTensor on the Poincaré ball (projected on creation)."""
    manifold = get_manifold(c, device=data.device)
    return ManifoldTensor(manifold.projx(data), manifold=manifold)


def get_riemannian_optimizer(
    params: Any,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    **kwargs: Any,
) -> RiemannianOptimizer:
    """Return RiemannianAdam (default) or RiemannianSGD for the given params."""
    if optimizer_type == "adam":
        return RiemannianAdam(params, lr=lr, **kwargs)
    elif optimizer_type == "sgd":
        return RiemannianSGD(params, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def poincare_distance_matrix(z: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Compute all (n, n) pairwise Poincaré distances (vectorized)."""
    manifold = get_manifold(c, device=z.device)
    return manifold.dist(z.unsqueeze(1), z.unsqueeze(0), keepdim=False)
