# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Poincare Ball geometry with geoopt backend.

This module provides numerically stable hyperbolic geometry operations
using geoopt's C++ backend for optimal performance.

IMPORTANT: All functions automatically use the device of input tensors.
The manifold cache is keyed by (curvature, device) to prevent device mismatches.

Actively Used Functions (by losses, models, train.py):
    - poincare_distance: Hyperbolic distance between points
    - hyperbolic_radius: Distance from origin (used by all losses for radii)
    - exp_map_zero: Tangent space → manifold (used by HyperbolicProjection)
    - log_map_zero: Manifold → tangent space (used by VAE decoder)
    - lambda_x: Conformal factor (used by HyperbolicKLDivergence)
    - get_riemannian_optimizer: RiemannianAdam/SGD factory
    - ManifoldParameter: Learnable hyperbolic embeddings

Available Utilities (not currently used but available):
    - project_to_poincare: Clamp points to ball
    - mobius_add: Hyperbolic translation
    - parallel_transport: Move tangent vectors
    - geodesic, geodesic_interpolation: Interpolation along geodesics
    - poincare_distance_matrix: All pairwise distances

Reference:
    Nickel & Kiela (2017) "Poincare Embeddings for Learning Hierarchical Representations"
    Mathieu et al. (2019) "Continuous Hierarchical Representations with Poincare VAEs"
"""

# geoopt is a required dependency
import geoopt
import torch
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt import PoincareBall as GeooptPoincareBall
from geoopt.optim import RiemannianAdam, RiemannianSGD

# Global manifold cache for efficiency - keyed by (curvature, device)
_manifold_cache = {}


def get_manifold(c: float = 1.0, device: torch.device | str | None = None) -> GeooptPoincareBall:
    """Get a PoincareBall manifold with specified curvature and device.

    IMPORTANT: Always pass device explicitly via `device=x.device` to ensure
    the manifold's internal tensors are on the correct device. Using device=None
    defaults to CPU which may cause device mismatch errors.

    Args:
        c: Curvature parameter (c > 0 for hyperbolic space)
        device: Device for manifold tensors. Pass tensor.device to match tensor.
                Defaults to CPU if None (not recommended for GPU training).

    Returns:
        geoopt.PoincareBall manifold with internal tensors on the specified device
    """
    # Normalize device to string for cache key
    if device is None:
        device_str = "cpu"
    elif isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    cache_key = (c, device_str)
    if cache_key not in _manifold_cache:
        manifold = geoopt.PoincareBall(c=c)
        # Move entire manifold (including k buffer) to the target device
        # PoincareBall is an nn.Module, so .to() works
        if device_str != "cpu":
            manifold = manifold.to(device_str)
        _manifold_cache[cache_key] = manifold

    return _manifold_cache[cache_key]


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, keepdim: bool = False) -> torch.Tensor:
    """Compute Poincare distance between points.

    Uses geoopt for numerical stability.

    Args:
        x: First set of points, shape (..., dim)
        y: Second set of points, shape (..., dim)
        c: Curvature parameter
        keepdim: Whether to keep the last dimension

    Returns:
        Poincare distances, shape (...) or (..., 1) if keepdim
    """
    manifold = get_manifold(c, device=x.device)
    return manifold.dist(x, y, keepdim=keepdim)


def hyperbolic_radius(z: torch.Tensor, c: float = 1.0, keepdim: bool = False) -> torch.Tensor:
    """Compute hyperbolic distance from origin (radius in Poincaré ball).

    This is the canonical way to compute radii for hierarchy losses.
    Avoids creating temporary zero tensors and ensures correct device handling.

    The hyperbolic radius is NOT the same as Euclidean norm. Near the boundary,
    hyperbolic radius grows faster than Euclidean norm due to the metric.

    Args:
        z: Points on Poincaré ball, shape (..., dim)
        c: Curvature parameter
        keepdim: Whether to keep the last dimension

    Returns:
        Hyperbolic radii (distances from origin), shape (...) or (..., 1)

    Example:
        >>> z = model.encode(x)  # Get hyperbolic embeddings
        >>> radii = hyperbolic_radius(z, c=1.0)  # Compute radii for loss
    """
    origin = torch.zeros_like(z)
    return poincare_distance(z, origin, c=c, keepdim=keepdim)


def project_to_poincare(z: torch.Tensor, max_norm: float = 0.95, c: float = 1.0) -> torch.Tensor:
    """Project points onto the Poincare ball.

    Uses geoopt.projx for stability at boundary.

    Args:
        z: Points to project, shape (..., dim)
        max_norm: Maximum norm (< 1/sqrt(c))
        c: Curvature parameter

    Returns:
        Projected points on Poincare ball
    """
    manifold = get_manifold(c, device=z.device)
    # geoopt's projx clamps to 1-eps boundary
    z_proj = manifold.projx(z)

    # Apply additional max_norm constraint if needed
    norm = torch.norm(z_proj, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / (norm + 1e-10), torch.ones_like(norm))
    return z_proj * scale


def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map from tangent space at origin to Poincare ball.

    exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)

    Args:
        v: Tangent vectors at origin, shape (..., dim)
        c: Curvature parameter

    Returns:
        Points on Poincare ball
    """
    manifold = get_manifold(c, device=v.device)
    origin = torch.zeros_like(v)
    return manifold.expmap(origin, v)


def log_map_zero(z: torch.Tensor, c: float = 1.0, max_norm: float = 0.95) -> torch.Tensor:
    """Logarithmic map from Poincare ball to tangent space at origin.

    log_0(z) = arctanh(sqrt(c) * ||z||) * z / (sqrt(c) * ||z||)

    Args:
        z: Points on Poincare ball, shape (..., dim)
        c: Curvature parameter
        max_norm: Maximum norm (for clamping near boundary)

    Returns:
        Tangent vectors at origin
    """
    manifold = get_manifold(c, device=z.device)
    origin = torch.zeros_like(z)
    return manifold.logmap(origin, z)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Mobius addition on Poincare ball.

    x (+) y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
              (1 + 2c<x,y> + c^2||x||^2||y||^2)

    Args:
        x: First operand, shape (..., dim)
        y: Second operand, shape (..., dim)
        c: Curvature parameter

    Returns:
        Result of Mobius addition
    """
    manifold = get_manifold(c, device=x.device)
    return manifold.mobius_add(x, y)


def lambda_x(x: torch.Tensor, c: float = 1.0, keepdim: bool = True) -> torch.Tensor:
    """Compute conformal factor lambda_x = 2 / (1 - c * ||x||^2).

    This factor relates Euclidean and Riemannian metrics at point x.

    Args:
        x: Points on Poincare ball, shape (..., dim)
        c: Curvature parameter
        keepdim: Whether to keep last dimension

    Returns:
        Conformal factors
    """
    manifold = get_manifold(c, device=x.device)
    return manifold.lambda_x(x, keepdim=keepdim)


def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Parallel transport tangent vector v from x to y.

    Args:
        x: Source point, shape (..., dim)
        y: Target point, shape (..., dim)
        v: Tangent vector at x, shape (..., dim)
        c: Curvature parameter

    Returns:
        Transported tangent vector at y
    """
    manifold = get_manifold(c, device=x.device)
    return manifold.transp(x, y, v)


def geodesic(x: torch.Tensor, y: torch.Tensor, t: float, c: float = 1.0) -> torch.Tensor:
    """Interpolate along geodesic from x to y at parameter t.

    The geodesic is the shortest path between two points on the Poincaré ball.
    Unlike linear interpolation, this follows the curved manifold.

    Args:
        x: Start point on Poincaré ball, shape (..., dim)
        y: End point on Poincaré ball, shape (..., dim)
        t: Interpolation parameter in [0, 1]. t=0 gives x, t=1 gives y
        c: Curvature parameter

    Returns:
        Point on geodesic at parameter t
    """
    manifold = get_manifold(c, device=x.device)
    return manifold.geodesic(t, x, y)


def geodesic_interpolation(x: torch.Tensor, y: torch.Tensor, steps: int = 10, c: float = 1.0) -> torch.Tensor:
    """Generate points along geodesic from x to y.

    Useful for visualization and analysis of paths between embeddings.

    Args:
        x: Start point on Poincaré ball, shape (..., dim)
        y: End point on Poincaré ball, shape (..., dim)
        steps: Number of interpolation steps
        c: Curvature parameter

    Returns:
        Points along geodesic, shape (steps, ..., dim)
    """
    manifold = get_manifold(c, device=x.device)
    t_values = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
    return torch.stack([manifold.geodesic(t.item(), x, y) for t in t_values])


def create_manifold_parameter(data: torch.Tensor, c: float = 1.0, requires_grad: bool = True) -> ManifoldParameter:
    """Create a learnable parameter that lives on the Poincare ball.

    This wraps a tensor as a ManifoldParameter, which:
    - Automatically projects data onto the manifold
    - Enables Riemannian gradient updates via RiemannianAdam
    - Handles boundary conditions automatically

    Args:
        data: Initial data (will be projected to manifold)
        c: Curvature parameter
        requires_grad: Whether parameter requires gradients

    Returns:
        ManifoldParameter on the Poincare ball
    """
    manifold = get_manifold(c, device=data.device)
    # Project data onto manifold for safety
    data_proj = manifold.projx(data)
    return ManifoldParameter(data_proj, manifold=manifold, requires_grad=requires_grad)


def create_manifold_tensor(data: torch.Tensor, c: float = 1.0) -> ManifoldTensor:
    """Create a non-learnable tensor on the Poincare ball.

    Like ManifoldParameter but without gradients. Useful for
    intermediate computations that should respect manifold geometry.

    Args:
        data: Initial data (will be projected to manifold)
        c: Curvature parameter

    Returns:
        ManifoldTensor on the Poincare ball
    """
    manifold = get_manifold(c, device=data.device)
    data_proj = manifold.projx(data)
    return ManifoldTensor(data_proj, manifold=manifold)


def get_riemannian_optimizer(params, lr: float = 1e-3, optimizer_type: str = "adam", **kwargs):
    """Get a Riemannian optimizer for hyperbolic parameters.

    Args:
        params: Model parameters
        lr: Learning rate
        optimizer_type: 'adam' or 'sgd'
        **kwargs: Additional optimizer arguments

    Returns:
        Riemannian optimizer
    """
    if optimizer_type == "adam":
        return RiemannianAdam(params, lr=lr, **kwargs)
    elif optimizer_type == "sgd":
        return RiemannianSGD(params, lr=lr, **kwargs)

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def poincare_distance_matrix(z: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Compute all pairwise Poincare distances (vectorized).

    Uses geoopt for numerical stability at the ball boundary.

    Args:
        z: Points on Poincare ball, shape (n, dim)
        c: Curvature parameter

    Returns:
        Distance matrix of shape (n, n)
    """
    manifold = get_manifold(c, device=z.device)

    # Expand for pairwise computation: (n, 1, dim) and (1, n, dim)
    z_i = z.unsqueeze(1)  # (n, 1, dim)
    z_j = z.unsqueeze(0)  # (1, n, dim)

    # Use geoopt's stable distance computation
    # Broadcasting: (n, 1, dim) vs (1, n, dim) -> (n, n)
    return manifold.dist(z_i, z_j, keepdim=False)


__all__ = [
    # Core functions (actively used)
    "get_manifold",
    "poincare_distance",
    "hyperbolic_radius",
    "exp_map_zero",
    "log_map_zero",
    "lambda_x",
    "get_riemannian_optimizer",
    "ManifoldParameter",
    # Utility functions (available but not currently used)
    "poincare_distance_matrix",
    "project_to_poincare",
    "mobius_add",
    "parallel_transport",
    "geodesic",
    "geodesic_interpolation",
    "create_manifold_parameter",
    "create_manifold_tensor",
    "ManifoldTensor",
    "RiemannianAdam",
    "RiemannianSGD",
]
