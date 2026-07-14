# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Internal utilities for p-adic VAE losses."""

import math
from typing import Callable, Optional, Tuple, Union

import torch


def make_zero_loss(
    device: torch.device, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return a scalar zero tensor on the given device, suitable as a loss value."""
    return torch.tensor(0.0, device=device, dtype=dtype)


def default_valuation_fn(fn: Optional[Callable], default_fn: Callable) -> Callable:
    """Return fn if explicitly provided, else default_fn (e.g. TERNARY.valuation)."""
    return fn if fn is not None else default_fn


def sample_random_pairs(
    batch_size: int, n_pairs: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n_pairs random index pairs (i, j), guaranteeing i != j via cyclic shift."""
    i_idx = torch.randint(0, batch_size, (n_pairs,), device=device)
    j_idx = torch.randint(0, batch_size, (n_pairs,), device=device)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % batch_size
    return i_idx, j_idx


def safe_corrcoef(
    a: torch.Tensor, b: torch.Tensor, nan_if_insufficient: bool = False
) -> torch.Tensor:
    """Pearson correlation between two 1-D tensors.

    With fewer than 2 elements the correlation is mathematically undefined:
    returns NaN when nan_if_insufficient=True (F-08: makes the "undefined"
    case visibly distinct from an actual zero correlation), else 0.0. If
    corrcoef itself produces NaN (e.g. constant input), always returns 0.0.
    """
    if a.numel() < 2:
        if nan_if_insufficient:
            return torch.tensor(float("nan"), device=a.device, dtype=a.dtype)
        return make_zero_loss(a.device, a.dtype)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    if torch.isnan(corr):
        return make_zero_loss(a.device, a.dtype)
    return corr


def weight_to_log_sigma(w: float) -> float:
    """Convert a positive loss weight to its log_sigma initializer for learnable weighting.

    Derived from the homoscedastic uncertainty formula: weight = 0.5 * exp(-2 * log_sigma).
    Solving for log_sigma: log_sigma = -0.5 * log(2 * w).
    """
    return -0.5 * math.log(max(2.0 * w, 1e-8))


def compute_coverage_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy reconstruction loss supporting (B,9,3) and (B,27) logit shapes."""
    targets_shifted = (targets + 1).long().clamp(0, 2)
    if logits.shape[-1] == 3:
        return torch.nn.functional.cross_entropy(logits.view(-1, 3), targets_shifted.view(-1))
    elif logits.shape[-1] == 27:
        return torch.nn.functional.cross_entropy(
            logits.view(-1, 9, 3).permute(0, 2, 1), targets_shifted
        )
    return make_zero_loss(logits.device)


def _exponential_target_radii(
    max_valuation: int, inner_radius: float, outer_radius: float, scale: float = 3.0
) -> torch.Tensor:
    """Precompute target radii using exponential decay matching p-adic structure."""
    levels = torch.arange(max_valuation + 1, dtype=torch.float64)
    exp_levels = torch.exp(-levels / scale)
    exp_max = math.exp(-max_valuation / scale)
    denom = 1.0 - exp_max
    return inner_radius + (outer_radius - inner_radius) * (exp_levels - exp_max) / denom


def normalize_to_direction(z_hyp: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Return unit-direction vectors for Poincaré ball points.

    Divides each point by its radius, clamping small radii to avoid division
    by zero.  Used by angular losses that operate in direction space.

    Args:
        z_hyp: Points on Poincaré ball, shape (B, D)
        r:     Euclidean radii, shape (B,)

    Returns:
        Unit-direction tensors, shape (B, D)
    """
    return z_hyp / r.unsqueeze(-1).clamp(min=1e-10)


def _euclidean_to_hyperbolic_radius(
    r_euclid: torch.Tensor, c: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """Convert Poincaré ball Euclidean radius to hyperbolic geodesic distance."""
    c_val = c if isinstance(c, float) else c.item()
    limit = 0.999 / (c_val ** 0.5 if c_val > 1e-6 else 1000.0)
    r_safe = r_euclid.clamp(min=0.0, max=limit)

    if isinstance(c, torch.Tensor):
        sqrt_c = torch.sqrt(c.clamp(min=1e-6))
        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * r_safe)
    else:
        sqrt_c_float = math.sqrt(max(c, 1e-6))
        return (2.0 / sqrt_c_float) * torch.atanh(sqrt_c_float * r_safe)


def compute_target_radii(
    max_valuation: int,
    inner_radius: float,
    outer_radius: float,
    c: Union[float, torch.Tensor],
    device: torch.device,
    scale: float = 3.0,
) -> torch.Tensor:
    """Target hyperbolic radii per valuation level (exponential decay, then hyperbolic-mapped)."""
    return _euclidean_to_hyperbolic_radius(
        _exponential_target_radii(max_valuation, inner_radius, outer_radius, scale=scale).to(device),
        c=c,
    )


def phase_gated_zero(z_ref: torch.Tensor, metrics: dict) -> Tuple[torch.Tensor, dict]:
    """Build a (zero_loss, metrics) pair for a phase-gated loss that hasn't activated yet."""
    return make_zero_loss(z_ref.device, z_ref.dtype), dict(metrics)
