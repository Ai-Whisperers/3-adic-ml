# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Internal utilities for p-adic VAE losses."""

from typing import Union

import torch


def _exponential_target_radii(
    max_valuation: int, inner_radius: float, outer_radius: float, scale: float = 3.0
) -> torch.Tensor:
    """Precompute target radii using exponential decay matching p-adic structure."""
    import math

    levels = torch.arange(max_valuation + 1, dtype=torch.float64)
    exp_levels = torch.exp(-levels / scale)
    exp_max = math.exp(-max_valuation / scale)
    denom = 1.0 - exp_max
    return inner_radius + (outer_radius - inner_radius) * (exp_levels - exp_max) / denom


def _euclidean_to_hyperbolic_radius(
    r_euclid: torch.Tensor, c: Union[float, torch.Tensor] = 1.0
) -> torch.Tensor:
    """Convert Poincaré ball Euclidean radius to hyperbolic geodesic distance."""
    # Clamp to ensure it stays within the ball radius 1/sqrt(c)
    c_val = c if isinstance(c, float) else c.item()
    limit = 0.999 / (c_val ** 0.5 if c_val > 1e-6 else 1000.0)
    r_safe = r_euclid.clamp(min=0.0, max=limit)

    if isinstance(c, torch.Tensor):
        sqrt_c = torch.sqrt(c.clamp(min=1e-6))
        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * r_safe)
    else:
        import math
        sqrt_c = math.sqrt(max(c, 1e-6))
        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * r_safe)
