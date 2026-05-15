# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Valuation-conditioned prior losses for p-adic VAE."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..core import TERNARY
from ..utils.scatter_utils import level_has_data, level_scatter_mean
from .base import HierarchyLossBase, MetricsDict
from .utils import _exponential_target_radii


class ValuationPriorLoss(HierarchyLossBase):
    """Valuation-conditioned mean and variance prior loss."""

    def __init__(
        self,
        curvature: float = 1.0,
        inner_radius: float = 0.08,
        outer_radius: float = 0.85,
        scale: float = 3.0,
        sigma_base: float = 0.5,
        sigma_scale: float = 0.1,
        max_valuation: int = 9,
        valuation_fn=None,
    ):
        super().__init__()
        self.curvature_init = curvature
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.scale = scale
        self.sigma_base = sigma_base
        self.sigma_scale = sigma_scale
        self.max_valuation = max_valuation
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
        
        target_r_euclid = _exponential_target_radii(
            max_valuation, inner_radius, outer_radius, scale
        )
        self.register_buffer('target_r_euclid', target_r_euclid)

        v_indices = torch.arange(max_valuation + 1, dtype=torch.float64)
        target_sigmas = sigma_base * torch.exp(-v_indices * sigma_scale)
        self.register_buffer('target_sigmas', target_sigmas)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: Optional[torch.Tensor],
        batch_indices: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if curvature is None:
            curvature = self.curvature_init

        mu = mu.to(torch.float64)
        device = mu.device
        target_r = self.target_r_euclid.to(device)

        if isinstance(curvature, torch.Tensor):
            sqrt_c = torch.sqrt(curvature.clamp(min=1e-6))
            target_tangent_norms = torch.atanh(target_r.clamp(max=0.9999)) / sqrt_c
        else:
            import math
            sqrt_c = math.sqrt(max(curvature, 1e-6))
            target_tangent_norms = torch.atanh(target_r.clamp(max=0.9999)) / sqrt_c

        valuations = self._valuation_fn(batch_indices).long().clamp(0, self.max_valuation)
        
        # 1. Mean Prior Loss
        target_norms = target_tangent_norms[valuations.cpu()].to(device)
        mu_norms = torch.norm(mu, dim=-1)
        mean_loss = F.mse_loss(mu_norms, target_norms)

        # 2. Variance Prior Loss
        var_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        avg_sigma = 0.0
        if logvar is not None:
            logvar = logvar.to(torch.float64)
            sigmas = torch.exp(0.5 * logvar)
            target_s = self.target_sigmas.to(device)[valuations]
            target_s_expanded = target_s.unsqueeze(-1).expand_as(sigmas)
            var_loss = F.mse_loss(sigmas, target_s_expanded)
            with torch.no_grad():
                avg_sigma = sigmas.mean().item()

        loss = mean_loss + var_loss

        dim_size = self.max_valuation + 1
        present_mask = level_has_data(valuations, dim_size=dim_size)
        mean_norms_all = level_scatter_mean(mu_norms, valuations, dim_size=dim_size)
        gaps_all = (mean_norms_all - target_tangent_norms.to(device)).abs()

        per_level_gap_tensors: Dict[str, torch.Tensor] = {}
        per_level_gaps: Dict[str, float] = {}
        per_level_norms: Dict[str, float] = {}
        per_level_sigmas: Dict[str, float] = {}

        if logvar is not None:
            sigmas_m = torch.exp(0.5 * logvar).mean(dim=-1)
            mean_sigmas_all = level_scatter_mean(sigmas_m, valuations, dim_size=dim_size)
            for v in present_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                per_level_sigmas[f'vp_sigma_v{v}'] = mean_sigmas_all[v].detach().item()

        for v in present_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            per_level_gap_tensors[f'vp_gap_tensor_v{v}'] = gaps_all[v]
            per_level_gaps[f'vp_gap_v{v}'] = gaps_all[v].detach().item()
            per_level_norms[f'vp_mu_norm_v{v}'] = mean_norms_all[v].detach().item()

        with torch.no_grad():
            mean_mu_norm = mu_norms.mean().item()
            mean_target = target_norms.mean().item()

        metrics: Dict[str, Any] = {
            'vp_mean_mu_norm': mean_mu_norm,
            'vp_mean_target': mean_target,
            'vp_gap': abs(mean_mu_norm - mean_target),
            'vp_mean_sigma': avg_sigma,
            **per_level_norms,
            **per_level_gaps,
            **per_level_sigmas,
        }
        metrics.update(per_level_gap_tensors)

        return loss, metrics
