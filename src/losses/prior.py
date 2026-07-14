# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Valuation-conditioned prior losses for p-adic VAE."""

from typing import Any, cast, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..core import TERNARY
from ..utils.scatter_utils import level_has_data, level_scatter_mean
from .base import HierarchyLossBase, MetricsDict
from .utils import (
    _euclidean_to_hyperbolic_radius,
    _exponential_target_radii,
    default_valuation_fn,
    make_zero_loss,
)


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
        self._valuation_fn = default_valuation_fn(valuation_fn, TERNARY.valuation)

        target_r_euclid = _exponential_target_radii(
            max_valuation, inner_radius, outer_radius, scale
        )
        self.register_buffer('target_r_euclid', target_r_euclid)

        v_indices = torch.arange(max_valuation + 1, dtype=torch.float64)
        target_sigmas = sigma_base * torch.exp(-v_indices * sigma_scale)
        self.register_buffer('target_sigmas', target_sigmas)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Forward pass matching HierarchyLossBase contract.

        Args:
            z_hyp: Hyperbolic embeddings (mu in tangent space for this loss)
            batch_indices: Operation indices
            **kwargs: Must contain 'logvar' (Optional[Tensor]) and 'curvature'

        Returns:
            Tuple of (loss, metrics)
        """
        mu = z_hyp
        logvar: Optional[torch.Tensor] = kwargs.get("logvar")
        curvature: Optional[Union[float, torch.Tensor]] = kwargs.get("curvature")

        if curvature is None:
            curvature = self.curvature_init

        mu = mu.to(torch.float64)
        device = mu.device
        target_r = self.target_r_euclid.to(device)
        target_tangent_norms = _euclidean_to_hyperbolic_radius(target_r, c=curvature)

        vals_raw = self._valuation_fn(batch_indices)
        valuations = cast(torch.Tensor, vals_raw).long().clamp(0, self.max_valuation)

        # 1. Mean Prior Loss
        target_norms = target_tangent_norms[valuations.cpu()].to(device)
        mu_norms = torch.norm(mu, dim=-1)
        mean_loss = F.mse_loss(mu_norms, target_norms)

        # 2. Variance Prior Loss
        var_loss = make_zero_loss(device)
        avg_sigma = 0.0
        if logvar is not None:
            logvar = logvar.to(torch.float64)
            sigmas = torch.exp(0.5 * logvar)
            target_s = self.target_sigmas.to(device)[valuations]
            target_s_expanded = target_s.unsqueeze(-1).expand_as(sigmas)
            var_loss = F.mse_loss(sigmas, target_s_expanded)
            with torch.no_grad():
                avg_sigma = float(sigmas.mean().item())

        loss = mean_loss + var_loss

        dim_size = self.max_valuation + 1
        valuations_long = cast(torch.LongTensor, valuations)
        present_mask = level_has_data(valuations_long, dim_size=dim_size)
        mean_norms_all = level_scatter_mean(mu_norms, valuations_long, dim_size=dim_size)
        gaps_all = (mean_norms_all - target_tangent_norms.to(device)).abs()

        metrics: MetricsDict = {
            'vp_mean_mu_norm': float(mu_norms.mean().item()),
            'vp_mean_target': float(target_norms.mean().item()),
            'vp_gap': float(abs(mu_norms.mean().item() - target_norms.mean().item())),
            'vp_mean_sigma': avg_sigma,
        }

        if logvar is not None:
            sigmas_m = torch.exp(0.5 * logvar).mean(dim=-1)
            mean_sigmas_all = level_scatter_mean(sigmas_m, valuations_long, dim_size=dim_size)
            for v in present_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                metrics[f'vp_sigma_v{v}'] = float(mean_sigmas_all[v].detach().item())

        for v in present_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            metrics[f'vp_gap_v{v}'] = float(gaps_all[v].detach().item())
            metrics[f'vp_mu_norm_v{v}'] = float(mean_norms_all[v].detach().item())
            # In-graph tensor for Lagrangian dual penalties
            metrics[f'vp_gap_tensor_v{v}'] = gaps_all[v]

        return loss, metrics
