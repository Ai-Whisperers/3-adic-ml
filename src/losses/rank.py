# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Ranking-based losses for p-adic VAE hierarchy."""

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from ..core import TERNARY
from ..geometry import hyperbolic_radius
from .base import HierarchyLossBase, MetricsDict


class GlobalRankLoss(HierarchyLossBase):
    """Global Rank Loss - enforces monotonic radius ordering by valuation."""

    def __init__(
        self,
        temperature: float = 0.1,
        n_pairs: int = 2000,
        use_all_pairs: bool = False,
        curvature: float = 1.0,
        valuation_fn=None,
        seed: int = 42,
        scatter_weight: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_pairs = n_pairs
        self.use_all_pairs = use_all_pairs
        self.curvature = curvature
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.scatter_weight = scatter_weight

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        device = z_hyp.device
        batch_size = z_hyp.size(0)
        cur_c = kwargs.get("curvature", self.curvature)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {}

        actual_radius = hyperbolic_radius(z_hyp, c=cur_c)
        valuations = self._valuation_fn(batch_indices).double()

        if self.use_all_pairs:
            i_idx, j_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
        else:
            n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
            i_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(device)
            j_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(device)
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % batch_size

        v_i, v_j = valuations[i_idx], valuations[j_idx]
        r_i, r_j = actual_radius[i_idx], actual_radius[j_idx]

        higher_v_mask = v_i > v_j
        lower_v_mask = v_i < v_j
        same_v_mask = v_i == v_j

        loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_viol = 0

        if higher_v_mask.any():
            # v_i > v_j => r_i should be < r_j.
            # Violation if r_i - r_j > 0.
            viol_high = F.sigmoid((r_i[higher_v_mask] - r_j[higher_v_mask]) / self.temperature)
            loss = loss + viol_high.mean()
            n_viol += (r_i[higher_v_mask] > r_j[higher_v_mask]).sum().item()

        if lower_v_mask.any():
            # v_i < v_j => r_i should be > r_j.
            # Violation if r_j - r_i > 0.
            viol_low = F.sigmoid((r_j[lower_v_mask] - r_i[lower_v_mask]) / self.temperature)
            loss = loss + viol_low.mean()
            n_viol += (r_j[lower_v_mask] > r_i[lower_v_mask]).sum().item()

        scatter_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if self.scatter_weight > 0 and same_v_mask.any():
            scatter_loss = F.mse_loss(r_i[same_v_mask], r_j[same_v_mask])
            loss = loss + self.scatter_weight * scatter_loss

        n_pairs_total = i_idx.numel()
        metrics = {
            "n_pairs": n_pairs_total,
            "rank_violations": float(n_viol),
            "violation_rate": float(n_viol / n_pairs_total) if n_pairs_total > 0 else 0.0,
            "rank_loss": loss.item(),
            "scatter_loss": scatter_loss.item(),
        }

        # Per-level scatter diagnostics
        if same_v_mask.any():
            from ..utils.scatter_utils import level_scatter_mean
            v_same = v_i[same_v_mask].long()
            r_diff_sq = (r_i[same_v_mask] - r_j[same_v_mask])**2
            scatter_all = level_scatter_mean(r_diff_sq, v_same, dim_size=10)
            for v in range(10):
                if (v_same == v).any():
                    metrics[f'scatter_v{v}'] = scatter_all[v].item()

        return loss, metrics
