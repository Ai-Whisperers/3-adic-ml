# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Hyperbolic p-Adic Contrastive Loss using InfoNCE-style prefix similarity alignment."""

from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn

from ..core import TERNARY
from ..geometry import poincare_distance_matrix
from .base import HierarchyLossBase, MetricsDict


class HyperbolicContrastiveLoss(HierarchyLossBase):
    """Hyperbolic p-Adic Contrastive Loss.

    Identifies positive prefix-cohort pairs sharing at least prefix_k digits in
    3-adic representation, and aligns their Poincaré distance via InfoNCE.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        prefix_k: int = 3,
        curvature: float = 1.0,
        valuation_fn=None,
    ):
        super().__init__()
        self._validate_positive(temperature, "temperature", self.__class__.__name__)
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        if not (1 <= prefix_k <= 9):
            raise ValueError(f"prefix_k must be in [1, 9], got {prefix_k}")

        self.temperature = temperature
        self.prefix_k = prefix_k
        self.curvature = curvature
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation_of_difference

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        batch_size = z_hyp.size(0)
        device = z_hyp.device
        cur_c = kwargs.get("curvature", self.curvature)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_pairs": 0,
                "n_pos_pairs": 0,
                "contrastive_loss": 0.0,
            }

        # 1. Compute pairwise valuations of difference to determine prefix similarity
        # batch_indices has shape (N,) -> idx_i (1, N), idx_j (N, 1)
        idx_i = batch_indices.unsqueeze(0)
        idx_j = batch_indices.unsqueeze(1)
        
        # v_diff has shape (N, N)
        if self._valuation_fn == TERNARY.valuation_of_difference:
            v_diff = self._valuation_fn(idx_i, idx_j)
        else:
            diff = torch.abs(idx_i.long() - idx_j.long())
            diff = torch.clamp(diff, 0, TERNARY.N_OPERATIONS - 1)
            try:
                v_diff = self._valuation_fn(diff)
            except Exception:
                v_diff = TERNARY.valuation_of_difference(idx_i, idx_j)

        # 2. Define positive mask (prefix match >= prefix_k, excluding self-pairs)
        pos_mask = (v_diff >= self.prefix_k).double()
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.double)
        pos_mask = pos_mask * (1.0 - diag_mask)

        # 3. Compute pairwise hyperbolic distance matrix using geoopt stable backend
        dist_matrix = poincare_distance_matrix(z_hyp, cur_c)

        # 4. InfoNCE formulation: sim_ij = -d_ij / temp
        sim = -dist_matrix / self.temperature

        # Subtract max for numerical stability
        sim_max, _ = torch.max(sim, dim=-1, keepdim=True)
        sim_exp = torch.exp(sim - sim_max.detach())
        sim_exp = sim_exp * (1.0 - diag_mask)  # mask self-similarity

        denom_sum = torch.sum(sim_exp, dim=-1, keepdim=True)
        denom_sum = torch.clamp(denom_sum, min=1e-12)

        # log(prob)
        log_prob = (sim - sim_max.detach()) - torch.log(denom_sum)

        pos_counts = torch.sum(pos_mask, dim=-1)
        valid_queries = pos_counts > 0

        if not valid_queries.any():
            # If no positive pairs in the batch, return dummy loss of 0 with grad
            dummy_loss = dist_matrix.mean() * 0.0
            return dummy_loss, {
                "n_pairs": batch_size * (batch_size - 1),
                "n_pos_pairs": 0,
                "contrastive_loss": 0.0,
            }

        pos_loss = -torch.sum(log_prob * pos_mask, dim=-1)
        pos_loss = pos_loss[valid_queries] / pos_counts[valid_queries]
        loss = torch.mean(pos_loss)

        metrics = {
            "n_pairs": batch_size * (batch_size - 1),
            "n_pos_pairs": int(pos_mask.sum().item()),
            "contrastive_loss": float(loss.item()),
            "mean_pos_dist": float(dist_matrix[pos_mask > 0].mean().item()) if pos_mask.sum() > 0 else 0.0,
            "mean_neg_dist": float(dist_matrix[pos_mask == 0].mean().item()) if (pos_mask == 0).sum() > 0 else 0.0,
        }

        return loss, metrics
