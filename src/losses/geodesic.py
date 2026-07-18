# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""P-Adic Geodesic Loss for hyperbolic embedding alignment."""

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from ..core import TERNARY
from ..geometry import poincare_distance
from .base import HierarchyLossBase, MetricsDict
from .utils import (
    default_valuation_fn,
    make_zero_loss,
    phase_gated_zero,
    safe_corrcoef,
    sample_random_pairs,
)


class PAdicGeodesicLoss(HierarchyLossBase):
    """Unified P-Adic Geodesic Loss.

    Aligns pairwise Poincaré distances with 3-adic valuations.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_target_distance: float = 3.0,
        valuation_scale: float = 3.0,
        n_pairs: int = 2000,
        use_smooth_l1: bool = True,
        use_individual_valuation: bool = False,
        valuation_fn=None,
        seed: int = 42,
    ):
        super().__init__()
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        self._validate_positive(max_target_distance, "max_target_distance", self.__class__.__name__)
        self._validate_positive(valuation_scale, "valuation_scale", self.__class__.__name__)
        if n_pairs < 1:
            raise ValueError(f"PAdicGeodesicLoss: n_pairs must be >= 1, got {n_pairs}")
        self.curvature = curvature
        self.max_target = max_target_distance
        self.valuation_scale = valuation_scale
        self.n_pairs = n_pairs
        self.use_smooth_l1 = use_smooth_l1
        self.use_individual_valuation = use_individual_valuation
        self._valuation_fn = default_valuation_fn(valuation_fn, TERNARY.valuation)
        self.max_valuation = float(TERNARY.MAX_VALUATION)
        # No CPU generator — randint uses device=device directly to avoid
        # the index-tensor transfer from CPU to training device each forward pass.

    def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
        """Map 3-adic valuation to target hyperbolic distance."""
        return self.max_target * torch.exp(-valuation / self.valuation_scale)

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
            return phase_gated_zero(z_hyp, {"n_pairs": 0})

        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        i_idx, j_idx = sample_random_pairs(batch_size, n_pairs, device)

        d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], cur_c)

        if self.use_individual_valuation:
            v_i = self._valuation_fn(batch_indices[i_idx]).double()
            v_j = self._valuation_fn(batch_indices[j_idx]).double()
            val_diff = torch.abs(v_i - v_j)
            cross_mask = val_diff > 0
            if not cross_mask.any():
                return phase_gated_zero(z_hyp, {"n_pairs": 0})
            d_actual = d_actual[cross_mask]
            val_diff = val_diff[cross_mask]
            d_target = self.max_target * val_diff / self.max_valuation
            valuation = val_diff
        else:
            diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())
            valuation = TERNARY.valuation(diff).double()
            d_target = self.target_distance(valuation)

        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(d_actual, d_target)
        else:
            loss = F.mse_loss(d_actual, d_target)

        with torch.no_grad():
            corr = safe_corrcoef(d_actual, d_target, nan_if_insufficient=True)

            mean_d_low_v = d_actual[valuation < 2].mean() if (valuation < 2).any() else make_zero_loss(device)
            mean_d_high_v = d_actual[valuation >= 4].mean() if (valuation >= 4).any() else make_zero_loss(device)

        metrics = {
            "n_pairs": n_pairs,
            "mean_d_actual": d_actual.mean().item(),
            "mean_d_target": d_target.mean().item(),
            "distance_correlation": corr.item(),
            "mean_d_low_valuation": mean_d_low_v.item(),
            "mean_d_high_valuation": mean_d_high_v.item(),
        }

        return loss, metrics


class TaxonomyGeodesicLoss(HierarchyLossBase):
    """Fase 1 of docs/plans/TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md ("Condition D").

    Structurally mirrors PAdicGeodesicLoss (same pair-sampling + smooth-L1-to-
    target-distance shape), but d_target comes from real taxonomic distance
    between the *species* each sample's window belongs to, not from
    v_3(index). Requires a per-sample species_id tensor passed via
    forward(..., species_ids=...) -- unlike batch_indices (ternary index),
    species identity isn't derivable from the sample itself, since many
    windows across different species collapse to the same ternary index
    (see index_collision_report in evaluate_phylogeny_recovery.py).
    """

    def __init__(
        self,
        taxonomic_distance: torch.Tensor,
        max_target_distance: float = 4.0,
        n_pairs: int = 500,
        use_smooth_l1: bool = True,
    ):
        super().__init__()
        self._validate_positive(max_target_distance, "max_target_distance", self.__class__.__name__)
        if n_pairs < 1:
            raise ValueError(f"TaxonomyGeodesicLoss: n_pairs must be >= 1, got {n_pairs}")
        if taxonomic_distance.dim() != 2 or taxonomic_distance.shape[0] != taxonomic_distance.shape[1]:
            raise ValueError(
                f"TaxonomyGeodesicLoss: taxonomic_distance must be square (n_species, n_species), "
                f"got shape {tuple(taxonomic_distance.shape)}"
            )
        self.max_target = max_target_distance
        self.n_pairs = n_pairs
        self.use_smooth_l1 = use_smooth_l1
        self.register_buffer(
            "taxonomic_distance", taxonomic_distance.to(torch.float64)
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        species_ids = kwargs.get("species_ids")
        if species_ids is None:
            raise ValueError(
                "TaxonomyGeodesicLoss.forward requires species_ids=... (per-sample "
                "index into the same species ordering as taxonomic_distance) -- "
                "batch_indices (ternary index) is not sufficient, since it collides "
                "across species."
            )

        batch_size = z_hyp.size(0)
        device = z_hyp.device
        cur_c = kwargs.get("curvature", 1.0)

        if batch_size < 2:
            return phase_gated_zero(z_hyp, {"n_pairs": 0})

        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        i_idx, j_idx = sample_random_pairs(batch_size, n_pairs, device)

        sp_i = species_ids[i_idx].long()
        sp_j = species_ids[j_idx].long()
        tax_dist_pair = self.taxonomic_distance[sp_i, sp_j]

        d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], cur_c)
        d_target = self.max_target * tax_dist_pair / self.taxonomic_distance.max()

        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(d_actual, d_target)
        else:
            loss = F.mse_loss(d_actual, d_target)

        with torch.no_grad():
            corr = safe_corrcoef(d_actual, d_target, nan_if_insufficient=True)
            same_species = tax_dist_pair == 0
            mean_d_same = d_actual[same_species].mean() if same_species.any() else make_zero_loss(device)
            mean_d_diff = d_actual[~same_species].mean() if (~same_species).any() else make_zero_loss(device)

        metrics = {
            "n_pairs": n_pairs,
            "mean_d_actual": d_actual.mean().item(),
            "mean_d_target": d_target.mean().item(),
            "distance_correlation": corr.item(),
            "mean_d_same_species": mean_d_same.item(),
            "mean_d_diff_species": mean_d_diff.item(),
        }

        return loss, metrics
