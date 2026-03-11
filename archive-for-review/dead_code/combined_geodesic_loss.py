# ARCHIVED: 2026-03-10
# REASON: Curriculum-blending wrapper for geodesic + radial losses. Superseded by
#          the CombinedLoss factory (src/losses/combined.py) which provides
#          config-driven composition of all loss functions. Could be reintegrated
#          if simpler two-loss blending is needed for experiments.
# ORIGINAL LOCATION: src/losses/padic_geodesic.py (lines 312-370)
# DEPENDENCIES: PAdicGeodesicLoss, RadialHierarchyLoss

from typing import Tuple

import torch
import torch.nn as nn


class CombinedGeodesicLoss(nn.Module):
    """Combined Geodesic + Radial Loss for V5.11.

    Wraps both losses with curriculum-based blending:
    - Early: More radial loss (establish hierarchy)
    - Late: More geodesic loss (refine correlation)

    The tau parameter controls the blend (can be learned by controller).
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_target_distance: float = 3.0,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        n_pairs: int = 2000,
    ):
        super().__init__()
        # NOTE: These imports would need updating if reintegrated
        from src.losses.padic_geodesic import PAdicGeodesicLoss, RadialHierarchyLoss

        self.geodesic_loss = PAdicGeodesicLoss(
            curvature=curvature,
            max_target_distance=max_target_distance,
            n_pairs=n_pairs,
        )
        self.radial_loss = RadialHierarchyLoss(
            inner_radius=inner_radius, outer_radius=outer_radius
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        tau: float = 0.5,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss with curriculum blending.

        Args:
            z_hyp: Points in Poincare ball
            batch_indices: Operation indices
            tau: Blend factor (0 = pure radial, 1 = pure geodesic)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        geo_loss, geo_metrics = self.geodesic_loss(z_hyp, batch_indices)
        rad_loss, rad_metrics = self.radial_loss(z_hyp, batch_indices)

        # Curriculum blend
        total_loss = (1 - tau) * rad_loss + tau * geo_loss

        # Merge metrics
        metrics = {
            "geodesic_loss": geo_loss.item(),
            "radial_loss": rad_loss.item(),
            "tau": tau,
            **{f"geo_{k}": v for k, v in geo_metrics.items()},
            **{f"rad_{k}": v for k, v in rad_metrics.items()},
        }

        return total_loss, metrics
