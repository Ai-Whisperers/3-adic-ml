# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Combined Loss Module - Config-driven loss composition.

This module provides a CombinedLoss class that reads a config dictionary
and instantiates/combines the appropriate loss functions.

Usage:
    from src.losses.combined import CombinedLoss

    loss_fn = CombinedLoss(config['loss'], curvature=1.0)
    losses = loss_fn(z_hyp, indices, logits, targets, epoch=10)
    total_loss = losses['total']
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import TERNARY
from src.geometry import poincare_distance

from .padic_geodesic import (
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    GlobalRankLoss,
    MonotonicRadialLoss,
    RichHierarchyLoss,
)


class CombinedLoss(nn.Module):
    """Config-driven combined loss function.

    Reads a loss configuration dictionary and instantiates the appropriate
    loss functions. Combines them with configured weights.

    Supported losses (from config):
        - rich_hierarchy: RichHierarchyLoss (hierarchy + coverage + separation)
        - radial: RadialHierarchyLoss (direct radius enforcement)
        - geodesic: PAdicGeodesicLoss (poincare distance alignment)
        - rank: GlobalRankLoss (soft ranking violation)
        - monotonic: MonotonicRadialLoss (level-wise ordering)

    Example config:
        loss:
          rich_hierarchy:
            enabled: true
            hierarchy_weight: 5.0
            coverage_weight: 1.0
          radial:
            enabled: true
            inner_radius: 0.1
            outer_radius: 0.85
            weight: 1.0
          geodesic:
            enabled: true
            phase_start_epoch: 50
            weight: 0.3
    """

    def __init__(
        self,
        loss_config: Dict[str, Any],
        curvature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize CombinedLoss from config.

        Args:
            loss_config: Dictionary with loss configuration
            curvature: Hyperbolic curvature parameter
            device: Device to place loss modules on
        """
        super().__init__()
        self.config = loss_config
        self.curvature = curvature
        self.device = device

        # Initialize enabled losses
        self._init_losses()

    def _init_losses(self):
        """Initialize loss modules based on config."""

        # RichHierarchyLoss (primary unified loss)
        rich_cfg = self.config.get('rich_hierarchy', {})
        if rich_cfg.get('enabled', False):
            self.rich_hierarchy = RichHierarchyLoss(
                inner_radius=rich_cfg.get('inner_radius', 0.1),
                outer_radius=rich_cfg.get('outer_radius', 0.85),
                curvature=self.curvature,
                separation_margin=rich_cfg.get('separation_margin', 0.01),
            )
            self.rich_hierarchy_weights = {
                'hierarchy': rich_cfg.get('hierarchy_weight', 5.0),
                'coverage': rich_cfg.get('coverage_weight', 1.0),
                'separation': rich_cfg.get('separation_weight', 3.0),
            }
        else:
            self.rich_hierarchy = None

        # RadialHierarchyLoss
        radial_cfg = self.config.get('radial', {})
        if radial_cfg.get('enabled', False):
            self.radial_loss = RadialHierarchyLoss(
                inner_radius=radial_cfg.get('inner_radius', 0.1),
                outer_radius=radial_cfg.get('outer_radius', 0.85),
                margin_weight=radial_cfg.get('margin_weight', 1.0),
                curvature=self.curvature,
                valuation_weight_exponent=radial_cfg.get('valuation_weight_exponent', 0.25),
                margin_step_factor=radial_cfg.get('margin_step_factor', 0.5),
            )
            self.radial_weight = radial_cfg.get('weight', 1.0)
        else:
            self.radial_loss = None
            self.radial_weight = 0.0

        # PAdicGeodesicLoss (phase-gated)
        geodesic_cfg = self.config.get('geodesic', {})
        if geodesic_cfg.get('enabled', False):
            self.geodesic_loss = PAdicGeodesicLoss(
                curvature=geodesic_cfg.get('curvature', self.curvature),
                max_target_distance=geodesic_cfg.get('max_target_distance', 3.0),
                n_pairs=geodesic_cfg.get('n_pairs', 2000),
                use_smooth_l1=geodesic_cfg.get('use_smooth_l1', True),
            )
            self.geodesic_weight = geodesic_cfg.get('weight', 0.3)
            self.geodesic_phase_start = geodesic_cfg.get('phase_start_epoch', 0)
        else:
            self.geodesic_loss = None
            self.geodesic_weight = 0.0
            self.geodesic_phase_start = 0

        # GlobalRankLoss
        rank_cfg = self.config.get('rank', {})
        if rank_cfg.get('enabled', False):
            self.rank_loss = GlobalRankLoss(
                temperature=rank_cfg.get('temperature', 0.1),
                n_pairs=rank_cfg.get('n_pairs', 2000),
                curvature=self.curvature,
            )
            self.rank_weight = rank_cfg.get('weight', 0.5)
        else:
            self.rank_loss = None
            self.rank_weight = 0.0

        # MonotonicRadialLoss
        monotonic_cfg = self.config.get('monotonic', {})
        if monotonic_cfg.get('enabled', False):
            self.monotonic_loss = MonotonicRadialLoss(
                inner_radius=monotonic_cfg.get('inner_radius', 0.1),
                outer_radius=monotonic_cfg.get('outer_radius', 0.85),
                min_margin=monotonic_cfg.get('min_margin', 0.02),
                curvature=self.curvature,
                target_loss_weight=monotonic_cfg.get('target_loss_weight', 0.5),
            )
            self.monotonic_weight = monotonic_cfg.get('weight', 1.0)
        else:
            self.monotonic_loss = None
            self.monotonic_weight = 0.0

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            z_hyp: Hyperbolic embeddings (B, latent_dim)
            indices: Operation indices (B,)
            logits: Decoder logits (B, 27) or (B, 9, 3)
            targets: Target ternary operations (B, 9) with values in {-1,0,1}
            epoch: Current epoch (for phase-gated losses)

        Returns:
            Dict with 'total' loss and individual loss components
        """
        device = z_hyp.device
        losses = {}
        total = torch.tensor(0.0, device=device)

        # 1. RichHierarchyLoss (if enabled)
        if self.rich_hierarchy is not None:
            rich_out = self.rich_hierarchy(z_hyp, indices, logits, targets)

            # Apply configured weights
            weighted_rich = (
                self.rich_hierarchy_weights['hierarchy'] * rich_out['hierarchy'] +
                self.rich_hierarchy_weights['coverage'] * rich_out['coverage'] +
                self.rich_hierarchy_weights['separation'] * rich_out['separation']
            )

            losses['rich_hierarchy'] = weighted_rich
            losses['rich_hierarchy_detail'] = rich_out
            total = total + weighted_rich

        # 2. RadialHierarchyLoss (if enabled)
        if self.radial_loss is not None:
            radial_out, radial_metrics = self.radial_loss(z_hyp, indices)
            losses['radial'] = radial_out
            losses['radial_metrics'] = radial_metrics
            total = total + self.radial_weight * radial_out

        # 3. PAdicGeodesicLoss (phase-gated)
        if self.geodesic_loss is not None and epoch >= self.geodesic_phase_start:
            geodesic_out, geodesic_metrics = self.geodesic_loss(z_hyp, indices)
            losses['geodesic'] = geodesic_out
            losses['geodesic_metrics'] = geodesic_metrics
            total = total + self.geodesic_weight * geodesic_out

        # 4. GlobalRankLoss (if enabled)
        if self.rank_loss is not None:
            rank_out, rank_metrics = self.rank_loss(z_hyp, indices)
            losses['rank'] = rank_out
            losses['rank_metrics'] = rank_metrics
            total = total + self.rank_weight * rank_out

        # 5. MonotonicRadialLoss (if enabled)
        if self.monotonic_loss is not None:
            monotonic_out, monotonic_metrics = self.monotonic_loss(z_hyp, indices)
            losses['monotonic'] = monotonic_out
            losses['monotonic_metrics'] = monotonic_metrics
            total = total + self.monotonic_weight * monotonic_out

        # 6. Fallback: Basic coverage loss if no rich_hierarchy
        if self.rich_hierarchy is None:
            coverage_loss = self._compute_coverage_loss(logits, targets)
            losses['coverage'] = coverage_loss
            total = total + coverage_loss

        losses['total'] = total
        return losses

    def _compute_coverage_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute basic coverage (reconstruction) loss.

        Args:
            logits: Decoder logits (B, 27) or (B, 9, 3)
            targets: Target ternary operations (B, 9) with values in {-1,0,1}

        Returns:
            CrossEntropy loss
        """
        device = logits.device

        # Handle different logit shapes
        if logits.shape[-1] == 3:
            # (B, 9, 3) format
            targets_shifted = (targets + 1).long().clamp(0, 2)
            return F.cross_entropy(
                logits.view(-1, 3),
                targets_shifted.view(-1),
            )
        elif logits.shape[-1] == 27:
            # (B, 27) format - reshape to (B, 9, 3)
            logits_reshaped = logits.view(-1, 9, 3)
            targets_shifted = (targets + 1).long().clamp(0, 2)
            return F.cross_entropy(
                logits_reshaped.permute(0, 2, 1),  # (B, 3, 9)
                targets_shifted,  # (B, 9)
            )
        else:
            # Unsupported shape - return zero loss with warning
            return torch.tensor(0.0, device=device)

    def get_enabled_losses(self) -> list:
        """Return list of enabled loss names."""
        enabled = []
        if self.rich_hierarchy is not None:
            enabled.append('rich_hierarchy')
        if self.radial_loss is not None:
            enabled.append('radial')
        if self.geodesic_loss is not None:
            enabled.append('geodesic')
        if self.rank_loss is not None:
            enabled.append('rank')
        if self.monotonic_loss is not None:
            enabled.append('monotonic')
        return enabled

    def __repr__(self) -> str:
        enabled = self.get_enabled_losses()
        return f"CombinedLoss(enabled={enabled})"


__all__ = ["CombinedLoss"]
