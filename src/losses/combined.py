# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Combined Loss Module - Config-driven loss composition.

This module provides a CombinedLoss class that reads a config dictionary
and instantiates/combines the appropriate loss functions.

V6.1 Feature: Learnable Loss Weights (Uncertainty Weighting)
    When `learnable_weights: true` in config, loss weights become trainable
    nn.Parameters using homoscedastic uncertainty weighting (Kendall et al. 2018).

    Instead of fixed weights, the network learns log-variance parameters:
        effective_weight = 1 / (2 * exp(2 * log_sigma))
        regularization = -log_sigma  (prevents weights going to zero)

    This allows the model to automatically balance competing objectives
    based on gradient flow, rather than relying on hand-tuned weights.

Usage:
    from src.losses.combined import CombinedLoss

    loss_fn = CombinedLoss(config['loss'], curvature=1.0)
    losses = loss_fn(z_hyp, indices, logits, targets, epoch=10)
    total_loss = losses['total']

    # With learnable weights:
    # loss_fn.get_learned_weights()  # returns current effective weights
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic_kl import HyperbolicKLDivergence
from .padic_geodesic import (
    GlobalRankLoss,
    MonotonicRadialLoss,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    RichHierarchyLoss,
)
from .radius_defaults import (
    auto_share_radius_config,
    compare_radius_configs,
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

    # Class-level attribute annotations for mypy type narrowing.
    # These are set in _init_losses(); declaring here makes isinstance checks
    # and Optional[X] narrowing visible to the type checker.
    rich_hierarchy: Optional[RichHierarchyLoss]
    radial_loss: Optional[RadialHierarchyLoss]
    geodesic_loss: Optional[PAdicGeodesicLoss]
    rank_loss: Optional[GlobalRankLoss]
    monotonic_loss: Optional[MonotonicRadialLoss]
    kl_loss: Optional[HyperbolicKLDivergence]

    def __init__(
        self,
        loss_config: Dict[str, Any],
        curvature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
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

        # Learnable weights configuration
        self.use_learnable_weights = loss_config.get('learnable_weights', False)

        # Initialize enabled losses
        self._init_losses()

        # Initialize learnable weight parameters (if enabled)
        if self.use_learnable_weights:
            self._init_learnable_weights()

        # Move all child modules (and their registered buffers) to device.
        # This is the single correct place: after all children are initialized,
        # one .to() call recursively moves every register_buffer in every child.
        if device is not None:
            self.to(device)

    def _init_losses(self) -> None:
        """Initialize loss modules based on config."""

        # Initialize losses
        self.rich_hierarchy = None
        self.radial_loss = None
        self.geodesic_loss = None
        self.rank_loss = None
        self.monotonic_loss = None
        
        # Collect enabled loss configs for radius sharing
        loss_configs = {}
        for name in ['rich_hierarchy', 'radial', 'monotonic']:
            cfg = self.config.get(name, {})
            if cfg.get('enabled', False):
                loss_configs[name] = cfg
        
        # Auto-share radius hyperparameters across all radial losses
        radius_configs = auto_share_radius_config(loss_configs)
        
        # Validate consistency
        consistent, msg = compare_radius_configs(radius_configs)
        if not consistent:
            print(f"[CombinedLoss] {msg}")
        
        # RichHierarchyLoss (primary unified loss)
        rich_cfg = self.config.get('rich_hierarchy', {})
        if rich_cfg.get('enabled', False):
            radius_cfg = radius_configs['rich_hierarchy']
            self.rich_hierarchy = RichHierarchyLoss(
                inner_radius=radius_cfg.inner_radius,
                outer_radius=radius_cfg.outer_radius,
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
                inner_radius=radius_configs['radial'].inner_radius,
                outer_radius=radius_configs['radial'].outer_radius,
                margin_weight=radial_cfg.get('margin_weight', 1.0),
                curvature=self.curvature,
                valuation_weight_exponent=radial_cfg.get('valuation_weight_exponent', 0.25),
                margin_step_factor=radial_cfg.get('margin_step_factor', 0.5),
                seed=43,  # Distinct seed: avoids identical pairs with geodesic_loss (seed=42)
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
                seed=44,  # Distinct seed: avoids identical pairs with geodesic_loss (42) and radial_loss (43)
            )
            self.rank_weight = rank_cfg.get('weight', 0.5)
        else:
            self.rank_loss = None
            self.rank_weight = 0.0

        # MonotonicRadialLoss
        monotonic_cfg = self.config.get('monotonic', {})
        if monotonic_cfg.get('enabled', False):
            self.monotonic_loss = MonotonicRadialLoss(
                inner_radius=radius_configs['monotonic'].inner_radius,
                outer_radius=radius_configs['monotonic'].outer_radius,
                min_margin=monotonic_cfg.get('min_margin', 0.02),
                curvature=self.curvature,
                target_loss_weight=monotonic_cfg.get('target_loss_weight', 0.5),
            )
            self.monotonic_weight = monotonic_cfg.get('weight', 1.0)
        else:
            self.monotonic_loss = None
            self.monotonic_weight = 0.0

        # HyperbolicKLDivergence (makes this a true VAE, not a deterministic AE)
        kl_cfg = self.config.get('hyperbolic_kl', {})
        if kl_cfg.get('enabled', False):
            self.kl_loss = HyperbolicKLDivergence(
                curvature=self.curvature,
                beta=kl_cfg.get('beta', 1.0),
                free_bits=kl_cfg.get('free_bits', 0.0),
            )
            self.kl_weight = kl_cfg.get('weight', 0.01)
        else:
            self.kl_loss = None
            self.kl_weight = 0.0

        # Guard: at least one loss must be enabled, or training will be gradient-free
        active = [
            self.rich_hierarchy, self.radial_loss, self.geodesic_loss,
            self.rank_loss, self.monotonic_loss, self.kl_loss,
        ]
        if not any(x is not None for x in active):
            raise ValueError(
                "CombinedLoss: all losses are disabled. "
                "Set at least one loss 'enabled: true' in config."
            )

        # Validate all weights are non-negative (negative weights invert gradients)
        weight_checks = [
            ("radial.weight", self.radial_weight),
            ("geodesic.weight", self.geodesic_weight),
            ("rank.weight", self.rank_weight),
            ("monotonic.weight", self.monotonic_weight),
            ("hyperbolic_kl.weight", self.kl_weight),
        ]
        for name, w in weight_checks:
            if w < 0.0:
                raise ValueError(
                    f"CombinedLoss: {name} is negative ({w}). "
                    "Negative weights invert loss gradients."
                )
        if self.rich_hierarchy is not None:
            for k, w in self.rich_hierarchy_weights.items():
                if w < 0.0:
                    raise ValueError(
                        f"CombinedLoss: rich_hierarchy.{k}_weight is negative ({w})."
                    )

    def _init_learnable_weights(self) -> None:
        """Initialize learnable log-sigma parameters for uncertainty weighting.

        Uses homoscedastic uncertainty weighting (Kendall et al. 2018):
            effective_weight = 1 / (2 * exp(2 * log_sigma))
            loss_contribution = effective_weight * loss - log_sigma

        The -log_sigma regularization prevents weights from collapsing to zero.
        Initial log_sigma=0 gives effective_weight=0.5, which is a neutral starting point.
        """
        # Map from loss name to initial log_sigma (derived from config weights)
        # log_sigma = -0.5 * log(2 * weight) so that 1/(2*exp(2*log_sigma)) = weight
        import math

        def weight_to_log_sigma(w: float) -> float:
            """Convert fixed weight to initial log_sigma."""
            # effective_weight = 1 / (2 * exp(2 * log_sigma))
            # w = 1 / (2 * exp(2 * s))
            # 2w = 1 / exp(2s)
            # exp(2s) = 1 / (2w)
            # 2s = -log(2w)
            # s = -0.5 * log(2w)
            return -0.5 * math.log(max(2 * w, 1e-6))

        # RichHierarchy sub-components
        if self.rich_hierarchy is not None:
            self.log_sigma_hierarchy = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.rich_hierarchy_weights['hierarchy']), dtype=torch.float64)
            )
            self.log_sigma_coverage = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.rich_hierarchy_weights['coverage']), dtype=torch.float64)
            )
            self.log_sigma_separation = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.rich_hierarchy_weights['separation']), dtype=torch.float64)
            )

        # Other losses
        if self.radial_loss is not None:
            self.log_sigma_radial = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.radial_weight), dtype=torch.float64)
            )

        if self.geodesic_loss is not None:
            self.log_sigma_geodesic = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.geodesic_weight), dtype=torch.float64)
            )

        if self.rank_loss is not None:
            self.log_sigma_rank = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.rank_weight), dtype=torch.float64)
            )

        if self.monotonic_loss is not None:
            self.log_sigma_monotonic = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.monotonic_weight), dtype=torch.float64)
            )

        if self.kl_loss is not None:
            self.log_sigma_kl = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.kl_weight), dtype=torch.float64)
            )

    def _uncertainty_weight(self, log_sigma: nn.Parameter) -> torch.Tensor:
        """Compute effective weight from log_sigma using uncertainty weighting.

        Args:
            log_sigma: Learnable log-variance parameter

        Returns:
            Effective weight = 1 / (2 * exp(2 * log_sigma))
        """
        return 0.5 * torch.exp(-2 * log_sigma)

    def _weighted_loss(
        self,
        loss: torch.Tensor,
        log_sigma: nn.Parameter,
    ) -> torch.Tensor:
        """Apply uncertainty weighting to a loss.

        Args:
            loss: Raw loss value
            log_sigma: Learnable log-variance parameter

        Returns:
            Weighted loss with regularization: weight * loss - log_sigma
        """
        weight = self._uncertainty_weight(log_sigma)
        # The -log_sigma term is regularization that prevents sigma from growing
        # (which would make the weight go to zero)
        return weight * loss - log_sigma

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute combined loss.

        Args:
            z_hyp: Hyperbolic embeddings (B, latent_dim)
            indices: Operation indices (B,)
            logits: Decoder logits (B, 27) or (B, 9, 3)
            targets: Target ternary operations (B, 9) with values in {-1,0,1}
            epoch: Current epoch (for phase-gated losses)
            mu: Mean of approximate posterior (B, latent_dim) for KL divergence
            logvar: Log-variance of approximate posterior (B, latent_dim) for KL

        Returns:
            Dict with 'total' loss and individual loss components
        """
        device = z_hyp.device
        losses = {}
        total = torch.tensor(0.0, device=device, dtype=torch.float64)

        # 1. RichHierarchyLoss (if enabled)
        # Returns (raw_tensors_dict, metrics_dict) per HierarchyLossBase contract.
        if self.rich_hierarchy is not None:
            rich_raw, rich_metrics = self.rich_hierarchy(
                z_hyp, indices, logits=logits, targets=targets
            )

            if self.use_learnable_weights:
                weighted_rich = (
                    self._weighted_loss(rich_raw['hierarchy'], self.log_sigma_hierarchy) +
                    self._weighted_loss(rich_raw['coverage'], self.log_sigma_coverage) +
                    self._weighted_loss(rich_raw['separation'], self.log_sigma_separation)
                )
            else:
                weighted_rich = (
                    self.rich_hierarchy_weights['hierarchy'] * rich_raw['hierarchy'] +
                    self.rich_hierarchy_weights['coverage'] * rich_raw['coverage'] +
                    self.rich_hierarchy_weights['separation'] * rich_raw['separation']
                )

            losses['rich_hierarchy'] = weighted_rich
            losses['rich_hierarchy_detail'] = rich_metrics
            total = total + weighted_rich

        # 2. RadialHierarchyLoss (if enabled)
        if self.radial_loss is not None:
            radial_out, radial_metrics = self.radial_loss(z_hyp, indices)
            losses['radial'] = radial_out
            losses['radial_metrics'] = radial_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(radial_out, self.log_sigma_radial)
            else:
                total = total + self.radial_weight * radial_out

        # 3. PAdicGeodesicLoss (phase-gated)
        if self.geodesic_loss is not None and epoch >= self.geodesic_phase_start:
            geodesic_out, geodesic_metrics = self.geodesic_loss(z_hyp, indices)
            losses['geodesic'] = geodesic_out
            losses['geodesic_metrics'] = geodesic_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(geodesic_out, self.log_sigma_geodesic)
            else:
                total = total + self.geodesic_weight * geodesic_out

        # 4. GlobalRankLoss (if enabled)
        if self.rank_loss is not None:
            rank_out, rank_metrics = self.rank_loss(z_hyp, indices)
            losses['rank'] = rank_out
            losses['rank_metrics'] = rank_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(rank_out, self.log_sigma_rank)
            else:
                total = total + self.rank_weight * rank_out

        # 5. MonotonicRadialLoss (if enabled)
        if self.monotonic_loss is not None:
            monotonic_out, monotonic_metrics = self.monotonic_loss(z_hyp, indices)
            losses['monotonic'] = monotonic_out
            losses['monotonic_metrics'] = monotonic_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(monotonic_out, self.log_sigma_monotonic)
            else:
                total = total + self.monotonic_weight * monotonic_out

        # 6. KL Divergence (makes this a true VAE)
        if self.kl_loss is not None and mu is not None and logvar is not None:
            kl_out = self.kl_loss(mu, logvar, z_hyp)
            if self.use_learnable_weights and hasattr(self, 'log_sigma_kl'):
                kl_contribution = self._weighted_loss(kl_out, self.log_sigma_kl)
                losses['kl'] = kl_contribution
                total = total + kl_contribution
            else:
                kl_contribution = self.kl_weight * kl_out
                losses['kl'] = kl_contribution
                total = total + kl_contribution

        # 7. Fallback: Basic coverage loss if no rich_hierarchy
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
            return torch.tensor(0.0, device=device, dtype=torch.float64)

    def get_enabled_losses(self) -> List[str]:
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
        if self.kl_loss is not None:
            enabled.append('kl')
        return enabled

    def get_learned_weights(self) -> Dict[str, float]:
        """Return current effective weights (if using learnable weights).

        Returns:
            Dict mapping loss name to effective weight.
            Returns config weights if not using learnable weights.
        """
        if not self.use_learnable_weights:
            # Return fixed weights from config
            weights = {}
            if self.rich_hierarchy is not None:
                weights.update(self.rich_hierarchy_weights)
            if self.radial_loss is not None:
                weights['radial'] = self.radial_weight
            if self.geodesic_loss is not None:
                weights['geodesic'] = self.geodesic_weight
            if self.rank_loss is not None:
                weights['rank'] = self.rank_weight
            if self.monotonic_loss is not None:
                weights['monotonic'] = self.monotonic_weight
            if self.kl_loss is not None:
                weights['kl'] = self.kl_weight
            return weights

        # Compute effective weights from learnable log_sigma
        weights = {}
        with torch.no_grad():
            if self.rich_hierarchy is not None:
                weights['hierarchy'] = self._uncertainty_weight(self.log_sigma_hierarchy).item()
                weights['coverage'] = self._uncertainty_weight(self.log_sigma_coverage).item()
                weights['separation'] = self._uncertainty_weight(self.log_sigma_separation).item()
            if self.radial_loss is not None:
                weights['radial'] = self._uncertainty_weight(self.log_sigma_radial).item()
            if self.geodesic_loss is not None:
                weights['geodesic'] = self._uncertainty_weight(self.log_sigma_geodesic).item()
            if self.rank_loss is not None:
                weights['rank'] = self._uncertainty_weight(self.log_sigma_rank).item()
            if self.monotonic_loss is not None:
                weights['monotonic'] = self._uncertainty_weight(self.log_sigma_monotonic).item()
            if self.kl_loss is not None and hasattr(self, 'log_sigma_kl'):
                weights['kl'] = self._uncertainty_weight(self.log_sigma_kl).item()
        return weights

    def get_log_sigmas(self) -> Dict[str, float]:
        """Return raw log_sigma values for debugging/logging.

        Returns:
            Dict mapping loss name to log_sigma value.
            Returns empty dict if not using learnable weights.
        """
        if not self.use_learnable_weights:
            return {}

        sigmas = {}
        with torch.no_grad():
            if self.rich_hierarchy is not None:
                sigmas['hierarchy'] = self.log_sigma_hierarchy.item()
                sigmas['coverage'] = self.log_sigma_coverage.item()
                sigmas['separation'] = self.log_sigma_separation.item()
            if self.radial_loss is not None:
                sigmas['radial'] = self.log_sigma_radial.item()
            if self.geodesic_loss is not None:
                sigmas['geodesic'] = self.log_sigma_geodesic.item()
            if self.rank_loss is not None:
                sigmas['rank'] = self.log_sigma_rank.item()
            if self.monotonic_loss is not None:
                sigmas['monotonic'] = self.log_sigma_monotonic.item()
            if self.kl_loss is not None and hasattr(self, 'log_sigma_kl'):
                sigmas['kl'] = self.log_sigma_kl.item()
        return sigmas

    def __repr__(self) -> str:
        enabled = self.get_enabled_losses()
        learnable = "learnable" if self.use_learnable_weights else "fixed"
        return f"CombinedLoss(enabled={enabled}, weights={learnable})"

    def get_radius_config(self) -> Dict[str, Dict[str, float]]:
        """Return radius configuration for each enabled loss.

        Returns:
            Dict mapping loss name to {'inner_radius': x, 'outer_radius': y}
        """
        radius_config = {}
        if self.rich_hierarchy is not None:
            radius_config['rich_hierarchy'] = {
                'inner_radius': self.rich_hierarchy.inner_radius,
                'outer_radius': self.rich_hierarchy.outer_radius,
            }
        if self.radial_loss is not None:
            radius_config['radial'] = {
                'inner_radius': self.radial_loss.inner_radius,
                'outer_radius': self.radial_loss.outer_radius,
            }
        if self.monotonic_loss is not None:
            radius_config['monotonic'] = {
                'inner_radius': self.monotonic_loss.inner_radius,
                'outer_radius': self.monotonic_loss.outer_radius,
            }
        return radius_config


__all__ = ["CombinedLoss"]
