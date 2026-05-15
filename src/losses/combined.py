# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
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

from typing import Any, Dict, List, Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import CombinedLossOutput
from src.core.ternary import get_valuation_fn

from .algebraic import (
    AlgebraicAdditionLoss,
    AlgebraicCoherenceLoss,
    AngularCoherenceLoss,
)
from .geodesic import PAdicGeodesicLoss
from .hierarchy import (
    MonotonicRadialLoss,
    RadialHierarchyLoss,
    RichHierarchyLoss,
    WithinLevelContrastiveLoss,
)
from .hyperbolic_kl import HyperbolicKLDivergence
from .prior import ValuationPriorLoss
from .radius_defaults import (
    auto_share_radius_config,
    compare_radius_configs,
)
from .rank import GlobalRankLoss


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
    valuation_prior: Optional[ValuationPriorLoss]
    wlc_loss: Optional[WithinLevelContrastiveLoss]
    angular_coherence: Optional[AngularCoherenceLoss]
    algebraic_coherence_loss: Optional[AlgebraicCoherenceLoss]

    def __init__(
        self,
        loss_config: Dict[str, Any],
        curvature: float = 1.0,
        device: Optional[torch.device] = None,
        valuation_type: str = "index",
    ) -> None:
        """Initialize CombinedLoss from config.

        Args:
            loss_config: Dictionary with loss configuration
            curvature: Hyperbolic curvature parameter
            device: Device to place loss modules on
            valuation_type: "index" for 3-adic v_3(n), "digit_count" for
                zero_count_valuation (content-based hierarchy — Option B).
        """
        super().__init__()
        self.config = loss_config
        self.curvature = curvature
        self.device = device
        self._valuation_fn = get_valuation_fn(valuation_type)

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
        for name in ['rich_hierarchy', 'radial', 'monotonic', 'valuation_prior']:
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
                separation_margin=rich_cfg.get('separation_margin', 0.1),
                variance_weight=rich_cfg.get('variance_weight', 0.1),
                valuation_fn=self._valuation_fn,
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
            radius_cfg = radius_configs['radial']
            self.radial_loss = RadialHierarchyLoss(
                inner_radius=radius_cfg.inner_radius,
                outer_radius=radius_cfg.outer_radius,
                margin_weight=radial_cfg.get('margin_weight', 1.0),
                curvature=self.curvature,
                valuation_weight_exponent=radial_cfg.get('valuation_weight_exponent', 0.3),
                margin_step_factor=radial_cfg.get('margin_step_factor', 0.01),
                seed=43,
                valuation_fn=self._valuation_fn,
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
                valuation_scale=geodesic_cfg.get('valuation_scale', 3.0),
                n_pairs=geodesic_cfg.get('n_pairs', 2000),
                use_smooth_l1=geodesic_cfg.get('use_smooth_l1', True),
                use_individual_valuation=geodesic_cfg.get('use_individual_valuation', False),
                valuation_fn=self._valuation_fn,
            )
            self.geodesic_weight = geodesic_cfg.get('weight', 0.4)
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
                use_all_pairs=rank_cfg.get('use_all_pairs', False),
                curvature=self.curvature,
                seed=44,
                scatter_weight=rank_cfg.get('scatter_weight', 0.0),
                valuation_fn=self._valuation_fn,
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
                margin_scale=monotonic_cfg.get('margin_scale', 1.0),
                use_soft_margin=monotonic_cfg.get('use_soft_margin', True),
                temperature=monotonic_cfg.get('temperature', 0.05),
                curvature=self.curvature,
                target_loss_weight=monotonic_cfg.get('target_loss_weight', 0.5),
                valuation_fn=self._valuation_fn,
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
                variance_only=kl_cfg.get('variance_only', False),
            )
            self.kl_weight = kl_cfg.get('weight', 0.01)
        else:
            self.kl_loss = None
            self.kl_weight = 0.0

        # ValuationPriorLoss (replaces N(0,I) mean term with valuation-conditioned prior)
        vp_cfg = self.config.get('valuation_prior', {})
        if vp_cfg.get('enabled', False):
            # Share inner/outer radius from rich_hierarchy if available
            rh_cfg = self.config.get('rich_hierarchy', {})
            inner_r = vp_cfg.get('inner_radius', rh_cfg.get('inner_radius', 0.08))
            outer_r = vp_cfg.get('outer_radius', rh_cfg.get('outer_radius', 0.85))
            self.valuation_prior = ValuationPriorLoss(
                curvature=self.curvature,
                inner_radius=inner_r,
                outer_radius=outer_r,
                scale=vp_cfg.get('scale', 3.0),
                sigma_base=vp_cfg.get('sigma_base', 0.5),
                sigma_scale=vp_cfg.get('sigma_scale', 0.1),
                valuation_fn=self._valuation_fn,
            )
            self.valuation_prior_weight = vp_cfg.get('weight', 1.0)
        else:
            self.valuation_prior = None
            self.valuation_prior_weight = 0.0

        # WithinLevelContrastiveLoss (pull same-valuation points geodesically together)
        wlc_cfg = self.config.get('within_level_contrastive', {})
        if wlc_cfg.get('enabled', False):
            self.wlc_loss = WithinLevelContrastiveLoss(
                curvature=self.curvature,
                max_pairs_per_level=wlc_cfg.get('max_pairs_per_level', 500),
                weight=wlc_cfg.get('weight', 1.0),
                valuation_fn=self._valuation_fn,
            )
        else:
            self.wlc_loss = None

        # AngularCoherenceLoss (pull same-prefix operations together in direction space)
        # NOTE: AC loss requires a factored latent (model.factored=True) so that the
        # radial component r is available in forward(). When r=None (non-factored mode)
        # the loss is silently skipped. A one-time warning is emitted in that case.
        ac_cfg = self.config.get('angular_coherence', {})
        if ac_cfg.get('enabled', False):
            self.angular_coherence = AngularCoherenceLoss(
                weight=ac_cfg.get('weight', 0.3),
                n_pairs=ac_cfg.get('n_pairs', 1000),
                prefix_k=ac_cfg.get('prefix_k', 2),
                phase_start_epoch=ac_cfg.get('phase_start_epoch', 50),
                level_prefix_k=ac_cfg.get('level_prefix_k', None),
                target_sim=ac_cfg.get('target_sim', 1.0),
                valuation_fn=self._valuation_fn,
            )
            self._ac_warned_no_r = False  # emit the missing-r warning at most once
            self._ac_skip_count = 0      # counts forward passes with r=None (visible in losses dict)
        else:
            self.angular_coherence = None
            self._ac_warned_no_r = True
            self._ac_skip_count = 0

        # AlgebraicCoherenceLoss (group by algebraic signature, attract same-class directions)
        alg_cfg = self.config.get('algebraic_coherence', {})
        if alg_cfg.get('enabled', False):
            self.algebraic_coherence_loss = AlgebraicCoherenceLoss(
                weight=alg_cfg.get('weight', 1.0),
                n_pairs=alg_cfg.get('n_pairs', 2000),
                target_sim=alg_cfg.get('target_sim', 0.70),
                phase_start_epoch=alg_cfg.get('phase_start_epoch', 20),
                min_class_size=alg_cfg.get('min_class_size', 3),
            )
            self.alg_coherence_weight = alg_cfg.get('weight', 1.0)
            self._alg_warned_no_r = False
            self._alg_skip_count = 0
        else:
            self.algebraic_coherence_loss = None
            self.alg_coherence_weight = 0.0
            self._alg_warned_no_r = True
            self._alg_skip_count = 0

        # Algebraic Addition Loss (z(a) + z(b) \approx z(a+b))
        aa_cfg = self.config.get('algebraic_addition', {})
        if aa_cfg.get('enabled', False):
            self.algebraic_addition_loss = AlgebraicAdditionLoss(
                weight=aa_cfg.get('weight', 1.0),
                n_pairs=aa_cfg.get('n_pairs', 512),
                phase_start_epoch=aa_cfg.get('phase_start_epoch', 0),
            )
        else:
            self.algebraic_addition_loss = None

        # Guard: at least one loss must be enabled, or training will be gradient-free
        active = [
            self.rich_hierarchy, self.radial_loss, self.geodesic_loss,
            self.rank_loss, self.monotonic_loss, self.kl_loss, self.valuation_prior,
            self.wlc_loss, self.angular_coherence, self.algebraic_coherence_loss,
            self.algebraic_addition_loss,
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
        if self.alg_coherence_weight < 0.0:
            raise ValueError(
                f"CombinedLoss: algebraic_coherence.weight is negative ({self.alg_coherence_weight})."
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
        curvature: Optional[Union[float, torch.Tensor]] = None,
        dual_weights: Optional[Dict[str, List[float]]] = None,
        r: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> CombinedLossOutput:
        """Compute combined loss.

        Args:
            z_hyp: Hyperbolic embeddings (B, latent_dim)
            indices: Operation indices (B,)
            logits: Decoder logits (B, 27) or (B, 9, 3)
            targets: Target ternary operations (B, 9) with values in {-1,0,1}
            epoch: Current epoch (for phase-gated losses)
            mu: Mean of approximate posterior (B, latent_dim) for KL divergence
            logvar: Log-variance of approximate posterior (B, latent_dim) for KL
            curvature: Current curvature value (pass model's learned curvature when
                       learnable_curvature=True). Falls back to init-time curvature.
            dual_weights: Optional dict from LagrangianDualState.get_dual_weights().
                Keys: 'lambda_margin' (list[9]), 'lambda_scatter' (list[10]),
                'lambda_prior' (list[10]). When provided, additive per-level
                penalties are applied using in-graph violation tensors from
                the metric dicts of MonotonicRadialLoss, GlobalRankLoss, and
                ValuationPriorLoss. None = no Lagrangian penalties (default).

        Returns:
            Dict with 'total' loss and individual loss components
        """
        device = z_hyp.device
        losses: CombinedLossOutput = {}
        total = torch.tensor(0.0, device=device, dtype=torch.float64)

        # 1. RichHierarchyLoss (if enabled)
        # Returns (raw_tensors_dict, metrics_dict) per HierarchyLossBase contract.
        # Pass logits=None when coverage_weight=0.0: skips the F.cross_entropy
        # forward inside RichHierarchyLoss, eliminating wasted compute when
        # coverage is deliberately disabled (e.g. loss_fn_b for VAE-B).
        cur_c = curvature if curvature is not None else self.curvature
        if self.rich_hierarchy is not None:
            _call_logits = (
                logits if self.rich_hierarchy_weights.get('coverage', 0.0) > 0.0
                else None
            )
            rich_raw, rich_metrics = self.rich_hierarchy(
                z_hyp, indices, logits=_call_logits, targets=targets,
                curvature=cur_c
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
            radial_out, radial_metrics = self.radial_loss(z_hyp, indices, curvature=cur_c)
            losses['radial'] = radial_out
            losses['radial_metrics'] = radial_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(radial_out, self.log_sigma_radial)
            else:
                total = total + self.radial_weight * radial_out

        # 3. PAdicGeodesicLoss (phase-gated)
        if self.geodesic_loss is not None and epoch >= self.geodesic_phase_start:
            geodesic_out, geodesic_metrics = self.geodesic_loss(z_hyp, indices, curvature=cur_c)
            losses['geodesic'] = geodesic_out
            losses['geodesic_metrics'] = geodesic_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(geodesic_out, self.log_sigma_geodesic)
            else:
                total = total + self.geodesic_weight * geodesic_out

        # 4. GlobalRankLoss (if enabled)
        if self.rank_loss is not None:
            rank_out, rank_metrics = self.rank_loss(z_hyp, indices, curvature=cur_c)
            losses['rank'] = rank_out
            losses['rank_metrics'] = rank_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(rank_out, self.log_sigma_rank)
            else:
                total = total + self.rank_weight * rank_out

        # 5. MonotonicRadialLoss (if enabled)
        if self.monotonic_loss is not None:
            monotonic_out, monotonic_metrics = self.monotonic_loss(z_hyp, indices, curvature=cur_c)
            losses['monotonic'] = monotonic_out
            losses['monotonic_metrics'] = monotonic_metrics
            if self.use_learnable_weights:
                total = total + self._weighted_loss(monotonic_out, self.log_sigma_monotonic)
            else:
                total = total + self.monotonic_weight * monotonic_out

        # 6. KL Divergence (makes this a true VAE)
        if self.kl_loss is not None and mu is not None and logvar is not None:
            kl_out = self.kl_loss(mu, logvar, z_hyp, curvature=cur_c)
            if self.use_learnable_weights and hasattr(self, 'log_sigma_kl'):
                kl_contribution = self._weighted_loss(kl_out, self.log_sigma_kl)
                losses['kl'] = kl_contribution
                total = total + kl_contribution
            else:
                kl_contribution = self.kl_weight * kl_out
                losses['kl'] = kl_contribution
                total = total + kl_contribution

        # 7. ValuationPriorLoss (valuation-conditioned μ/σ prior)
        # Requires mu and indices. Uses current curvature when learnable.
        if self.valuation_prior is not None and mu is not None:
            cur_c = curvature if curvature is not None else self.curvature
            vp_out, vp_metrics = self.valuation_prior(mu, logvar, indices, curvature=cur_c)
            vp_contribution = self.valuation_prior_weight * vp_out
            losses['valuation_prior'] = vp_contribution
            losses['valuation_prior_metrics'] = vp_metrics
            total = total + vp_contribution

        # 8. Lagrangian dual penalties (optional, from outer-loop dual ascent).
        # Each lambda_v * violation_tensor_v term is additive and in-graph:
        # the tensor violations come from per-level metrics dicts computed above,
        # and lambda values are plain floats (not optimiser parameters).
        if dual_weights is not None:
            lam_margin = dual_weights.get('lambda_margin', [])
            lam_scatter = dual_weights.get('lambda_scatter', [])
            lam_prior = dual_weights.get('lambda_prior', [])

            # Margin penalties from MonotonicRadialLoss
            if 'monotonic_metrics' in losses:
                mono_m = losses['monotonic_metrics']
                lagrangian_margin_total = torch.tensor(0.0, device=device, dtype=torch.float64)
                for v in range(9):
                    vt = mono_m.get(f'gap_viol_tensor_v{v}')
                    if vt is not None and v < len(lam_margin) and lam_margin[v] > 0.0:
                        lagrangian_margin_total = lagrangian_margin_total + lam_margin[v] * vt
                if lagrangian_margin_total.item() > 0.0:
                    losses['lagrangian_margin'] = lagrangian_margin_total
                    total = total + lagrangian_margin_total

            # Scatter penalties from GlobalRankLoss
            if 'rank_metrics' in losses:
                rank_m = losses['rank_metrics']
                lagrangian_scatter_total = torch.tensor(0.0, device=device, dtype=torch.float64)
                for v in range(10):
                    st = rank_m.get(f'scatter_tensor_v{v}')
                    if st is not None and v < len(lam_scatter) and lam_scatter[v] > 0.0:
                        lagrangian_scatter_total = lagrangian_scatter_total + lam_scatter[v] * st
                if lagrangian_scatter_total.item() > 0.0:
                    losses['lagrangian_scatter'] = lagrangian_scatter_total
                    total = total + lagrangian_scatter_total

            # Prior norm penalties from ValuationPriorLoss
            if 'valuation_prior_metrics' in losses:
                vp_m = losses['valuation_prior_metrics']
                lagrangian_prior_total = torch.tensor(0.0, device=device, dtype=torch.float64)
                for v in range(10):
                    gt = vp_m.get(f'vp_gap_tensor_v{v}')
                    if gt is not None and v < len(lam_prior) and lam_prior[v] > 0.0:
                        lagrangian_prior_total = lagrangian_prior_total + lam_prior[v] * gt
                if lagrangian_prior_total.item() > 0.0:
                    losses['lagrangian_prior'] = lagrangian_prior_total
                    total = total + lagrangian_prior_total

        # 9. Within-level contrastive loss (pull same-valuation points geodesically together)
        if self.wlc_loss is not None:
            wlc_out, wlc_metrics = self.wlc_loss(z_hyp, indices)
            losses['within_level_contrastive'] = wlc_out
            losses['wlc_metrics'] = wlc_metrics
            total = total + wlc_out

        # 10. Angular coherence loss (sharpen direction sub-clusters by digit prefix)
        # Requires r (radial component from factored latent). In non-factored mode
        # r=None, making AC structurally inapplicable: there is no separate direction
        # space to align, only the combined z_hyp vector.
        if self.angular_coherence is not None and r is not None:
            ac_out, ac_metrics = self.angular_coherence(z_hyp, r, indices, epoch)
            losses['angular_coherence'] = ac_out
            losses['angular_coherence_metrics'] = ac_metrics
            total = total + ac_out
        elif self.angular_coherence is not None and r is None:
            self._ac_skip_count += 1
            losses['ac_skipped_no_r'] = self._ac_skip_count
            if not self._ac_warned_no_r:
                warnings.warn(
                    "AngularCoherenceLoss is enabled (angular_coherence.enabled=true) but "
                    "r=None was passed to CombinedLoss.forward(). AC loss is producing ZERO "
                    "gradient. This happens when model.factored=False — the model does not "
                    "produce a separate radial component. Either set model.factored=True or "
                    "disable angular_coherence in your config.",
                    UserWarning,
                    stacklevel=2,
                )
                self._ac_warned_no_r = True

        # 11. AlgebraicCoherenceLoss (requires factored latent, same as AC)
        if self.algebraic_coherence_loss is not None and r is not None:
            alg_out, alg_metrics = self.algebraic_coherence_loss(z_hyp, r, indices, epoch)
            losses['algebraic_coherence'] = alg_out
            losses['alg_coherence_metrics'] = alg_metrics
            total = total + alg_out
        elif self.algebraic_coherence_loss is not None and r is None:
            self._alg_skip_count += 1
            losses['alg_skipped_no_r'] = self._alg_skip_count
            if not self._alg_warned_no_r:
                warnings.warn(
                    "AlgebraicCoherenceLoss requires factored latent (r=None). "
                    "Set model.factored=True or disable algebraic_coherence in config.",
                    UserWarning,
                    stacklevel=2,
                )
                self._alg_warned_no_r = True

        # 12. AlgebraicAdditionLoss (requires mu and model)
        if self.algebraic_addition_loss is not None and mu is not None and model is not None:
            aa_out, aa_metrics = self.algebraic_addition_loss(mu, indices, model, epoch)
            losses['algebraic_addition'] = aa_out
            losses['alg_addition_metrics'] = aa_metrics
            total = total + aa_out

        # 13. Fallback: Basic coverage loss if no rich_hierarchy
        if self.rich_hierarchy is None:
            # Respect coverage_weight from config even if rich_hierarchy is disabled
            coverage_weight = self.config.get('rich_hierarchy', {}).get('coverage_weight', 1.0)
            if coverage_weight > 0.0:
                coverage_loss = self._compute_coverage_loss(logits, targets)
                losses['coverage'] = coverage_loss
                total = total + coverage_weight * coverage_loss

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
        if self.valuation_prior is not None:
            enabled.append('valuation_prior')
        if self.wlc_loss is not None:
            enabled.append('within_level_contrastive')
        if self.angular_coherence is not None:
            enabled.append('angular_coherence')
        if self.algebraic_coherence_loss is not None:
            enabled.append('algebraic_coherence')
        if self.algebraic_addition_loss is not None:
            enabled.append('algebraic_addition')
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
            if self.algebraic_coherence_loss is not None:
                weights['algebraic_coherence'] = self.alg_coherence_weight
            if self.algebraic_addition_loss is not None:
                weights['algebraic_addition'] = self.algebraic_addition_loss.weight
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
