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
        regularization = +log_sigma  (Kendall & Gal 2018 — stable equilibrium at log_sigma = 0.5*log(loss))

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

from .utils import compute_coverage_loss, make_zero_loss, weight_to_log_sigma

from src.core.contracts import CombinedLossOutput
from src.core import get_valuation_fn

from .algebraic import (
    AlgebraicAdditionLoss,
    AlgebraicCoherenceLoss,
    AlgebraicDistributiveLoss,
    AlgebraicMultiplicationLoss,
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
from .contrastive import HyperbolicContrastiveLoss
from .surrogate import SurrogatePropertyLoss, SurrogateRegressor




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
    algebraic_addition_loss: Optional[AlgebraicAdditionLoss]
    algebraic_multiplication_loss: Optional[AlgebraicMultiplicationLoss]
    algebraic_distributive_loss: Optional[AlgebraicDistributiveLoss]

    def __init__(
        self,
        loss_config: Dict[str, Any],
        curvature: float = 1.0,
        device: Optional[torch.device] = None,
        valuation_type: str = "index",
        latent_dim: int = 128,
    ) -> None:
        """Initialize CombinedLoss from config.

        Args:
            loss_config: Dictionary with loss configuration
            curvature: Hyperbolic curvature parameter
            device: Device to place loss modules on
            valuation_type: "index" for 3-adic v_3(n), "digit_count" for
                zero_count_valuation (content-based hierarchy — Option B).
            latent_dim: Latent dimension size.
        """
        super().__init__()
        self.config = loss_config
        self.curvature = curvature
        self.device = device
        self.latent_dim = latent_dim
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
            self.rich_hierarchy_phase_start = rich_cfg.get('phase_start_epoch', 0)
        else:
            self.rich_hierarchy = None
            self.rich_hierarchy_phase_start = 0

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
            self.radial_phase_start = radial_cfg.get('phase_start_epoch', 0)
        else:
            self.radial_loss = None
            self.radial_weight = 0.0
            self.radial_phase_start = 0

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
            self.rank_phase_start = rank_cfg.get('phase_start_epoch', 0)
        else:
            self.rank_loss = None
            self.rank_weight = 0.0
            self.rank_phase_start = 0

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
            self.monotonic_phase_start = monotonic_cfg.get('phase_start_epoch', 0)
        else:
            self.monotonic_loss = None
            self.monotonic_weight = 0.0
            self.monotonic_phase_start = 0

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
            self.kl_warmup_epochs = max(0, int(kl_cfg.get('warmup_epochs', 0)))
        else:
            self.kl_loss = None
            self.kl_weight = 0.0
            self.kl_warmup_epochs = 0

        # ValuationPriorLoss (replaces N(0,I) mean term with valuation-conditioned prior)
        vp_cfg = self.config.get('valuation_prior', {})
        if vp_cfg.get('enabled', False):
            # Share inner/outer radius from rich_hierarchy or radius_configs if available
            inner_r = vp_cfg.get('inner_radius')
            outer_r = vp_cfg.get('outer_radius')
            if inner_r is None or outer_r is None:
                if 'valuation_prior' in radius_configs:
                    inner_r = inner_r if inner_r is not None else radius_configs['valuation_prior'].inner_radius
                    outer_r = outer_r if outer_r is not None else radius_configs['valuation_prior'].outer_radius
                else:
                    rh_cfg = self.config.get('rich_hierarchy') or {}
                    inner_r = inner_r if inner_r is not None else rh_cfg.get('inner_radius') or 0.08
                    outer_r = outer_r if outer_r is not None else rh_cfg.get('outer_radius') or 0.85
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
            ac_target_sim = ac_cfg.get('target_sim')
            if ac_target_sim is None:
                ac_target_sim = 1.0
            self.angular_coherence = AngularCoherenceLoss(
                weight=ac_cfg.get('weight', 0.3),
                n_pairs=ac_cfg.get('n_pairs', 1000),
                prefix_k=ac_cfg.get('prefix_k', 2),
                phase_start_epoch=ac_cfg.get('phase_start_epoch', 50),
                level_prefix_k=ac_cfg.get('level_prefix_k', None),
                target_sim=ac_target_sim,
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
                target_sim=alg_cfg.get('target_sim', 0.85),
                phase_start_epoch=alg_cfg.get('phase_start_epoch', 20),
                min_global_size=alg_cfg.get('min_global_size', alg_cfg.get('min_class_size', 2)),
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

        # Algebraic Multiplication Loss (z(a) * z(b) \approx z(a*b))
        am_cfg = self.config.get('algebraic_multiplication', {})
        if am_cfg.get('enabled', False):
            self.algebraic_multiplication_loss = AlgebraicMultiplicationLoss(
                weight=am_cfg.get('weight', 1.0),
                n_pairs=am_cfg.get('n_pairs', 512),
                phase_start_epoch=am_cfg.get('phase_start_epoch', 0),
            )
        else:
            self.algebraic_multiplication_loss = None

        # Algebraic Distributive Loss (z(a * (b + c)) \approx z(a) * (z(b) + z(c)))
        ad_cfg = self.config.get('algebraic_distributive', {})
        if ad_cfg.get('enabled', False):
            self.algebraic_distributive_loss = AlgebraicDistributiveLoss(
                weight=ad_cfg.get('weight', 1.0),
                n_triplets=ad_cfg.get('n_triplets', 512),
                phase_start_epoch=ad_cfg.get('phase_start_epoch', 0),
            )
        else:
            self.algebraic_distributive_loss = None

        # dynamic curriculum setting
        dc_cfg = self.config.get('dynamic_curriculum', {})
        self.dynamic_curriculum_enabled = dc_cfg.get('enabled', False)
        # If enabled, biological losses are locked until explicitly activated by the engine
        self.biological_losses_active = not self.dynamic_curriculum_enabled

        # HyperbolicContrastiveLoss
        hc_cfg = self.config.get('hyperbolic_contrastive', {})
        if hc_cfg.get('enabled', False):
            self.hyperbolic_contrastive = HyperbolicContrastiveLoss(
                temperature=hc_cfg.get('temperature', 0.1),
                prefix_k=hc_cfg.get('prefix_k', 3),
                curvature=self.curvature,
                valuation_fn=self._valuation_fn
            )
            self.hyperbolic_contrastive_weight = hc_cfg.get('weight', 1.0)
            self.hyperbolic_contrastive_phase_start = hc_cfg.get('phase_start_epoch', 0)
        else:
            self.hyperbolic_contrastive = None
            self.hyperbolic_contrastive_weight = 0.0
            self.hyperbolic_contrastive_phase_start = 0

        # SurrogatePropertyLoss
        sp_cfg = self.config.get('surrogate_property', {})
        if sp_cfg.get('enabled', False):
            # Register regressor as submodule of CombinedLoss
            self.surrogate_regressor = SurrogateRegressor(
                latent_dim=self.latent_dim,
                hidden_dim=sp_cfg.get('hidden_dim', 128)
            )
            self.surrogate_loss = SurrogatePropertyLoss(self.surrogate_regressor)
            self.surrogate_weight = sp_cfg.get('weight', 1.0)
            self.surrogate_phase_start = sp_cfg.get('phase_start_epoch', 0)
        else:
            self.surrogate_regressor = None
            self.surrogate_loss = None
            self.surrogate_weight = 0.0
            self.surrogate_phase_start = 0

        # Guard: at least one loss must be enabled, or training will be gradient-free

        active = [
            self.rich_hierarchy, self.radial_loss, self.geodesic_loss,
            self.rank_loss, self.monotonic_loss, self.kl_loss, self.valuation_prior,
            self.wlc_loss, self.angular_coherence, self.algebraic_coherence_loss,
            self.algebraic_addition_loss, self.algebraic_multiplication_loss,
            self.algebraic_distributive_loss,
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

        if self.hyperbolic_contrastive is not None:
            self.log_sigma_contrastive = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.hyperbolic_contrastive_weight), dtype=torch.float64)
            )

        if self.surrogate_loss is not None:
            self.log_sigma_surrogate = nn.Parameter(
                torch.tensor(weight_to_log_sigma(self.surrogate_weight), dtype=torch.float64)
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
            Weighted loss with regularization: weight * loss + log_sigma
        """
        weight = self._uncertainty_weight(log_sigma)
        # +log_sigma is the Kendall & Gal regularization — creates a stable equilibrium
        # at log_sigma = 0.5*log(loss). Using -log_sigma causes log_sigma to diverge to
        # +inf (weight → 0) because dL/d(log_sigma) = -exp(-2*log_sigma)*loss - 1 < 0 always.
        return weight * loss + log_sigma

    def _apply_weight(
        self,
        loss_val: torch.Tensor,
        base_weight: float,
        log_sigma: Optional[nn.Parameter],
        epoch: int,
        device: torch.device,
        phase_start: int = 0,
    ) -> torch.Tensor:
        """Apply weight and phase gating to a loss value."""
        if epoch < phase_start:
            return make_zero_loss(device)
        if self.use_learnable_weights and log_sigma is not None:
            return self._weighted_loss(loss_val, log_sigma)
        return base_weight * loss_val

    def _forward_biological_losses(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        cur_c: Any,
        losses: CombinedLossOutput,
    ) -> torch.Tensor:
        """Compute losses 1–5: the radial/hierarchy group, gated by biological_losses_active."""
        device = z_hyp.device
        total = make_zero_loss(device)

        if not self.biological_losses_active:
            return total

        # 1. RichHierarchyLoss
        if self.rich_hierarchy is not None and epoch >= self.rich_hierarchy_phase_start:
            _call_logits = logits if self.rich_hierarchy_weights.get('coverage', 0.0) > 0.0 else None
            rich_raw, rich_metrics = self.rich_hierarchy(
                z_hyp, indices, logits=_call_logits, targets=targets, curvature=cur_c
            )
            ls = self.log_sigma_hierarchy if self.use_learnable_weights else None
            lc = self.log_sigma_coverage if self.use_learnable_weights else None
            lsp = self.log_sigma_separation if self.use_learnable_weights else None
            weighted_rich = (
                self._apply_weight(rich_raw['hierarchy'], self.rich_hierarchy_weights.get('hierarchy', 5.0), ls, epoch, device) +
                self._apply_weight(rich_raw['coverage'], self.rich_hierarchy_weights.get('coverage', 1.0), lc, epoch, device) +
                self._apply_weight(rich_raw['separation'], self.rich_hierarchy_weights.get('separation', 3.0), lsp, epoch, device)
            )
            losses['rich_hierarchy'] = weighted_rich
            losses['rich_hierarchy_detail'] = rich_metrics
            total = total + weighted_rich

        # 2. RadialHierarchyLoss
        if self.radial_loss is not None and epoch >= self.radial_phase_start:
            radial_out, radial_metrics = self.radial_loss(z_hyp, indices, curvature=cur_c)
            losses['radial'] = radial_out
            losses['radial_metrics'] = radial_metrics
            ls = self.log_sigma_radial if self.use_learnable_weights else None
            total = total + self._apply_weight(radial_out, self.radial_weight, ls, epoch, device)

        # 3. PAdicGeodesicLoss
        if self.geodesic_loss is not None and epoch >= self.geodesic_phase_start:
            geodesic_out, geodesic_metrics = self.geodesic_loss(z_hyp, indices, curvature=cur_c)
            losses['geodesic'] = geodesic_out
            losses['geodesic_metrics'] = geodesic_metrics
            ls = self.log_sigma_geodesic if self.use_learnable_weights else None
            total = total + self._apply_weight(geodesic_out, self.geodesic_weight, ls, epoch, device)

        # 4. GlobalRankLoss
        if self.rank_loss is not None and epoch >= self.rank_phase_start:
            rank_out, rank_metrics = self.rank_loss(z_hyp, indices, curvature=cur_c)
            losses['rank'] = rank_out
            losses['rank_metrics'] = rank_metrics
            ls = self.log_sigma_rank if self.use_learnable_weights else None
            total = total + self._apply_weight(rank_out, self.rank_weight, ls, epoch, device)

        # 5. MonotonicRadialLoss
        if self.monotonic_loss is not None and epoch >= self.monotonic_phase_start:
            monotonic_out, monotonic_metrics = self.monotonic_loss(z_hyp, indices, curvature=cur_c)
            losses['monotonic'] = monotonic_out
            losses['monotonic_metrics'] = monotonic_metrics
            ls = self.log_sigma_monotonic if self.use_learnable_weights else None
            total = total + self._apply_weight(monotonic_out, self.monotonic_weight, ls, epoch, device)

        return total

    def _forward_lagrangian_penalties(
        self,
        dual_weights: Dict[str, List[float]],
        losses: CombinedLossOutput,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute loss 8: Lagrangian dual penalties from outer-loop dual ascent."""
        total = make_zero_loss(device)
        lam_margin = dual_weights.get('lambda_margin', [])
        lam_scatter = dual_weights.get('lambda_scatter', [])
        lam_prior = dual_weights.get('lambda_prior', [])

        # Margin penalties from MonotonicRadialLoss
        if 'monotonic_metrics' in losses:
            lag = make_zero_loss(device)
            for v in range(9):
                vt = losses['monotonic_metrics'].get(f'gap_viol_tensor_v{v}')
                if vt is not None and v < len(lam_margin) and lam_margin[v] > 0.0:
                    lag = lag + lam_margin[v] * vt
            if lag.item() > 0.0:
                losses['lagrangian_margin'] = lag
                total = total + lag

        # Scatter penalties from GlobalRankLoss
        if 'rank_metrics' in losses:
            lag = make_zero_loss(device)
            for v in range(10):
                st = losses['rank_metrics'].get(f'scatter_tensor_v{v}')
                if st is not None and v < len(lam_scatter) and lam_scatter[v] > 0.0:
                    lag = lag + lam_scatter[v] * st
            if lag.item() > 0.0:
                losses['lagrangian_scatter'] = lag
                total = total + lag

        # Prior norm penalties from ValuationPriorLoss
        if 'valuation_prior_metrics' in losses:
            lag = make_zero_loss(device)
            for v in range(10):
                gt = losses['valuation_prior_metrics'].get(f'vp_gap_tensor_v{v}')
                if gt is not None and v < len(lam_prior) and lam_prior[v] > 0.0:
                    lag = lag + lam_prior[v] * gt
            if lag.item() > 0.0:
                losses['lagrangian_prior'] = lag
                total = total + lag

        return total

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
        """Compute combined loss with curriculum-based weight scheduling."""
        device = z_hyp.device
        losses: CombinedLossOutput = {}
        cur_c = curvature if curvature is not None else self.curvature
        total = make_zero_loss(device)

        # 1–5. Radial / hierarchy losses (gated by biological_losses_active)
        total = total + self._forward_biological_losses(z_hyp, indices, logits, targets, epoch, cur_c, losses)

        # 6. KL Divergence — linearly ramped over kl_warmup_epochs after phase_start
        kl_phase_start = self.config.get('hyperbolic_kl', {}).get('phase_start_epoch', 0)
        if self.kl_loss is not None and mu is not None and logvar is not None and epoch >= kl_phase_start:
            kl_out = self.kl_loss(mu, logvar, z_hyp, curvature=cur_c)
            if self.kl_warmup_epochs > 0 and not self.use_learnable_weights:
                warmup_factor = min(1.0, (epoch - kl_phase_start + 1) / self.kl_warmup_epochs)
                effective_kl_weight = self.kl_weight * warmup_factor
            else:
                warmup_factor = 1.0
                effective_kl_weight = self.kl_weight
            ls = self.log_sigma_kl if (self.use_learnable_weights and hasattr(self, 'log_sigma_kl')) else None
            total = total + self._apply_weight(kl_out, effective_kl_weight, ls, epoch, device)
            losses['kl'] = kl_out
            losses['kl_warmup_factor'] = warmup_factor

        # 7. ValuationPriorLoss
        vp_phase_start = self.config.get('valuation_prior', {}).get('phase_start_epoch', 0)
        if self.valuation_prior is not None and mu is not None and epoch >= vp_phase_start:
            vp_out, vp_metrics = self.valuation_prior(mu, indices, logvar=logvar, curvature=cur_c)
            losses['valuation_prior'] = vp_out
            losses['valuation_prior_metrics'] = vp_metrics
            total = total + self._apply_weight(vp_out, self.valuation_prior_weight, None, epoch, device)

        # 8. Lagrangian dual penalties
        if dual_weights is not None:
            total = total + self._forward_lagrangian_penalties(dual_weights, losses, device)

        # 9. Within-level contrastive loss
        if self.wlc_loss is not None:
            wlc_out, wlc_metrics = self.wlc_loss(z_hyp, indices)
            losses['within_level_contrastive'] = wlc_out
            losses['wlc_metrics'] = wlc_metrics
            total = total + wlc_out

        # 10. Angular coherence (requires factored latent r)
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

        # 11. AlgebraicCoherenceLoss (requires factored latent r)
        if self.algebraic_coherence_loss is not None and r is not None:
            alg_out, alg_metrics = self.algebraic_coherence_loss(z_hyp, r, indices, epoch, model=model)
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

        # 12–14. Algebraic homomorphism losses (require mu and model)
        if mu is not None and model is not None:
            if self.algebraic_addition_loss is not None:
                aa_out, aa_metrics = self.algebraic_addition_loss(mu, indices, model, epoch)
                losses['algebraic_addition'] = aa_out
                losses['alg_addition_metrics'] = aa_metrics
                total = total + aa_out

            if self.algebraic_multiplication_loss is not None:
                am_out, am_metrics = self.algebraic_multiplication_loss(mu, indices, model, epoch)
                losses['algebraic_multiplication'] = am_out
                losses['alg_multiplication_metrics'] = am_metrics
                total = total + am_out

            if self.algebraic_distributive_loss is not None:
                ad_out, ad_metrics = self.algebraic_distributive_loss(mu, indices, model, epoch)
                losses['algebraic_distributive'] = ad_out
                losses['alg_distributive_metrics'] = ad_metrics
                total = total + ad_out

        # 15. Fallback coverage loss when rich_hierarchy is absent or gated
        if self.rich_hierarchy is None or not self.biological_losses_active:
            coverage_weight = self.config.get('rich_hierarchy', {}).get('coverage_weight', 1.0)
            if coverage_weight > 0.0:
                coverage_loss = self._compute_coverage_loss(logits, targets)
                losses['coverage'] = coverage_loss
                total = total + coverage_weight * coverage_loss

        # 16. Hyperbolic p-Adic Contrastive Loss
        if self.hyperbolic_contrastive is not None and epoch >= self.hyperbolic_contrastive_phase_start:
            hc_out, hc_metrics = self.hyperbolic_contrastive(z_hyp, indices, curvature=cur_c)
            losses['hyperbolic_contrastive'] = hc_out
            losses['hyperbolic_contrastive_metrics'] = hc_metrics
            ls = self.log_sigma_contrastive if (self.use_learnable_weights and hasattr(self, 'log_sigma_contrastive')) else None
            total = total + self._apply_weight(hc_out, self.hyperbolic_contrastive_weight, ls, epoch, device)

        # 17. Surrogate Property Loss
        if self.surrogate_loss is not None and mu is not None and epoch >= self.surrogate_phase_start:
            sp_out, sp_metrics = self.surrogate_loss(mu, targets)
            losses['surrogate_property'] = sp_out
            losses['surrogate_property_metrics'] = sp_metrics
            ls = self.log_sigma_surrogate if (self.use_learnable_weights and hasattr(self, 'log_sigma_surrogate')) else None
            total = total + self._apply_weight(sp_out, self.surrogate_weight, ls, epoch, device)

        losses['total'] = total
        return losses

    def _compute_coverage_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return compute_coverage_loss(logits, targets)

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
        if self.algebraic_multiplication_loss is not None:
            enabled.append('algebraic_multiplication')
        if self.algebraic_distributive_loss is not None:
            enabled.append('algebraic_distributive')
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
