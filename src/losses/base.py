# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Abstract base classes for all hierarchy losses.

Two contracts are defined here:

HierarchyLossBase
    The standard contract: forward() returns (scalar_tensor, metrics_dict).
    Used by PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss,
    MonotonicRadialLoss.

RichHierarchyLossBase
    The component contract: forward() returns (component_tensors, metrics_dict).
    Used by RichHierarchyLoss only, because CombinedLoss applies its own
    weights to each component rather than receiving a pre-weighted scalar.

MetricsDict
    Type alias for the metrics return value. Values are numeric scalars
    (int or float) safe for logging. No gradient tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
try:
    from typing import TypedDict
except ImportError:  # Python <3.8 fallback (not expected, but safe)
    from typing_extensions import TypedDict

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Shared type aliases
# ------------------------------------------------------------------

# Metrics dicts primarily contain int or float scalars for logging.
# Any is allowed so that Lagrangian dual code can store per-level violation
# tensors (in-graph) alongside the usual float logging values.
# Convention: keys ending in '_tensor' hold torch.Tensor; all others are
# int/float and safe for .item() / logging.
MetricsDict = Dict[str, Any]


class CombinedLossOutput(TypedDict, total=False):
    """Typed return value of ``CombinedLoss.forward()``.

    All keys except ``total`` are optional — they are only present when the
    corresponding loss component is enabled in the config.  Keys ending in
    ``_metrics`` are MetricsDict sub-dicts (safe to log, no gradients).
    Keys ending in ``_tensor`` inside the sub-dicts are in-graph tensors
    used by the Lagrangian dual; all other values are float/int.

    Quick reference for loss component keys:
        total                   → scalar differentiable tensor (always present)
        kl                      → HyperbolicKLDivergence component
        rich_hierarchy          → RichHierarchyLoss weighted sum
        radial                  → RadialHierarchyLoss component
        geodesic                → PAdicGeodesicLoss component
        rank                    → GlobalRankLoss component
        monotonic               → MonotonicRadialLoss component
        angular_coherence       → AngularCoherenceLoss component
        valuation_prior         → ValuationPriorLoss component
        within_level_contrastive → WithinLevelContrastiveLoss component
        coverage                → fallback cross-entropy (only when rich_hierarchy disabled)
        lagrangian_margin       → Lagrangian margin penalty (if dual enabled)
        lagrangian_scatter      → Lagrangian scatter penalty (if dual enabled)
        lagrangian_prior        → Lagrangian prior penalty (if dual enabled)

    Metrics sub-dicts (float/int values, no gradients):
        rich_hierarchy_detail   → sub-dict from RichHierarchyLoss
        radial_metrics          → sub-dict from RadialHierarchyLoss
        geodesic_metrics        → sub-dict from PAdicGeodesicLoss
        rank_metrics            → sub-dict from GlobalRankLoss
        monotonic_metrics       → sub-dict from MonotonicRadialLoss (r_v0..r_v9)
        angular_coherence_metrics → sub-dict from AngularCoherenceLoss
        valuation_prior_metrics → sub-dict from ValuationPriorLoss
        wlc_metrics             → sub-dict from WithinLevelContrastiveLoss

    When ``learnable_weights=True`` the effective weights are logged via
    ``loss_fn.get_learned_weights()`` — they are not present in this dict.

    Usage::

        losses: CombinedLossOutput = loss_fn(z_hyp, indices, logits, targets, epoch=ep)
        losses["total"].backward()                       # always safe
        h_metrics = losses.get("rich_hierarchy_detail", {})
        r_v0 = h_metrics.get("r_v0", float("nan"))
        ac_m = losses.get("angular_coherence_metrics", {})
    """

    total: Any          # torch.Tensor — always present
    kl: Any
    rich_hierarchy: Any
    radial: Any
    geodesic: Any
    rank: Any
    monotonic: Any
    angular_coherence: Any
    valuation_prior: Any
    within_level_contrastive: Any
    coverage: Any
    lagrangian_margin: Any
    lagrangian_scatter: Any
    lagrangian_prior: Any
    rich_hierarchy_detail: MetricsDict
    radial_metrics: MetricsDict
    geodesic_metrics: MetricsDict
    rank_metrics: MetricsDict
    monotonic_metrics: MetricsDict
    angular_coherence_metrics: MetricsDict
    valuation_prior_metrics: MetricsDict
    wlc_metrics: MetricsDict


class HierarchyLossBase(ABC, nn.Module):
    """Abstract base for p-adic hierarchy losses returning a scalar loss.

    Contract:
        forward(z_hyp, batch_indices, **kwargs) -> Tuple[Tensor, MetricsDict]

            Returns:
                loss:    Scalar differentiable tensor on z_hyp.device
                metrics: MetricsDict — numeric logging values, no gradient

    Validation:
        All subclasses must call _validate_radii(), _validate_positive(), etc.
        from their __init__ before any computation.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Compute loss and return (scalar_loss, metrics_dict).

        Args:
            z_hyp: Points on Poincaré ball, shape (B, latent_dim), float64
            batch_indices: Ternary operation indices, shape (B,), int64
            **kwargs: Loss-specific inputs

        Returns:
            loss: Scalar differentiable tensor on z_hyp.device
            metrics: MetricsDict — logging values, no gradient
        """

    # ------------------------------------------------------------------
    # Shared validation helpers — call from subclass __init__
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_radii(inner_radius: float, outer_radius: float, cls_name: str) -> None:
        if not (0.0 < inner_radius < 1.0):
            raise ValueError(
                f"{cls_name}: inner_radius must be in (0, 1), got {inner_radius}"
            )
        if not (0.0 < outer_radius < 1.0):
            raise ValueError(
                f"{cls_name}: outer_radius must be in (0, 1), got {outer_radius}"
            )
        if inner_radius >= outer_radius:
            raise ValueError(
                f"{cls_name}: inner_radius ({inner_radius}) must be < outer_radius ({outer_radius})"
            )

    @staticmethod
    def _validate_positive(value: float, name: str, cls_name: str) -> None:
        if value <= 0.0:
            raise ValueError(
                f"{cls_name}: {name} must be positive, got {value}"
            )

    @staticmethod
    def _validate_weight(value: float, name: str, cls_name: str) -> None:
        if value < 0.0:
            raise ValueError(
                f"{cls_name}: {name} must be >= 0 (negative weights invert gradients), got {value}"
            )

    @staticmethod
    def _validate_probability(value: float, name: str, cls_name: str) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{cls_name}: {name} must be in [0, 1], got {value}"
            )


class RichHierarchyLossBase(HierarchyLossBase):
    """Abstract base for losses that return per-component tensors.

    Unlike HierarchyLossBase (which returns a pre-weighted scalar),
    subclasses return a dict of raw component tensors so that CombinedLoss
    can apply its own configurable weights to each component.

    Contract:
        forward(z_hyp, batch_indices, **kwargs)
            -> Tuple[Dict[str, Tensor], MetricsDict]

            Returns:
                raw: Dict mapping component name -> differentiable tensor
                     e.g. {"hierarchy": t, "coverage": t, "separation": t}
                metrics: MetricsDict — float/int logging values, no gradient
    """

    @abstractmethod
    def forward(  # type: ignore[override]
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], MetricsDict]:
        """Return component tensors and metrics.

        Args:
            z_hyp: Points on Poincaré ball, shape (B, latent_dim), float64
            batch_indices: Ternary operation indices, shape (B,), int64
            **kwargs: Must contain 'logits' and 'targets'

        Returns:
            raw: Component tensors — each differentiable, on z_hyp.device
            metrics: MetricsDict — logging values, no gradient
        """


__all__ = [
    "HierarchyLossBase",
    "RichHierarchyLossBase",
    "MetricsDict",
    "CombinedLossOutput",
]
