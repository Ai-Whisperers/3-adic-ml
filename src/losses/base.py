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
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.core.contracts import CombinedLossOutput, MetricsDict



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
