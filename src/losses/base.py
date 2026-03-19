# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Abstract base class for all hierarchy losses.

Enforces a consistent interface contract across all loss implementations:
- forward() always returns (torch.Tensor, dict) — scalar loss + metrics dict
- All constructors validate their parameters at construction time
- __repr__ is meaningful for inspection

This prevents the silent divergence found in v6.2 audit where RichHierarchyLoss
returned only a dict while all other losses returned (tensor, dict), causing
CombinedLoss to handle it as a special case and making interface assumptions
invisible to static analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn


class HierarchyLossBase(ABC, nn.Module):
    """Abstract base for all p-adic hierarchy losses.

    Contract:
        forward(z_hyp, batch_indices, **kwargs) -> Tuple[torch.Tensor, Dict]
            z_hyp:         Points on Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices used for 3-adic valuation (batch,)
            **kwargs:      Loss-specific additional inputs (logits, targets, epoch, etc.)

            Returns:
                loss:    Scalar tensor, differentiable, on z_hyp.device
                metrics: Dict of float scalars for logging (detached, no grad)

    Validation:
        All subclasses must call _validate_radii() and _validate_positive()
        from their __init__ before any computation.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and return (scalar_loss, metrics_dict).

        Args:
            z_hyp: Points on Poincaré ball, shape (B, latent_dim), float64
            batch_indices: Ternary operation indices, shape (B,), int64
            **kwargs: Loss-specific inputs

        Returns:
            loss: Scalar differentiable tensor on z_hyp.device
            metrics: Dict[str, float] — logging values, no gradient
        """

    # ------------------------------------------------------------------
    # Shared validation helpers — call from subclass __init__
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_radii(inner_radius: float, outer_radius: float, cls_name: str) -> None:
        """Validate Poincaré ball radius parameters.

        Raises:
            ValueError: If radii are outside valid range or inverted.
        """
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
        """Validate that a parameter is strictly positive.

        Raises:
            ValueError: If value <= 0.
        """
        if value <= 0.0:
            raise ValueError(
                f"{cls_name}: {name} must be positive, got {value}"
            )

    @staticmethod
    def _validate_weight(value: float, name: str, cls_name: str) -> None:
        """Validate that a loss weight is non-negative.

        Raises:
            ValueError: If value < 0 (negative weights invert gradients).
        """
        if value < 0.0:
            raise ValueError(
                f"{cls_name}: {name} must be >= 0 (negative weights invert gradients), got {value}"
            )

    @staticmethod
    def _validate_probability(value: float, name: str, cls_name: str) -> None:
        """Validate that a value is in [0, 1].

        Raises:
            ValueError: If value outside [0, 1].
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{cls_name}: {name} must be in [0, 1], got {value}"
            )
