# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""QSPR Surrogate Property Loss for co-training representation space with sequence properties."""

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import HierarchyLossBase, MetricsDict


class SurrogateRegressor(nn.Module):
    """Euclidean MLP surrogate for sequence property prediction from VAE tangent space."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        # Enforce float64 internally
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # Target: [net_hydropathy, transition_complexity]
        )
        self.double()

    def forward(self, z_tangent: torch.Tensor) -> torch.Tensor:
        return self.net(z_tangent.double())


class SurrogatePropertyLoss(HierarchyLossBase):
    """Surrogate Property Loss.

    Co-trains the latent space to differentiably align with physical sequence properties
    such as net hydropathy and transition complexity.
    """

    def __init__(self, regressor: SurrogateRegressor):
        super().__init__()
        self.regressor = regressor

    def forward(
        self,
        z_tangent: torch.Tensor,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """
        z_tangent: VAE latent tangent space vector, shape (N, latent_dim)
        x: input ternary sequences, shape (N, 9) with values in {-1, 0, 1}
        """
        # Ensure float64
        x = x.to(torch.float64)
        z_tangent = z_tangent.to(torch.float64)

        # 1. Compute target properties from ternary input x
        # Property 1: Net hydropathy (GRAVY proxy)
        target_hydropathy = x.mean(dim=-1, keepdim=True)  # (N, 1)

        # Property 2: Sign-transition complexity (flexibility/stability proxy)
        # Measures sequence fluctuations
        target_complexity = torch.abs(x[:, 1:] - x[:, :-1]).mean(dim=-1, keepdim=True)  # (N, 1)

        # Combine into shape (N, 2)
        targets = torch.cat([target_hydropathy, target_complexity], dim=-1)

        # 2. Predict using the regressor MLP
        predictions = self.regressor(z_tangent)

        # 3. Compute MSE Loss
        loss = F.mse_loss(predictions, targets)

        metrics = {
            "surrogate_mse_loss": float(loss.item()),
            "mean_target_hydropathy": float(target_hydropathy.mean().item()),
            "mean_pred_hydropathy": float(predictions[:, 0].mean().item()),
            "mean_target_complexity": float(target_complexity.mean().item()),
            "mean_pred_complexity": float(predictions[:, 1].mean().item()),
        }

        return loss, metrics
