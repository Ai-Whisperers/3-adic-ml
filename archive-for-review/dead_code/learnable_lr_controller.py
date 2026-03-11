# ARCHIVED: 2026-03-10
# REASON: Experimental MLP-based LR controller. Never used in production training.
#          MetricBasedLR is the active controller. This class could be reintegrated
#          if meta-learning of LR schedules proves valuable.
# ORIGINAL LOCATION: src/models/lr_controller.py (lines 473-557)
# DEPENDENCIES: torch, torch.nn, LRController ABC, TrainingMetrics dataclass

import torch
import torch.nn as nn
from typing import Any, Dict


class LearnableLRController(nn.Module):
    """Learnable LR controller (differentiable, experimental).

    A small MLP that takes metrics and outputs LR multipliers.
    Can be trained end-to-end with the VAE using meta-learning.

    Warning: This is experimental and may destabilize training.
    """

    def __init__(
        self,
        n_metrics: int = 6,
        hidden_dim: int = 32,
        n_components: int = 4,
        temperature: float = 1.0,
    ):
        """Initialize learnable controller.

        Args:
            n_metrics: Number of input metrics
            hidden_dim: Hidden dimension for MLP
            n_components: Number of output LR scales
            temperature: Softmax temperature (higher = softer)
        """
        super().__init__()
        self.n_components = n_components
        self.temperature = temperature

        self.mlp = nn.Sequential(
            nn.Linear(n_metrics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_components),
        )

        # Initialize to output near-1.0 scales
        with torch.no_grad():
            self.mlp[-1].bias.fill_(2.0)  # sigmoid(2) ~ 0.88

        self.component_names = ["encoder_a", "encoder_b", "projections", "decoders"]

    def forward(self, metrics_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass: metrics -> LR scales.

        Args:
            metrics_tensor: (batch, n_metrics) or (n_metrics,)

        Returns:
            LR scales in [0, 1], shape (batch, n_components) or (n_components,)
        """
        logits = self.mlp(metrics_tensor)
        return torch.sigmoid(logits / self.temperature)

    def get_lr_scales(self, metrics) -> Dict[str, float]:
        """Get LR scales from metrics."""
        metrics_tensor = torch.tensor(
            [
                metrics.coverage,
                metrics.hierarchy_a,
                metrics.hierarchy_b,
                metrics.dist_corr_a,
                metrics.q_value,
                metrics.grad_norm_projections,
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            scales = self.forward(metrics_tensor)

        return {
            name: scales[i].item()
            for i, name in enumerate(self.component_names[: self.n_components])
        }

    def update(self, metrics) -> Dict[str, Any]:
        """Return scales (state is implicit in weights)."""
        scales = self.get_lr_scales(metrics)
        return {
            "lr_scales": scales,
            "events": [],
            "type": "learnable",
        }
