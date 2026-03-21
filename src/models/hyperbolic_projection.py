# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic Projection Layer.

True hyperbolic projection using geoopt expmap0/logmap0.

Architecture (V6.0 - True Hyperbolic):
  z_tangent → [tangent_transform] → z_tangent_scaled
            → expmap0(z_tangent_scaled) → z_hyp (on Poincaré manifold)

The tangent space at origin T₀M IS Euclidean ℝⁿ, so:
- Encoder outputs live in tangent space (Euclidean MLPs work)
- We transform in tangent space to learn hierarchy
- expmap0 projects to manifold (true hyperbolic)
- logmap0 returns to tangent space for decoder

Reference:
  Mathieu et al. (2019) "Continuous Hierarchical Representations with Poincaré VAEs"
"""

from typing import Tuple, Union, cast

import geoopt
import torch
import torch.nn as nn

from src.geometry import ManifoldParameter, exp_map_zero


class HyperbolicProjection(nn.Module):
    """True hyperbolic projection using expmap0.

    Transforms tangent vectors and projects to Poincaré ball via expmap0.

    Architecture:
        z_tangent → tangent_net (residual MLP) → z_transformed
                  → expmap0(z_transformed) → z_hyp (on manifold)

    The tangent space at origin is Euclidean, so standard MLPs work.
    The tangent_net learns to scale/rotate vectors to achieve correct
    radial hierarchy (magnitude → distance from origin on manifold).

    Key insight: ||expmap0(v)|| increases with ||v||, so learning the
    right tangent magnitudes achieves the right radial positions.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        init_identity: bool = False,
        tangent_scale_init: float = 0.1,
        n_layers: int = 1,
        dropout: float = 0.0,
        learnable_curvature: bool = False,
    ):
        """Initialize HyperbolicProjection.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for tangent network
            max_radius: Maximum radius constraint (applied after expmap)
            curvature: Hyperbolic curvature parameter
            init_identity: If True, initialize tangent_net as identity
            tangent_scale_init: Initial value for learnable tangent_scale (default 0.1)
            n_layers: Number of hidden layers
            dropout: Dropout rate
            learnable_curvature: If True, curvature is learnable
        """
        super().__init__()
        if not (0.0 < max_radius < 1.0):
            raise ValueError(
                f"HyperbolicProjection: max_radius must be in (0, 1) for a valid "
                f"Poincaré ball constraint, got {max_radius}"
            )
        if curvature <= 0.0:
            raise ValueError(
                f"HyperbolicProjection: curvature must be > 0, got {curvature}"
            )
        if not (0.0 <= tangent_scale_init):
            raise ValueError(
                f"HyperbolicProjection: tangent_scale_init must be >= 0, got {tangent_scale_init}"
            )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.learnable_curvature = learnable_curvature

        # Create manifold
        self.manifold = geoopt.PoincareBall(c=curvature, learnable=learnable_curvature)
        self.curvature = self.manifold.c

        # Tangent space transform network (residual architecture)
        # Learns to scale/rotate tangent vectors for correct hierarchy
        layers = []
        for i in range(n_layers):
            in_dim = latent_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.tangent_net = nn.Sequential(*layers)

        # Tangent scale: controls initial magnitude of tangent vectors entering expmap0.
        # Encoder outputs have norm ~4.0; expmap0(tanh) saturates for large norms.
        # tangent_scale_init=0.1 gives tangent norm ~0.4 → expmap0 radius ~0.38,
        # well within target range [inner_radius, outer_radius].
        # Value is passed from config (model.tangent_scale in v6.yaml).
        self.tangent_scale = nn.Parameter(
            torch.tensor(tangent_scale_init, dtype=torch.float64)
        )

        if init_identity:
            self._init_identity()

        # Enforce float64 precision for numerical stability
        self.to(torch.float64)

    def _init_identity(self):
        """Initialize tangent_net as identity (zero residual)."""
        with torch.no_grad():
            cast(nn.Linear, self.tangent_net[-1]).weight.zero_()
            cast(nn.Linear, self.tangent_net[-1]).bias.zero_()

    def forward(
        self, z_tangent: torch.Tensor, as_manifold: bool = False
    ) -> Union[torch.Tensor, "ManifoldParameter"]:
        """Project tangent vector to Poincaré ball via expmap0.

        Args:
            z_tangent: Tangent vectors at origin (batch, latent_dim)
            as_manifold: If True, return ManifoldParameter

        Returns:
            z_hyp: Points on Poincaré ball
        """
        # Enforce float64 precision
        z_tangent = z_tangent.to(torch.float64)

        # Scale tangent vectors + residual transform
        # tangent_scale is learned; init=0.1 prevents expmap0 saturation at init
        z_scaled = self.tangent_scale * z_tangent
        z_transformed = z_scaled + self.tangent_net(z_scaled)

        # Project to manifold via expmap0 using model's own manifold instance.
        # Using self.manifold.expmap() instead of the global exp_map_zero() cache
        # ensures gradients flow through self.manifold.c (learnable curvature).
        # The global cache uses a detached float key and returns a static manifold.
        origin = torch.zeros_like(z_transformed)
        z_hyp = self.manifold.expmap(origin, z_transformed)

        # Apply max_radius constraint (explicit float64 for numerical stability)
        norm = torch.norm(z_hyp, dim=-1, keepdim=True)
        eps = torch.tensor(1e-10, device=z_hyp.device, dtype=torch.float64)
        scale = torch.where(
            norm > self.max_radius,
            self.max_radius / (norm + eps),
            torch.ones(1, device=z_hyp.device, dtype=torch.float64)
        )
        z_hyp = z_hyp * scale

        if as_manifold:
            z_hyp = self.manifold.projx(z_hyp)
            return ManifoldParameter(z_hyp, manifold=self.manifold)

        return z_hyp

    def get_curvature(self) -> float:
        """Get current curvature value."""
        c = self.manifold.c
        return c.item() if hasattr(c, "item") else float(c)

    def forward_with_components(
        self, z_tangent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project with diagnostic outputs.

        Returns:
            Tuple of (z_hyp, z_transformed, tangent_norm)
        """
        # Enforce float64 precision
        z_tangent = z_tangent.to(torch.float64)

        z_scaled = self.tangent_scale * z_tangent
        z_transformed = z_scaled + self.tangent_net(z_scaled)
        origin = torch.zeros_like(z_transformed)
        z_hyp = self.manifold.expmap(origin, z_transformed)

        # Apply max_radius constraint (explicit float64 for numerical stability)
        norm = torch.norm(z_hyp, dim=-1, keepdim=True)
        eps = torch.tensor(1e-10, device=z_hyp.device, dtype=torch.float64)
        scale = torch.where(
            norm > self.max_radius,
            self.max_radius / (norm + eps),
            torch.ones(1, device=z_hyp.device, dtype=torch.float64)
        )
        z_hyp = z_hyp * scale

        tangent_norm = torch.norm(z_transformed, dim=-1)
        return z_hyp, z_transformed, tangent_norm


class DualHyperbolicProjection(nn.Module):
    """Dual hyperbolic projections for VAE-A and VAE-B.

    Each VAE gets its own projection to the Poincaré ball via expmap0.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        n_layers: int = 1,
        dropout: float = 0.0,
        learnable_curvature: bool = False,
        init_identity: bool = False,
        tangent_scale_init: float = 0.1,
    ):
        """Initialize DualHyperbolicProjection.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for tangent networks
            max_radius: Maximum Poincaré ball radius
            curvature: Hyperbolic curvature
            n_layers: Number of hidden layers
            dropout: Dropout rate
            learnable_curvature: If True, curvature is learnable
            init_identity: If True, initialize tangent_net as identity (zero residual)
            tangent_scale_init: Initial value for learnable tangent_scale parameter
        """
        super().__init__()
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.learnable_curvature = learnable_curvature

        # VAE-A projection
        self.proj_A = HyperbolicProjection(
            latent_dim,
            hidden_dim,
            max_radius,
            curvature,
            n_layers=n_layers,
            dropout=dropout,
            learnable_curvature=learnable_curvature,
            init_identity=init_identity,
            tangent_scale_init=tangent_scale_init,
        )

        # VAE-B projection (separate)
        self.proj_B = HyperbolicProjection(
            latent_dim,
            hidden_dim,
            max_radius,
            curvature,
            n_layers=n_layers,
            dropout=dropout,
            learnable_curvature=False,  # Share curvature with A
            init_identity=init_identity,
            tangent_scale_init=tangent_scale_init,
        )

    def forward(
        self, z_A: torch.Tensor, z_B: torch.Tensor, as_manifold: bool = False
    ) -> Tuple[
        Union[torch.Tensor, "ManifoldParameter"],
        Union[torch.Tensor, "ManifoldParameter"],
    ]:
        """Project both tangent vectors to Poincaré ball.

        Args:
            z_A: VAE-A tangent vector (batch, latent_dim)
            z_B: VAE-B tangent vector (batch, latent_dim)
            as_manifold: If True, return ManifoldParameters

        Returns:
            Tuple of (z_A_hyp, z_B_hyp) on the manifold
        """
        z_A_hyp = self.proj_A(z_A, as_manifold=as_manifold)
        z_B_hyp = self.proj_B(z_B, as_manifold=as_manifold)
        return z_A_hyp, z_B_hyp

    def get_curvature(self) -> float:
        """Get current curvature value."""
        return self.proj_A.get_curvature()


__all__ = ["HyperbolicProjection", "DualHyperbolicProjection"]
