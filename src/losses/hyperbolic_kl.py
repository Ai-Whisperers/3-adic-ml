# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic KL Divergence for Poincaré VAEs.

This module provides a curvature-corrected KL divergence for VAEs with
hyperbolic latent spaces. The standard Gaussian KL doesn't account for
the curved geometry of the Poincaré ball.

Reference:
    Mathieu et al. (2019) "Continuous Hierarchical Representations with
    Poincaré Variational Auto-Encoders"

Usage:
    from src.losses import HyperbolicKLDivergence

    kl_loss = HyperbolicKLDivergence(curvature=1.0, beta=1.0)
    loss = kl_loss(mu, logvar, z_hyp)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.geometry import lambda_x


class HyperbolicKLDivergence(nn.Module):
    """Curvature-corrected KL divergence for hyperbolic VAEs.

    In hyperbolic space, the standard Gaussian KL divergence needs to be
    adjusted by the conformal factor λ(x) = 2 / (1 - c||x||²) to account
    for the metric distortion near the boundary.

    The corrected KL is:
        KL_hyp = 0.5 * (λ(μ)² * σ² + ||μ||² - log(σ²) - d)

    where d is the latent dimension.

    Args:
        curvature: Poincaré ball curvature (default: 1.0)
        beta: KL weight for β-VAE (default: 1.0)
        free_bits: Minimum KL per dimension (default: 0.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        curvature: float = 1.0,
        beta: float = 1.0,
        free_bits: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.curvature = curvature
        self.beta = beta
        self.free_bits = free_bits
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z_hyp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute curvature-corrected KL divergence.

        Args:
            mu: Mean of the approximate posterior, shape (B, d)
            logvar: Log-variance of the approximate posterior, shape (B, d)
            z_hyp: Optional hyperbolic embedding for conformal factor.
                   If None, uses mu directly.

        Returns:
            KL divergence loss (scalar if reduction='mean' or 'sum')
        """
        # Use z_hyp for conformal factor if provided, else use mu
        # (mu is in tangent space, z_hyp is on manifold)
        point_for_lambda = z_hyp if z_hyp is not None else mu

        # Compute conformal factor: λ(x) = 2 / (1 - c||x||²)
        # lambda_x returns shape (B, 1) with keepdim=True
        conf_factor = lambda_x(point_for_lambda, c=self.curvature, keepdim=True)

        # Variance from log-variance
        var = torch.exp(logvar)

        # Curvature-corrected KL divergence per dimension:
        # KL_d = 0.5 * (λ² * σ² + μ² - log(σ²) - 1)
        kl_per_dim = 0.5 * (
            conf_factor.pow(2) * var +  # Scaled variance term
            mu.pow(2) -                  # Mean squared term
            logvar -                     # Log-variance term
            1.0                          # Dimension term
        )

        # Apply free bits (minimum KL per dimension)
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over dimensions to get KL per sample
        kl_per_sample = kl_per_dim.sum(dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            kl = kl_per_sample.mean()
        elif self.reduction == "sum":
            kl = kl_per_sample.sum()
        else:  # 'none'
            kl = kl_per_sample

        return self.beta * kl

    def extra_repr(self) -> str:
        return f"curvature={self.curvature}, beta={self.beta}, free_bits={self.free_bits}"


class StandardKLDivergence(nn.Module):
    """Standard Gaussian KL divergence (for comparison).

    KL(q(z|x) || p(z)) where q = N(μ, σ²) and p = N(0, I)

    KL = 0.5 * (σ² + μ² - log(σ²) - 1)

    Args:
        beta: KL weight for β-VAE (default: 1.0)
        free_bits: Minimum KL per dimension (default: 0.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z_hyp: Optional[torch.Tensor] = None,  # Ignored, for API compatibility
    ) -> torch.Tensor:
        """Compute standard Gaussian KL divergence.

        Args:
            mu: Mean of the approximate posterior, shape (B, d)
            logvar: Log-variance of the approximate posterior, shape (B, d)
            z_hyp: Ignored (for API compatibility with HyperbolicKLDivergence)

        Returns:
            KL divergence loss
        """
        var = torch.exp(logvar)

        # Standard KL per dimension
        kl_per_dim = 0.5 * (var + mu.pow(2) - logvar - 1.0)

        # Apply free bits
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over dimensions
        kl_per_sample = kl_per_dim.sum(dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            kl = kl_per_sample.mean()
        elif self.reduction == "sum":
            kl = kl_per_sample.sum()
        else:
            kl = kl_per_sample

        return self.beta * kl

    def extra_repr(self) -> str:
        return f"beta={self.beta}, free_bits={self.free_bits}"


__all__ = [
    "HyperbolicKLDivergence",
    "StandardKLDivergence",
]
