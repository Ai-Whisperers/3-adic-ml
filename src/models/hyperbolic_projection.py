# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

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

import math
from typing import Optional, Tuple, Union

import geoopt
import torch
import torch.nn as nn

from src.geometry import ManifoldParameter, clamp_to_max_norm


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

    tangent_net: nn.Sequential
    linear_r: Optional[nn.Linear]

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: Union[float, torch.Tensor, geoopt.PoincareBall] = 1.0,
        init_identity: bool = False,
        tangent_scale_init: float = 0.1,
        n_layers: int = 1,
        dropout: float = 0.0,
        learnable_curvature: bool = False,
        factored: bool = False,
        radial_dims: int = 4,
    ):
        """Initialize HyperbolicProjection.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for tangent network
            max_radius: Maximum radius constraint (applied after expmap)
            curvature: Hyperbolic curvature parameter, tensor, or existing manifold
            init_identity: If True, initialize tangent_net as identity
            tangent_scale_init: Initial value for learnable tangent_scale (default 0.1)
            n_layers: Number of hidden layers
            dropout: Dropout rate
            learnable_curvature: If True, curvature is learnable
            factored: If True, use V7 factored latent (z_r ⊕ z_θ → r * normalize(z_θ))
            radial_dims: Number of dimensions dedicated to radius in factored mode (k)
        """
        super().__init__()
        if not (0.0 < max_radius < 1.0):
            raise ValueError(
                f"HyperbolicProjection: max_radius must be in (0, 1) for a valid "
                f"Poincaré ball constraint, got {max_radius}"
            )
        # Validate curvature only if it's a float/tensor (manifold handles its own c)
        if not isinstance(curvature, geoopt.PoincareBall):
            c_val = curvature if isinstance(curvature, float) else curvature.item()
            if c_val <= 0.0:
                raise ValueError(
                    f"HyperbolicProjection: curvature must be > 0, got {c_val}"
                )

        if tangent_scale_init <= 0:
            raise ValueError(
                f"HyperbolicProjection: tangent_scale_init must be > 0, got {tangent_scale_init}"
            )
        if factored and radial_dims >= latent_dim:
            raise ValueError(
                f"HyperbolicProjection: radial_dims ({radial_dims}) must be < latent_dim ({latent_dim})"
            )
        if n_layers == 0:
            raise ValueError(
                "HyperbolicProjection: n_layers=0 is not supported. "
                "With 0 hidden layers the tangent_net output layer expects hidden_dim "
                f"({hidden_dim}) inputs but z_tangent has {latent_dim if not factored else latent_dim - radial_dims} dims. "
                "Use n_layers >= 1 (default: 1)."
            )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.learnable_curvature = learnable_curvature
        self.factored = factored
        self.radial_dims = radial_dims

        # Create or assign manifold (used for non-factored mode; kept for curvature tracking)
        if isinstance(curvature, geoopt.PoincareBall):
            self.manifold = curvature
        else:
            self.manifold = geoopt.PoincareBall(c=curvature, learnable=learnable_curvature)
        self.curvature = self.manifold.c

        # In factored mode:
        #   - tangent_net processes z_θ (direction dims: latent_dim - radial_dims)
        #   - linear_r maps z_r (radial_dims) → scalar r via sigmoid * max_radius
        # In non-factored mode:
        #   - tangent_net processes full z_tangent (latent_dim) for expmap0 approach
        net_in_dim = (latent_dim - radial_dims) if factored else latent_dim

        # Tangent space transform network (residual architecture)
        layers = []
        for i in range(n_layers):
            in_dim = net_in_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, net_in_dim))
        self.tangent_net = nn.Sequential(*layers)

        # Factored mode: linear_r maps radial dims to a scalar pre-sigmoid.
        # Small random weight init ensures gradient flows from hierarchy losses
        # through r back to z_r (and thus to the encoder) from epoch 1.
        # Bias=0 → sigmoid(~0) ≈ 0.5 → r ≈ 0.5 * max_radius at initialization.
        if factored:
            self.linear_r = nn.Linear(radial_dims, 1, bias=True)
            nn.init.normal_(self.linear_r.weight, std=0.1)
            nn.init.zeros_(self.linear_r.bias)
        else:
            self.linear_r = None

        # Tangent scale: in non-factored mode scales encoder outputs before expmap0.
        # In factored mode scales z_θ before direction residual network.
        # Stored in log-space so exp(log_tangent_scale) is always strictly positive,
        # preventing the degenerate fixed point where scale→0 collapses all directions.
        self.log_tangent_scale = nn.Parameter(
            torch.tensor(math.log(tangent_scale_init), dtype=torch.float64)
        )

        if init_identity:
            self._init_identity_manual()

        # Enforce float64 precision for numerical stability
        self.to(torch.float64)

    def _expmap_and_clamp(self, z_transformed: torch.Tensor) -> torch.Tensor:
        """Apply expmap0 and enforce max_radius constraint."""
        origin = torch.zeros_like(z_transformed)
        z_hyp = self.manifold.expmap(origin, z_transformed)
        return clamp_to_max_norm(z_hyp, self.max_radius)

    def _residual_tangent_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Scale by tangent_scale, then add the residual tangent_net output.

        z_scaled = tangent_scale * z; return z_scaled + tangent_net(z_scaled).
        Shared by the non-factored path (applied to the full z_tangent) and
        the factored path (applied to z_theta only) — identical operation,
        different slice of the input.
        """
        z_scaled = self.tangent_scale * z
        return z_scaled + self.tangent_net(z_scaled)

    def _transform_and_project(
        self, z_tangent: torch.Tensor, need_tangent_norm: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Non-factored tangent transform + expmap0 projection.

        Returns (z_hyp, z_transformed, tangent_norm), shared by forward() and
        forward_with_components(). forward() doesn't use tangent_norm, so it
        passes need_tangent_norm=False to skip that extra norm on every call.
        """
        z_transformed = self._residual_tangent_transform(z_tangent)
        z_hyp = self._expmap_and_clamp(z_transformed)
        tangent_norm = torch.norm(z_transformed, dim=-1) if need_tangent_norm else None
        return z_hyp, z_transformed, tangent_norm

    def _init_identity_manual(self):
        """Initialize tangent_net as identity (zero residual)."""
        with torch.no_grad():
            last_layer = self.tangent_net[-1]
            if isinstance(last_layer, nn.Linear):
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    @property
    def tangent_scale(self) -> torch.Tensor:
        """Effective tangent scale = exp(log_tangent_scale), clamped to (~5e-5, 20).

        Clamping here (not in forward) ensures the bound is enforced on every access,
        including forward_with_components and any future entry points.
        """
        with torch.no_grad():
            self.log_tangent_scale.data.clamp_(-10.0, 3.0)
        return self.log_tangent_scale.exp()

    def _clamp_curvature(self) -> None:
        """Enforce safe range on learnable curvature [0.1, 5.0]."""
        if self.learnable_curvature:
            with torch.no_grad():
                # Some manifolds use softplus/exp transforms on internal parameters.
                # We check the effective curvature and clamp the underlying data
                # to keep it in a numerically stable 'active' region.
                curr_c = self.manifold.c
                if curr_c < 0.1 or curr_c > 5.0:
                    for name, p in self.manifold.named_parameters():
                        if any(key in name for key in ["c", "k", "kappa"]):
                            # Clamp parameter to a safe 'latent' range
                            # For softplus, 0.1 effective c corresponds to ~ -2.25 raw k
                            # For softplus, 5.0 effective c corresponds to ~ 4.99 raw k
                            p.data.clamp_(min=-3.0, max=5.0)

    def forward(
        self, z_tangent: torch.Tensor, as_manifold: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], "ManifoldParameter"]:
        """Project tangent vector to Poincaré ball.

        Non-factored mode: expmap0 approach (V6).
        Factored mode: z_r → sigmoid → r, z_θ → normalize → dir, z_hyp = r * dir (V7).

        Args:
            z_tangent: Tangent vectors at origin (batch, latent_dim)
            as_manifold: If True, return ManifoldParameter (non-factored only)

        Returns:
            Non-factored: z_hyp (batch, latent_dim) — Poincaré ball points
            Factored: (z_hyp, r) tuple where z_hyp (batch, latent_dim - radial_dims),
                      r (batch,) is the explicit Poincaré radius. Gradients from
                      ||z_hyp|| are isolated to z_r (not z_θ) by construction:
                      d(||r*dir||)/d(z_θ) = 0 because F.normalize Jacobian is
                      orthogonal to its output, and r*dir is collinear with dir.
        """
        # Enforce safe curvature range
        self._clamp_curvature()

        # Enforce float64 precision
        z_tangent = z_tangent.to(torch.float64)

        if self.factored:
            return self._forward_factored(z_tangent)

        # --- Non-factored mode (V6 expmap0 approach) ---
        z_hyp, _, _ = self._transform_and_project(z_tangent, need_tangent_norm=False)

        if as_manifold:
            z_hyp = self.manifold.projx(z_hyp)
            return ManifoldParameter(z_hyp, manifold=self.manifold)

        return z_hyp

    def _forward_factored(
        self, z_tangent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factored forward pass (V7).

        Splits z_tangent into:
          - z_r  (first radial_dims): controls Poincaré radius only
          - z_θ  (remaining dims):   controls direction only

        Hierarchy/radial losses operate on r; reconstruction operates on z_θ.
        The gradient isolation holds because d(||r*dir||)/d(z_θ) = 0:
        the norm of r*dir equals r regardless of dir's value, and F.normalize's
        Jacobian (I - dir*dirᵀ)/||z_θ|| projects the incoming gradient r*dir
        onto the tangent space of the unit sphere, which is orthogonal to dir.

        Args:
            z_tangent: (batch, latent_dim) tangent vectors

        Returns:
            (z_hyp, r) where z_hyp has shape (batch, latent_dim - radial_dims)
            and r has shape (batch,) with values in (0, max_radius).
        """
        z_r = z_tangent[:, :self.radial_dims]          # (B, k)
        z_theta = z_tangent[:, self.radial_dims:]       # (B, D-k)

        # Radius: sigmoid maps to (0,1), scaled to (0, max_radius)
        assert self.linear_r is not None
        r = torch.sigmoid(self.linear_r(z_r).squeeze(-1)) * self.max_radius  # (B,)

        # Direction: residual transform of z_θ, then normalize to unit sphere
        dir_unnorm = self._residual_tangent_transform(z_theta)
        dir_norm = torch.norm(dir_unnorm, dim=-1, keepdim=True).clamp(min=1e-10)
        dir = dir_unnorm / dir_norm                     # (B, D-k), unit norm

        # Poincaré ball point: ||z_hyp|| = r < max_radius < 1 by construction
        z_hyp = r.unsqueeze(-1) * dir                  # (B, D-k)

        return z_hyp, r

    def get_curvature(self) -> torch.Tensor:
        """Get current curvature parameter/tensor.

        Returns the actual parameter to ensure that when this is passed
        to loss functions, gradients can flow back to the learnable
        curvature during backpropagation.
        """
        return self.manifold.c

    def forward_with_components(
        self, z_tangent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project with diagnostic outputs.

        Returns:
            Non-factored: (z_hyp, z_transformed, tangent_norm) — tangent_norm is L2 norm of z_transformed
            Factored:     (z_hyp, z_hyp, r)                    — third element is Poincaré radius r, not a norm
        """
        if self.factored:
            z_hyp, r = self._forward_factored(z_tangent.to(torch.float64))
            return z_hyp, z_hyp, r  # third element is radius r, not tangent_norm
        return self._transform_and_project(z_tangent.to(torch.float64))


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
        factored: bool = False,
        radial_dims: int = 4,
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
            factored: If True, use V7 factored latent architecture
            radial_dims: Number of radial dimensions (k) in factored mode
        """
        super().__init__()
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.learnable_curvature = learnable_curvature
        self.factored = factored

        def _make_proj(
            proj_curvature: Union[float, geoopt.PoincareBall], proj_learnable_curvature: bool
        ) -> HyperbolicProjection:
            return HyperbolicProjection(
                latent_dim,
                hidden_dim,
                max_radius,
                curvature=proj_curvature,
                n_layers=n_layers,
                dropout=dropout,
                learnable_curvature=proj_learnable_curvature,
                init_identity=init_identity,
                tangent_scale_init=tangent_scale_init,
                factored=factored,
                radial_dims=radial_dims,
            )

        # VAE-A projection
        self.proj_A = _make_proj(curvature, learnable_curvature)

        # VAE-B projection (shares the exact same manifold instance from proj_A)
        # This ensures they share the same curvature parameter and manifold logic.
        # learnable_curvature is ignored when a manifold (not a raw float) is passed.
        self.proj_B = _make_proj(self.proj_A.manifold, False)

    def forward(
        self, z_A: torch.Tensor, z_B: torch.Tensor, as_manifold: bool = False
    ) -> Tuple:
        """Project both tangent vectors to Poincaré ball.

        Args:
            z_A: VAE-A tangent vector (batch, latent_dim)
            z_B: VAE-B tangent vector (batch, latent_dim)
            as_manifold: If True, return ManifoldParameters (non-factored only)

        Returns:
            Non-factored: (z_A_hyp, z_B_hyp, None, None)
            Factored: (z_A_hyp, z_B_hyp, r_A, r_B) where r_A, r_B are (batch,) radii
        """
        if self.factored:
            z_A_hyp, r_A = self.proj_A(z_A, as_manifold=as_manifold)
            z_B_hyp, r_B = self.proj_B(z_B, as_manifold=as_manifold)
            return z_A_hyp, z_B_hyp, r_A, r_B
        else:
            z_A_hyp = self.proj_A(z_A, as_manifold=as_manifold)
            z_B_hyp = self.proj_B(z_B, as_manifold=as_manifold)
            return z_A_hyp, z_B_hyp, None, None

    def get_curvature(self) -> Union[torch.Tensor, float]:
        """Get current curvature value."""
        return self.proj_A.get_curvature()


__all__ = ["HyperbolicProjection", "DualHyperbolicProjection"]
