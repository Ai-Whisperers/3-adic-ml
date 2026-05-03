# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Ternary VAE with True Hyperbolic Geometry.

Architecture (V6.0 - Dual VAE):
    Two parallel VAEs share input but learn complementary objectives:
    - VAE-A: Coverage (reconstruction accuracy)
    - VAE-B: Hierarchy (radial structure in Poincaré ball)

Data Flow:
    Input x (B, 9) ternary {-1, 0, 1}
        │
        ├─► encoder_A ─► fc_mu_A, fc_logvar_A ─► z_A_tangent
        │                                            │
        └─► encoder_B ─► fc_mu_B, fc_logvar_B ─► z_B_tangent
                                                     │
                              ┌──────────────────────┴──────────────────────┐
                              ▼                                             ▼
                    projections.proj_A                            projections.proj_B
                    (tangent_net + expmap0)                       (tangent_net + expmap0)
                              │                                             │
                              ▼                                             ▼
                         z_A_hyp ◄── Losses computed here ──► z_B_hyp
                         (Poincaré)   (true hyperbolic)       (Poincaré)
                              │                                             │
                          logmap0                                       logmap0
                              │                                             │
                              ▼                                             ▼
                         decoder_A ─► logits_A (27)              decoder_B ─► logits_B (27)

Key Insight:
    Tangent space at origin T₀M IS Euclidean, so standard MLPs work there.
    The expmap0/logmap0 operations provide non-Euclidean structure.

StateNet Integration (TernaryVAEV6Controllable):
    - encoder_a_trainable: Coverage-gated (fix when coverage drops) → head_A
    - encoder_b_trainable: Hierarchy-gated (fix when hierarchy plateaus) → head_B
    - controller_trainable: Gradient-gated (fix when stable) → projections

Reference:
    Mathieu et al. (2019) "Continuous Hierarchical Representations with Poincaré VAEs"
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.geometry import log_map_zero
from src.models.hyperbolic_projection import DualHyperbolicProjection

# =============================================================================
# EncoderHead: Modular encoder with trainability control
# =============================================================================


class EncoderHead(nn.Module):
    """Encoder backbone + mu/logvar projection heads with trainability control.

    Bundles the encoder MLP with its distribution parameter heads (mu, logvar).
    Provides unified trainability control for StateNet integration.

    Architecture:
        x (B, input_dim) → backbone → h (B, hidden_dim)
                                   ├─► fc_mu → mu (B, latent_dim)
                                   └─► fc_logvar → logvar (B, latent_dim)

    Args:
        hidden_dim: Hidden dimension for backbone (64 recommended)
        latent_dim: Output latent dimension (16 recommended)
        encoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU)
        input_dim: Input feature dimension (9 default, 18 with positional encoding)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        encoder_type: str = "improved",
        input_dim: int = 9,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.input_dim = input_dim

        # Backbone output dim depends on type
        enc_out_dim = hidden_dim if encoder_type == "improved" else 64

        # Build components
        self.backbone = _build_encoder_backbone(hidden_dim, encoder_type, input_dim)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Trainability state
        self._trainable = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Tuple of (mu, logvar), each (B, latent_dim)
        """
        h = self.backbone(x)
        mu = self.fc_mu(h)
        # Clamp logvar to prevent σ explosion: exp(0.5*logvar) stays in [exp(-5), exp(1)]
        # Unclamped logvar→+∞ gives σ→∞, making z_tangent = μ + σε arbitrarily large.
        logvar = self.fc_logvar(h).clamp(-10.0, 2.0)
        return mu, logvar

    def set_trainable(self, trainable: bool) -> None:
        """Set trainability for all parameters in this head.

        Args:
            trainable: If True, parameters receive gradients. If False, frozen.
        """
        self._trainable = trainable
        for p in self.parameters():
            p.requires_grad = trainable

    @property
    def is_trainable(self) -> bool:
        """Whether this encoder head is currently trainable."""
        return self._trainable

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get list of parameters that currently require gradients.

        Returns:
            List of parameters with requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]


# =============================================================================
# Encoder/Decoder Builders
# =============================================================================


def _build_encoder_backbone(
    hidden_dim: int, encoder_type: str = "improved", input_dim: int = 9
) -> nn.Sequential:
    """Build encoder backbone network (internal helper).

    Maps input to hidden representation. Does NOT include the mu/logvar heads.

    Architecture (improved):
        input_dim → hidden_dim*2 → LayerNorm → SiLU
                  → hidden_dim*2 → LayerNorm → SiLU
                  → hidden_dim → SiLU

    Architecture (standard):
        input_dim → 256 → ReLU → 128 → ReLU → 64 → ReLU

    Args:
        hidden_dim: Hidden dimension (64 recommended for improved type)
        encoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU)
        input_dim: Input feature dimension (9 default, 18 with positional encoding)

    Returns:
        Sequential module outputting (B, hidden_dim) or (B, 64) for standard
    """
    if encoder_type == "improved":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
    else:  # "standard" - matches v5.5 architecture
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )


def _build_decoder(
    latent_dim: int, hidden_dim: int, decoder_type: str = "improved"
) -> nn.Sequential:
    """Build decoder network.

    Maps latent vector (from tangent space via logmap0) to reconstruction
    logits. Output is 27 = 9 positions × 3 classes for ternary classification.

    Architecture (improved):
        latent_dim → hidden_dim → SiLU
                   → hidden_dim*2 → LayerNorm → SiLU
                   → 27 (raw logits)

    Architecture (standard):
        latent_dim → 32 → ReLU → 64 → ReLU → 27

    Args:
        latent_dim: Latent dimension (16 recommended)
        hidden_dim: Hidden dimension (64 recommended)
        decoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU)

    Returns:
        Sequential module outputting (B, 27) logits
    """
    if decoder_type == "improved":
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 27),
        )
    else:  # "standard" - matches v5.5 architecture
        return nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 27),
        )


# =============================================================================
# TernaryVAEV6
# =============================================================================


class TernaryVAEV6(nn.Module):
    """Dual Ternary VAE with true hyperbolic geometry.

    Architecture:
        Encoder → μ, logvar (tangent space at origin)
        z_tangent = μ + ε * σ (sample in tangent space)
        z_hyp = expmap0(transform(z_tangent)) (project to manifold)
        logmap0(z_hyp) → Decoder (back to tangent space)

    Two parallel VAEs:
        - VAE-A: Optimized for coverage (reconstruction accuracy)
        - VAE-B: Optimized for hierarchy (radial structure in Poincaré ball)

    Args:
        latent_dim: Latent space dimension (default: 16)
        hidden_dim: Hidden layer dimension (default: 64)
        max_radius: Maximum radius in Poincaré ball (default: 0.95)
        curvature: Hyperbolic curvature (default: 1.0)
        encoder_type: "improved" or "standard" (default: "improved")
        decoder_type: "improved" or "standard" (default: "improved")
        n_projection_layers: Projection network depth (default: 1)
        projection_dropout: Dropout in projection networks (default: 0.0)
        learnable_curvature: Allow curvature to be learned (default: False)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        encoder_type: str = "improved",
        decoder_type: str = "improved",
        n_projection_layers: int = 1,
        projection_dropout: float = 0.0,
        learnable_curvature: bool = False,
        init_identity: bool = True,
        tangent_scale_init: float = 0.1,
        factored: bool = False,
        radial_dims: int = 4,
        positional_encoding: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.factored = factored
        self.positional_encoding = positional_encoding

        # Positional significance weights: pos_weights[k] = 1/3^k.
        # Position 0 is most predictive of v_3(n) (determines v_3=0 vs >0 for
        # 66% of the dataset), so it receives weight 1.0.
        if positional_encoding:
            self.register_buffer(
                "pos_weights",
                torch.tensor([1.0 / (3 ** k) for k in range(9)], dtype=torch.float64),
                persistent=False,
            )

        # Encoder heads (backbone + mu/logvar projections)
        input_dim = 18 if positional_encoding else 9
        self.head_A = EncoderHead(hidden_dim, latent_dim, encoder_type, input_dim)
        self.head_B = EncoderHead(hidden_dim, latent_dim, encoder_type, input_dim)

        # Hyperbolic projections (tangent → manifold via expmap0, or factored r*dir)
        self.projections = DualHyperbolicProjection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            curvature=curvature,
            n_layers=n_projection_layers,
            dropout=projection_dropout,
            learnable_curvature=learnable_curvature,
            init_identity=init_identity,
            tangent_scale_init=tangent_scale_init,
            factored=factored,
            radial_dims=radial_dims,
        )

        # Decoders (input from tangent space via logmap0)
        self.decoder_A = _build_decoder(latent_dim, hidden_dim, decoder_type)
        self.decoder_B = _build_decoder(latent_dim, hidden_dim, decoder_type)

        # Enforce float64 precision for numerical stability
        self.to(torch.float64)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters for both VAEs.

        Each encoder head produces mu and logvar in tangent space at origin.
        These define the approximate posterior q(z|x) = N(mu, exp(logvar)).

        Args:
            x: Input tensor (B, 9) with ternary values

        Returns:
            Tuple of (mu_A, logvar_A, mu_B, logvar_B), each (B, latent_dim)
        """
        mu_A, logvar_A = self.head_A(x)
        mu_B, logvar_B = self.head_B(x)
        return mu_A, logvar_A, mu_B, logvar_B

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling.

        Samples z = mu + eps * std where eps ~ N(0, I). This allows gradients
        to flow through the sampling operation.

        Note: Sampling occurs in tangent space T₀M at origin, which IS Euclidean.
        The non-Euclidean structure comes from expmap0 projection afterward.

        Args:
            mu: Mean of approximate posterior (B, latent_dim)
            logvar: Log variance of approximate posterior (B, latent_dim)

        Returns:
            Sampled latent z_tangent (B, latent_dim) in tangent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, decode_b: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through both VAEs.

        Args:
            x: Input ternary operations (B, 9) with values in {-1, 0, 1}
            decode_b: If False, skip decoder_B (saves ~15% compute when
                      VAE-B loss has coverage_weight=0.0 and logits_B unused)

        Returns:
            Dict with logits, latents, and hyperbolic projections
        """
        # Enforce float64 precision
        x = x.to(torch.float64)

        # Positional significance encoding: concatenate position-scaled features.
        # x_aug = [x, x * pos_weights] where pos_weights[k] = 1/3^k.
        # Gives the encoder explicit signal about which digit positions matter
        # most for 3-adic valuation without changing any other component.
        if self.positional_encoding:
            x = torch.cat([x, x * self.pos_weights], dim=-1)

        mu_A, logvar_A, mu_B, logvar_B = self.encode(x)

        # Sample in tangent space (Euclidean at origin)
        z_A_tangent = self.reparameterize(mu_A, logvar_A)
        z_B_tangent = self.reparameterize(mu_B, logvar_B)

        z_A_hyp, z_B_hyp, r_A, r_B = self.projections(
            z_A_tangent, z_B_tangent, as_manifold=False
        )

        # Decoder receives z_tangent directly (not logmap0(z_hyp)).
        # logmap0(expmap0(v)) = v, so there is no information difference, but
        # feeding logmap0(z_hyp) coupled the decoder to tangent_scale, causing
        # reconstruction loss to collapse tangent_scale toward 0 and prevent
        # points from reaching target Poincaré radii.
        logits_A = self.decoder_A(z_A_tangent)
        if decode_b:
            logits_B = self.decoder_B(z_B_tangent)
        else:
            logits_B = None  # decoder_B skipped; coverage must be disabled in caller

        return {
            "logits": logits_A,
            "logits_A": logits_A,
            "logits_B": logits_B,
            "mu_A": mu_A,
            "logvar_A": logvar_A,
            "mu_B": mu_B,
            "logvar_B": logvar_B,
            "z_A_tangent": z_A_tangent,
            "z_B_tangent": z_B_tangent,
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
            "r_A": r_A,  # explicit Poincaré radius (factored mode only, else None)
            "r_B": r_B,
        }

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer."""
        return [{"params": self.parameters(), "lr": base_lr}]

    def get_mu_representations(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Get raw mu representations for given indices.
        
        Used by AlgebraicAdditionLoss to compute representations of sums
        within the forward pass while preserving gradients.
        """
        # Convert indices to ternary
        from src.core import TERNARY
        x = TERNARY.to_ternary(indices).to(device).to(torch.float64)
        
        # Apply positional encoding if enabled
        if self.positional_encoding:
            x = torch.cat([x, x * self.pos_weights], dim=-1)
            
        # Get mu from head_A (primary coverage pathway)
        mu_A, _ = self.head_A(x)
        return mu_A


# =============================================================================
# TernaryVAEV6Controllable
# =============================================================================


class TernaryVAEV6Controllable(TernaryVAEV6):
    """VAE with dynamic trainability control for StateNet integration.

    Implements Complementary Learning Systems theory:
        - Slow pathway (encoders): Consolidate, fix when objectives met
        - Fast pathway (projections/decoders): Continuously adapt

    StateNet controls three component groups:
        - encoder_a_trainable: Coverage-gated (fix when coverage drops)
        - encoder_b_trainable: Hierarchy-gated (fix when hierarchy plateaus)
        - controller_trainable: Gradient-gated (projections - fix when stable)

    Additional Args:
        encoder_a_lr_scale: LR multiplier for encoder A (default: 0.05)
        encoder_b_lr_scale: LR multiplier for encoder B (default: 0.1)
        projections_lr_scale: LR multiplier for projections (default: 1.0)
        encoder_a_trainable: Initial trainability for encoder A (default: False)
        encoder_b_trainable: Initial trainability for encoder B (default: True)
        projections_trainable: Initial trainability for projections (default: True)
    """

    def __init__(
        self,
        encoder_a_lr_scale: float = 0.05,
        encoder_b_lr_scale: float = 0.1,
        projections_lr_scale: float = 1.0,
        encoder_a_trainable: bool = False,
        encoder_b_trainable: bool = True,
        projections_trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_a_lr_scale = encoder_a_lr_scale
        self.encoder_b_lr_scale = encoder_b_lr_scale
        self.projections_lr_scale = projections_lr_scale

        # Projections trainability state (encoders use head_A/head_B._trainable)
        self._projections_trainable = True

        # Apply initial trainability states
        self.set_encoder_a_trainable(encoder_a_trainable)
        self.set_encoder_b_trainable(encoder_b_trainable)
        self.set_projections_trainable(projections_trainable)

    def set_encoder_a_trainable(self, trainable: bool) -> None:
        """Set encoder A (head_A) trainability."""
        self.head_A.set_trainable(trainable)

    def set_encoder_b_trainable(self, trainable: bool) -> None:
        """Set encoder B (head_B) trainability."""
        self.head_B.set_trainable(trainable)

    def set_projections_trainable(self, trainable: bool) -> None:
        """Set projections (controller) trainability.

        This is the 'controller' component that StateNet's controller_trainable
        state controls. The projections transform tangent vectors before expmap0.
        """
        self._projections_trainable = trainable
        for p in self.projections.parameters():
            p.requires_grad = trainable

    def apply_statenet_state(self, state: Dict[str, Any]) -> None:
        """Apply trainability states from StateNet controller.

        Args:
            state: Dict with trainability flags from StateNet.update()
                - encoder_a_trainable: Controls head_A
                - encoder_b_trainable: Controls head_B
                - controller_trainable: Controls projections
        """
        if "encoder_a_trainable" in state:
            self.set_encoder_a_trainable(state["encoder_a_trainable"])
        if "encoder_b_trainable" in state:
            self.set_encoder_b_trainable(state["encoder_b_trainable"])
        if "controller_trainable" in state:
            self.set_projections_trainable(state["controller_trainable"])

    def get_trainability_summary(self) -> str:
        """Get human-readable trainability state."""
        a = "train" if self.head_A.is_trainable else "fixed"
        b = "train" if self.head_B.is_trainable else "fixed"
        p = "train" if self._projections_trainable else "fixed"
        return f"A:{a} B:{b} P:{p}"

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups with differential learning rates.

        Only includes parameters that currently require gradients.
        Groups are named for logging/debugging.

        Args:
            base_lr: Base learning rate to scale from

        Returns:
            List of param group dicts for optimizer
        """
        groups = []

        # Encoder A (slow learner)
        enc_a_params = self.head_A.get_trainable_params()
        if enc_a_params:
            groups.append(
                {
                    "params": enc_a_params,
                    "lr": base_lr * self.encoder_a_lr_scale,
                    "name": "encoder_a",
                }
            )

        # Encoder B (medium learner)
        enc_b_params = self.head_B.get_trainable_params()
        if enc_b_params:
            groups.append(
                {
                    "params": enc_b_params,
                    "lr": base_lr * self.encoder_b_lr_scale,
                    "name": "encoder_b",
                }
            )

        # Projections / Controller (fast adapter)
        proj_params = [p for p in self.projections.parameters() if p.requires_grad]
        if proj_params:
            groups.append(
                {
                    "params": proj_params,
                    "lr": base_lr * self.projections_lr_scale,
                    "name": "projections",
                }
            )

        # Decoders (always trainable, full LR)
        dec_params = [p for p in self.decoder_A.parameters() if p.requires_grad]
        dec_params += [p for p in self.decoder_B.parameters() if p.requires_grad]
        if dec_params:
            groups.append(
                {
                    "params": dec_params,
                    "lr": base_lr,
                    "name": "decoders",
                }
            )

        return groups


__all__ = [
    "EncoderHead",
    "TernaryVAEV6",
    "TernaryVAEV6Controllable",
]
