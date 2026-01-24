# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
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
    - encoder_a_trainable: Coverage-gated (fix when coverage drops)
    - encoder_b_trainable: Hierarchy-gated (fix when hierarchy plateaus)
    - controller_trainable: Intended for projections (NOT YET WIRED)

Reference:
    Mathieu et al. (2019) "Continuous Hierarchical Representations with Poincaré VAEs"
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.geometry import log_map_zero
from src.models.hyperbolic_projection import DualHyperbolicProjection


# =============================================================================
# Encoder/Decoder Builders
# =============================================================================

def build_encoder(hidden_dim: int, encoder_type: str = "improved") -> nn.Sequential:
    """Build encoder backbone network.

    Maps 9-dim ternary input to hidden representation. Does NOT include
    the mu/logvar heads - those are separate nn.Linear layers.

    Architecture (improved):
        9 → hidden_dim*2 → LayerNorm → SiLU
          → hidden_dim*2 → LayerNorm → SiLU
          → hidden_dim → SiLU

    Architecture (standard):
        9 → 256 → ReLU → 128 → ReLU → 64 → ReLU

    Args:
        hidden_dim: Hidden dimension (64 recommended for improved type)
        encoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU)

    Returns:
        Sequential module outputting (B, hidden_dim) or (B, 64) for standard
    """
    if encoder_type == "improved":
        return nn.Sequential(
            nn.Linear(9, hidden_dim * 2),
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
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )


def build_decoder(latent_dim: int, hidden_dim: int, decoder_type: str = "improved") -> nn.Sequential:
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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        # Encoder output dim depends on type
        enc_out_dim = hidden_dim if encoder_type == "improved" else 64

        # Encoders (output to tangent space at origin)
        self.encoder_A = build_encoder(hidden_dim, encoder_type)
        self.fc_mu_A = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar_A = nn.Linear(enc_out_dim, latent_dim)

        self.encoder_B = build_encoder(hidden_dim, encoder_type)
        self.fc_mu_B = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar_B = nn.Linear(enc_out_dim, latent_dim)

        # Hyperbolic projections (tangent → manifold via expmap0)
        self.projections = DualHyperbolicProjection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            curvature=curvature,
            n_layers=n_projection_layers,
            dropout=projection_dropout,
            learnable_curvature=learnable_curvature,
        )

        # Decoders (input from tangent space via logmap0)
        self.decoder_A = build_decoder(latent_dim, hidden_dim, decoder_type)
        self.decoder_B = build_decoder(latent_dim, hidden_dim, decoder_type)

        # Enforce float64 precision for numerical stability
        self.to(torch.float64)

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent distribution parameters for both VAEs.

        Each encoder produces mu and logvar in tangent space at origin.
        These define the approximate posterior q(z|x) = N(mu, exp(logvar)).

        Args:
            x: Input tensor (B, 9) with ternary values

        Returns:
            Tuple of (mu_A, logvar_A, mu_B, logvar_B), each (B, latent_dim)
        """
        h_A = self.encoder_A(x)
        mu_A = self.fc_mu_A(h_A)
        logvar_A = self.fc_logvar_A(h_A)

        h_B = self.encoder_B(x)
        mu_B = self.fc_mu_B(h_B)
        logvar_B = self.fc_logvar_B(h_B)

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through both VAEs.

        Args:
            x: Input ternary operations (B, 9) with values in {-1, 0, 1}

        Returns:
            Dict with logits, latents, and hyperbolic projections
        """
        # Enforce float64 precision
        x = x.to(torch.float64)

        mu_A, logvar_A, mu_B, logvar_B = self.encode(x)

        # Sample in tangent space (Euclidean at origin)
        z_A_tangent = self.reparameterize(mu_A, logvar_A)
        z_B_tangent = self.reparameterize(mu_B, logvar_B)

        # Project to Poincaré manifold via expmap0
        # as_manifold=True returns ManifoldParameter for type safety and constraint enforcement
        z_A_hyp, z_B_hyp = self.projections(z_A_tangent, z_B_tangent, as_manifold=True)

        # Map back to tangent space for decoder (logmap0)
        z_A_dec = log_map_zero(z_A_hyp, c=self.curvature, max_norm=self.max_radius)
        z_B_dec = log_map_zero(z_B_hyp, c=self.curvature, max_norm=self.max_radius)

        logits_A = self.decoder_A(z_A_dec)
        logits_B = self.decoder_B(z_B_dec)

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
        }

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer."""
        return [{"params": self.parameters(), "lr": base_lr}]


# =============================================================================
# TernaryVAEV6Controllable
# =============================================================================

class TernaryVAEV6Controllable(TernaryVAEV6):
    """V5.11 VAE with dynamic trainability control for StateNet.

    Supports:
        - Independent trainability control of encoder A and B
        - Differential learning rates per component
        - StateNet state application

    Additional Args:
        encoder_a_lr_scale: LR multiplier for encoder A (default: 0.05)
        encoder_b_lr_scale: LR multiplier for encoder B (default: 0.1)
        encoder_b_trainable: Initial trainability for encoder B (default: True)
    """

    def __init__(
        self,
        encoder_a_lr_scale: float = 0.05,
        encoder_b_lr_scale: float = 0.1,
        encoder_b_trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_a_lr_scale = encoder_a_lr_scale
        self.encoder_b_lr_scale = encoder_b_lr_scale

        # Trainability state tracking
        self._encoder_a_trainable = True
        self._encoder_b_trainable = True

        if not encoder_b_trainable:
            self.set_encoder_b_trainable(False)

    def set_encoder_a_trainable(self, trainable: bool):
        """Set encoder A trainability (True = parameters update)."""
        self._encoder_a_trainable = trainable
        for p in self.encoder_A.parameters():
            p.requires_grad = trainable
        for p in self.fc_mu_A.parameters():
            p.requires_grad = trainable
        for p in self.fc_logvar_A.parameters():
            p.requires_grad = trainable

    def set_encoder_b_trainable(self, trainable: bool):
        """Set encoder B trainability (True = parameters update)."""
        self._encoder_b_trainable = trainable
        for p in self.encoder_B.parameters():
            p.requires_grad = trainable
        for p in self.fc_mu_B.parameters():
            p.requires_grad = trainable
        for p in self.fc_logvar_B.parameters():
            p.requires_grad = trainable

    def apply_statenet_state(self, state: Dict[str, Any]):
        """Apply trainability states from StateNet controller."""
        if "encoder_a_trainable" in state:
            self.set_encoder_a_trainable(state["encoder_a_trainable"])
        if "encoder_b_trainable" in state:
            self.set_encoder_b_trainable(state["encoder_b_trainable"])

    def get_trainability_summary(self) -> str:
        """Get human-readable trainability state."""
        a = "train" if self._encoder_a_trainable else "fixed"
        b = "train" if self._encoder_b_trainable else "fixed"
        return f"A:{a} B:{b}"

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups with differential learning rates."""
        groups = []

        # Encoder A (scaled LR)
        enc_a_params = [p for p in self.encoder_A.parameters() if p.requires_grad]
        enc_a_params += [p for p in self.fc_mu_A.parameters() if p.requires_grad]
        enc_a_params += [p for p in self.fc_logvar_A.parameters() if p.requires_grad]
        if enc_a_params:
            groups.append({
                "params": enc_a_params,
                "lr": base_lr * self.encoder_a_lr_scale,
                "name": "encoder_A",
            })

        # Encoder B (scaled LR)
        enc_b_params = [p for p in self.encoder_B.parameters() if p.requires_grad]
        enc_b_params += [p for p in self.fc_mu_B.parameters() if p.requires_grad]
        enc_b_params += [p for p in self.fc_logvar_B.parameters() if p.requires_grad]
        if enc_b_params:
            groups.append({
                "params": enc_b_params,
                "lr": base_lr * self.encoder_b_lr_scale,
                "name": "encoder_B",
            })

        # Projections (full LR)
        proj_params = [p for p in self.projections.parameters() if p.requires_grad]
        if proj_params:
            groups.append({
                "params": proj_params,
                "lr": base_lr,
                "name": "projections",
            })

        # Decoders (full LR)
        dec_params = [p for p in self.decoder_A.parameters() if p.requires_grad]
        dec_params += [p for p in self.decoder_B.parameters() if p.requires_grad]
        if dec_params:
            groups.append({
                "params": dec_params,
                "lr": base_lr,
                "name": "decoders",
            })

        return groups


__all__ = [
    "TernaryVAEV6",
    "TernaryVAEV6Controllable",
]
