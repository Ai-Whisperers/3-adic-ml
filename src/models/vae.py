# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE Models for V5.11+.

This module implements the core VAE architectures used in the V5.11 and V5.12
pipelines. These models feature:
- Dual VAE structure (VAE-A for coverage, VAE-B for hierarchy)
- Hyperbolic projections (Poincaré ball)
- Flexible encoder/decoder architectures
- Support for homeostatic/statenet control (partial freezing)

Classes:
    TernaryVAEV5_11: Standard dual VAE.
    TernaryVAEV5_11_PartialFreeze: VAE with specific support for component freezing
                                   and differential learning rates.
"""

from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hyperbolic_projection import DualHyperbolicProjection


class TernaryVAEV5_11(nn.Module):
    """Standard V5.11 Dual Ternary VAE.

    Consists of two VAEs (A and B) sharing a latent space (or having parallel ones)
    mapped to a hyperbolic manifold.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        use_controller: bool = True,
        use_dual_projection: bool = True,
        n_projection_layers: int = 1,
        projection_dropout: float = 0.0,
        learnable_curvature: bool = False,
        manifold_aware: bool = False,
        encoder_type: str = "improved",
        decoder_type: str = "improved",
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_dual_projection = use_dual_projection
        self.manifold_aware = manifold_aware

        # --- Encoder A (Coverage Focus) ---
        self.encoder_A = self._build_encoder(9, hidden_dim, latent_dim, encoder_type)
        self.fc_mu_A = nn.Linear(hidden_dim if encoder_type == "improved" else 64, latent_dim)
        self.fc_logvar_A = nn.Linear(hidden_dim if encoder_type == "improved" else 64, latent_dim)

        # --- Encoder B (Hierarchy Focus) ---
        self.encoder_B = self._build_encoder(9, hidden_dim, latent_dim, encoder_type)
        self.fc_mu_B = nn.Linear(hidden_dim if encoder_type == "improved" else 64, latent_dim)
        self.fc_logvar_B = nn.Linear(hidden_dim if encoder_type == "improved" else 64, latent_dim)

        # --- Hyperbolic Projections ---
        self.projections = DualHyperbolicProjection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            curvature=curvature,
            share_direction=not use_dual_projection,
            n_layers=n_projection_layers,
            dropout=projection_dropout,
            learnable_curvature=learnable_curvature,
        )

        # --- Decoder A (Reconstruction) ---
        self.decoder_A = self._build_decoder(latent_dim, hidden_dim, 27, decoder_type)

        # --- Decoder B (Optional/Auxiliary) ---
        # Usually structurally identical to A
        self.decoder_B = self._build_decoder(latent_dim, hidden_dim, 27, decoder_type)

    def _build_encoder(self, input_dim: int, hidden_dim: int, output_dim: int, type: str) -> nn.Module:
        if type == "improved":
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
        else:  # "standard" / legacy
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )

    def _build_decoder(self, input_dim: int, hidden_dim: int, output_dim: int, type: str) -> nn.Module:
        if type == "improved":
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, output_dim),
            )
        else:  # "standard" / legacy
            return nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

    def encode(self, x: torch.Tensor):
        # Encoder A
        h_A = self.encoder_A(x)
        mu_A = self.fc_mu_A(h_A)
        logvar_A = self.fc_logvar_A(h_A)

        # Encoder B
        h_B = self.encoder_B(x)
        mu_B = self.fc_mu_B(h_B)
        logvar_B = self.fc_logvar_B(h_B)

        return mu_A, logvar_A, mu_B, logvar_B

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, compute_control: bool = False) -> Dict[str, torch.Tensor]:
        mu_A, logvar_A, mu_B, logvar_B = self.encode(x)

        # Euclidean Latents
        z_A_euc = self.reparameterize(mu_A, logvar_A)
        z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Hyperbolic Projections
        # Pass as_manifold=self.manifold_aware if supported by trainer
        z_A_hyp, z_B_hyp = self.projections(z_A_euc, z_B_euc, as_manifold=self.manifold_aware)

        # Decode (usually from Euclidean z_A for reconstruction)
        # Some variants might decode from hyp, but standard v5.11 decodes from z_A_euc or mu_A
        logits_A = self.decoder_A(z_A_euc)
        logits_B = self.decoder_B(z_B_euc)

        return {
            "logits": logits_A, # Primary logits
            "logits_A": logits_A,
            "logits_B": logits_B,
            "mu_A": mu_A,
            "logvar_A": logvar_A,
            "mu_B": mu_B,
            "logvar_B": logvar_B,
            "z_A_euc": z_A_euc,
            "z_B_euc": z_B_euc,
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
        }

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer (standard uniform LR)."""
        return [{"params": self.parameters(), "lr": base_lr}]


class TernaryVAEV5_11_PartialFreeze(TernaryVAEV5_11):
    """V5.11 VAE with Partial Freezing & Differential Learning Rates.

    Supports the 'StateNet' controller logic where specific components
    (encoders) can be frozen while others (projections, decoders) continue
    to adapt.
    """

    def __init__(
        self,
        freeze_encoder_b: bool = False,
        encoder_b_lr_scale: float = 0.1,
        encoder_a_lr_scale: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_b_lr_scale = encoder_b_lr_scale
        self.encoder_a_lr_scale = encoder_a_lr_scale

        # Initial freeze state
        if freeze_encoder_b:
            self.set_encoder_b_frozen(True)

        self._encoder_a_frozen = False
        self._encoder_b_frozen = freeze_encoder_b

    def set_encoder_a_frozen(self, frozen: bool):
        self._encoder_a_frozen = frozen
        for param in self.encoder_A.parameters():
            param.requires_grad = not frozen
        for param in self.fc_mu_A.parameters():
            param.requires_grad = not frozen
        for param in self.fc_logvar_A.parameters():
            param.requires_grad = not frozen

    def set_encoder_b_frozen(self, frozen: bool):
        self._encoder_b_frozen = frozen
        for param in self.encoder_B.parameters():
            param.requires_grad = not frozen
        for param in self.fc_mu_B.parameters():
            param.requires_grad = not frozen
        for param in self.fc_logvar_B.parameters():
            param.requires_grad = not frozen

    def apply_statenet_state(self, state: Dict[str, Any]):
        """Apply freeze states from StateNet controller."""
        if "encoder_a_frozen" in state:
            self.set_encoder_a_frozen(state["encoder_a_frozen"])
        if "encoder_b_frozen" in state:
            self.set_encoder_b_frozen(state["encoder_b_frozen"])
        # Controller freezing logic (if applicable to model params) would go here

    def get_freeze_state_summary(self) -> str:
        return f"A:{'F' if self._encoder_a_frozen else 'T'} B:{'F' if self._encoder_b_frozen else 'T'}"

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups with differential learning rates."""
        groups = []

        # Encoder A (Scaled LR)
        enc_a_params = list(self.encoder_A.parameters()) + \
                       list(self.fc_mu_A.parameters()) + \
                       list(self.fc_logvar_A.parameters())
        if enc_a_params:
             groups.append({
                "params": filter(lambda p: p.requires_grad, enc_a_params),
                "lr": base_lr * self.encoder_a_lr_scale,
                "name": "encoder_A"
            })

        # Encoder B (Scaled LR)
        enc_b_params = list(self.encoder_B.parameters()) + \
                       list(self.fc_mu_B.parameters()) + \
                       list(self.fc_logvar_B.parameters())
        if enc_b_params:
            groups.append({
                "params": filter(lambda p: p.requires_grad, enc_b_params),
                "lr": base_lr * self.encoder_b_lr_scale,
                "name": "encoder_B"
            })

        # Projections (Full LR - Fast adaptation)
        proj_params = list(self.projections.parameters())
        if proj_params:
            groups.append({
                "params": filter(lambda p: p.requires_grad, proj_params),
                "lr": base_lr,
                "name": "projections"
            })

        # Decoders (Full LR)
        dec_params = list(self.decoder_A.parameters()) + list(self.decoder_B.parameters())
        if dec_params:
             groups.append({
                "params": filter(lambda p: p.requires_grad, dec_params),
                "lr": base_lr,
                "name": "decoders"
            })

        return groups
