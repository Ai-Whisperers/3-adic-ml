# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ternary VAE Models for V5.11+.

Architecture:
    - Dual VAE structure (VAE-A for coverage, VAE-B for hierarchy)
    - Hyperbolic projections to Poincaré ball
    - Encoder types: "improved" (SiLU+LayerNorm) or "standard" (ReLU, v5.5 compatible)
    - Support for StateNet partial freeze control

Checkpoint Compatibility:
    V5.5 checkpoints use nested keys (encoder_A.encoder.X, encoder_A.fc_mu).
    V5.11 uses flat keys (encoder_A.X, fc_mu_A).
    Use load_v5_5_checkpoint() for proper key mapping.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.models.hyperbolic_projection import DualHyperbolicProjection


# =============================================================================
# Key Mapping: V5.5 → V5.11
# =============================================================================

V5_5_TO_V5_11_KEY_MAP = {
    # Encoder A
    "encoder_A.encoder.": "encoder_A.",
    "encoder_A.fc_mu.": "fc_mu_A.",
    "encoder_A.fc_logvar.": "fc_logvar_A.",
    # Encoder B
    "encoder_B.encoder.": "encoder_B.",
    "encoder_B.fc_mu.": "fc_mu_B.",
    "encoder_B.fc_logvar.": "fc_logvar_B.",
    # Decoder A
    "decoder_A.decoder.": "decoder_A.",
    # Decoder B (v5.5 has different structure, skip)
}


def map_v5_5_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map v5.5 checkpoint keys to V5.11 format.

    Args:
        state_dict: V5.5 checkpoint state dict

    Returns:
        State dict with V5.11 compatible keys
    """
    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in V5_5_TO_V5_11_KEY_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        mapped[new_key] = value
    return mapped


# =============================================================================
# Decoder Mapping Layer (Option C: Learnable z_hyp → decoder bridge)
# =============================================================================


class DecoderMappingLayer(nn.Module):
    """Learnable mapping from hyperbolic latent (z_hyp) to decoder input space.

    Bridges the geometric gap between:
    - z_hyp: Bounded in Poincaré ball (||z|| < max_radius), shaped by hierarchy loss
    - Decoder expectation: Originally trained on unbounded Gaussian z_euc

    Architecture:
        Residual MLP that starts as identity and learns corrections.
        z_mapped = z_hyp + MLP(z_hyp)

    Why residual?
        1. Initializes as identity (no regression at training start)
        2. Gradients flow directly through skip connection
        3. Network learns corrections, not full transform
        4. Matches HyperbolicProjection precedent in codebase

    Why SiLU activation?
        - Smooth (better gradients than ReLU)
        - Self-gated (natural gradient scaling)
        - No dead neurons

    Args:
        latent_dim: Dimension of latent space (default: 16)
        hidden_dim: Hidden layer dimension (default: 32)

    Example:
        >>> mapping = DecoderMappingLayer(latent_dim=16)
        >>> z_hyp = torch.randn(32, 16).clamp(-0.95, 0.95)  # Bounded
        >>> z_mapped = mapping(z_hyp)  # Initially ≈ z_hyp
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP with SiLU activation
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        # Initialize as identity: fc2 outputs zero initially
        # So z_mapped = z_hyp + 0 = z_hyp at start
        self._init_as_identity()

    def _init_as_identity(self):
        """Initialize output layer to zero for identity behavior."""
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            # fc1 uses default init (small random) - that's fine

    def forward(self, z_hyp: torch.Tensor) -> torch.Tensor:
        """Map z_hyp to decoder input space.

        Args:
            z_hyp: Hyperbolic latent (B, latent_dim), bounded in Poincaré ball

        Returns:
            z_mapped: Decoder input (B, latent_dim), adapted for reconstruction
        """
        # Residual: preserve z_hyp, add learned correction
        correction = self.fc2(self.act(self.fc1(z_hyp)))
        return z_hyp + correction

    def extra_repr(self) -> str:
        return f"latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}"


# =============================================================================
# Encoder/Decoder Builders
# =============================================================================

def build_encoder(hidden_dim: int, encoder_type: str = "improved") -> nn.Sequential:
    """Build encoder network.

    Args:
        hidden_dim: Hidden dimension (64 recommended)
        encoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU, v5.5 compat)

    Returns:
        Encoder sequential module (9 → hidden_dim output)
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

    Args:
        latent_dim: Latent dimension (16 recommended)
        hidden_dim: Hidden dimension (64 recommended)
        decoder_type: "improved" (SiLU+LayerNorm) or "standard" (ReLU, v5.5 compat)

    Returns:
        Decoder sequential module (latent_dim → 27 output)
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
# TernaryVAEV5_11
# =============================================================================

class TernaryVAEV5_11(nn.Module):
    """Dual Ternary VAE with hyperbolic projections.

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
        use_decoder_mapping: Use learnable z_hyp → decoder mapping (default: False)
            When True, decoder receives mapped z_hyp instead of z_euc.
            This creates geometric coherence: both losses and decoder use hyperbolic latent.
        mapping_hidden_dim: Hidden dimension for decoder mapping layer (default: 32)
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
        use_decoder_mapping: bool = False,
        mapping_hidden_dim: int = 32,
        # Unused kwargs for compatibility
        use_controller: bool = True,
        use_dual_projection: bool = True,
        manifold_aware: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_decoder_mapping = use_decoder_mapping

        # Encoder output dim depends on type
        enc_out_dim = hidden_dim if encoder_type == "improved" else 64

        # Encoders
        self.encoder_A = build_encoder(hidden_dim, encoder_type)
        self.fc_mu_A = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar_A = nn.Linear(enc_out_dim, latent_dim)

        self.encoder_B = build_encoder(hidden_dim, encoder_type)
        self.fc_mu_B = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar_B = nn.Linear(enc_out_dim, latent_dim)

        # Hyperbolic projections
        self.projections = DualHyperbolicProjection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            curvature=curvature,
            share_direction=False,
            n_layers=n_projection_layers,
            dropout=projection_dropout,
            learnable_curvature=learnable_curvature,
        )

        # Decoders (both output 27 logits for 9 positions × 3 classes)
        self.decoder_A = build_decoder(latent_dim, hidden_dim, decoder_type)
        self.decoder_B = build_decoder(latent_dim, hidden_dim, decoder_type)

        # Decoder mapping layers (Option C: z_hyp → decoder input)
        # When enabled, decoder receives mapped z_hyp instead of z_euc
        if use_decoder_mapping:
            self.decoder_mapping_A = DecoderMappingLayer(latent_dim, mapping_hidden_dim)
            self.decoder_mapping_B = DecoderMappingLayer(latent_dim, mapping_hidden_dim)
        else:
            self.decoder_mapping_A = None
            self.decoder_mapping_B = None

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent parameters."""
        h_A = self.encoder_A(x)
        mu_A = self.fc_mu_A(h_A)
        logvar_A = self.fc_logvar_A(h_A)

        h_B = self.encoder_B(x)
        mu_B = self.fc_mu_B(h_B)
        logvar_B = self.fc_logvar_B(h_B)

        return mu_A, logvar_A, mu_B, logvar_B

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, compute_control: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through both VAEs.

        Args:
            x: Input ternary operations (B, 9) with values in {-1, 0, 1}
            compute_control: Unused, kept for API compatibility

        Returns:
            Dict with logits, latents, and hyperbolic projections
        """
        mu_A, logvar_A, mu_B, logvar_B = self.encode(x)

        # Sample latents (Euclidean)
        z_A_euc = self.reparameterize(mu_A, logvar_A)
        z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Project to Poincaré ball
        z_A_hyp, z_B_hyp = self.projections(z_A_euc, z_B_euc)

        # Decode: use mapped z_hyp if enabled, otherwise z_euc (legacy)
        if self.use_decoder_mapping and self.decoder_mapping_A is not None:
            # Option C: Map z_hyp to decoder input space
            z_A_dec = self.decoder_mapping_A(z_A_hyp)
            z_B_dec = self.decoder_mapping_B(z_B_hyp)
        else:
            # Legacy: decode from Euclidean latents (geometric inconsistency)
            z_A_dec = z_A_euc
            z_B_dec = z_B_euc

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
            "z_A_euc": z_A_euc,
            "z_B_euc": z_B_euc,
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
        }

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Return parameter groups for optimizer."""
        return [{"params": self.parameters(), "lr": base_lr}]

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device = torch.device("cpu"),
        **model_kwargs,
    ) -> "TernaryVAEV5_11":
        """Create model and load v5.5 checkpoint with key mapping.

        Args:
            checkpoint_path: Path to v5.5 .pt file
            device: Device to load model to
            **model_kwargs: Override default model parameters

        Returns:
            Model with v5.5 weights loaded
        """
        # Force standard type for v5.5 compatibility
        model_kwargs.setdefault("encoder_type", "standard")
        model_kwargs.setdefault("decoder_type", "standard")
        model_kwargs.setdefault("latent_dim", 16)
        model_kwargs.setdefault("hidden_dim", 64)

        model = cls(**model_kwargs).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Map keys
        mapped_state = map_v5_5_keys(state_dict)

        # Load with strict=False (projections won't match)
        missing, unexpected = model.load_state_dict(mapped_state, strict=False)

        return model


# =============================================================================
# TernaryVAEV5_11_PartialFreeze
# =============================================================================

class TernaryVAEV5_11_PartialFreeze(TernaryVAEV5_11):
    """V5.11 VAE with partial freezing for StateNet control.

    Supports:
        - Independent freeze/unfreeze of encoder A and B
        - Differential learning rates per component
        - StateNet state application

    Additional Args:
        encoder_a_lr_scale: LR multiplier for encoder A (default: 0.05)
        encoder_b_lr_scale: LR multiplier for encoder B (default: 0.1)
        freeze_encoder_b: Initial freeze state for encoder B (default: False)
    """

    def __init__(
        self,
        encoder_a_lr_scale: float = 0.05,
        encoder_b_lr_scale: float = 0.1,
        freeze_encoder_b: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_a_lr_scale = encoder_a_lr_scale
        self.encoder_b_lr_scale = encoder_b_lr_scale

        # Freeze state tracking
        self._encoder_a_frozen = False
        self._encoder_b_frozen = False

        if freeze_encoder_b:
            self.set_encoder_b_frozen(True)

    def set_encoder_a_frozen(self, frozen: bool):
        """Freeze/unfreeze encoder A."""
        self._encoder_a_frozen = frozen
        for p in self.encoder_A.parameters():
            p.requires_grad = not frozen
        for p in self.fc_mu_A.parameters():
            p.requires_grad = not frozen
        for p in self.fc_logvar_A.parameters():
            p.requires_grad = not frozen

    def set_encoder_b_frozen(self, frozen: bool):
        """Freeze/unfreeze encoder B."""
        self._encoder_b_frozen = frozen
        for p in self.encoder_B.parameters():
            p.requires_grad = not frozen
        for p in self.fc_mu_B.parameters():
            p.requires_grad = not frozen
        for p in self.fc_logvar_B.parameters():
            p.requires_grad = not frozen

    def apply_statenet_state(self, state: Dict[str, Any]):
        """Apply freeze states from StateNet controller."""
        if "encoder_a_frozen" in state:
            self.set_encoder_a_frozen(state["encoder_a_frozen"])
        if "encoder_b_frozen" in state:
            self.set_encoder_b_frozen(state["encoder_b_frozen"])

    def get_freeze_state_summary(self) -> str:
        """Get human-readable freeze state."""
        a = "F" if self._encoder_a_frozen else "T"
        b = "F" if self._encoder_b_frozen else "T"
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

        # Decoder mappings (full LR) - Option C bridge layers
        if self.decoder_mapping_A is not None:
            mapping_params = [p for p in self.decoder_mapping_A.parameters() if p.requires_grad]
            mapping_params += [p for p in self.decoder_mapping_B.parameters() if p.requires_grad]
            if mapping_params:
                groups.append({
                    "params": mapping_params,
                    "lr": base_lr,
                    "name": "decoder_mappings",
                })

        return groups

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: torch.device = torch.device("cpu"),
        **model_kwargs,
    ) -> "TernaryVAEV5_11_PartialFreeze":
        """Create model and load v5.5 checkpoint with key mapping."""
        model_kwargs.setdefault("encoder_type", "standard")
        model_kwargs.setdefault("decoder_type", "standard")
        model_kwargs.setdefault("latent_dim", 16)
        model_kwargs.setdefault("hidden_dim", 64)

        model = cls(**model_kwargs).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        mapped_state = map_v5_5_keys(state_dict)
        model.load_state_dict(mapped_state, strict=False)

        return model


__all__ = [
    "DecoderMappingLayer",
    "TernaryVAEV5_11",
    "TernaryVAEV5_11_PartialFreeze",
    "map_v5_5_keys",
]
