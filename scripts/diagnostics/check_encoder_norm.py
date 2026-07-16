#!/usr/bin/env python3
"""Check the norm of the encoder output."""

import torch

from src.config import StateNetConfig
from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable

# Create a sample model
config = StateNetConfig()
model = TernaryVAEV6Controllable(
    latent_dim=16,
    hidden_dim=128,
    max_radius=0.99,
    curvature=1.0,
    factored=True,
    radial_dims=4
)

# Create a batch of ternary operation indices
batch_indices = torch.randint(0, 19683, (8,))  # Batch of 8

# Convert to ternary representation
batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (8, 9)

# Forward pass
out = model(batch_ternary)

# Pre-projection tangent-space norms (see check_actual_encoder_norms.py for
# the post-projection Poincaré ball norms).
z_A_norm = torch.norm(out["z_A_tangent"], dim=-1)
z_B_norm = torch.norm(out["z_B_tangent"], dim=-1)

print(f"Encoder A z_tangent norm: {z_A_norm.mean().item():.4f}")
print(f"Encoder B z_tangent norm: {z_B_norm.mean().item():.4f}")
