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

# Latents
z_hyp = out["z_hyp"]
z_norm = torch.norm(z_hyp, dim=-1)

print(f"Latent norm: {z_norm.mean().item():.6f} ± {z_norm.std().item():.6f}")
