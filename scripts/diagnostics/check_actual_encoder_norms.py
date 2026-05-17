#!/usr/bin/env python3
"""Check the actual encoder norms of VAE-A vs VAE-B."""

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
batch_indices = torch.randint(0, 19683, (16,))  # Batch of 16

batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (16, 9)

# Forward pass
out = model(batch_ternary)

# VAE-A latents
z_A = out["z_A_hyp"]
z_A_norm = torch.norm(z_A, dim=-1)

# VAE-B latents
z_B = out["z_B_hyp"]
z_B_norm = torch.norm(z_B, dim=-1)

print(f"VAE-A norm: {z_A_norm.mean().item():.6f} ± {z_A_norm.std().item():.6f}")
print(f"VAE-B norm: {z_B_norm.mean().item():.6f} ± {z_B_norm.std().item():.6f}")
