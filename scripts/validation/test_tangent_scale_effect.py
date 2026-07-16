#!/usr/bin/env python3
"""Check the effect of tangent_scale on the initial radius."""

import math
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
batch_indices = torch.randint(0, 19683, (16,))

batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (16, 9)

# Forward pass
out = model(batch_ternary)

# Latents (VAE-A; VAE-B follows the same shape)
z_norm = torch.norm(out["z_A_hyp"], dim=-1)

print(f"Latent norm: {z_norm.mean().item():.6f} ± {z_norm.std().item():.6f}")

# Change effective tangent_scale to 10.0 (stored as log(10) in log_tangent_scale)
model.projections.proj_A.log_tangent_scale.data.fill_(math.log(10.0))
model.projections.proj_B.log_tangent_scale.data.fill_(math.log(10.0))

# Forward pass again
out2 = model(batch_ternary)
z_norm2 = torch.norm(out2["z_A_hyp"], dim=-1)

print(f"Latent norm (scale=10): {z_norm2.mean().item():.6f} ± {z_norm2.std().item():.6f}")
