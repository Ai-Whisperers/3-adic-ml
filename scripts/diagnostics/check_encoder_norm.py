#!/usr/bin/env python3
"""Check the actual norm of encoder outputs"""

import torch
from src.models.vae import TernaryVAEV6Controllable
from src.config import StateNetConfig

# Create a minimal config
config = {
    "model": {
        "name": "TernaryVAEV6Controllable",
        "latent_dim": 16,
        "hidden_dim": 64,
    },
    "statenet": {
        "enabled": True,
        "initial": {
            "encoder_a_trainable": True,
            "encoder_b_trainable": True,
            "projections_trainable": True,
        },
    },
    "option_c": {
        "enabled": True,
        "encoder_a_lr_scale": 0.05,
        "encoder_b_lr_scale": 0.1,
        "projections_lr_scale": 1.0,
    },
}

# Create model
sn_config = StateNetConfig.from_dict(config.get("statenet", {}))
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    projections_trainable=sn_config.initial.projections_trainable,
)

# Create a sample input (ternary operation indices)
# All 19,683 ternary operations with values {-1, 0, 1}
# Let's just take a small batch
batch_indices = torch.randint(0, 19683, (8,))  # Batch of 8 random indices

# Convert to ternary representation
from src.core import TERNARY

batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (8, 9)

print(f"Input ternary shape: {batch_ternary.shape}")
print(f"Input ternary sample: {batch_ternary[0]}")

# Run through encoder A
with torch.no_grad():
    mu_A, log_A = model.head_A(batch_ternary)
    z_tangent_A = mu_A + torch.exp(0.5 * log_A) * torch.randn_like(mu_A)

    mu_B, log_B = model.head_B(batch_ternary)
    z_tangent_B = mu_B + torch.exp(0.5 * log_B) * torch.randn_like(mu_B)

print(f"Encoder A z_tangent norm: {torch.norm(z_tangent_A, dim=-1).mean().item():.4f}")
print(f"Encoder B z_tangent norm: {torch.norm(z_tangent_B, dim=-1).mean().item():.4f}")

# Also check the deterministic output (just mu)
print(f"Encoder A mu norm: {torch.norm(mu_A, dim=-1).mean().item():.4f}")
print(f"Encoder B mu norm: {torch.norm(mu_B, dim=-1).mean().item():.4f}")
