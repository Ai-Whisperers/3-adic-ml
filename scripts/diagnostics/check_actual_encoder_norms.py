#!/usr/bin/env python3
"""Check what the actual encoder outputs look like when passed to HyperbolicProjection"""

import torch

from src.config import StateNetConfig
from src.models.hyperbolic_projection import HyperbolicProjection
from src.models.vae import TernaryVAEV6Controllable

# Create model
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
sn_config = StateNetConfig.from_dict(config.get("statenet", {}))
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    projections_trainable=sn_config.initial.projections_trainable,
)

# Create hyperbolic projection (as used in the model)
proj = HyperbolicProjection(
    latent_dim=16,
    hidden_dim=64,
    init_identity=False,  # Our fix
    # tangent_scale defaults to 0.05 now
)

print(f"HyperbolicProjection tangent_scale: {proj.tangent_scale.item()}")

# Create a batch of ternary operation indices
batch_indices = torch.randint(0, 19683, (16,))  # Batch of 16
from src.core import TERNARY

batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (16, 9)

print(f"Input ternary shape: {batch_ternary.shape}")

# Get encoder outputs (deterministic part - mu)
with torch.no_grad():
    mu_A, log_A = model.head_A(batch_ternary)
    mu_B, log_B = model.head_B(batch_ternary)

    # Full stochastic sample
    z_tangent_A = mu_A + torch.exp(0.5 * log_A) * torch.randn_like(mu_A)
    z_tangent_B = mu_B + torch.exp(0.5 * log_B) * torch.randn_like(mu_B)

print(f"Encoder A mu norm: {torch.norm(mu_A, dim=-1).mean().item():.4f}")
print(f"Encoder B mu norm: {torch.norm(mu_B, dim=-1).mean().item():.4f}")
print(f"Encoder A z_tangent norm: {torch.norm(z_tangent_A, dim=-1).mean().item():.4f}")
print(f"Encoder B z_tangent norm: {torch.norm(z_tangent_B, dim=-1).mean().item():.4f}")

# Pass through hyperbolic projection
with torch.no_grad():
    z_A_hyp = proj(z_tangent_A, as_manifold=False)
    z_B_hyp = proj(z_tangent_B, as_manifold=False)

    hyp_norm_A = torch.norm(z_A_hyp, dim=-1)
    hyp_norm_B = torch.norm(z_B_hyp, dim=-1)

print("\nAfter HyperbolicProjection:")
print(
    f"Encoder A Poincaré norm: mean={hyp_norm_A.mean().item():.4f}, std={hyp_norm_A.std().item():.4f}"
)
print(
    f"Encoder B Poincaré norm: mean={hyp_norm_B.mean().item():.4f}, std={hyp_norm_B.std().item():.4f}"
)
print(
    f"Encoder A points near boundary (>0.9): {(hyp_norm_A > 0.9).float().mean().item() * 100:.1f}%"
)
print(
    f"Encoder B points near boundary (>0.9): {(hyp_norm_B > 0.9).float().mean().item() * 100:.1f}%"
)

# Also test what happens with just the mu (deterministic) output
with torch.no_grad():
    z_A_hyp_mu = proj(mu_A, as_manifold=False)
    z_B_hyp_mu = proj(mu_B, as_manifold=False)

    hyp_norm_A_mu = torch.norm(z_A_hyp_mu, dim=-1)
    hyp_norm_B_mu = torch.norm(z_B_hyp_mu, dim=-1)

print("\nWith deterministic mu output:")
print(
    f"Encoder A Poincaré norm: mean={hyp_norm_A_mu.mean().item():.4f}, std={hyp_norm_A_mu.std().item():.4f}"
)
print(
    f"Encoder B Poincaré norm: mean={hyp_norm_B_mu.mean().item():.4f}, std={hyp_norm_B_mu.std().item():.4f}"
)
print(
    f"Encoder A points near boundary (>0.9): {(hyp_norm_A_mu > 0.9).float().mean().item() * 100:.1f}%"
)
print(
    f"Encoder B points near boundary (>0.9): {(hyp_norm_B_mu > 0.9).float().mean().item() * 100:.1f}%"
)
