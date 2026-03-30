#!/usr/bin/env python3
"""Test effect of tangent_scale on initial Poincaré norm"""

import torch
from src.models.hyperbolic_projection import HyperbolicProjection
from src.models.vae import TernaryVAEV6Controllable
from src.config import StateNetConfig

# Create model to get realistic encoder outputs
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

# Get a batch of ternary operation indices
batch_indices = torch.randint(0, 19683, (16,))
from src.core import TERNARY

batch_ternary = TERNARY.to_ternary(batch_indices)  # Shape: (16, 9)

# Get encoder outputs (stochastic)
with torch.no_grad():
    mu_A, log_A = model.head_A(batch_ternary)
    z_tangent_A = mu_A + torch.exp(0.5 * log_A) * torch.randn_like(mu_A)

print(f"Encoder A z_tangent norm: {torch.norm(z_tangent_A, dim=-1).mean().item():.4f}")

# Test different tangent_scale values
for ts in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    proj = HyperbolicProjection(
        latent_dim=16,
        hidden_dim=64,
        init_identity=False,
        # We'll set the tangent_scale manually after creation
    )
    # Override the tangent_scale parameter
    with torch.no_grad():
        proj.tangent_scale.copy_(torch.tensor(ts, dtype=torch.float64))

    with torch.no_grad():
        z_scaled = proj.tangent_scale * z_tangent_A
        z_transformed = z_scaled + proj.tangent_net(z_scaled)
        z_hyp = proj(z_tangent_A, as_manifold=False)
        hyp_norm = torch.norm(z_hyp, dim=-1)

    print(
        f"tangent_scale={ts:.2f}: "
        f"scaled norm={torch.norm(z_scaled, dim=-1).mean().item():.4f}, "
        f"transformed norm={torch.norm(z_transformed, dim=-1).mean().item():.4f}, "
        f"Poincaré norm mean={hyp_norm.mean().item():.4f}, "
        f"std={hyp_norm.std().item():.4f}"
    )
