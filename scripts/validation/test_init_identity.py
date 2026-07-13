#!/usr/bin/env python3
"""Test the effect of init_identity fix alone"""

import torch

from src.models.hyperbolic_projection import HyperbolicProjection

print("Testing HyperbolicProjection with init_identity=False but tangent_scale=0.1")

# Test with our init_identity fix but original tangent_scale
proj = HyperbolicProjection(
    latent_dim=16,
    hidden_dim=64,
    init_identity=False,  # Our fix
    # tangent_scale defaults to 0.1
)

# Check tangent_scale
print(f"tangent_scale: {proj.tangent_scale.item()}")

# Test with encoder-like input (norm ~4.0 for the deterministic part, but let's use
# the full stochastic sample which has higher norm)
# From our earlier check, the stochastic sample z_tangent has norm ~4.0 * sqrt(16) = 16.0
z_tangent = torch.randn(8, 16, dtype=torch.float64) * 4.0  # This gives norm ~16.0

print(f"Input z_tangent norm: {torch.norm(z_tangent, dim=-1).mean().item():.4f}")

with torch.no_grad():
    z_scaled = proj.tangent_scale * z_tangent
    z_transformed = z_scaled + proj.tangent_net(z_scaled)
    residual = proj.tangent_net(z_scaled)

    # Project to Poincaré ball
    z_hyp = proj(z_tangent, as_manifold=False)
    hyp_norm = torch.norm(z_hyp, dim=-1)

print(f"Scaled input norm: {torch.norm(z_scaled, dim=-1).mean().item():.4f}")
print(f"Residual norm: {torch.norm(residual, dim=-1).mean().item():.4f}")
print(f"Transformed norm: {torch.norm(z_transformed, dim=-1).mean().item():.4f}")
print(f"Poincaré ball norm: {hyp_norm.mean().item():.4f}")
print(f"Poincaré ball norm std: {hyp_norm.std().item():.4f}")
print(
    f"Points near boundary (>0.9): {(hyp_norm > 0.9).float().mean().item() * 100:.1f}%"
)

# Also test what the original (init_identity=True) would give
print("\n--- For comparison, init_identity=True (original) ---")
proj_orig = HyperbolicProjection(
    latent_dim=16,
    hidden_dim=64,
    init_identity=True,  # Original
    # tangent_scale defaults to 0.1
)

with torch.no_grad():
    z_scaled_orig = proj_orig.tangent_scale * z_tangent
    # With init_identity=True, residual = 0
    z_transformed_orig = z_scaled_orig  # + 0
    z_hyp_orig = proj_orig(z_tangent, as_manifold=False)
    hyp_norm_orig = torch.norm(z_hyp_orig, dim=-1)

print(
    f"Original scaled input norm: {torch.norm(z_scaled_orig, dim=-1).mean().item():.4f}"
)
print(
    f"Original transformed norm: {torch.norm(z_transformed_orig, dim=-1).mean().item():.4f}"
)
print(f"Original Poincaré ball norm: {hyp_norm_orig.mean().item():.4f}")
print(f"Original Poincaré ball norm std: {hyp_norm_orig.std().item():.4f}")
print(
    f"Original points near boundary (>0.9): {(hyp_norm_orig > 0.9).float().mean().item() * 100:.1f}%"
)
