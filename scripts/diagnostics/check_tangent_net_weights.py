#!/usr/bin/env python3
"""Check the tangent_net weights at initialization"""

import torch

from src.models.hyperbolic_projection import HyperbolicProjection

# Create projection with our fixes
proj = HyperbolicProjection(
    latent_dim=16,
    hidden_dim=64,
    init_identity=False,  # Our fix
    # tangent_scale_init defaults to 0.1
)

print("Tangent net architecture:")
for i, layer in enumerate(proj.tangent_net):
    print(f"  Layer {i}: {layer}")

print("\nWeight statistics:")
for i, layer in enumerate(proj.tangent_net):
    if hasattr(layer, "weight"):
        w = layer.weight
        b = layer.bias if layer.bias is not None else torch.tensor(0.0)
        print(
            f"  Layer {i} weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}"
        )
        print(
            f"  Layer {i} bias:   mean={b.mean().item():.6f}, std={b.std().item():.6f}"
        )

# Check the final layer specifically (which was zeroed out in init_identity=True)
final_layer = proj.tangent_net[-1]
print(f"\nFinal layer ({len(proj.tangent_net) - 1}): {final_layer}")
if hasattr(final_layer, "weight"):
    w = final_layer.weight
    b = final_layer.bias if final_layer.bias is not None else torch.tensor(0.0)
    print(f"  Final layer weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}")
    print(f"  Final layer bias:   mean={b.mean().item():.6f}, std={b.std().item():.6f}")

    # Check if it's close to zero (it shouldn't be with init_identity=False)
    weight_norm = torch.norm(w).item()
    bias_norm = torch.norm(b).item() if b.numel() > 0 else 0.0
    print(f"  Final layer weight norm: {weight_norm:.6f}")
    print(f"  Final layer bias norm: {bias_norm:.6f}")

    # With init_identity=False, these should NOT be zero
    is_zero_weight = weight_norm < 1e-6
    is_zero_bias = bias_norm < 1e-6
    print(f"  Is weight essentially zero? {is_zero_weight}")
    print(f"  Is bias essentially zero? {is_zero_bias}")
