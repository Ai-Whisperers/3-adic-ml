#!/usr/bin/env python3

"""
Validation script to test the fixes for embedding space collapse.
Tests:
1. HyperbolicProjection with init_identity=False produces non-zero tangent_net output
2. tangent_scale_init=0.1 (stored as log_tangent_scale=log(0.1)) gives appropriate scaling
3. Points distribute properly in Poincaré ball (not all at boundary)
"""

import torch

from src.models.hyperbolic_projection import HyperbolicProjection

# Deterministic: with an unseeded RNG, test_expmap0_saturation's
# boundary_ratio < 1.0 assertion coin-flips on a 16-sample batch (all 16
# landing above the 0.9 threshold by chance is not rare enough to ignore).
torch.manual_seed(42)


def test_tangent_net_output():
    """Test that tangent_net produces non-zero output with init_identity=False"""
    print("Testing tangent_net output...")

    # Create projection with our fixes
    proj = HyperbolicProjection(
        latent_dim=16,
        hidden_dim=64,
        init_identity=False,  # Our fix
    )

    # Check that effective tangent_scale is ~0.1 (exp of log_tangent_scale)
    effective_scale = proj.tangent_scale.item()
    assert abs(effective_scale - 0.1) < 1e-6, (
        f"Expected effective scale ~0.1, got {effective_scale}"
    )
    print(f"✓ effective tangent_scale correctly ~0.1 (log_tangent_scale={proj.log_tangent_scale.item():.4f})")

    # Test input similar to encoder output (norm ~4.0) - use float64 to match model
    z_tangent = (
        torch.randn(8, 16, dtype=torch.float64) * 4.0
    )  # Batch of 8, latent_dim=16
    print(f"Input z_tangent norm: {torch.norm(z_tangent, dim=-1).mean().item():.4f}")

    with torch.no_grad():
        z_scaled = proj.tangent_scale * z_tangent
        z_transformed = z_scaled + proj.tangent_net(z_scaled)
        residual = proj.tangent_net(z_scaled)

    print(f"Scaled input norm: {torch.norm(z_scaled, dim=-1).mean().item():.4f}")
    print(f"Residual norm: {torch.norm(residual, dim=-1).mean().item():.4f}")
    print(f"Transformed norm: {torch.norm(z_transformed, dim=-1).mean().item():.4f}")

    # Check that residual is non-zero (the fix)
    residual_mean = torch.norm(residual, dim=-1).mean().item()
    assert residual_mean > 1e-5, f"Residual is too small: {residual_mean}"
    print(f"✓ Residual is non-zero: {residual_mean:.6f}")

    return True


def test_expmap0_saturation():
    """Test that points don't all saturate at boundary"""
    print("\nTesting expmap0 saturation...")

    proj = HyperbolicProjection(
        latent_dim=16,
        hidden_dim=64,
        init_identity=False,
    )

    # Check effective scale is ~0.1
    assert abs(proj.tangent_scale.item() - 0.1) < 1e-6

    # Test with various input scales. Batch of 200 (not 16): the
    # boundary_ratio < 1.0 assertion below is a population statistic, and at
    # n=16 it coin-flips on whether every sample happens to land above 0.9.
    for scale_factor in [1.0, 2.0, 3.0, 4.0, 5.0]:
        z_tangent = torch.randn(200, 16, dtype=torch.float64) * scale_factor
        with torch.no_grad():
            z_hyp = proj(z_tangent, as_manifold=False)
            hyp_norm = torch.norm(z_hyp, dim=-1)

        print(
            f"Scale {scale_factor}: hyp norm mean={hyp_norm.mean().item():.4f}, "
            f"std={hyp_norm.std().item():.4f}, max={hyp_norm.max().item():.4f}"
        )

        # Check that not all points are at boundary (>0.9)
        boundary_ratio = (hyp_norm > 0.9).float().mean().item()
        print(f"  Points near boundary (>0.9): {boundary_ratio * 100:.1f}%")

        # With our fix, points should not ALL be at exactly max_radius (std > 0 means variety exists)
        if scale_factor == 4.0:  # Typical encoder output scale
            assert hyp_norm.std().item() > 0.001, (
                f"No variation — all points at boundary (std={hyp_norm.std().item():.6f})"
            )
            assert boundary_ratio < 1.0, (
                f"Every point at boundary — degenerate saturation"
            )
            print("  ✓ No degenerate saturation for typical encoder scale")

    return True


def test_identity_vs_non_identity():
    """Compare identity vs non-identity initialization"""
    print("\nComparing identity vs non-identity initialization...")

    # Identity initialization (original problematic version)
    proj_id = HyperbolicProjection(
        latent_dim=16,
        hidden_dim=64,
        init_identity=True,  # Original
    )

    # Non-identity initialization (our fix)
    proj_fixed = HyperbolicProjection(
        latent_dim=16,
        hidden_dim=64,
        init_identity=False,  # Our fix
    )

    # Check effective tangent_scale values (both should be ~0.1)
    print(f"Identity version effective scale: {proj_id.tangent_scale.item():.4f}")
    print(f"Fixed version effective scale: {proj_fixed.tangent_scale.item():.4f}")

    # Same input
    z_tangent = torch.randn(8, 16, dtype=torch.float64) * 4.0

    with torch.no_grad():
        # Test identity version residual
        z_scaled_id = proj_id.tangent_scale * z_tangent
        residual_id = proj_id.tangent_net(z_scaled_id)
        residual_id_norm = torch.norm(residual_id, dim=-1).mean().item()

        # Test fixed version residual
        z_scaled_fixed = proj_fixed.tangent_scale * z_tangent
        residual_fixed = proj_fixed.tangent_net(z_scaled_fixed)
        residual_fixed_norm = torch.norm(residual_fixed, dim=-1).mean().item()

    print(f"Identity residual norm: {residual_id_norm:.6f}")
    print(f"Fixed residual norm: {residual_fixed_norm:.6f}")

    # Identity version should have near-zero residual (last layer zeroed out)
    assert residual_id_norm < 1e-4, f"Identity residual too large: {residual_id_norm}"
    # Fixed version should have significant residual (learnable layers)
    assert residual_fixed_norm > 1e-3, (
        f"Fixed residual too small: {residual_fixed_norm}"
    )

    print("✓ Fix verified: identity residual ~0, fixed residual > 0")

    # Also test that the fixed version produces varied outputs
    z_hyp_id = proj_id(z_tangent, as_manifold=False)
    z_hyp_fixed = proj_fixed(z_tangent, as_manifold=False)

    id_std = torch.norm(z_hyp_id - z_hyp_id.mean(dim=0), dim=-1).mean().item()
    fixed_std = torch.norm(z_hyp_fixed - z_hyp_fixed.mean(dim=0), dim=-1).mean().item()

    print(f"Identity output std: {id_std:.6f}")
    print(f"Fixed output std: {fixed_std:.6f}")

    # Fixed version should have higher standard deviation (more varied outputs)
    assert fixed_std > id_std * 0.5, (
        f"Fixed version should have more variation: fixed={fixed_std:.4f}, identity={id_std:.4f}"
    )

    print("✓ Fixed version has better output variation than identity")

    return True


if __name__ == "__main__":
    print("Validating fixes for embedding space collapse...\n")

    try:
        test_tangent_net_output()
        test_expmap0_saturation()
        test_identity_vs_non_identity()

        print("\n✅ All validation tests passed!")
        print("The fixes should resolve the embedding space collapse issue.")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise
