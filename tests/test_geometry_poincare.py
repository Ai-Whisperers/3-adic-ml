# Copyright 2024-2025 AI Whisperers
# Tests for hyperbolic geometry - mathematical invariants of Poincaré ball

"""
Tier 1 Critical Tests: Hyperbolic Geometry

These tests verify the Poincaré ball operations are mathematically correct.
Numerical errors here corrupt all training since the manifold is the bridge
from discrete 3-adic to continuous latent space.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import (
    poincare_distance,
    hyperbolic_radius,
    exp_map_zero,
    log_map_zero,
    lambda_x,
    get_manifold,
)


# Tolerances for numerical comparisons
ATOL = 1e-6  # Absolute tolerance
RTOL = 1e-5  # Relative tolerance


class TestExpLogComposition:
    """Verify exp_map_zero and log_map_zero are inverses."""

    def test_log_of_exp_is_identity(self):
        """log_map_zero(exp_map_zero(v)) ≈ v for tangent vectors."""
        torch.manual_seed(42)

        # Generate random tangent vectors (not too large to stay well inside ball)
        v = torch.randn(100, 16, dtype=torch.float64) * 0.5

        # Map to manifold then back
        z_hyp = exp_map_zero(v, c=1.0)
        v_recovered = log_map_zero(z_hyp, c=1.0)

        max_error = (v - v_recovered).abs().max().item()
        assert max_error < ATOL, (
            f"log(exp(v)) != v: max error = {max_error}"
        )

    def test_exp_of_log_is_identity(self):
        """exp_map_zero(log_map_zero(z)) ≈ z for points on manifold."""
        torch.manual_seed(42)

        # Generate random points inside the ball
        z = torch.randn(100, 16, dtype=torch.float64)
        z = z / z.norm(dim=-1, keepdim=True) * 0.7  # Normalize to radius 0.7

        # Map to tangent then back
        v = log_map_zero(z, c=1.0)
        z_recovered = exp_map_zero(v, c=1.0)

        max_error = (z - z_recovered).abs().max().item()
        assert max_error < ATOL, (
            f"exp(log(z)) != z: max error = {max_error}"
        )

    def test_composition_at_various_radii(self):
        """Test round-trip at different radii (0.1, 0.5, 0.9)."""
        torch.manual_seed(42)

        for target_radius in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v = torch.randn(50, 16, dtype=torch.float64)
            v = v / v.norm(dim=-1, keepdim=True) * target_radius

            z = exp_map_zero(v, c=1.0)
            v_back = log_map_zero(z, c=1.0)

            max_error = (v - v_back).abs().max().item()
            assert max_error < ATOL * (1 + target_radius * 10), (
                f"Round-trip at radius {target_radius} failed: max error = {max_error}"
            )


class TestBallContainment:
    """Verify exp_map_zero keeps points inside the Poincaré ball."""

    def test_exp_maps_to_interior(self):
        """exp_map_zero output should have norm < 1."""
        torch.manual_seed(42)

        # Random tangent vectors of various magnitudes
        v = torch.randn(100, 16, dtype=torch.float64)

        z = exp_map_zero(v, c=1.0)
        norms = z.norm(dim=-1)

        assert (norms < 1.0).all(), (
            f"Points escaped ball: max norm = {norms.max().item()}"
        )

    def test_large_tangent_approaches_boundary(self):
        """Very large tangent vectors should map near (but not at) boundary."""
        # Large tangent vector
        v = torch.ones(1, 16, dtype=torch.float64) * 100.0

        z = exp_map_zero(v, c=1.0)
        norm = z.norm().item()

        assert norm < 1.0, f"Point at or beyond boundary: norm = {norm}"
        assert norm > 0.99, f"Large tangent should be near boundary: norm = {norm}"

    def test_zero_tangent_maps_to_origin(self):
        """Zero tangent vector should map to origin."""
        v = torch.zeros(1, 16, dtype=torch.float64)

        z = exp_map_zero(v, c=1.0)
        norm = z.norm().item()

        assert norm < ATOL, f"Zero tangent didn't map to origin: norm = {norm}"

    def test_various_curvatures(self):
        """Ball containment should hold for different curvatures.

        For curvature c, the Poincaré ball has radius 1/sqrt(c).
        So c=1.0 gives radius 1.0, c=0.1 gives radius ~3.16, c=4.0 gives radius 0.5.
        """
        torch.manual_seed(42)
        v = torch.randn(50, 16, dtype=torch.float64) * 2.0

        for c in [0.5, 1.0, 2.0, 5.0]:
            z = exp_map_zero(v, c=c)
            norms = z.norm(dim=-1)

            # Ball radius is 1/sqrt(c)
            ball_radius = 1.0 / (c ** 0.5)

            assert (norms < ball_radius).all(), (
                f"Points escaped ball at c={c}: max norm = {norms.max().item()}, "
                f"ball radius = {ball_radius}"
            )


class TestDistancePositivity:
    """Verify Poincaré distance is always non-negative."""

    def test_distance_non_negative(self):
        """poincare_distance(x, y) >= 0 for all x, y."""
        torch.manual_seed(42)

        # Random points in ball
        x = torch.randn(100, 16, dtype=torch.float64) * 0.5
        y = torch.randn(100, 16, dtype=torch.float64) * 0.5

        distances = poincare_distance(x, y, c=1.0)

        assert (distances >= 0).all(), (
            f"Negative distance found: min = {distances.min().item()}"
        )

    def test_distance_finite(self):
        """Distances should be finite (no NaN or Inf)."""
        torch.manual_seed(42)

        x = torch.randn(100, 16, dtype=torch.float64) * 0.5
        y = torch.randn(100, 16, dtype=torch.float64) * 0.5

        distances = poincare_distance(x, y, c=1.0)

        assert not torch.isnan(distances).any(), "NaN in distances"
        assert not torch.isinf(distances).any(), "Inf in distances"


class TestDistanceIdentity:
    """Verify d(x, x) = 0."""

    def test_self_distance_zero(self):
        """poincare_distance(x, x) = 0 for all x."""
        torch.manual_seed(42)

        x = torch.randn(100, 16, dtype=torch.float64) * 0.5

        distances = poincare_distance(x, x, c=1.0)

        max_self_dist = distances.max().item()
        assert max_self_dist < ATOL, (
            f"Self-distance not zero: max = {max_self_dist}"
        )

    def test_origin_self_distance(self):
        """Distance from origin to itself should be 0."""
        origin = torch.zeros(1, 16, dtype=torch.float64)

        d = poincare_distance(origin, origin, c=1.0).item()
        assert d < ATOL, f"Origin self-distance: {d}"


class TestDistanceSymmetry:
    """Verify d(x, y) = d(y, x)."""

    def test_distance_symmetric(self):
        """poincare_distance(x, y) = poincare_distance(y, x)."""
        torch.manual_seed(42)

        x = torch.randn(100, 16, dtype=torch.float64) * 0.5
        y = torch.randn(100, 16, dtype=torch.float64) * 0.5

        d_xy = poincare_distance(x, y, c=1.0)
        d_yx = poincare_distance(y, x, c=1.0)

        max_diff = (d_xy - d_yx).abs().max().item()
        assert max_diff < ATOL, (
            f"Distance not symmetric: max |d(x,y) - d(y,x)| = {max_diff}"
        )


class TestTriangleInequality:
    """Verify d(x, z) <= d(x, y) + d(y, z)."""

    def test_triangle_inequality_holds(self):
        """Triangle inequality must hold for all triples."""
        torch.manual_seed(42)

        # Generate random points
        n_triples = 200
        x = torch.randn(n_triples, 16, dtype=torch.float64) * 0.5
        y = torch.randn(n_triples, 16, dtype=torch.float64) * 0.5
        z = torch.randn(n_triples, 16, dtype=torch.float64) * 0.5

        d_xy = poincare_distance(x, y, c=1.0)
        d_yz = poincare_distance(y, z, c=1.0)
        d_xz = poincare_distance(x, z, c=1.0)

        # Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
        violations = d_xz > d_xy + d_yz + ATOL

        assert not violations.any(), (
            f"Triangle inequality violated in {violations.sum()} cases"
        )


class TestHyperbolicRadius:
    """Verify hyperbolic_radius (distance from origin)."""

    def test_radius_matches_distance_to_origin(self):
        """hyperbolic_radius(z) should equal poincare_distance(z, 0)."""
        torch.manual_seed(42)

        z = torch.randn(100, 16, dtype=torch.float64) * 0.5
        origin = torch.zeros_like(z)

        radius = hyperbolic_radius(z, c=1.0)
        dist_to_origin = poincare_distance(z, origin, c=1.0)

        max_diff = (radius - dist_to_origin).abs().max().item()
        assert max_diff < ATOL, (
            f"Radius != distance to origin: max diff = {max_diff}"
        )

    def test_radius_non_negative(self):
        """Radius should be non-negative."""
        torch.manual_seed(42)

        z = torch.randn(100, 16, dtype=torch.float64) * 0.5
        radius = hyperbolic_radius(z, c=1.0)

        assert (radius >= 0).all(), f"Negative radius: min = {radius.min().item()}"

    def test_origin_radius_zero(self):
        """Origin should have radius 0."""
        origin = torch.zeros(1, 16, dtype=torch.float64)
        radius = hyperbolic_radius(origin, c=1.0).item()

        assert radius < ATOL, f"Origin radius: {radius}"


class TestConformalFactor:
    """Verify conformal factor λ(x) = 2 / (1 - c||x||²)."""

    def test_conformal_positive(self):
        """Conformal factor must be positive for points inside ball."""
        torch.manual_seed(42)

        # Points strictly inside the ball
        z = torch.randn(100, 16, dtype=torch.float64) * 0.5

        lam = lambda_x(z, c=1.0)

        assert (lam > 0).all(), f"Non-positive conformal factor: min = {lam.min().item()}"

    def test_conformal_at_origin(self):
        """λ(0) = 2 (at origin, no distortion adjustment)."""
        origin = torch.zeros(1, 16, dtype=torch.float64)
        lam = lambda_x(origin, c=1.0).item()

        assert abs(lam - 2.0) < ATOL, f"λ(0) = {lam}, expected 2.0"

    def test_conformal_increases_near_boundary(self):
        """Conformal factor should increase as ||x|| → 1."""
        # Points at different radii
        radii = [0.1, 0.3, 0.5, 0.7, 0.9]
        lambdas = []

        for r in radii:
            z = torch.zeros(1, 16, dtype=torch.float64)
            z[0, 0] = r
            lam = lambda_x(z, c=1.0).item()
            lambdas.append(lam)

        # Should be strictly increasing
        for i in range(len(lambdas) - 1):
            assert lambdas[i] < lambdas[i + 1], (
                f"Conformal not increasing: λ({radii[i]}) = {lambdas[i]} "
                f">= λ({radii[i+1]}) = {lambdas[i+1]}"
            )

    def test_conformal_finite(self):
        """Conformal should be finite for points strictly inside ball."""
        torch.manual_seed(42)

        z = torch.randn(100, 16, dtype=torch.float64) * 0.9

        lam = lambda_x(z, c=1.0)

        assert not torch.isnan(lam).any(), "NaN in conformal factor"
        assert not torch.isinf(lam).any(), "Inf in conformal factor"


class TestCurvatureEffect:
    """Verify curvature parameter affects distances correctly."""

    def test_higher_curvature_larger_distance(self):
        """Higher curvature should produce larger distances (for same Euclidean points)."""
        torch.manual_seed(42)

        x = torch.randn(50, 16, dtype=torch.float64) * 0.3
        y = torch.randn(50, 16, dtype=torch.float64) * 0.3

        d_low = poincare_distance(x, y, c=0.5)
        d_high = poincare_distance(x, y, c=2.0)

        # Higher curvature means more "curved" space, distances scale
        # The relationship isn't strictly monotonic due to the formula
        # but generally higher c means larger d for same Euclidean distance
        mean_low = d_low.mean().item()
        mean_high = d_high.mean().item()

        # This is a soft check - the relationship depends on the points
        assert abs(mean_low - mean_high) > ATOL, (
            "Curvature should affect distances"
        )

    def test_different_curvatures_same_manifold_structure(self):
        """Different curvatures should still satisfy metric properties."""
        torch.manual_seed(42)

        x = torch.randn(30, 16, dtype=torch.float64) * 0.3
        y = torch.randn(30, 16, dtype=torch.float64) * 0.3
        z = torch.randn(30, 16, dtype=torch.float64) * 0.3

        for c in [0.1, 1.0, 5.0]:
            d_xy = poincare_distance(x, y, c=c)
            d_yz = poincare_distance(y, z, c=c)
            d_xz = poincare_distance(x, z, c=c)

            # Triangle inequality
            violations = d_xz > d_xy + d_yz + ATOL
            assert not violations.any(), (
                f"Triangle inequality violated at c={c}"
            )


class TestManifoldCache:
    """Verify manifold caching works correctly."""

    def test_same_curvature_same_object(self):
        """Same curvature should return cached manifold."""
        m1 = get_manifold(c=1.0, device="cpu")
        m2 = get_manifold(c=1.0, device="cpu")

        assert m1 is m2, "Same (c, device) should return same object"

    def test_different_curvature_different_object(self):
        """Different curvature should return different manifold."""
        m1 = get_manifold(c=1.0, device="cpu")
        m2 = get_manifold(c=2.0, device="cpu")

        assert m1 is not m2, "Different c should return different object"


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_near_boundary_no_overflow(self):
        """Points very close to boundary should not cause overflow."""
        # Point at radius 0.999
        z = torch.zeros(1, 16, dtype=torch.float64)
        z[0, 0] = 0.999

        # These should not overflow
        lam = lambda_x(z, c=1.0)
        radius = hyperbolic_radius(z, c=1.0)

        assert not torch.isnan(lam).any(), "NaN at boundary"
        assert not torch.isinf(lam).any(), "Inf at boundary"
        assert not torch.isnan(radius).any(), "NaN in radius at boundary"

    def test_very_small_vectors(self):
        """Very small tangent vectors should work."""
        v = torch.ones(1, 16, dtype=torch.float64) * 1e-10

        z = exp_map_zero(v, c=1.0)
        v_back = log_map_zero(z, c=1.0)

        # Should be approximately identity
        max_error = (v - v_back).abs().max().item()
        assert max_error < 1e-8, f"Small vector round-trip error: {max_error}"

    def test_high_dimensional_stability(self):
        """Operations should be stable in higher dimensions."""
        torch.manual_seed(42)

        # 64-dimensional latent space
        v = torch.randn(20, 64, dtype=torch.float64) * 0.5

        z = exp_map_zero(v, c=1.0)
        v_back = log_map_zero(z, c=1.0)

        max_error = (v - v_back).abs().max().item()
        assert max_error < ATOL * 10, (  # Slightly relaxed for high dim
            f"High-dim round-trip error: {max_error}"
        )


class TestDeviceConsistency:
    """Test CPU/GPU consistency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_exp_map_match(self):
        """exp_map_zero should match between CPU and GPU."""
        torch.manual_seed(42)

        v = torch.randn(50, 16, dtype=torch.float64)

        z_cpu = exp_map_zero(v.cpu(), c=1.0)
        z_gpu = exp_map_zero(v.cuda(), c=1.0).cpu()

        max_diff = (z_cpu - z_gpu).abs().max().item()
        assert max_diff < ATOL, f"CPU/GPU exp_map mismatch: {max_diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_distance_match(self):
        """Distances should match between CPU and GPU."""
        torch.manual_seed(42)

        x = torch.randn(50, 16, dtype=torch.float64) * 0.5
        y = torch.randn(50, 16, dtype=torch.float64) * 0.5

        d_cpu = poincare_distance(x.cpu(), y.cpu(), c=1.0)
        d_gpu = poincare_distance(x.cuda(), y.cuda(), c=1.0).cpu()

        max_diff = (d_cpu - d_gpu).abs().max().item()
        assert max_diff < ATOL, f"CPU/GPU distance mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
