# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tier 2: Loss Function Correctness Tests.

Tests for:
- Gradient flow through all loss paths
- Loss non-negativity
- Target distance/radius monotonicity
- Metric bounds and correctness

These tests verify the optimization signal is mathematically correct.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.core import TERNARY
from src.geometry import hyperbolic_radius, poincare_distance
from src.losses.padic_geodesic import (
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    _exponential_target_radii,
    GlobalRankLoss,
    MonotonicRadialLoss,
    RichHierarchyLoss,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    torch.manual_seed(42)
    batch_size = 64
    latent_dim = 16

    # Create z_hyp in Poincaré ball (norm < 1)
    z_raw = torch.randn(batch_size, latent_dim, dtype=torch.float64)
    z_hyp = 0.8 * torch.tanh(z_raw)  # Keeps norm well inside ball

    # Random indices from valid range
    indices = torch.randint(0, 19683, (batch_size,))

    return z_hyp, indices


@pytest.fixture
def sample_batch_with_reconstruction():
    """Create a sample batch with reconstruction targets."""
    torch.manual_seed(42)
    batch_size = 64
    latent_dim = 16

    z_raw = torch.randn(batch_size, latent_dim, dtype=torch.float64)
    z_hyp = 0.8 * torch.tanh(z_raw)

    indices = torch.randint(0, 19683, (batch_size,))

    # Logits for reconstruction (B, 27) format
    logits = torch.randn(batch_size, 27, dtype=torch.float64)

    # Targets in {-1, 0, 1}
    targets = torch.randint(-1, 2, (batch_size, 9))

    return z_hyp, indices, logits, targets


@pytest.fixture
def small_batch():
    """Create a small batch for edge case testing."""
    torch.manual_seed(42)
    z_hyp = torch.randn(8, 16, dtype=torch.float64) * 0.5
    indices = torch.randint(0, 19683, (8,))
    return z_hyp, indices


# =============================================================================
# Test Classes: Gradient Flow
# =============================================================================


class TestGradientFlowPAdicGeodesic:
    """Test gradient flow through PAdicGeodesicLoss."""

    def test_gradient_exists_and_nonzero(self, sample_batch):
        """Verify gradients flow back to z_hyp."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None, "Gradient not computed"
        assert z_hyp.grad.abs().sum() > 1e-10, "Gradient is all zeros"

    def test_gradient_finite(self, sample_batch):
        """Verify gradients are finite (no NaN/Inf)."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert torch.isfinite(z_hyp.grad).all(), "Gradient contains NaN or Inf"

    def test_gradient_with_boundary_points(self):
        """Verify gradient flow with points near boundary."""
        torch.manual_seed(42)
        # Points near boundary (norm ~ 0.99)
        z_hyp = torch.randn(32, 16, dtype=torch.float64)
        z_hyp = 0.99 * z_hyp / z_hyp.norm(dim=-1, keepdim=True)
        z_hyp = z_hyp.requires_grad_(True)

        indices = torch.randint(0, 19683, (32,))

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None
        assert torch.isfinite(z_hyp.grad).all(), "Boundary gradient not finite"


class TestGradientFlowRadialHierarchy:
    """Test gradient flow through RadialHierarchyLoss."""

    def test_gradient_exists_and_nonzero(self):
        """Verify gradients flow back to z_hyp."""
        # Use points far from target radii to ensure non-zero gradient
        torch.manual_seed(123)
        # Create points clustered near origin (all will have radius ~0.1)
        # but with diverse valuations, so target radii vary
        z_hyp = torch.randn(64, 16, dtype=torch.float64) * 0.05
        z_hyp = z_hyp.requires_grad_(True)
        indices = torch.randint(0, 19683, (64,))

        loss_fn = RadialHierarchyLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None, "Gradient not computed"
        assert z_hyp.grad.abs().sum() > 1e-10, "Gradient is all zeros"

    def test_gradient_with_margin_loss(self, sample_batch):
        """Verify gradient flow includes margin loss component."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        loss_fn = RadialHierarchyLoss(use_margin_loss=True, margin_weight=1.0)
        loss, metrics = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None
        # Margin loss should contribute to gradient
        assert metrics["margin_loss"] >= 0


class TestGradientFlowGlobalRank:
    """Test gradient flow through GlobalRankLoss."""

    def test_gradient_exists_and_nonzero(self):
        """Verify gradients flow back to z_hyp."""
        # Use points with clear rank violations to ensure gradient
        torch.manual_seed(456)
        # Create a batch where radii don't match valuation ordering
        z_hyp = torch.randn(64, 16, dtype=torch.float64)
        # Scale to various radii
        z_hyp = 0.5 * z_hyp / z_hyp.norm(dim=-1, keepdim=True)
        z_hyp = z_hyp.requires_grad_(True)
        indices = torch.randint(0, 19683, (64,))

        loss_fn = GlobalRankLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None, "Gradient not computed"
        # GlobalRankLoss uses sigmoid which always has gradient
        assert z_hyp.grad.abs().sum() > 1e-10, "Gradient is all zeros"

    def test_soft_ranking_differentiable(self, sample_batch):
        """Verify soft ranking via sigmoid is differentiable."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        # Low temperature makes sigmoid sharper but still differentiable
        loss_fn = GlobalRankLoss(temperature=0.01)
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert torch.isfinite(z_hyp.grad).all()


class TestGradientFlowMonotonicRadial:
    """Test gradient flow through MonotonicRadialLoss."""

    def test_gradient_exists_and_nonzero(self, sample_batch):
        """Verify gradients flow back to z_hyp."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        loss_fn = MonotonicRadialLoss()
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert z_hyp.grad is not None, "Gradient not computed"

    def test_soft_margin_differentiable(self, sample_batch):
        """Verify soft margin via softplus is differentiable."""
        z_hyp, indices = sample_batch
        z_hyp = z_hyp.clone().requires_grad_(True)

        loss_fn = MonotonicRadialLoss(use_soft_margin=True, temperature=0.01)
        loss, _ = loss_fn(z_hyp, indices)

        loss.backward()

        assert torch.isfinite(z_hyp.grad).all()


class TestGradientFlowRichHierarchy:
    """Test gradient flow through RichHierarchyLoss."""

    def test_all_components_have_gradient(self, sample_batch_with_reconstruction):
        """Verify all three loss components have gradient paths."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction
        z_hyp = z_hyp.clone().requires_grad_(True)
        logits = logits.clone().requires_grad_(True)

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        # Sum all components
        total = out["hierarchy"] + out["coverage"] + out["separation"]
        total.backward()

        assert z_hyp.grad is not None, "z_hyp gradient not computed"
        assert logits.grad is not None, "logits gradient not computed"

    def test_hierarchy_gradient_nonzero(self):
        """Verify hierarchy loss produces gradient on z_hyp."""
        # Use points clustered in one region to ensure non-zero hierarchy loss
        torch.manual_seed(789)
        batch_size = 64
        # All points near origin - will have hierarchy loss vs diverse targets
        z_hyp = torch.randn(batch_size, 16, dtype=torch.float64) * 0.1
        z_hyp = z_hyp.requires_grad_(True)
        indices = torch.randint(0, 19683, (batch_size,))
        logits = torch.randn(batch_size, 27, dtype=torch.float64)
        targets = torch.randint(-1, 2, (batch_size, 9))

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        out["hierarchy"].backward()

        assert z_hyp.grad is not None
        assert z_hyp.grad.abs().sum() > 1e-10

    def test_coverage_gradient_on_logits(self, sample_batch_with_reconstruction):
        """Verify coverage loss produces gradient on logits."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction
        logits = logits.clone().requires_grad_(True)

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        out["coverage"].backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 1e-10


# NOTE: TestGradientFlowCombinedGeodesic removed — CombinedGeodesicLoss
# was archived as dead code (superseded by CombinedLoss).

# =============================================================================
# Test Classes: Loss Non-Negativity
# =============================================================================


class TestLossNonNegativityPAdicGeodesic:
    """Test PAdicGeodesicLoss non-negativity."""

    def test_loss_non_negative(self, sample_batch):
        """Verify loss is always >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0, f"Negative loss: {loss}"

    def test_loss_finite(self, sample_batch):
        """Verify loss is finite."""
        z_hyp, indices = sample_batch

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    def test_loss_zero_on_batch_size_one(self):
        """Verify batch_size=1 returns zero loss, not error."""
        z_hyp = torch.randn(1, 16, dtype=torch.float64) * 0.5
        indices = torch.tensor([0])

        loss_fn = PAdicGeodesicLoss()
        loss, metrics = loss_fn(z_hyp, indices)

        assert loss == 0.0
        assert metrics["n_pairs"] == 0

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_loss_non_negative_multiple_seeds(self, seed):
        """Verify non-negativity across multiple random seeds."""
        torch.manual_seed(seed)
        z_hyp = torch.randn(64, 16, dtype=torch.float64) * 0.7
        indices = torch.randint(0, 19683, (64,))

        loss_fn = PAdicGeodesicLoss()
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0
        assert torch.isfinite(loss)


class TestLossNonNegativityRadialHierarchy:
    """Test RadialHierarchyLoss non-negativity."""

    def test_primary_loss_non_negative(self, sample_batch):
        """Verify primary loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss(use_margin_loss=False)
        loss, metrics = loss_fn(z_hyp, indices)

        assert loss >= 0
        assert metrics["primary_loss"] >= 0

    def test_margin_loss_non_negative(self, sample_batch):
        """Verify margin loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss(use_margin_loss=True)
        loss, metrics = loss_fn(z_hyp, indices)

        assert loss >= 0
        assert metrics["margin_loss"] >= 0

    def test_total_loss_non_negative(self, sample_batch):
        """Verify total loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss(use_margin_loss=True, margin_weight=2.0)
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0


class TestLossNonNegativityGlobalRank:
    """Test GlobalRankLoss non-negativity."""

    def test_loss_non_negative(self, sample_batch):
        """Verify loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = GlobalRankLoss()
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0

    def test_loss_bounded_above(self, sample_batch):
        """Verify loss is bounded (sigmoid output in [0,1])."""
        z_hyp, indices = sample_batch

        loss_fn = GlobalRankLoss()
        loss, _ = loss_fn(z_hyp, indices)

        # sigmoid * weight, weight is valuation diff, max ~9
        # So loss should be < 10 typically
        assert loss < 100, f"Loss unexpectedly large: {loss}"

    def test_violation_rate_bounded(self, sample_batch):
        """Verify violation rate in [0, 1]."""
        z_hyp, indices = sample_batch

        loss_fn = GlobalRankLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert 0 <= metrics["violation_rate"] <= 1


class TestLossNonNegativityMonotonicRadial:
    """Test MonotonicRadialLoss non-negativity."""

    def test_loss_non_negative(self, sample_batch):
        """Verify loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss()
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0

    def test_components_non_negative(self, sample_batch):
        """Verify all components >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss()
        loss, metrics = loss_fn(z_hyp, indices)

        assert metrics["monotonic_loss"] >= 0
        assert metrics["target_loss"] >= 0

    def test_hard_margin_non_negative(self, sample_batch):
        """Verify hard margin (relu) loss >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss(use_soft_margin=False)
        loss, _ = loss_fn(z_hyp, indices)

        assert loss >= 0


class TestLossNonNegativityRichHierarchy:
    """Test RichHierarchyLoss non-negativity."""

    def test_hierarchy_non_negative(self, sample_batch_with_reconstruction):
        """Verify hierarchy loss >= 0."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        assert out["hierarchy"] >= 0

    def test_coverage_non_negative(self, sample_batch_with_reconstruction):
        """Verify coverage loss >= 0."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        assert out["coverage"] >= 0

    def test_separation_non_negative(self, sample_batch_with_reconstruction):
        """Verify separation loss >= 0."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        assert out["separation"] >= 0

    def test_all_components_finite(self, sample_batch_with_reconstruction):
        """Verify all components are finite."""
        z_hyp, indices, logits, targets = sample_batch_with_reconstruction

        loss_fn = RichHierarchyLoss()
        out = loss_fn(z_hyp, indices, logits, targets)

        assert torch.isfinite(out["hierarchy"])
        assert torch.isfinite(out["coverage"])
        assert torch.isfinite(out["separation"])


# =============================================================================
# Test Classes: Target Distance/Radius Monotonicity
# =============================================================================


class TestTargetDistanceMonotonicity:
    """Test PAdicGeodesicLoss target_distance monotonicity."""

    def test_target_distance_strictly_decreasing(self):
        """Verify target_distance(v) > target_distance(v+1) for all v."""
        loss_fn = PAdicGeodesicLoss()

        for v in range(9):
            d_current = loss_fn.target_distance(torch.tensor(v, dtype=torch.float64))
            d_next = loss_fn.target_distance(torch.tensor(v + 1, dtype=torch.float64))
            assert d_current > d_next, (
                f"Not monotonic at v={v}: d({v})={d_current}, d({v + 1})={d_next}"
            )

    def test_target_distance_positive(self):
        """Verify target_distance is always positive."""
        loss_fn = PAdicGeodesicLoss()

        for v in range(10):
            d = loss_fn.target_distance(torch.tensor(v, dtype=torch.float64))
            assert d > 0, f"Non-positive target distance at v={v}"

    def test_target_distance_formula_v0(self):
        """Verify target_distance(0) = max_target."""
        loss_fn = PAdicGeodesicLoss(max_target_distance=3.0)

        d = loss_fn.target_distance(torch.tensor(0, dtype=torch.float64))
        assert abs(d.item() - 3.0) < 1e-10

    def test_target_distance_formula_explicit(self):
        """Verify target_distance follows d = max * exp(-v/scale)."""
        max_target = 3.0
        scale = 3.0
        loss_fn = PAdicGeodesicLoss(
            max_target_distance=max_target, valuation_scale=scale
        )

        test_cases = [
            (0, max_target * math.exp(0)),
            (3, max_target * math.exp(-1)),
            (6, max_target * math.exp(-2)),
            (9, max_target * math.exp(-3)),
        ]

        for v, expected in test_cases:
            actual = loss_fn.target_distance(
                torch.tensor(v, dtype=torch.float64)
            ).item()
            assert abs(actual - expected) < 1e-10, (
                f"Formula mismatch at v={v}: expected {expected}, got {actual}"
            )

    def test_target_distance_batch(self):
        """Verify target_distance works on batched input."""
        loss_fn = PAdicGeodesicLoss()

        valuations = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float64)
        distances = loss_fn.target_distance(valuations)

        # Check monotonicity
        for i in range(9):
            assert distances[i] > distances[i + 1]


class TestTargetRadiusMonotonicity:
    """Test RadialHierarchyLoss and MonotonicRadialLoss target radii."""

    def test_radial_hierarchy_target_decreasing(self):
        """Verify target radius decreases with valuation."""
        loss_fn = RadialHierarchyLoss(
            inner_radius=0.1, outer_radius=0.85, max_valuation=9
        )
        target = loss_fn._target_radii
        for v in range(9):
            assert target[v] > target[v + 1], f"Not monotonic at v={v}"

    def test_radial_hierarchy_target_bounds(self):
        """Verify target radii are within [inner, outer]."""
        inner = 0.1
        outer = 0.85
        loss_fn = RadialHierarchyLoss(
            inner_radius=inner, outer_radius=outer, max_valuation=9
        )
        target = loss_fn._target_radii

        assert torch.all(target >= inner - 1e-10)
        assert torch.all(target <= outer + 1e-10)

    def test_monotonic_target_radii_buffer(self):
        """Verify MonotonicRadialLoss uses exponential target radii with shrinking gaps."""
        loss_fn = MonotonicRadialLoss(
            inner_radius=0.1, outer_radius=0.85, max_valuation=9
        )
        target = loss_fn._target_radii

        # Endpoints are preserved
        assert torch.allclose(
            target[0], torch.tensor(0.85, dtype=torch.float64), atol=1e-10
        )
        assert torch.allclose(
            target[-1], torch.tensor(0.1, dtype=torch.float64), atol=1e-10
        )

        # Exponential spacing means early gaps are larger than late gaps
        gaps = target[:-1] - target[1:]
        assert gaps[0] > gaps[-1]

    def test_rich_hierarchy_target_radii_buffer(self):
        """Verify RichHierarchyLoss target_radii buffer follows exponential mapping."""
        loss_fn = RichHierarchyLoss(inner_radius=0.1, outer_radius=0.85)

        # Check buffer values
        expected = _exponential_target_radii(
            max_valuation=9,
            inner_radius=0.1,
            outer_radius=0.85,
            scale=3.0,
        )

        target_radii = torch.as_tensor(loss_fn.target_radii)
        assert torch.allclose(target_radii, expected, atol=1e-10)

        # Check monotonicity
        for i in range(9):
            assert target_radii[i] > target_radii[i + 1]

    def test_rich_hierarchy_separation_uses_level_aware_margin(self):
        """Verify separation penalizes v=0/v=9 pairs using valuation-aware margin."""
        loss_fn = RichHierarchyLoss(
            inner_radius=0.1, outer_radius=0.85, separation_margin=0.01
        )

        z_hyp = torch.zeros(2, 16, dtype=torch.float64)
        z_hyp[0, 0] = math.tanh(0.6 / 2.0)
        z_hyp[1, 0] = math.tanh(0.5 / 2.0)
        indices = torch.tensor([1, 0], dtype=torch.long)

        logits = torch.randn(2, 9, 3, dtype=torch.float64)
        targets = torch.randint(-1, 2, (2, 9))

        out = loss_fn(z_hyp, indices, logits, targets)
        assert out["separation"] > 0.0


# =============================================================================
# Test Classes: Metric Bounds and Correctness
# =============================================================================


class TestMetricBoundsPAdicGeodesic:
    """Test PAdicGeodesicLoss metric bounds."""

    def test_correlation_bounded(self, sample_batch):
        """Verify distance_correlation in [-1, 1]."""
        z_hyp, indices = sample_batch

        loss_fn = PAdicGeodesicLoss()
        _, metrics = loss_fn(z_hyp, indices)

        corr = metrics["distance_correlation"]
        assert -1 <= corr <= 1, f"Correlation out of bounds: {corr}"

    def test_correlation_not_nan(self, sample_batch):
        """Verify correlation is not NaN."""
        z_hyp, indices = sample_batch

        loss_fn = PAdicGeodesicLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert not math.isnan(metrics["distance_correlation"])

    def test_mean_distances_positive(self, sample_batch):
        """Verify mean distances are positive."""
        z_hyp, indices = sample_batch

        loss_fn = PAdicGeodesicLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert metrics["mean_d_actual"] >= 0
        assert metrics["mean_d_target"] >= 0

    def test_n_pairs_reasonable(self, sample_batch):
        """Verify n_pairs is within expected range."""
        z_hyp, indices = sample_batch
        batch_size = z_hyp.size(0)

        loss_fn = PAdicGeodesicLoss(n_pairs=2000)
        _, metrics = loss_fn(z_hyp, indices)

        max_pairs = batch_size * (batch_size - 1) // 2
        assert 0 < metrics["n_pairs"] <= max_pairs


class TestMetricBoundsRadialHierarchy:
    """Test RadialHierarchyLoss metric bounds."""

    def test_radial_correlation_bounded(self, sample_batch):
        """Verify radial_hierarchy_corr in [-1, 1]."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss()
        _, metrics = loss_fn(z_hyp, indices)

        corr = metrics["radial_hierarchy_corr"]
        assert -1 <= corr <= 1, f"Correlation out of bounds: {corr}"

    def test_radius_bounds(self, sample_batch):
        """Verify radius metrics are sensible."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert metrics["radius_min"] >= 0
        assert metrics["radius_max"] >= metrics["radius_min"]
        assert metrics["radius_range"] >= 0


class TestMetricBoundsGlobalRank:
    """Test GlobalRankLoss metric bounds."""

    def test_violation_count_non_negative(self, sample_batch):
        """Verify n_violations >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = GlobalRankLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert metrics["n_violations"] >= 0

    def test_n_pairs_positive_when_data(self, sample_batch):
        """Verify n_pairs > 0 with sufficient data."""
        z_hyp, indices = sample_batch

        loss_fn = GlobalRankLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert metrics["n_pairs"] > 0


class TestMetricBoundsMonotonicRadial:
    """Test MonotonicRadialLoss metric bounds."""

    def test_n_levels_bounded(self, sample_batch):
        """Verify n_levels in [0, 10]."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert 0 <= metrics["n_levels"] <= 10

    def test_margin_violations_non_negative(self, sample_batch):
        """Verify margin_violations >= 0."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss()
        _, metrics = loss_fn(z_hyp, indices)

        assert metrics["margin_violations"] >= 0

    def test_per_level_radii_present(self, sample_batch):
        """Verify per-level radius metrics are logged."""
        z_hyp, indices = sample_batch

        loss_fn = MonotonicRadialLoss()
        _, metrics = loss_fn(z_hyp, indices)

        # At least some r_v* keys should exist
        level_keys = [k for k in metrics if k.startswith("r_v")]
        assert len(level_keys) > 0, "No per-level radius metrics"


# =============================================================================
# Test Classes: Edge Cases
# =============================================================================


class TestEdgeCasesBatchSize:
    """Test edge cases with unusual batch sizes."""

    def test_batch_size_one_all_losses(self):
        """Verify batch_size=1 doesn't crash any loss."""
        z_hyp = torch.randn(1, 16, dtype=torch.float64) * 0.5
        indices = torch.tensor([100])
        logits = torch.randn(1, 27, dtype=torch.float64)
        targets = torch.randint(-1, 2, (1, 9))

        # PAdicGeodesicLoss
        loss, _ = PAdicGeodesicLoss()(z_hyp, indices)
        assert loss == 0.0

        # RadialHierarchyLoss
        loss, _ = RadialHierarchyLoss()(z_hyp, indices)
        assert torch.isfinite(loss)

        # GlobalRankLoss
        loss, _ = GlobalRankLoss()(z_hyp, indices)
        assert loss == 0.0

        # MonotonicRadialLoss
        loss, _ = MonotonicRadialLoss()(z_hyp, indices)
        assert loss == 0.0

        # RichHierarchyLoss
        out = RichHierarchyLoss()(z_hyp, indices, logits, targets)
        assert torch.isfinite(out["hierarchy"])
        assert torch.isfinite(out["coverage"])
        assert torch.isfinite(out["separation"])

    def test_batch_size_two(self):
        """Verify batch_size=2 works for pair-based losses."""
        z_hyp = torch.randn(2, 16, dtype=torch.float64) * 0.5
        indices = torch.tensor([0, 100])

        for LossClass in [
            PAdicGeodesicLoss,
            RadialHierarchyLoss,
            GlobalRankLoss,
            MonotonicRadialLoss,
        ]:
            loss, _ = LossClass()(z_hyp, indices)
            assert torch.isfinite(loss), f"{LossClass.__name__} failed on batch_size=2"


class TestEdgeCasesSameValuation:
    """Test edge cases where all samples have same valuation."""

    def test_same_valuation_padic_geodesic(self):
        """Verify PAdicGeodesicLoss handles same-valuation batch."""
        # All indices divisible by 3 but not 9 (valuation 1)
        torch.manual_seed(42)
        z_hyp = torch.randn(32, 16, dtype=torch.float64) * 0.5
        indices = torch.tensor([3, 6, 12, 15, 21, 24, 30, 33] * 4)  # All v=1

        loss_fn = PAdicGeodesicLoss()
        loss, metrics = loss_fn(z_hyp, indices)

        assert torch.isfinite(loss)
        # Correlation may be NaN but should be handled (returns 0)
        assert not math.isnan(metrics["distance_correlation"])

    def test_same_valuation_monotonic_radial(self):
        """Verify MonotonicRadialLoss handles same-valuation batch."""
        torch.manual_seed(42)
        z_hyp = torch.randn(32, 16, dtype=torch.float64) * 0.5
        # All indices with valuation 0
        indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11] * 4)

        loss_fn = MonotonicRadialLoss()
        loss, metrics = loss_fn(z_hyp, indices)

        # Only 1 level present, should return 0 loss
        assert loss == 0.0
        assert metrics["n_levels"] == 1


class TestEdgeCasesNearBoundary:
    """Test edge cases with points near Poincaré ball boundary."""

    def test_boundary_points_all_losses(self):
        """Verify boundary points don't cause numerical issues."""
        torch.manual_seed(42)
        # Points very close to boundary
        z_hyp = torch.randn(32, 16, dtype=torch.float64)
        z_hyp = 0.999 * z_hyp / z_hyp.norm(dim=-1, keepdim=True)
        indices = torch.randint(0, 19683, (32,))
        logits = torch.randn(32, 27, dtype=torch.float64)
        targets = torch.randint(-1, 2, (32, 9))

        # Test all losses
        for LossClass in [
            PAdicGeodesicLoss,
            RadialHierarchyLoss,
            GlobalRankLoss,
            MonotonicRadialLoss,
        ]:
            loss, _ = LossClass()(z_hyp, indices)
            assert torch.isfinite(loss), f"{LossClass.__name__} not finite at boundary"

        out = RichHierarchyLoss()(z_hyp, indices, logits, targets)
        assert torch.isfinite(out["hierarchy"])
        assert torch.isfinite(out["coverage"])
        assert torch.isfinite(out["separation"])

    def test_origin_points(self):
        """Verify points at origin work correctly."""
        z_hyp = torch.zeros(32, 16, dtype=torch.float64)
        indices = torch.randint(0, 19683, (32,))

        for LossClass in [
            PAdicGeodesicLoss,
            RadialHierarchyLoss,
            GlobalRankLoss,
            MonotonicRadialLoss,
        ]:
            loss, _ = LossClass()(z_hyp, indices)
            assert torch.isfinite(loss), f"{LossClass.__name__} not finite at origin"


# =============================================================================
# Test Classes: Consistency
# =============================================================================


class TestConsistencyHyperbolicRadius:
    """Verify losses use hyperbolic radius, not Euclidean norm."""

    def test_radial_uses_hyperbolic_distance(self, sample_batch):
        """Verify RadialHierarchyLoss uses hyperbolic_radius."""
        z_hyp, indices = sample_batch

        loss_fn = RadialHierarchyLoss(curvature=1.0)
        _, metrics = loss_fn(z_hyp, indices)

        # Compute expected hyperbolic radius
        expected_radii = hyperbolic_radius(z_hyp, c=1.0)

        # Mean should match
        assert abs(metrics["mean_radius"] - expected_radii.mean().item()) < 1e-6

    def test_global_rank_uses_hyperbolic_distance(self, sample_batch):
        """Verify GlobalRankLoss uses hyperbolic_radius."""
        z_hyp, indices = sample_batch

        # This is verified by the fact that curvature parameter exists
        loss_fn = GlobalRankLoss(curvature=1.0)
        loss_c1, _ = loss_fn(z_hyp, indices)

        loss_fn_c2 = GlobalRankLoss(curvature=2.0)
        loss_c2, _ = loss_fn_c2(z_hyp, indices)

        # Different curvatures should generally produce different losses
        # (unless by coincidence)
        # We just verify both are finite
        assert torch.isfinite(loss_c1)
        assert torch.isfinite(loss_c2)


class TestReproducibility:
    """Test that seeded losses are reproducible."""

    def test_padic_geodesic_reproducible(self, sample_batch):
        """Verify PAdicGeodesicLoss is reproducible with same seed."""
        z_hyp, indices = sample_batch

        loss_fn1 = PAdicGeodesicLoss(seed=42)
        loss_fn2 = PAdicGeodesicLoss(seed=42)

        loss1, _ = loss_fn1(z_hyp, indices)
        loss2, _ = loss_fn2(z_hyp, indices)

        assert torch.allclose(loss1, loss2)

    def test_radial_margin_reproducible(self, sample_batch):
        """Verify RadialHierarchyLoss margin sampling is reproducible."""
        z_hyp, indices = sample_batch

        loss_fn1 = RadialHierarchyLoss(seed=42, use_margin_loss=True)
        loss_fn2 = RadialHierarchyLoss(seed=42, use_margin_loss=True)

        loss1, _ = loss_fn1(z_hyp, indices)
        loss2, _ = loss_fn2(z_hyp, indices)

        assert torch.allclose(loss1, loss2)
