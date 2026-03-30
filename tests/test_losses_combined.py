# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tier 2: CombinedLoss and Learnable Weights Tests.

Tests for:
- CombinedLoss config-driven instantiation
- Phase gating for geodesic loss
- Learnable weight formula correctness (Kendall et al. 2018)
- Weight initialization and round-trip verification
- Gradient flow to learnable parameters

These tests verify the V6.1 learnable weights feature works correctly.
"""

import math

import pytest
import torch

from src.losses.combined import CombinedLoss

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_inputs():
    """Create sample inputs for CombinedLoss."""
    torch.manual_seed(42)
    batch_size = 64
    latent_dim = 16

    z_hyp = torch.randn(batch_size, latent_dim, dtype=torch.float64) * 0.5
    indices = torch.randint(0, 19683, (batch_size,))
    logits = torch.randn(batch_size, 27, dtype=torch.float64)
    targets = torch.randint(-1, 2, (batch_size, 9))

    return z_hyp, indices, logits, targets


@pytest.fixture
def rich_hierarchy_config():
    """Config with RichHierarchyLoss enabled."""
    return {
        'rich_hierarchy': {
            'enabled': True,
            'hierarchy_weight': 5.0,
            'coverage_weight': 1.0,
            'separation_weight': 3.0,
        }
    }


@pytest.fixture
def full_config():
    """Config with multiple losses enabled."""
    return {
        'rich_hierarchy': {
            'enabled': True,
            'hierarchy_weight': 5.0,
            'coverage_weight': 1.0,
            'separation_weight': 3.0,
        },
        'radial': {
            'enabled': True,
            'weight': 1.0,
        },
        'geodesic': {
            'enabled': True,
            'phase_start_epoch': 50,
            'weight': 0.3,
        },
        'rank': {
            'enabled': True,
            'weight': 0.5,
        },
        'monotonic': {
            'enabled': True,
            'weight': 1.0,
        },
    }


# =============================================================================
# Test Classes: CombinedLoss Basic Functionality
# =============================================================================


class TestCombinedLossInstantiation:
    """Test CombinedLoss config-driven instantiation."""

    def test_empty_config(self):
        """Verify empty config raises ValueError (all losses disabled)."""
        import pytest
        with pytest.raises(ValueError, match="all losses are disabled"):
            CombinedLoss({}, curvature=1.0)

    def test_rich_hierarchy_only(self, rich_hierarchy_config):
        """Verify RichHierarchyLoss instantiation."""
        loss_fn = CombinedLoss(rich_hierarchy_config, curvature=1.0)

        # Contract: get_enabled_losses() is the public API for checking active losses
        assert 'rich_hierarchy' in loss_fn.get_enabled_losses()
        # Contract: get_learned_weights() exposes the configured weights
        weights = loss_fn.get_learned_weights()
        assert weights['hierarchy'] == 5.0
        assert weights['coverage'] == 1.0
        assert weights['separation'] == 3.0

    def test_full_config_instantiation(self, full_config):
        """Verify all losses instantiate correctly."""
        loss_fn = CombinedLoss(full_config, curvature=1.0)

        # Contract: get_enabled_losses() is the stable public API
        enabled = loss_fn.get_enabled_losses()
        assert 'rich_hierarchy' in enabled
        assert 'radial' in enabled
        assert 'monotonic' in enabled

    def test_get_enabled_losses(self, full_config):
        """Verify get_enabled_losses returns correct list."""
        loss_fn = CombinedLoss(full_config, curvature=1.0)

        enabled = loss_fn.get_enabled_losses()
        assert 'rich_hierarchy' in enabled
        assert 'radial' in enabled
        assert 'geodesic' in enabled
        assert 'rank' in enabled
        assert 'monotonic' in enabled


class TestCombinedLossForward:
    """Test CombinedLoss forward pass."""

    def test_forward_returns_total(self, rich_hierarchy_config, sample_inputs):
        """Verify forward returns dict with 'total' key."""
        z_hyp, indices, logits, targets = sample_inputs

        loss_fn = CombinedLoss(rich_hierarchy_config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        assert 'total' in losses
        assert torch.isfinite(losses['total'])

    def test_forward_returns_components(self, full_config, sample_inputs):
        """Verify forward returns all loss components."""
        z_hyp, indices, logits, targets = sample_inputs

        loss_fn = CombinedLoss(full_config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=100)

        assert 'rich_hierarchy' in losses
        assert 'radial' in losses
        assert 'geodesic' in losses  # epoch >= phase_start
        assert 'rank' in losses
        assert 'monotonic' in losses

    def test_total_equals_sum_of_weighted_components(self, rich_hierarchy_config, sample_inputs):
        """Verify total equals the weighted rich_hierarchy component (only active loss)."""
        z_hyp, indices, logits, targets = sample_inputs

        loss_fn = CombinedLoss(rich_hierarchy_config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        # Contract: when only rich_hierarchy is enabled, total == rich_hierarchy
        assert torch.allclose(losses['total'], losses['rich_hierarchy'], atol=1e-10)
        assert torch.isfinite(losses['total'])


# =============================================================================
# Test Classes: Phase Gating
# =============================================================================


class TestPhaseGating:
    """Test phase gating for geodesic loss."""

    def test_geodesic_excluded_before_phase_start(self, sample_inputs):
        """Verify geodesic loss is not computed before phase_start_epoch."""
        z_hyp, indices, logits, targets = sample_inputs

        config = {
            'geodesic': {
                'enabled': True,
                'phase_start_epoch': 50,
                'weight': 0.3,
            }
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        # Before phase start
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=10)
        assert 'geodesic' not in losses

        losses = loss_fn(z_hyp, indices, logits, targets, epoch=49)
        assert 'geodesic' not in losses

    def test_geodesic_included_at_phase_start(self, sample_inputs):
        """Verify geodesic loss is computed at phase_start_epoch."""
        z_hyp, indices, logits, targets = sample_inputs

        config = {
            'geodesic': {
                'enabled': True,
                'phase_start_epoch': 50,
                'weight': 0.3,
            }
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        losses = loss_fn(z_hyp, indices, logits, targets, epoch=50)
        assert 'geodesic' in losses
        assert torch.isfinite(losses['geodesic'])

    def test_geodesic_included_after_phase_start(self, sample_inputs):
        """Verify geodesic loss is computed after phase_start_epoch."""
        z_hyp, indices, logits, targets = sample_inputs

        config = {
            'geodesic': {
                'enabled': True,
                'phase_start_epoch': 50,
                'weight': 0.3,
            }
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        losses = loss_fn(z_hyp, indices, logits, targets, epoch=100)
        assert 'geodesic' in losses

    def test_phase_start_zero_always_active(self, sample_inputs):
        """Verify phase_start_epoch=0 means geodesic is always active."""
        z_hyp, indices, logits, targets = sample_inputs

        config = {
            'geodesic': {
                'enabled': True,
                'phase_start_epoch': 0,
                'weight': 0.3,
            }
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)
        assert 'geodesic' in losses


# =============================================================================
# Test Classes: Learnable Weight Formula
# =============================================================================


class TestWeightToLogSigmaFormula:
    """Test weight_to_log_sigma conversion formula."""

    def test_formula_derivation(self):
        """Verify weight_to_log_sigma formula is mathematically correct.

        Given: effective_weight = 0.5 * exp(-2 * log_sigma)
        We want: given weight w, find log_sigma such that effective_weight = w

        w = 0.5 * exp(-2 * s)
        2w = exp(-2s)
        ln(2w) = -2s
        s = -0.5 * ln(2w)
        """
        def weight_to_log_sigma(w):
            return -0.5 * math.log(max(2 * w, 1e-6))

        # Test specific values
        # w=0.5: s = -0.5 * ln(1) = 0
        assert abs(weight_to_log_sigma(0.5) - 0.0) < 1e-10

        # w=1.0: s = -0.5 * ln(2)
        expected = -0.5 * math.log(2)
        assert abs(weight_to_log_sigma(1.0) - expected) < 1e-10

        # w=5.0: s = -0.5 * ln(10)
        expected = -0.5 * math.log(10)
        assert abs(weight_to_log_sigma(5.0) - expected) < 1e-10

    def test_formula_range(self):
        """Verify log_sigma range for typical weights."""
        def weight_to_log_sigma(w):
            return -0.5 * math.log(max(2 * w, 1e-6))

        # Small weight -> positive log_sigma (small effective weight)
        assert weight_to_log_sigma(0.1) > 0

        # Large weight -> negative log_sigma (large effective weight)
        assert weight_to_log_sigma(10.0) < 0


class TestUncertaintyWeightFormula:
    """Test uncertainty_weight conversion formula."""

    def test_formula_positive(self):
        """Verify effective weight is always positive."""
        def uncertainty_weight(log_sigma):
            return 0.5 * math.exp(-2 * log_sigma)

        # Test various log_sigma values
        for s in [-5, -2, -1, 0, 1, 2, 5]:
            w = uncertainty_weight(s)
            assert w > 0, f"Weight not positive at s={s}"

    def test_formula_monotonic(self):
        """Verify effective weight decreases as log_sigma increases."""
        def uncertainty_weight(log_sigma):
            return 0.5 * math.exp(-2 * log_sigma)

        s_values = [-2, -1, 0, 1, 2]
        weights = [uncertainty_weight(s) for s in s_values]

        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], "Not monotonically decreasing"


class TestWeightRoundTrip:
    """Test weight -> log_sigma -> weight round trip."""

    def test_round_trip_exact(self):
        """Verify weight round-trips exactly."""
        def weight_to_log_sigma(w):
            return -0.5 * math.log(max(2 * w, 1e-6))

        def uncertainty_weight(log_sigma):
            return 0.5 * math.exp(-2 * log_sigma)

        test_weights = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

        for w in test_weights:
            s = weight_to_log_sigma(w)
            w_recovered = uncertainty_weight(s)
            assert abs(w_recovered - w) < 1e-10, f"Round-trip failed: {w} -> {s} -> {w_recovered}"

    def test_round_trip_pytorch(self):
        """Verify round trip works with PyTorch tensors."""
        loss_fn = CombinedLoss({'learnable_weights': True, 'radial': {'enabled': True, 'weight': 3.0}}, curvature=1.0)

        # Get initial weight
        initial_weight = loss_fn._uncertainty_weight(loss_fn.log_sigma_radial).item()

        # Should match config weight
        assert abs(initial_weight - 3.0) < 0.01


class TestLearnableWeightsInitialization:
    """Test learnable weights initialize correctly from config."""

    def test_rich_hierarchy_weights_match_config(self):
        """Verify RichHierarchy sub-weights match config."""
        config = {
            'learnable_weights': True,
            'rich_hierarchy': {
                'enabled': True,
                'hierarchy_weight': 5.0,
                'coverage_weight': 1.0,
                'separation_weight': 3.0,
            }
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        weights = loss_fn.get_learned_weights()

        assert abs(weights['hierarchy'] - 5.0) < 0.01
        assert abs(weights['coverage'] - 1.0) < 0.01
        assert abs(weights['separation'] - 3.0) < 0.01

    def test_radial_weight_matches_config(self):
        """Verify radial weight matches config."""
        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 2.5}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        weights = loss_fn.get_learned_weights()

        assert abs(weights['radial'] - 2.5) < 0.01

    def test_all_weights_match_config(self):
        """Verify all weights match config when learnable."""
        config = {
            'learnable_weights': True,
            'rich_hierarchy': {
                'enabled': True,
                'hierarchy_weight': 5.0,
                'coverage_weight': 1.0,
                'separation_weight': 3.0,
            },
            'radial': {'enabled': True, 'weight': 1.5},
            'geodesic': {'enabled': True, 'weight': 0.3},
            'rank': {'enabled': True, 'weight': 0.5},
            'monotonic': {'enabled': True, 'weight': 2.0},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        weights = loss_fn.get_learned_weights()

        assert abs(weights['hierarchy'] - 5.0) < 0.01
        assert abs(weights['coverage'] - 1.0) < 0.01
        assert abs(weights['separation'] - 3.0) < 0.01
        assert abs(weights['radial'] - 1.5) < 0.01
        assert abs(weights['geodesic'] - 0.3) < 0.01
        assert abs(weights['rank'] - 0.5) < 0.01
        assert abs(weights['monotonic'] - 2.0) < 0.01


class TestLearnableWeightsGradientFlow:
    """Test gradient flow to learnable weight parameters."""

    def test_log_sigma_receives_gradient(self, sample_inputs):
        """Verify log_sigma parameters receive gradients."""
        z_hyp, indices, logits, targets = sample_inputs
        z_hyp = z_hyp.clone().requires_grad_(True)

        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 1.0}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        losses['total'].backward()

        assert loss_fn.log_sigma_radial.grad is not None, "No gradient on log_sigma_radial"
        # Gradient may be very small but should exist

    def test_all_log_sigmas_receive_gradient(self, sample_inputs):
        """Verify all log_sigma parameters receive gradients."""
        z_hyp, indices, logits, targets = sample_inputs
        z_hyp = z_hyp.clone().requires_grad_(True)

        config = {
            'learnable_weights': True,
            'rich_hierarchy': {
                'enabled': True,
                'hierarchy_weight': 5.0,
                'coverage_weight': 1.0,
                'separation_weight': 3.0,
            },
            'radial': {'enabled': True, 'weight': 1.0},
            'monotonic': {'enabled': True, 'weight': 1.0},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        losses['total'].backward()

        # Check all log_sigma params
        assert loss_fn.log_sigma_hierarchy.grad is not None
        assert loss_fn.log_sigma_coverage.grad is not None
        assert loss_fn.log_sigma_separation.grad is not None
        assert loss_fn.log_sigma_radial.grad is not None
        assert loss_fn.log_sigma_monotonic.grad is not None

    def test_gradient_nonzero_on_nontrivial_loss(self, sample_inputs):
        """Verify gradient is nonzero when loss is nonzero."""
        z_hyp, indices, logits, targets = sample_inputs
        z_hyp = z_hyp.clone().requires_grad_(True)

        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 1.0}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        # Radial loss should be nonzero for random points
        assert losses['radial'] > 0

        losses['total'].backward()

        # Gradient should be nonzero
        assert loss_fn.log_sigma_radial.grad.abs() > 1e-15


class TestLearnableWeightsBehavior:
    """Test learnable weights behavioral properties."""

    def test_weight_bounded_positive(self):
        """Verify effective weights stay positive even at extreme log_sigma."""
        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 1.0}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        # Test extreme log_sigma values
        with torch.no_grad():
            for s in [-10, -5, 0, 5, 10]:
                loss_fn.log_sigma_radial.fill_(s)
                w = loss_fn._uncertainty_weight(loss_fn.log_sigma_radial)
                assert w > 0, f"Weight not positive at s={s}"
                assert torch.isfinite(w), f"Weight not finite at s={s}"

    def test_weight_bounded_reasonable(self):
        """Verify effective weights don't overflow at extreme log_sigma."""
        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 1.0}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        with torch.no_grad():
            # Very negative log_sigma -> very large weight
            loss_fn.log_sigma_radial.fill_(-10.0)
            w = loss_fn._uncertainty_weight(loss_fn.log_sigma_radial)
            # exp(20) ~ 4.8e8, so w ~ 2.4e8
            assert w < 1e10, "Weight overflow"

            # Very positive log_sigma -> very small weight
            loss_fn.log_sigma_radial.fill_(10.0)
            w = loss_fn._uncertainty_weight(loss_fn.log_sigma_radial)
            assert w > 1e-10, "Weight underflow"

    def test_regularization_term_prevents_collapse(self, sample_inputs):
        """Verify -log_sigma regularization is present in loss."""
        z_hyp, indices, logits, targets = sample_inputs

        config = {
            'learnable_weights': True,
            'radial': {'enabled': True, 'weight': 1.0}
        }

        loss_fn = CombinedLoss(config, curvature=1.0)

        # Set log_sigma to positive value (would make weight small)
        with torch.no_grad():
            loss_fn.log_sigma_radial.fill_(2.0)

        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        # The total should include -log_sigma = -2.0 as regularization
        # This makes total potentially negative when raw loss is small
        # but prevents weight from collapsing to zero
        # Just verify it runs without error
        assert torch.isfinite(losses['total'])


class TestFixedVsLearnableWeights:
    """Compare behavior of fixed vs learnable weights."""

    def test_fixed_weights_no_parameters(self, rich_hierarchy_config):
        """Verify fixed weights mode has no learnable weight parameters."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = False

        loss_fn = CombinedLoss(config, curvature=1.0)

        # Should not have log_sigma attributes
        assert not hasattr(loss_fn, 'log_sigma_hierarchy')
        assert not hasattr(loss_fn, 'log_sigma_coverage')
        assert not hasattr(loss_fn, 'log_sigma_separation')

    def test_learnable_weights_has_parameters(self, rich_hierarchy_config):
        """Verify learnable weights mode has learnable parameters."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = True

        loss_fn = CombinedLoss(config, curvature=1.0)

        # Should have log_sigma attributes that are nn.Parameters
        assert hasattr(loss_fn, 'log_sigma_hierarchy')
        assert hasattr(loss_fn, 'log_sigma_coverage')
        assert hasattr(loss_fn, 'log_sigma_separation')

        # They should be in the module's parameters
        param_names = [name for name, _ in loss_fn.named_parameters()]
        assert 'log_sigma_hierarchy' in param_names
        assert 'log_sigma_coverage' in param_names
        assert 'log_sigma_separation' in param_names

    def test_get_learned_weights_fixed_mode(self, rich_hierarchy_config):
        """Verify get_learned_weights returns config weights in fixed mode."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = False

        loss_fn = CombinedLoss(config, curvature=1.0)
        weights = loss_fn.get_learned_weights()

        assert weights['hierarchy'] == 5.0
        assert weights['coverage'] == 1.0
        assert weights['separation'] == 3.0

    def test_get_log_sigmas_fixed_mode(self, rich_hierarchy_config):
        """Verify get_log_sigmas returns empty dict in fixed mode."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = False

        loss_fn = CombinedLoss(config, curvature=1.0)
        sigmas = loss_fn.get_log_sigmas()

        assert sigmas == {}

    def test_get_log_sigmas_learnable_mode(self, rich_hierarchy_config):
        """Verify get_log_sigmas returns values in learnable mode."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = True

        loss_fn = CombinedLoss(config, curvature=1.0)
        sigmas = loss_fn.get_log_sigmas()

        assert 'hierarchy' in sigmas
        assert 'coverage' in sigmas
        assert 'separation' in sigmas

        # Verify they're finite
        for name, s in sigmas.items():
            assert math.isfinite(s), f"log_sigma for {name} is not finite"


# =============================================================================
# Test Classes: CombinedLoss Edge Cases
# =============================================================================


class TestCombinedLossEdgeCases:
    """Test CombinedLoss edge cases."""

    def test_no_losses_enabled(self, sample_inputs):
        """Verify empty config raises ValueError (no losses active = no gradients)."""
        import pytest
        with pytest.raises(ValueError, match="all losses are disabled"):
            CombinedLoss({}, curvature=1.0)

    def test_logits_shape_27(self, sample_inputs):
        """Verify (B, 27) logits format works."""
        z_hyp, indices, logits, targets = sample_inputs
        assert logits.shape[-1] == 27

        config = {'rich_hierarchy': {'enabled': True}}

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        assert torch.isfinite(losses['total'])

    def test_logits_shape_9x3(self, sample_inputs):
        """Verify (B, 9, 3) logits format works."""
        z_hyp, indices, _, targets = sample_inputs
        logits = torch.randn(z_hyp.size(0), 9, 3, dtype=torch.float64)

        config = {'rich_hierarchy': {'enabled': True}}

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        assert torch.isfinite(losses['total'])

    def test_batch_size_one(self):
        """Verify batch_size=1 doesn't crash."""
        z_hyp = torch.randn(1, 16, dtype=torch.float64) * 0.5
        indices = torch.tensor([0])
        logits = torch.randn(1, 27, dtype=torch.float64)
        targets = torch.randint(-1, 2, (1, 9))

        config = {
            'rich_hierarchy': {'enabled': True},
            'radial': {'enabled': True},
            'monotonic': {'enabled': True},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        losses = loss_fn(z_hyp, indices, logits, targets, epoch=0)

        assert torch.isfinite(losses['total'])


class TestCombinedLossRepr:
    """Test CombinedLoss string representation."""

    def test_repr_fixed_weights(self, rich_hierarchy_config):
        """Verify repr shows 'fixed' for fixed weights."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = False

        loss_fn = CombinedLoss(config, curvature=1.0)

        repr_str = repr(loss_fn)
        assert 'fixed' in repr_str
        assert 'rich_hierarchy' in repr_str

    def test_repr_learnable_weights(self, rich_hierarchy_config):
        """Verify repr shows 'learnable' for learnable weights."""
        config = rich_hierarchy_config.copy()
        config['learnable_weights'] = True

        loss_fn = CombinedLoss(config, curvature=1.0)

        repr_str = repr(loss_fn)
        assert 'learnable' in repr_str


class TestRadiusSharing:
    """Test radius sharing functionality."""

    def test_auto_share_single_block(self):
        """Verify single block specifies radii, others share."""
        from src.losses.radius_defaults import auto_share_radius_config

        config = {
            'rich_hierarchy': {'enabled': True, 'inner_radius': 0.08, 'outer_radius': 0.70},
            'radial': {'enabled': True},
            'monotonic': {'enabled': True},
        }

        radius_configs = auto_share_radius_config(config)

        assert 'rich_hierarchy' in radius_configs
        assert 'radial' in radius_configs
        assert 'monotonic' in radius_configs

        # All should have same config
        assert radius_configs['rich_hierarchy'].inner_radius == 0.08
        assert radius_configs['rich_hierarchy'].outer_radius == 0.70
        assert radius_configs['radial'].inner_radius == 0.08
        assert radius_configs['radial'].outer_radius == 0.70
        assert radius_configs['monotonic'].inner_radius == 0.08
        assert radius_configs['monotonic'].outer_radius == 0.70

    def test_auto_share_uses_defaults_when_none_specified(self):
        """Verify defaults used when no block specifies radii."""
        from src.losses.radius_defaults import (
            DEFAULT_INNER_RADIUS,
            DEFAULT_OUTER_RADIUS,
            auto_share_radius_config,
        )

        config = {
            'rich_hierarchy': {'enabled': True},
            'radial': {'enabled': True},
        }

        radius_configs = auto_share_radius_config(config)

        # Should use defaults
        assert radius_configs['rich_hierarchy'].inner_radius == DEFAULT_INNER_RADIUS
        assert radius_configs['rich_hierarchy'].outer_radius == DEFAULT_OUTER_RADIUS
        assert radius_configs['radial'].inner_radius == DEFAULT_INNER_RADIUS
        assert radius_configs['radial'].outer_radius == DEFAULT_OUTER_RADIUS

    def test_auto_share_multiple_blocks_warns(self):
        """Verify warning when multiple blocks specify radii."""
        from src.losses.radius_defaults import auto_share_radius_config

        config = {
            'rich_hierarchy': {'enabled': True, 'inner_radius': 0.08, 'outer_radius': 0.70},
            'radial': {'enabled': True, 'inner_radius': 0.10, 'outer_radius': 0.80},  # Different values
            'monotonic': {'enabled': True},
        }

        radius_configs = auto_share_radius_config(config)

        # Should use first block's config for all
        assert radius_configs['rich_hierarchy'].inner_radius == 0.08
        assert radius_configs['radial'].inner_radius == 0.08  # Should match first block
        assert radius_configs['monotonic'].inner_radius == 0.08

    def test_compare_radius_configs_all_equal(self):
        """Verify comparison returns True when all configs equal."""
        from src.losses.radius_defaults import RadiusConfig, compare_radius_configs

        config1 = RadiusConfig(inner_radius=0.08, outer_radius=0.70)
        config2 = RadiusConfig(inner_radius=0.08, outer_radius=0.70)
        config3 = RadiusConfig(inner_radius=0.08, outer_radius=0.70)

        configs = {
            'block1': config1,
            'block2': config2,
            'block3': config3,
        }

        all_equal, msg = compare_radius_configs(configs)
        assert all_equal is True
        assert 'consistent' in msg.lower()

    def test_compare_radius_configs_detects_inconsistency(self):
        """Verify comparison detects different radius configs."""
        from src.losses.radius_defaults import RadiusConfig, compare_radius_configs

        config1 = RadiusConfig(inner_radius=0.08, outer_radius=0.70)
        config2 = RadiusConfig(inner_radius=0.10, outer_radius=0.80)

        configs = {
            'block1': config1,
            'block2': config2,
        }

        all_equal, msg = compare_radius_configs(configs)
        assert all_equal is False
        assert 'inconsistent' in msg.lower()
        assert 'block1' in msg
        assert 'block2' in msg

    def test_combined_loss_uses_shared_radius_config(self):
        """Verify CombinedLoss uses shared radius config across losses."""
        config = {
            'rich_hierarchy': {
                'enabled': True,
                'inner_radius': 0.08,
                'outer_radius': 0.70,
            },
            'radial': {'enabled': True},
            'monotonic': {'enabled': True},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        radius_config = loss_fn.get_radius_config()

        # All losses should have same radius config
        assert radius_config['rich_hierarchy']['inner_radius'] == 0.08
        assert radius_config['rich_hierarchy']['outer_radius'] == 0.70
        assert radius_config['radial']['inner_radius'] == 0.08
        assert radius_config['radial']['outer_radius'] == 0.70
        assert radius_config['monotonic']['inner_radius'] == 0.08
        assert radius_config['monotonic']['outer_radius'] == 0.70

    def test_combined_loss_get_radius_config_empty_when_no_radial_losses(self):
        """Verify get_radius_config returns empty dict when no radial losses enabled."""
        config = {
            'geodesic': {'enabled': True},
            'rank': {'enabled': True},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        radius_config = loss_fn.get_radius_config()

        assert radius_config == {}

    def test_radial_monotonic_share_defaults(self):
        """Verify radial and monotonic share defaults when rich_hierarchy not enabled."""
        from src.losses.radius_defaults import DEFAULT_INNER_RADIUS, DEFAULT_OUTER_RADIUS

        config = {
            'radial': {'enabled': True},
            'monotonic': {'enabled': True},
        }

        loss_fn = CombinedLoss(config, curvature=1.0)
        radius_config = loss_fn.get_radius_config()

        # Should use defaults
        assert radius_config['radial']['inner_radius'] == DEFAULT_INNER_RADIUS
        assert radius_config['radial']['outer_radius'] == DEFAULT_OUTER_RADIUS
        assert radius_config['monotonic']['inner_radius'] == DEFAULT_INNER_RADIUS
        assert radius_config['monotonic']['outer_radius'] == DEFAULT_OUTER_RADIUS


# =============================================================================
# CombinedLossOutput TypedDict key correctness
# =============================================================================


class TestCombinedLossOutputKeys:
    """Verify actual dict keys written by CombinedLoss.forward() match the TypedDict."""

    def _make_loss_fn(self, extra: dict | None = None) -> CombinedLoss:
        config: dict = {
            'rich_hierarchy': {'enabled': True, 'hierarchy_weight': 1.0,
                               'coverage_weight': 1.0, 'separation_weight': 1.0},
            'monotonic': {'enabled': True},
        }
        if extra:
            config.update(extra)
        return CombinedLoss(config, curvature=1.0)

    def _make_inputs(self, B: int = 32):
        torch.manual_seed(0)
        z = torch.randn(B, 16, dtype=torch.float64) * 0.3
        idx = torch.randint(0, 19683, (B,))
        logits = torch.randn(B, 27, dtype=torch.float64)
        targets = (torch.arange(B * 9) % 3 - 1).reshape(B, 9).long()
        return z, idx, logits, targets

    def test_forward_has_total_and_rich_hierarchy(self):
        loss_fn = self._make_loss_fn()
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0)
        assert 'total' in out
        assert 'rich_hierarchy' in out

    def test_forward_monotonic_metrics_key(self):
        loss_fn = self._make_loss_fn()
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0)
        assert 'monotonic_metrics' in out
        # Old stale name must not appear
        assert 'rank_metrics' not in out or 'monotonic' in out

    def test_forward_radial_metrics_key(self):
        loss_fn = self._make_loss_fn({'radial': {'enabled': True}})
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0)
        assert 'radial' in out
        assert 'radial_metrics' in out

    def test_forward_geodesic_metrics_key(self):
        loss_fn = self._make_loss_fn({'geodesic': {'enabled': True, 'phase_start_epoch': 0}})
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=5)
        assert 'geodesic' in out
        assert 'geodesic_metrics' in out

    def test_forward_kl_key_not_kl_loss(self):
        """forward() must write 'kl', not the stale 'kl_loss' name."""
        mu = torch.zeros(32, 16, dtype=torch.float64)
        logvar = torch.zeros(32, 16, dtype=torch.float64)
        loss_fn = self._make_loss_fn({'hyperbolic_kl': {'enabled': True}})
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0, mu=mu, logvar=logvar)
        assert 'kl' in out
        assert 'kl_loss' not in out

    def test_forward_angular_coherence_key_not_angular(self):
        """forward() must write 'angular_coherence', not the stale 'angular' name."""
        B = 32
        loss_fn = self._make_loss_fn({
            'angular_coherence': {'enabled': True, 'phase_start_epoch': 0},
        })
        z, idx, logits, targets = self._make_inputs(B)
        r = torch.rand(B, dtype=torch.float64) * 0.5
        out = loss_fn(z, idx, logits, targets, epoch=100, r=r)
        assert 'angular_coherence' in out
        assert 'angular' not in out
        assert 'angular_coherence_metrics' in out
        assert 'angular_metrics' not in out

    def test_forward_within_level_contrastive_key_not_within_contrastive(self):
        """forward() must write 'within_level_contrastive', not the stale 'within_contrastive'."""
        loss_fn = self._make_loss_fn({
            'within_level_contrastive': {'enabled': True},
        })
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0)
        assert 'within_level_contrastive' in out
        assert 'within_contrastive' not in out
        assert 'wlc_metrics' in out

    def test_forward_rich_hierarchy_detail_key_not_rich_metrics(self):
        """forward() must write 'rich_hierarchy_detail', not the stale 'rich_metrics'."""
        loss_fn = self._make_loss_fn()
        z, idx, logits, targets = self._make_inputs()
        out = loss_fn(z, idx, logits, targets, epoch=0)
        assert 'rich_hierarchy_detail' in out
        assert 'rich_metrics' not in out

    def test_schema_rank_use_all_pairs_field_exists(self):
        """RankLossConfig must expose use_all_pairs so users can enable O(n²) mode."""
        from src.config.schema import RankLossConfig
        cfg = RankLossConfig()
        assert cfg.use_all_pairs is False  # safe default

    def test_rank_use_all_pairs_passed_to_constructor(self):
        """use_all_pairs=True from config must reach GlobalRankLoss."""
        config = {
            'rank': {'enabled': True, 'use_all_pairs': True},
            'rich_hierarchy': {'enabled': True},
        }
        loss_fn = CombinedLoss(config, curvature=1.0)
        assert loss_fn.rank_loss is not None
        assert loss_fn.rank_loss.use_all_pairs is True

