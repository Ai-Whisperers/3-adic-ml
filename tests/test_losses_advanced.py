# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Advanced Loss Function Tests.

Targeted tests for mathematically complex loss modules:
- ValuationPriorLoss (prior.py)
- AngularCoherenceLoss, AlgebraicCoherenceLoss, AlgebraicAdditionLoss (algebraic.py)
- LagrangianDualState (lagrangian.py)
"""

from unittest.mock import MagicMock

import pytest
import torch

from src.losses.algebraic import (
    AlgebraicAdditionLoss,
    AlgebraicCoherenceLoss,
    AlgebraicDistributiveLoss,
    AlgebraicMultiplicationLoss,
    AngularCoherenceLoss,
)
from src.losses.lagrangian import LagrangianDualState
from src.losses.prior import ValuationPriorLoss

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample tensors for testing."""
    torch.manual_seed(42)
    batch_size = 32
    latent_dim = 16

    mu = torch.randn(batch_size, latent_dim, dtype=torch.float64)
    logvar = torch.randn(batch_size, latent_dim, dtype=torch.float64) * 0.1
    indices = torch.randint(0, 1000, (batch_size,))

    # Unit vectors for direction-based losses
    r = torch.norm(mu, dim=-1, keepdim=True)
    z_hyp = 0.5 * (mu / r.clamp(min=1e-6))

    return mu, logvar, indices, z_hyp, r.squeeze(-1)


# =============================================================================
# ValuationPriorLoss Tests
# =============================================================================

class TestValuationPriorLoss:
    def test_gradient_flow(self, sample_data):
        mu, logvar, indices, _, _ = sample_data
        mu.requires_grad_(True)
        logvar.requires_grad_(True)

        loss_fn = ValuationPriorLoss()
        loss, _ = loss_fn(mu, indices, logvar=logvar)

        loss.backward()

        assert mu.grad is not None
        assert logvar.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(logvar.grad).all()

    def test_metrics_presence(self, sample_data):
        mu, logvar, indices, _, _ = sample_data
        loss_fn = ValuationPriorLoss()
        _, metrics = loss_fn(mu, indices, logvar=logvar)

        expected_keys = ['vp_mean_mu_norm', 'vp_mean_target', 'vp_gap', 'vp_mean_sigma']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_no_logvar_works(self, sample_data):
        mu, _, indices, _, _ = sample_data
        loss_fn = ValuationPriorLoss()
        loss, metrics = loss_fn(mu, indices, logvar=None)

        assert loss > 0
        assert metrics['vp_mean_sigma'] == 0.0

    def test_per_level_metrics(self, sample_data):
        mu, logvar, indices, _, _ = sample_data
        # Ensure diverse valuations
        indices = torch.tensor([1, 3, 9, 27, 81] * 8)[:32]

        loss_fn = ValuationPriorLoss()
        _, metrics = loss_fn(mu, indices, logvar=logvar)

        # Check if per-level metrics like vp_gap_v1, vp_gap_v2 exist
        gap_keys = [k for k in metrics.keys() if k.startswith('vp_gap_v')]
        assert len(gap_keys) > 0


# =============================================================================
# Algebraic Loss Tests
# =============================================================================

class TestAngularCoherenceLoss:
    def test_phase_start_threshold(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        loss_fn = AngularCoherenceLoss(phase_start_epoch=10)

        # Before threshold
        loss_early, metrics_early = loss_fn(z_hyp, r, indices, epoch=5)
        assert loss_early.item() == 0.0
        assert metrics_early["angular_coherence_loss"] == 0.0

        # After threshold
        loss_late, metrics_late = loss_fn(z_hyp, r, indices, epoch=15)
        # Note: might be 0 if no pairs found, but should be callable
        assert isinstance(loss_late, torch.Tensor)

    def test_level_prefix_k_logic(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        # Mock diverse data
        indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11] * 4) # Valuation 0 mostly

        # Use specific k per level
        level_k = [1] * 10
        loss_fn = AngularCoherenceLoss(level_prefix_k=level_k, phase_start_epoch=0)

        loss, metrics = loss_fn(z_hyp, r, indices, epoch=1)
        assert isinstance(loss, torch.Tensor)

    def test_small_batch_returns_zero(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        loss_fn = AngularCoherenceLoss(phase_start_epoch=0)
        # B < 4 should return zero
        loss, metrics = loss_fn(z_hyp[:2], r[:2], indices[:2], epoch=1)
        assert loss.item() == 0.0


class TestAlgebraicCoherenceLoss:
    def test_gradient_flow(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        z_hyp.requires_grad_(True)

        loss_fn = AlgebraicCoherenceLoss(phase_start_epoch=0)
        loss, _ = loss_fn(z_hyp, r, indices, epoch=1)

        if loss > 0:
            loss.backward()
            assert z_hyp.grad is not None
            assert torch.isfinite(z_hyp.grad).all()

    def test_disabled_returns_zero(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        loss_fn = AlgebraicCoherenceLoss(weight=0.0)
        loss, metrics = loss_fn(z_hyp, r, indices, epoch=100)
        assert loss.item() == 0.0

    def test_min_class_size_guard(self, sample_data):
        _, _, indices, z_hyp, r = sample_data
        # Force all distinct signatures (unlikely with small sample but possible)
        # Here we just set min_class_size high
        # min_global_size=1000 forces all classes to be skipped (none have that many ops globally)
        loss_fn = AlgebraicCoherenceLoss(min_global_size=1000, phase_start_epoch=0)
        loss, metrics = loss_fn(z_hyp, r, indices, epoch=1)

        assert loss.item() == 0.0
        assert metrics["alg_coherence_pairs"] == 0


class TestAlgebraicAdditionLoss:
    def test_forward_with_model_mock(self, sample_data):
        mu, _, indices, _, _ = sample_data

        # Mock model with get_mu_representations
        model = MagicMock()
        def mock_get_mu(idx, device):
            return torch.randn(len(idx), 16, dtype=torch.float64, device=device)
        model.get_mu_representations.side_effect = mock_get_mu

        loss_fn = AlgebraicAdditionLoss()
        loss, metrics = loss_fn(mu, indices, model, epoch=0)

        assert isinstance(loss, torch.Tensor)
        assert "alg_addition_loss" in metrics
        assert "alg_addition_sim" in metrics

    def test_disabled_returns_zero(self, sample_data):
        mu, _, indices, _, _ = sample_data
        model = MagicMock()
        loss_fn = AlgebraicAdditionLoss(weight=0.0)
        loss, metrics = loss_fn(mu, indices, model, epoch=100)
        assert loss.item() == 0.0
        assert metrics["alg_addition_loss"] == 0.0

    def test_large_n_pairs_clamping(self, sample_data):
        """Verify that n_pairs is clamped to batch size."""
        mu, _, indices, _, _ = sample_data
        model = MagicMock()
        # Return the correct shape: (n_pairs, latent_dim)
        def mock_get_mu(idx, device):
            return torch.randn(len(idx), 16, dtype=torch.float64, device=device)
        model.get_mu_representations.side_effect = mock_get_mu

        # Request more pairs than possible; should be silently clamped
        loss_fn = AlgebraicAdditionLoss(n_pairs=1000)
        loss, metrics = loss_fn(mu, indices, model, epoch=1)
        assert "alg_addition_loss" in metrics

    def test_shape_mismatch_raises_runtime_error(self, sample_data):
        """Wrong-shape model output must raise RuntimeError, not silently broadcast."""
        mu, _, indices, _, _ = sample_data
        model = MagicMock()
        # Intentionally wrong: always returns a single row regardless of input size
        model.get_mu_representations.return_value = torch.randn(1, 16, dtype=torch.float64)

        loss_fn = AlgebraicAdditionLoss(n_pairs=8)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            loss_fn(mu, indices, model, epoch=1)


class TestAlgebraicMultiplicationLoss:
    def test_forward_with_model_mock(self, sample_data):
        mu, _, indices, _, _ = sample_data
        
        # Mock model with get_mu_representations
        model = MagicMock()
        def mock_get_mu(idx, device):
            # Deterministic but pseudo-random based on sum/prod to check Sim
            return torch.randn(len(idx), 16, dtype=torch.float64, device=device)
        model.get_mu_representations.side_effect = mock_get_mu
        
        loss_fn = AlgebraicMultiplicationLoss()
        loss, metrics = loss_fn(mu, indices, model, epoch=0)
        
        assert isinstance(loss, torch.Tensor)
        assert "alg_multiplication_loss" in metrics
        assert "alg_multiplication_sim" in metrics

    def test_phase_gate(self, sample_data):
        mu, _, indices, _, _ = sample_data
        model = MagicMock()
        loss_fn = AlgebraicMultiplicationLoss(phase_start_epoch=50)
        loss, metrics = loss_fn(mu, indices, model, epoch=10)
        assert loss.item() == 0.0
        assert metrics["alg_multiplication_loss"] == 0.0


class TestAlgebraicDistributiveLoss:
    def test_forward_with_model_mock(self, sample_data):
        mu, _, indices, _, _ = sample_data

        # Mock model with get_mu_representations
        model = MagicMock()
        def mock_get_mu(idx, device):
            return torch.randn(len(idx), 16, dtype=torch.float64, device=device)
        model.get_mu_representations.side_effect = mock_get_mu

        loss_fn = AlgebraicDistributiveLoss()
        loss, metrics = loss_fn(mu, indices, model, epoch=0)

        assert isinstance(loss, torch.Tensor)
        assert "alg_distributive_loss" in metrics
        assert "alg_distributive_sim" in metrics

    def test_phase_gate(self, sample_data):
        mu, _, indices, _, _ = sample_data
        model = MagicMock()
        loss_fn = AlgebraicDistributiveLoss(phase_start_epoch=50)
        loss, metrics = loss_fn(mu, indices, model, epoch=10)
        assert loss.item() == 0.0
        assert metrics["alg_distributive_loss"] == 0.0


# =============================================================================
# LagrangianDualState Tests
# =============================================================================

class TestLagrangianDualState:
    def test_initial_state(self):
        dual = LagrangianDualState(n_levels=10)
        weights = dual.get_dual_weights()

        assert all(w == 0.0 for w in weights['lambda_margin'])
        assert all(w == 0.0 for w in weights['lambda_scatter'])
        assert all(w == 0.0 for w in weights['lambda_prior'])
        assert not dual.is_active()

    def test_warmup_guard(self):
        dual = LagrangianDualState(warmup_epochs=10, lr=1.0)
        dual.step_epoch(5)

        # Should not update during warmup
        dual.update({'gap_viol_v0': 1.0, 'scatter_v0': 1.0, 'vp_gap_v0': 1.0})
        weights = dual.get_dual_weights()
        assert weights['lambda_margin'][0] == 0.0
        assert weights['lambda_scatter'][0] == 0.0
        assert weights['lambda_prior'][0] == 0.0

    def test_dual_ascent_update(self):
        dual = LagrangianDualState(warmup_epochs=0, lr=0.1, max_lambda=5.0)
        dual.step_epoch(1)

        # Update with violations
        dual.update({'gap_viol_v0': 2.0, 'scatter_v0': 5.0})
        weights = dual.get_dual_weights()

        assert weights['lambda_margin'][0] == pytest.approx(0.2)
        assert weights['lambda_scatter'][0] == pytest.approx(0.5)
        assert weights['lambda_prior'][0] == 0.0 # No key provided

    def test_clamping(self):
        dual = LagrangianDualState(warmup_epochs=0, lr=10.0, max_lambda=1.0)
        dual.step_epoch(1)

        # Positive violation -> hit max
        dual.update({'vp_gap_v0': 1.0})
        assert dual.lambda_prior[0] == 1.0

        # Negative violation -> hit 0
        dual.update({'vp_gap_v0': -2.0})
        assert dual.lambda_prior[0] == 0.0

    def test_serialization(self):
        dual = LagrangianDualState(n_levels=10, lr=0.05)
        dual.lambda_prior[5] = 0.8

        state = dual.state_dict()

        dual2 = LagrangianDualState(n_levels=10)
        dual2.load_state_dict(state)

        assert dual2.lr == 0.05
        assert dual2.lambda_prior[5] == 0.8
        assert dual2.state_dict() == state


# =============================================================================
# HyperbolicContrastiveLoss & SurrogatePropertyLoss Tests
# =============================================================================

from src.losses.contrastive import HyperbolicContrastiveLoss
from src.losses.surrogate import SurrogatePropertyLoss, SurrogateRegressor

class TestHyperbolicContrastiveLoss:
    def test_contrastive_loss(self, sample_data):
        _, _, indices, z_hyp, _ = sample_data
        z_hyp = z_hyp.clone().detach().requires_grad_(True)

        loss_fn = HyperbolicContrastiveLoss(temperature=0.1, prefix_k=3, curvature=1.0)
        loss, metrics = loss_fn(z_hyp, indices)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert "contrastive_loss" in metrics
        assert "n_pairs" in metrics

        # Verify gradient flow
        loss.backward()
        assert z_hyp.grad is not None


class TestSurrogatePropertyLoss:
    def test_surrogate_loss(self, sample_data):
        mu, _, _, _, _ = sample_data
        mu = mu.clone().detach().requires_grad_(True)

        batch_size = mu.size(0)
        # Create mock sequence inputs (N, 9) in {-1, 0, 1}
        x = torch.randint(-1, 2, (batch_size, 9), dtype=torch.float64)

        regressor = SurrogateRegressor(latent_dim=mu.size(1), hidden_dim=64)
        loss_fn = SurrogatePropertyLoss(regressor)
        loss, metrics = loss_fn(mu, x)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert "surrogate_mse_loss" in metrics
        assert "mean_pred_hydropathy" in metrics

        # Verify gradient flow
        loss.backward()
        assert mu.grad is not None
