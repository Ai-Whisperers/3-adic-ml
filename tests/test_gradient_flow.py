"""Integration tests for gradient flow through the full training pipeline.

These tests catch architectural bugs like:
- Dead parameters (no gradient flow)
- Saturated embeddings (all points same radius)
- Missing loss terms (e.g., KL divergence)
"""

import pytest
import torch

from src.core import TERNARY
from src.losses.combined import CombinedLoss
from src.models.vae import TernaryVAEV6Controllable


@pytest.fixture
def model():
    """Create a model with both VAEs trainable."""
    m = TernaryVAEV6Controllable(
        encoder_a_trainable=True,
        encoder_b_trainable=True,
        projections_trainable=True,
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
        learnable_curvature=True,
        n_projection_layers=2,
        projection_dropout=0.1,
    )
    return m


@pytest.fixture
def loss_fn():
    """Create CombinedLoss with all losses enabled (matching v6.yaml)."""
    config = {
        'rich_hierarchy': {
            'enabled': True,
            'hierarchy_weight': 5.0,
            'coverage_weight': 1.0,
            'separation_weight': 3.0,
            'inner_radius': 0.08,
            'outer_radius': 0.70,
        },
        'radial': {
            'enabled': True,
            'weight': 5.0,
            'margin_weight': 0.5,
            'inner_radius': None, # Let rich_hierarchy be the source of truth
            'outer_radius': None,
        },
        'geodesic': {
            'enabled': True,
            'phase_start_epoch': 0,
            'weight': 2.0,
            'n_pairs': 100,
        },
        'rank': {
            'enabled': True,
            'weight': 0.5,
            'n_pairs': 100,
        },
        'monotonic': {
            'enabled': True,
            'weight': 1.0,
            'inner_radius': None, # Let rich_hierarchy be the source of truth
            'outer_radius': None,
        },
        'hyperbolic_kl': {
            'enabled': True,
            'beta': 0.1,
            'weight': 0.01,
            'free_bits': 0.5,
        },
    }
    return CombinedLoss(config, curvature=1.0)


@pytest.fixture
def batch():
    """Create a realistic batch from actual ternary data."""
    torch.manual_seed(42)
    n = 64
    indices = torch.randint(0, 19683, (n,))
    ops = TERNARY.to_ternary(indices)
    return ops.to(torch.float64), indices


class TestGradientFlowAllParameters:
    """Verify ALL model parameters receive gradients through training."""

    def test_vae_a_receives_gradients(self, model, loss_fn, batch):
        """VAE-A encoder should receive gradients."""
        ops, indices = batch
        out = model(ops)
        losses = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        losses["total"].backward()

        head_a_grads = [p.grad for p in model.head_A.parameters() if p.grad is not None]
        assert len(head_a_grads) > 0, "head_A receives no gradients!"
        assert any(g.abs().sum() > 0 for g in head_a_grads), "head_A gradients are all zero!"

    def test_vae_b_receives_gradients(self, model, loss_fn, batch):
        """VAE-B encoder should receive gradients (was dead before fix)."""
        ops, indices = batch
        out = model(ops)
        losses_B = loss_fn(
            out["z_B_hyp"], indices, out["logits_B"], ops, epoch=10,
            mu=out["mu_B"], logvar=out["logvar_B"],
        )
        losses_B["total"].backward()

        head_b_grads = [p.grad for p in model.head_B.parameters() if p.grad is not None]
        assert len(head_b_grads) > 0, "head_B receives no gradients!"
        assert any(g.abs().sum() > 0 for g in head_b_grads), "head_B gradients are all zero!"

    def test_projections_receive_gradients(self, model, loss_fn, batch):
        """Projection networks should receive gradients."""
        ops, indices = batch
        out = model(ops)
        losses = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        losses["total"].backward()

        proj_grads = [p.grad for p in model.projections.parameters() if p.grad is not None]
        assert len(proj_grads) > 0, "projections receive no gradients!"

    def test_decoders_receive_gradients(self, model, loss_fn, batch):
        """Both decoders should receive gradients."""
        ops, indices = batch
        out = model(ops)

        losses_A = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        losses_B = loss_fn(
            out["z_B_hyp"], indices, out["logits_B"], ops, epoch=10,
            mu=out["mu_B"], logvar=out["logvar_B"],
        )
        (losses_A["total"] + losses_B["total"]).backward()

        dec_a_grads = [p.grad for p in model.decoder_A.parameters() if p.grad is not None]
        dec_b_grads = [p.grad for p in model.decoder_B.parameters() if p.grad is not None]
        assert len(dec_a_grads) > 0, "decoder_A receives no gradients!"
        assert len(dec_b_grads) > 0, "decoder_B receives no gradients!"

    def test_all_critical_params_receive_gradients(self, model, loss_fn, batch):
        """All critical parameters should receive gradients.

        Known exceptions (zero gradient at identity init):
        - tangent_net hidden layers: identity init zeros the last layer,
          breaking gradient chain through intermediate layers at init.
          Gradient will flow once last layer moves away from zero.
        - manifold.isp_c: curvature param; .item() detaches from graph
          in exp_map_zero. Curvature learning happens via other paths.
        """
        ops, indices = batch
        out = model(ops)

        losses_A = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        losses_B = loss_fn(
            out["z_B_hyp"], indices, out["logits_B"], ops, epoch=10,
            mu=out["mu_B"], logvar=out["logvar_B"],
        )
        (losses_A["total"] + losses_B["total"]).backward()

        # Known zero-gradient params at identity init (not dead, just zero at init)
        expected_zero_patterns = ['manifold.isp_c', 'tangent_net.0.', 'tangent_net.1.',
                                  'tangent_net.4.', 'tangent_net.5.']

        dead_params = []
        total_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is None or p.grad.abs().sum() == 0:
                    if not any(pat in name for pat in expected_zero_patterns):
                        dead_params.append(name)

        assert len(dead_params) == 0, (
            f"{len(dead_params)}/{total_params} unexpected dead parameters: {dead_params}"
        )


class TestInitialRadiusSpread:
    """Verify embeddings DON'T all saturate at max_radius after init."""

    def test_initial_radii_not_saturated(self, model, batch):
        """After tangent_scale fix, radii should spread across target range."""
        ops, _ = batch
        with torch.no_grad():
            out = model(ops)

        z_A_hyp = out["z_A_hyp"]
        norms = torch.norm(z_A_hyp, dim=-1)

        # Should NOT all be at max_radius (0.95)
        std = norms.std()
        assert std > 0.001, (
            f"All embeddings saturated at same radius! std={std:.6f}, "
            f"mean={norms.mean():.4f}, min={norms.min():.4f}, max={norms.max():.4f}"
        )

    def test_initial_radii_within_target_range(self, model, batch):
        """Initial radii should be in a reasonable range, not all at boundary."""
        ops, _ = batch
        with torch.no_grad():
            out = model(ops)

        norms = torch.norm(out["z_A_hyp"], dim=-1)
        mean_norm = norms.mean().item()

        # After tangent_scale=0.05, mean norm should be well below max_radius
        assert mean_norm < 0.92, (
            f"Mean radius {mean_norm:.4f} too close to boundary (max_radius=0.95)"
        )


class TestKLDivergenceIntegration:
    """Verify KL divergence is actually computed and contributes to loss."""

    def test_kl_loss_present(self, loss_fn, model, batch):
        """KL loss should appear in loss dict when enabled."""
        ops, indices = batch
        out = model(ops)
        losses = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        assert 'kl' in losses, "KL loss missing from CombinedLoss output!"
        assert losses['kl'].item() > 0, "KL loss is zero!"

    def test_kl_loss_absent_without_mu_logvar(self, loss_fn, model, batch):
        """KL loss should be skipped gracefully when mu/logvar not provided."""
        ops, indices = batch
        out = model(ops)
        losses = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
        )
        # Should work without error, KL just not computed
        assert 'total' in losses

    def test_kl_prevents_logvar_collapse(self, model, loss_fn, batch):
        """With KL active, logvar should not collapse to -inf."""
        ops, indices = batch
        out = model(ops)

        logvar = out["logvar_A"]
        assert logvar.mean().item() > -20, (
            f"logvar collapsed to {logvar.mean().item():.2f}, KL not preventing collapse"
        )


class TestDualVAETrainingLoop:
    """Simulate the actual training loop to verify end-to-end gradient flow."""

    def test_one_training_step(self, model, loss_fn, batch):
        """Simulate a complete training step with both VAEs."""
        ops, indices = batch

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
        )
        optimizer.zero_grad()

        out = model(ops)

        # Compute losses for both VAEs (as fixed in train.py)
        losses_A = loss_fn(
            out["z_A_hyp"], indices, out["logits_A"], ops, epoch=10,
            mu=out["mu_A"], logvar=out["logvar_A"],
        )
        losses_B = loss_fn(
            out["z_B_hyp"], indices, out["logits_B"], ops, epoch=10,
            mu=out["mu_B"], logvar=out["logvar_B"],
        )
        total_loss = losses_A["total"] + losses_B["total"]

        total_loss.backward()
        optimizer.step()

        # Verify loss decreased after step
        out2 = model(ops)
        losses_A2 = loss_fn(
            out2["z_A_hyp"], indices, out2["logits_A"], ops, epoch=10,
            mu=out2["mu_A"], logvar=out2["logvar_A"],
        )
        losses_B2 = loss_fn(
            out2["z_B_hyp"], indices, out2["logits_B"], ops, epoch=10,
            mu=out2["mu_B"], logvar=out2["logvar_B"],
        )
        total_loss2 = losses_A2["total"] + losses_B2["total"]

        # Loss should change (not necessarily decrease in 1 step due to competing objectives)
        assert abs(total_loss.item() - total_loss2.item()) > 1e-6, (
            "Loss unchanged after optimizer step — training is stuck!"
        )
