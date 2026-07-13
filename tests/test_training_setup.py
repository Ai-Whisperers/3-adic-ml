# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Tests for src/training/setup.py.

setup.py had no dedicated test coverage at all before this file: the four
setup_* helpers used by every training run (dataloaders, losses, optimizer,
scheduler) were only ever exercised indirectly through the full
train_model() smoke tests, and setup_controller / setup_lagrangian were
never called by any test.

The scheduler code carries its own documented historical bug in
_cosine_warmup_restarts_factor's docstring: an earlier implementation used
`ChainedScheduler([CosineAnnealingWarmRestarts(...), LambdaLR(...)])` to
combine per-phase LR scaling with cosine annealing, but LambdaLR evaluates
its lambda against the optimizer's original base_lr rather than the
previous scheduler's already-annealed output — so the cosine factor was
silently discarded and the "cosine" schedule was actually flat within each
phase. That bug class (component present in the graph but silently
producing a no-op) is exactly the same shape as the dual_state wiring bug
covered in test_training_engine.py, so it gets the same treatment here:
a test that would fail if the composition regressed to a no-op.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from src.config.statenet_config import StateNetConfig
from src.core import TERNARY, get_valuation_fn
from src.losses.lagrangian import LagrangianDualState
from src.models import MetricBasedLR, TernaryVAEV6Controllable
from src.training.setup import (
    _cosine_warmup_restarts_factor,
    setup_controller,
    setup_dataloaders,
    setup_lagrangian,
    setup_losses,
    setup_optimizer,
    setup_scheduler,
)


# ---------------------------------------------------------------------------
# _cosine_warmup_restarts_factor — pure function, analytical correctness
# ---------------------------------------------------------------------------

class TestCosineWarmupRestartsFactor:
    def test_t0_zero_or_negative_returns_constant_one(self):
        """Guards against T_0<=0 (misconfiguration) causing a ZeroDivisionError
        deep inside the modulo/geometric-growth logic."""
        assert _cosine_warmup_restarts_factor(0, 0, 2) == 1.0
        assert _cosine_warmup_restarts_factor(50, -1, 2) == 1.0

    def test_epoch_zero_is_always_the_peak(self):
        assert _cosine_warmup_restarts_factor(0, 10, 1) == pytest.approx(1.0)
        assert _cosine_warmup_restarts_factor(0, 5, 2) == pytest.approx(1.0)

    def test_t_mult_one_is_periodic_with_period_t0(self):
        """No restart growth: the factor must repeat exactly every T_0 epochs."""
        T_0 = 10
        for epoch in (1, 4, 7, 9):
            a = _cosine_warmup_restarts_factor(epoch, T_0, 1)
            b = _cosine_warmup_restarts_factor(epoch + T_0, T_0, 1)
            assert a == pytest.approx(b, abs=1e-9), (
                f"epoch={epoch} not periodic with T_0={T_0}: {a} vs {b}"
            )

    def test_t_mult_one_midpoint_is_halfway_down(self):
        """At T_cur = T_0/2 the cosine factor must be exactly 0.5 — the
        inflection point of the descent, not the minimum."""
        T_0 = 10
        assert _cosine_warmup_restarts_factor(T_0 // 2, T_0, 1) == pytest.approx(0.5, abs=1e-9)

    def test_t_mult_one_trough_just_before_restart(self):
        """The factor approaches its minimum at T_cur = T_0 - 1, then the
        warm restart snaps it back up to 1.0 at the next epoch."""
        T_0 = 10
        trough = _cosine_warmup_restarts_factor(T_0 - 1, T_0, 1)
        restart = _cosine_warmup_restarts_factor(T_0, T_0, 1)
        assert trough < 0.05
        assert restart == pytest.approx(1.0)

    def test_t_mult_two_restart_periods_grow_geometrically(self):
        """With T_mult=2, successive restart cycles double in length: for
        T_0=5 the restarts (factor==1.0, T_cur==0) land at epochs
        0, 5, 5+10=15, 15+20=35 — not every 5 epochs."""
        T_0, T_mult = 5, 2
        restart_epochs = [0, 5, 15, 35]
        for e in restart_epochs:
            assert _cosine_warmup_restarts_factor(e, T_0, T_mult) == pytest.approx(1.0, abs=1e-9), (
                f"Expected a restart peak at epoch={e}"
            )
        # And NOT at a naive multiple of T_0 that ignores the geometric growth.
        assert _cosine_warmup_restarts_factor(10, T_0, T_mult) != pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# setup_scheduler — multi_phase_cosine: the actual no-op regression guard
# ---------------------------------------------------------------------------

class TestMultiPhaseCosineScheduler:
    def _build(self, phases, T_0=10, T_mult=1, epochs=20, lr=1.0):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=lr)
        cfg = {
            "training": {
                "epochs": epochs,
                "scheduler": {
                    "type": "multi_phase_cosine",
                    "T_0": T_0, "T_mult": T_mult,
                    "phases": phases,
                },
            }
        }
        return optimizer, setup_scheduler(optimizer, cfg)

    def test_lr_matches_phase_scale_times_cosine_factor(self):
        """Direct regression guard for the ChainedScheduler no-op bug: the
        realized LR at each epoch must equal base_lr * phase_scale(epoch) *
        cosine_factor(epoch), not just phase_scale(epoch) alone."""
        phases = [
            {"epoch_range": [0, 10], "base_lr_scale": 1.0},
            {"epoch_range": [10, 20], "base_lr_scale": 0.1},
        ]
        _, scheduler = self._build(phases, T_0=10, T_mult=1)

        assert scheduler.get_last_lr()[0] == pytest.approx(1.0)  # epoch 0
        for epoch in range(1, 16):
            scheduler.step()
            phase_scale = 1.0 if epoch < 10 else 0.1
            expected = phase_scale * _cosine_warmup_restarts_factor(epoch, 10, 1)
            assert scheduler.get_last_lr()[0] == pytest.approx(expected, abs=1e-6), (
                f"epoch={epoch}: got {scheduler.get_last_lr()[0]}, expected {expected}"
            )

    def test_cosine_modulation_is_not_a_noop_within_a_phase(self):
        """If the composition regressed to the old ChainedScheduler bug, LR
        would be flat (== base_lr * phase_scale) for every epoch inside a
        phase, only jumping at phase boundaries. It must instead trace a
        proper cosine descent."""
        phases = [{"epoch_range": [0, 20], "base_lr_scale": 1.0}]
        _, scheduler = self._build(phases, T_0=20, T_mult=1)

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(9):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert len(set(round(x, 8) for x in lrs)) > 1, f"LR is flat within phase: {lrs}"
        assert lrs == sorted(lrs, reverse=True), (
            f"LR should monotonically decrease over the first half-cycle: {lrs}"
        )

    def test_phase_boundary_changes_scale_immediately(self):
        phases = [
            {"epoch_range": [0, 3], "base_lr_scale": 1.0},
            {"epoch_range": [3, 6], "base_lr_scale": 0.01},
        ]
        _, scheduler = self._build(phases, T_0=100, T_mult=1)  # huge T_0 -> cosine ~flat locally
        for _ in range(3):
            scheduler.step()
        lr_at_boundary = scheduler.get_last_lr()[0]
        assert lr_at_boundary < 0.02, (
            f"Expected phase-2 scale (0.01) to apply immediately at epoch 3, got {lr_at_boundary}"
        )

    def test_empty_phases_falls_back_to_plain_cosine(self):
        """phases=[] must not crash — falls through to CosineAnnealingLR."""
        _, scheduler = self._build(phases=[], epochs=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


class TestSchedulerTypeSelection:
    def test_default_type_is_plain_cosine_annealing(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        scheduler = setup_scheduler(optimizer, {"training": {"epochs": 10}})
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_warmup_restart_type_uses_builtin_scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        cfg = {"training": {"epochs": 10, "scheduler": {
            "type": "cosine_warmup_restart", "T_0": 5, "T_mult": 2,
        }}}
        scheduler = setup_scheduler(optimizer, cfg)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


# ---------------------------------------------------------------------------
# setup_controller
# ---------------------------------------------------------------------------

class TestSetupController:
    def test_disabled_returns_none(self):
        assert setup_controller({"statenet": {"enabled": False}}) is None

    def test_enabled_returns_metric_based_lr(self):
        controller = setup_controller({"statenet": {"enabled": True}})
        assert isinstance(controller, MetricBasedLR)

    def test_missing_statenet_key_defaults_to_enabled(self):
        """setup_controller defaults 'enabled' to True when the statenet
        block is absent entirely — different from an explicit False."""
        assert isinstance(setup_controller({}), MetricBasedLR)

    def test_option_c_overrides_lr_scales(self):
        cfg = {
            "statenet": {"enabled": True},
            "option_c": {
                "enabled": True,
                "encoder_a_lr_scale": 0.33,
                "encoder_b_lr_scale": 0.44,
                "projections_lr_scale": 0.55,
            },
        }
        controller = setup_controller(cfg)
        assert controller.config.lr_scales.encoder_a == pytest.approx(0.33)
        assert controller.config.lr_scales.encoder_b == pytest.approx(0.44)
        assert controller.config.lr_scales.projections == pytest.approx(0.55)

    def test_option_c_disabled_keeps_statenet_configured_scales(self):
        cfg = {
            "statenet": {"enabled": True, "lr_scales": {"encoder_a": 0.9}},
            "option_c": {"enabled": False, "encoder_a_lr_scale": 0.01},
        }
        controller = setup_controller(cfg)
        assert controller.config.lr_scales.encoder_a == pytest.approx(0.9), (
            "option_c.enabled=False must not override statenet.lr_scales"
        )


# ---------------------------------------------------------------------------
# setup_lagrangian
# ---------------------------------------------------------------------------

class TestSetupLagrangian:
    def test_missing_config_returns_none(self):
        assert setup_lagrangian({}) is None

    def test_disabled_returns_none(self):
        cfg = {"loss": {"lagrangian": {"enabled": False}}}
        assert setup_lagrangian(cfg) is None

    def test_enabled_returns_configured_dual_state(self):
        cfg = {"loss": {"lagrangian": {
            "enabled": True, "lr": 0.02, "warmup_epochs": 5,
            "max_lambda": 2.0, "n_levels": 7,
        }}}
        dual_state = setup_lagrangian(cfg)
        assert isinstance(dual_state, LagrangianDualState)
        assert dual_state.lr == pytest.approx(0.02)
        assert dual_state.warmup_epochs == 5
        assert dual_state.max_lambda == pytest.approx(2.0)
        assert dual_state.n_levels == 7


# ---------------------------------------------------------------------------
# setup_dataloaders — stratified sampling weights
# ---------------------------------------------------------------------------

class TestSetupDataloadersStratifiedWeights:
    def test_sample_weights_are_inverse_sqrt_of_level_count(self):
        """Rarer valuation levels must get proportionally higher per-sample
        weight (1/sqrt(count)) so WeightedRandomSampler oversamples them --
        without this, high-valuation levels (as few as 1 sample) would
        almost never appear in a batch."""
        idx = torch.arange(50)
        ops = TERNARY.to_ternary(idx)
        train_ds = TensorDataset(ops, idx)
        val_ds = TensorDataset(ops[:5], idx[:5])
        cfg = {"training": {"batch_size": 8, "num_workers": 0}, "data": {"valuation_type": "index"}}

        train_loader, _ = setup_dataloaders(train_ds, val_ds, cfg, seed=0)

        vfn = get_valuation_fn("index")
        valuations = vfn(idx)
        counts = torch.bincount(valuations, minlength=TERNARY.MAX_VALUATION + 1)

        weights = torch.tensor(list(train_loader.sampler.weights))
        for v in range(TERNARY.MAX_VALUATION + 1):
            if counts[v] == 0:
                continue
            mask = valuations == v
            expected = 1.0 / math.sqrt(counts[v].item())
            actual = weights[mask]
            assert actual.allclose(torch.full_like(actual, expected), atol=1e-6), (
                f"level v={v} (count={counts[v]}): expected weight {expected}, got {actual.unique()}"
            )

    def test_val_loader_is_not_shuffled_and_uses_plain_sampler(self):
        idx = torch.arange(50)
        ops = TERNARY.to_ternary(idx)
        train_ds = TensorDataset(ops, idx)
        val_ds = TensorDataset(ops[:5], idx[:5])
        cfg = {"training": {"batch_size": 8, "num_workers": 0}, "data": {"valuation_type": "index"}}

        _, val_loader = setup_dataloaders(train_ds, val_ds, cfg, seed=0)
        from torch.utils.data import SequentialSampler
        assert isinstance(val_loader.sampler, SequentialSampler)


# ---------------------------------------------------------------------------
# setup_losses — VAE-B config must be an independent deep copy
# ---------------------------------------------------------------------------

class TestSetupLosses:
    def test_vae_b_coverage_weight_forced_to_zero(self):
        cfg = {
            "loss": {"rich_hierarchy": {
                "enabled": True, "coverage_weight": 1.0,
                "hierarchy_weight": 5.0, "separation_weight": 3.0,
            }},
            "model": {"latent_dim": 8, "curvature": 1.0},
            "data": {"valuation_type": "index"},
        }
        loss_fn, loss_fn_b = setup_losses(cfg, torch.device("cpu"))
        assert loss_fn.rich_hierarchy_weights["coverage"] == pytest.approx(1.0)
        assert loss_fn_b.rich_hierarchy_weights["coverage"] == pytest.approx(0.0)

    def test_vae_b_override_does_not_mutate_input_config(self):
        """setup_losses deep-copies loss_cfg before mutating it for VAE-B --
        a shallow copy here would silently corrupt the caller's config dict
        (and, since VAE-A is built from the same dict object first, could
        even leak coverage_weight=0.0 back into VAE-A depending on
        construction order)."""
        cfg = {
            "loss": {"rich_hierarchy": {
                "enabled": True, "coverage_weight": 1.0,
                "hierarchy_weight": 5.0, "separation_weight": 3.0,
            }},
            "model": {"latent_dim": 8, "curvature": 1.0},
            "data": {"valuation_type": "index"},
        }
        setup_losses(cfg, torch.device("cpu"))
        assert cfg["loss"]["rich_hierarchy"]["coverage_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# setup_optimizer
# ---------------------------------------------------------------------------

class TestSetupOptimizer:
    def _model(self):
        torch.manual_seed(0)
        return TernaryVAEV6Controllable(
            latent_dim=8, hidden_dim=16, factored=False,
            encoder_a_trainable=True, encoder_b_trainable=True, projections_trainable=True,
        )

    def test_loss_params_included_as_own_param_group(self):
        model = self._model()
        loss_params = [torch.nn.Parameter(torch.zeros(3))]
        cfg = {"training": {"lr": 1e-3, "weight_decay": 1e-4}, "riemannian": {"enabled": False}}
        optimizer = setup_optimizer(model, cfg, loss_params)

        names = [g.get("name") for g in optimizer.param_groups]
        assert "loss_weights" in names
        loss_group = next(g for g in optimizer.param_groups if g.get("name") == "loss_weights")
        # AdamW's Optimizer.__init__ rewraps each group's params in a new
        # list, so compare by identity of the contained Parameters instead
        # of the list object.
        assert len(loss_group["params"]) == len(loss_params)
        assert all(p is q for p, q in zip(loss_group["params"], loss_params))

    def test_empty_loss_params_adds_no_extra_group(self):
        model = self._model()
        cfg = {"training": {"lr": 1e-3}, "riemannian": {"enabled": False}}
        optimizer = setup_optimizer(model, cfg, [])
        names = [g.get("name") for g in optimizer.param_groups]
        assert "loss_weights" not in names

    def test_standard_optimizer_is_adamw_by_default(self):
        model = self._model()
        cfg = {"training": {"lr": 1e-3}, "riemannian": {"enabled": False}}
        optimizer = setup_optimizer(model, cfg, [])
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_riemannian_enabled_returns_riemannian_optimizer(self):
        import geoopt.optim
        model = self._model()
        cfg = {"training": {"lr": 1e-3}, "riemannian": {"enabled": True, "stabilize": 5}}
        optimizer = setup_optimizer(model, cfg, [])
        assert isinstance(optimizer, geoopt.optim.RiemannianAdam)
