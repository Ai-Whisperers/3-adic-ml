# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Regression tests for the dual_state wiring in src/training/engine.py.

Historical bug (V23.0 audit): `LagrangianDualState.update()` was never called
during training — the dual ascent that adapts per-level constraint weights was
silently dead for every run before the fix. Nothing in the test suite caught
it because every engine test passed `dual_state=None`, and `LagrangianDualState`
itself was only unit-tested in isolation (its `update()` logic is correct, but
nobody verified the *call site* actually invokes it).

These tests target the wiring specifically:
  1. `train_epoch` must accumulate per-level violation floats returned by the
     loss functions (`monotonic_metrics` / `rank_metrics` /
     `valuation_prior_metrics`) into `dual_violation_acc` / `dual_violation_count`.
  2. `train_model` must call `dual_state.update(...)` once per epoch whenever
     violations were accumulated, and the dual variables must actually change
     as a result (not just receive a call that's a no-op due to warmup).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Unit test: train_epoch's violation-accumulation logic, in isolation.
# ---------------------------------------------------------------------------

def _make_losses_dict(total: torch.Tensor, **metric_blocks: Dict[str, Any]) -> Dict[str, Any]:
    d: Dict[str, Any] = {"total": total}
    d.update(metric_blocks)
    return d


class TestTrainEpochViolationAccumulation:
    """train_epoch must average per-level violation floats across all batches
    and expose them as dual_violation_acc / dual_violation_count, regardless
    of whether a real dual_state is passed in (engine.py only reads
    dual_weights during forward; accumulation happens independently)."""

    def _run(self, loss_fn, loss_fn_b, n_batches: int) -> Dict[str, Any]:
        from src.training.engine import train_epoch

        model = MagicMock(spec=torch.nn.Module)
        model.train.return_value = None
        model.projections = MagicMock()
        model.projections.get_curvature.return_value = 1.0

        batch_ops = torch.randint(-1, 2, (4, 9)).float()
        logits_A = torch.randn(4, 9, 3)
        out = {
            "z_A_hyp": torch.randn(4, 8),
            "z_B_hyp": torch.randn(4, 8),
            "logits_A": logits_A,
            "mu_A": torch.randn(4, 8),
            "logvar_A": torch.randn(4, 8),
            "mu_B": torch.randn(4, 8),
            "logvar_B": torch.randn(4, 8),
            "r_A": torch.rand(4),
            "r_B": torch.rand(4),
        }
        model.return_value = out

        loader = [(batch_ops, torch.arange(4)) for _ in range(n_batches)]
        optimizer = MagicMock()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        return train_epoch(
            epoch=25, model=model, loader=loader, optimizer=optimizer,
            loss_fn=loss_fn, loss_fn_b=loss_fn_b, device=torch.device("cpu"),
            scaler=scaler, max_grad_norm=1.0, use_amp=False,
            dual_state=None, current_dual_weights=None,
            hw_monitor=None, reporting=MagicMock(), global_step_start=0,
        )

    def test_violations_averaged_across_batches(self):
        """Two batches with differing gap_viol_v0 must average, not sum."""
        loss_fn = MagicMock()
        loss_fn.side_effect = [
            _make_losses_dict(
                torch.tensor(0.5, requires_grad=True),
                monotonic_metrics={"gap_viol_v0": 0.4},
            ),
            _make_losses_dict(
                torch.tensor(0.5, requires_grad=True),
                monotonic_metrics={"gap_viol_v0": 0.2},
            ),
        ]
        loss_fn.parameters.return_value = []

        loss_fn_b = MagicMock()
        loss_fn_b.return_value = _make_losses_dict(torch.tensor(0.1, requires_grad=True))
        loss_fn_b.parameters.return_value = []

        metrics = self._run(loss_fn, loss_fn_b, n_batches=2)

        assert metrics["dual_violation_count"] == 2
        assert metrics["dual_violation_acc"]["gap_viol_v0"] == pytest.approx(0.3)

    def test_violations_merged_across_metric_blocks_and_both_vaes(self):
        """gap_viol/scatter/vp_gap keys from monotonic+rank+valuation_prior metrics,
        on both VAE-A and VAE-B, must all land in the same accumulator."""
        loss_fn = MagicMock()
        loss_fn.return_value = _make_losses_dict(
            torch.tensor(0.5, requires_grad=True),
            monotonic_metrics={"gap_viol_v0": 1.0},
            rank_metrics={"scatter_v0": 2.0, "not_a_violation_key": 99.0},
        )
        loss_fn.parameters.return_value = []

        loss_fn_b = MagicMock()
        loss_fn_b.return_value = _make_losses_dict(
            torch.tensor(0.1, requires_grad=True),
            valuation_prior_metrics={"vp_gap_v0": 3.0},
        )
        loss_fn_b.parameters.return_value = []

        metrics = self._run(loss_fn, loss_fn_b, n_batches=1)

        acc = metrics["dual_violation_acc"]
        assert acc["gap_viol_v0"] == pytest.approx(1.0)
        assert acc["scatter_v0"] == pytest.approx(2.0)
        assert acc["vp_gap_v0"] == pytest.approx(3.0)
        assert "not_a_violation_key" not in acc

    def test_no_violation_keys_present_when_losses_report_none(self):
        """When neither loss reports any per-level violation floats, the
        dual_violation_* keys must be absent entirely (not zero/empty-dict) —
        this is the flag train_model reads via `.get(..., 0) > 0` to decide
        whether to call dual_state.update() at all."""
        loss_fn = MagicMock()
        loss_fn.return_value = _make_losses_dict(torch.tensor(0.5, requires_grad=True))
        loss_fn.parameters.return_value = []

        loss_fn_b = MagicMock()
        loss_fn_b.return_value = _make_losses_dict(torch.tensor(0.1, requires_grad=True))
        loss_fn_b.parameters.return_value = []

        metrics = self._run(loss_fn, loss_fn_b, n_batches=1)

        assert "dual_violation_acc" not in metrics
        assert "dual_violation_count" not in metrics


# ---------------------------------------------------------------------------
# 2. Integration test: train_model must actually call dual_state.update()
#    and the dual variables must move — the real historical bug.
# ---------------------------------------------------------------------------

class _DualStateSpy:
    """Transparent proxy around a real LagrangianDualState that records every
    update() call. Everything else (step_epoch, get_dual_weights, state_dict,
    is_active, ...) is forwarded untouched so train_model/reporting see a
    fully functional dual_state."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.update_calls: list[Dict[str, float]] = []

    def update(self, violations: Dict[str, float]) -> None:
        self.update_calls.append(dict(violations))
        self._real.update(violations)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.mark.integration
class TestDualStateWiredIntoTrainModel:
    """Runs the real training loop for a few epochs with monotonic + rank
    losses enabled (both produce per-level violation floats) and a
    zero-warmup LagrangianDualState. If the wiring in train_model regresses
    to "compute violations but never call update()", this test fails."""

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory):
        from src.losses.lagrangian import LagrangianDualState
        from src.models.vae import TernaryVAEV6Controllable
        from src.training.bootstrap import DataAuditor, set_determinism
        from src.training.engine import train_model
        from src.training.reporting import ReportingManager
        from src.training.setup import (
            setup_dataloaders,
            setup_losses,
            setup_optimizer,
            setup_scheduler,
        )

        set_determinism(seed=42)
        device = torch.device("cpu")

        cfg: Dict[str, Any] = {
            "device": {"use_amp": False},
            "data": {"valuation_type": "index"},
            "model": {
                "name": "TernaryVAEV6Controllable",
                "latent_dim": 8,
                "hidden_dim": 16,
                "max_radius": 0.95,
                "curvature": 1.0,
                "factored": False,
                "radial_dims": 4,
                "projection_layers": 1,
                "projection_dropout": 0.0,
                "init_identity": True,
                "tangent_scale": 0.1,
                "encoder_type": "improved",
                "decoder_type": "improved",
                "learnable_curvature": False,
                "positional_encoding": False,
            },
            "training": {
                "epochs": 2,
                "batch_size": 2048,   # large batch -> most valuation levels present
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "max_grad_norm": 1.0,
                "eval_every": 1,
                "save_every": 100,
                "val_frac": 0.1,
                "num_workers": 0,
                "seed": 42,
                "scheduler": {"type": "cosine"},
            },
            "loss": {
                "rich_hierarchy": {
                    "enabled": True, "hierarchy_weight": 1.0,
                    "coverage_weight": 1.0, "separation_weight": 0.5,
                },
                "monotonic": {"enabled": True, "weight": 1.0},
                "rank": {"enabled": True, "weight": 0.5, "n_pairs": 2000},
            },
            "statenet": {"enabled": False},
            "option_c": {"enabled": False},
            "riemannian": {"enabled": False},
        }

        auditor = DataAuditor(seed=42)
        train_ds, val_ds, _ = auditor.prepare_data(
            val_frac=cfg["training"]["val_frac"], device=device
        )
        train_loader, val_loader = setup_dataloaders(train_ds, val_ds, cfg, seed=42)

        model = TernaryVAEV6Controllable(
            latent_dim=cfg["model"]["latent_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            max_radius=cfg["model"]["max_radius"],
            curvature=cfg["model"]["curvature"],
            factored=cfg["model"]["factored"],
            radial_dims=cfg["model"]["radial_dims"],
            n_projection_layers=cfg["model"]["projection_layers"],
            projection_dropout=cfg["model"]["projection_dropout"],
            init_identity=cfg["model"]["init_identity"],
            tangent_scale_init=cfg["model"]["tangent_scale"],
            encoder_type=cfg["model"]["encoder_type"],
            decoder_type=cfg["model"]["decoder_type"],
            learnable_curvature=cfg["model"]["learnable_curvature"],
            positional_encoding=cfg["model"]["positional_encoding"],
            encoder_a_trainable=True,
            encoder_b_trainable=True,
            projections_trainable=True,
        ).to(device)

        loss_fn, loss_fn_b = setup_losses(cfg, device)
        loss_params = list(loss_fn.parameters())
        optimizer = setup_optimizer(model, cfg, loss_params)
        scheduler = setup_scheduler(optimizer, cfg)

        log_dir = tmp_path_factory.mktemp("dual_wiring_run")
        reporting = ReportingManager(log_dir, cfg, tb_logger=MagicMock())

        real_dual_state = LagrangianDualState(warmup_epochs=0, lr=0.05)
        spy = _DualStateSpy(real_dual_state)

        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=loss_fn, loss_fn_b=loss_fn_b, device=device,
            reporting=reporting, config=cfg,
            lr_controller=None, dual_state=spy, hw_monitor=None,
            grokking_detector=None, vis_pipeline=None, use_amp=False,
        )

        return spy, real_dual_state

    def test_dual_state_update_was_called(self, run_result):
        spy, _ = run_result
        assert len(spy.update_calls) >= 1, (
            "dual_state.update() was never called — the dual ascent wiring in "
            "train_model regressed to the pre-V23 dead-code state."
        )

    def test_update_received_nonempty_violations(self, run_result):
        spy, _ = run_result
        assert any(len(call) > 0 for call in spy.update_calls), (
            "dual_state.update() was called with an empty violations dict on "
            "every occasion — train_epoch's dual_violation_acc accumulation "
            "produced nothing usable."
        )

    def test_dual_variables_actually_moved(self, run_result):
        """The real regression: update() being called is not enough if the
        dual variables never leave zero (e.g. warmup gating bug, or violations
        dict keys not matching what LagrangianDualState.update() expects)."""
        _, real_dual_state = run_result
        weights = real_dual_state.get_dual_weights()
        all_lambdas = (
            weights["lambda_margin"] + weights["lambda_scatter"] + weights["lambda_prior"]
        )
        assert any(v > 0 for v in all_lambdas), (
            "All Lagrangian dual variables are still zero after training — "
            f"get_dual_weights()={weights}"
        )
