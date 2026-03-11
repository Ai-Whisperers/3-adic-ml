import torch

from src.config.statenet_config import StateNetConfig
from src.models.lr_controller import (
    MetricBasedLR,
    TrainingMetrics,
    compute_Q,
    get_optimizer_grad_stats,
    update_optimizer_lr_scales,
)


def _make_config(overrides: dict | None = None) -> StateNetConfig:
    base = {
        "timing": {
            "warmup_epochs": 0,
            "hysteresis_epochs": 0,
            "window_size": 2,
        },
        "initial": {
            "encoder_a_trainable": True,
            "encoder_b_trainable": True,
            "projections_trainable": True,
        },
        "coverage": {
            "fix_threshold": 0.995,
            "train_threshold": 1.0,
        },
        "hierarchy": {
            "plateau_threshold": 0.0005,
            "plateau_patience": 2,
            "stall_patience": 1,
        },
        "controller": {
            "grad_threshold": 0.5,
            "grad_patience": 2,
            "spike_multiplier": 1.5,
        },
        "lr_scales": {
            "encoder_a": 0.05,
            "encoder_b": 0.1,
            "projections": 1.0,
            "decoders": 1.0,
        },
    }
    if overrides:
        for section, values in overrides.items():
            if section not in base:
                base[section] = values
            elif isinstance(base[section], dict) and isinstance(values, dict):
                base[section].update(values)
            else:
                base[section] = values
    return StateNetConfig.from_dict(base)


def _metrics(
    epoch: int,
    coverage: float = 1.0,
    hierarchy_a: float = -0.8,
    hierarchy_b: float = -0.8,
    dist_corr_a: float = 0.0,
    q_value: float = 0.0,
    grad_norm_projections: float = 1.0,
) -> TrainingMetrics:
    return TrainingMetrics(
        epoch=epoch,
        coverage=coverage,
        hierarchy_a=hierarchy_a,
        hierarchy_b=hierarchy_b,
        dist_corr_a=dist_corr_a,
        q_value=q_value,
        grad_norm_projections=grad_norm_projections,
    )


class TestComputeQ:
    def test_compute_q_formula(self):
        q = compute_Q(dist_corr=0.3, hierarchy=-0.4)
        assert q == 0.3 + 1.5 * 0.4

    def test_compute_q_uses_abs_hierarchy(self):
        q_neg = compute_Q(dist_corr=0.1, hierarchy=-0.5)
        q_pos = compute_Q(dist_corr=0.1, hierarchy=0.5)
        assert q_neg == q_pos

    def test_compute_q_zero_inputs(self):
        assert compute_Q(0.0, 0.0) == 0.0


class TestTrainingMetrics:
    def test_from_dict_standard_keys(self):
        metrics = TrainingMetrics.from_dict(
            {
                "coverage": 0.9,
                "hierarchy_a": -0.7,
                "hierarchy_b": -0.8,
                "dist_corr_a": 0.2,
                "q_value": 1.1,
                "grad_norm_projections": 0.3,
            },
            epoch=5,
        )

        assert metrics.epoch == 5
        assert metrics.coverage == 0.9
        assert metrics.hierarchy_a == -0.7
        assert metrics.hierarchy_b == -0.8
        assert metrics.dist_corr_a == 0.2
        assert metrics.q_value == 1.1
        assert metrics.grad_norm_projections == 0.3

    def test_from_dict_alternate_keys(self):
        metrics = TrainingMetrics.from_dict(
            {
                "coverage": 0.91,
                "hierarchy_A": -0.6,
                "hierarchy_B": -0.75,
                "dist_corr_A": 0.25,
                "Q": 1.3,
            },
            epoch=7,
        )

        assert metrics.epoch == 7
        assert metrics.hierarchy_a == -0.6
        assert metrics.hierarchy_b == -0.75
        assert metrics.dist_corr_a == 0.25
        assert metrics.q_value == 1.3


class TestMetricBasedLR:
    def test_warmup_uses_initial_states(self):
        config = _make_config(
            {
                "timing": {"warmup_epochs": 3},
                "initial": {
                    "encoder_a_trainable": False,
                    "encoder_b_trainable": True,
                    "projections_trainable": False,
                },
            }
        )
        controller = MetricBasedLR(config)

        state = controller.update(_metrics(epoch=1))

        assert state["events"] == ["warmup"]
        assert state["lr_scales"]["encoder_a"] == 0.0
        assert state["lr_scales"]["encoder_b"] == config.lr_scales.encoder_b
        assert state["lr_scales"]["projections"] == 0.0
        assert state["active_states"]["encoder_a"] is False
        assert state["active_states"]["encoder_b"] is True
        assert state["active_states"]["projections"] is False

    def test_coverage_drop_freezes_encoder_a(self):
        config = _make_config({"initial": {"encoder_a_trainable": True}})
        controller = MetricBasedLR(config)

        state = controller.update(_metrics(epoch=5, coverage=0.9))

        assert state["lr_scales"]["encoder_a"] == 0.0
        assert state["active_states"]["encoder_a"] is False
        assert any("encoder_a frozen" in event for event in state["events"])

    def test_coverage_recovery_unfreezes_encoder_a_when_hierarchy_stalls(self):
        config = _make_config(
            {
                "initial": {"encoder_a_trainable": False},
                "hierarchy": {"stall_patience": 1, "plateau_threshold": 0.001},
                "timing": {
                    "window_size": 2,
                    "hysteresis_epochs": 0,
                    "warmup_epochs": 0,
                },
            }
        )
        controller = MetricBasedLR(config)

        controller.update(_metrics(epoch=1, coverage=1.0, hierarchy_a=-0.5))
        state = controller.update(_metrics(epoch=2, coverage=1.0, hierarchy_a=-0.5))

        assert state["lr_scales"]["encoder_a"] == config.lr_scales.encoder_a
        assert state["active_states"]["encoder_a"] is True
        assert any("encoder_a unfrozen" in event for event in state["events"])

    def test_hierarchy_plateau_freezes_encoder_b(self):
        config = _make_config(
            {
                "initial": {"encoder_b_trainable": True},
                "hierarchy": {
                    "plateau_patience": 2,
                    "plateau_threshold": 0.001,
                },
                "timing": {
                    "window_size": 2,
                    "warmup_epochs": 0,
                    "hysteresis_epochs": 0,
                },
            }
        )
        controller = MetricBasedLR(config)

        controller.update(_metrics(epoch=1, hierarchy_b=-0.8))
        controller.update(_metrics(epoch=2, hierarchy_b=-0.8))
        state = controller.update(_metrics(epoch=3, hierarchy_b=-0.8))

        assert state["lr_scales"]["encoder_b"] == 0.0
        assert state["active_states"]["encoder_b"] is False
        assert any("encoder_b frozen" in event for event in state["events"])

    def test_hierarchy_degradation_unfreezes_encoder_b(self):
        config = _make_config(
            {
                "initial": {"encoder_b_trainable": False},
                "timing": {"warmup_epochs": 0, "hysteresis_epochs": 0},
            }
        )
        controller = MetricBasedLR(config)

        controller.update(_metrics(epoch=1, hierarchy_b=-0.9))
        state = controller.update(_metrics(epoch=2, hierarchy_b=-0.75))

        assert state["lr_scales"]["encoder_b"] == config.lr_scales.encoder_b
        assert state["active_states"]["encoder_b"] is True
        assert any("encoder_b unfrozen" in event for event in state["events"])

    def test_low_gradient_freezes_projections(self):
        config = _make_config(
            {
                "initial": {"projections_trainable": True},
                "controller": {"grad_threshold": 0.5, "grad_patience": 2},
                "timing": {"warmup_epochs": 0, "hysteresis_epochs": 0},
            }
        )
        controller = MetricBasedLR(config)

        controller.update(_metrics(epoch=1, grad_norm_projections=0.4))
        controller.update(_metrics(epoch=2, grad_norm_projections=0.4))
        state = controller.update(_metrics(epoch=3, grad_norm_projections=0.4))

        assert state["lr_scales"]["projections"] == 0.0
        assert state["active_states"]["projections"] is False
        assert any("projections frozen" in event for event in state["events"])

    def test_gradient_spike_unfreezes_projections(self):
        config = _make_config(
            {
                "initial": {"projections_trainable": False},
                "controller": {"spike_multiplier": 1.5},
                "timing": {"warmup_epochs": 0, "hysteresis_epochs": 0},
            }
        )
        controller = MetricBasedLR(config)

        controller.update(_metrics(epoch=1, grad_norm_projections=0.2))
        controller.update(_metrics(epoch=2, grad_norm_projections=0.2))
        state = controller.update(_metrics(epoch=3, grad_norm_projections=1.0))

        assert state["lr_scales"]["projections"] == config.lr_scales.projections
        assert state["active_states"]["projections"] is True
        assert any("projections unfrozen" in event for event in state["events"])

    def test_hysteresis_blocks_rapid_state_flip(self):
        config = _make_config(
            {
                "timing": {
                    "warmup_epochs": 0,
                    "hysteresis_epochs": 3,
                    "window_size": 2,
                },
                "initial": {"encoder_a_trainable": True},
                "hierarchy": {"stall_patience": 1, "plateau_threshold": 0.001},
            }
        )
        controller = MetricBasedLR(config)

        state_freeze = controller.update(
            _metrics(epoch=5, coverage=0.9, hierarchy_a=-0.4)
        )
        assert state_freeze["active_states"]["encoder_a"] is False

        state_blocked = controller.update(
            _metrics(epoch=6, coverage=1.0, hierarchy_a=-0.4)
        )
        assert state_blocked["active_states"]["encoder_a"] is False
        assert not any(
            "encoder_a unfrozen" in event for event in state_blocked["events"]
        )

    def test_update_tracks_best_q(self):
        config = _make_config({"timing": {"warmup_epochs": 0}})
        controller = MetricBasedLR(config)

        state_1 = controller.update(_metrics(epoch=1, q_value=1.2))
        state_2 = controller.update(_metrics(epoch=2, q_value=0.8))
        state_3 = controller.update(_metrics(epoch=3, q_value=1.5))

        assert state_1["best_q"] == 1.2
        assert state_2["best_q"] == 1.2
        assert state_3["best_q"] == 1.5

    def test_reset_restores_initial_state(self):
        config = _make_config(
            {
                "initial": {
                    "encoder_a_trainable": False,
                    "encoder_b_trainable": True,
                    "projections_trainable": False,
                }
            }
        )
        controller = MetricBasedLR(config)

        controller.update(
            _metrics(epoch=1, coverage=0.9, q_value=1.0, grad_norm_projections=0.2)
        )
        controller.update(
            _metrics(epoch=2, coverage=0.9, q_value=1.1, grad_norm_projections=0.2)
        )

        controller.reset()

        assert controller._coverage_history == []
        assert controller._hierarchy_a_history == []
        assert controller._hierarchy_b_history == []
        assert controller._grad_norm_history == []
        assert controller._q_history == []
        assert controller._best_q == 0.0
        assert controller._active["encoder_a"] is False
        assert controller._active["encoder_b"] is True
        assert controller._active["projections"] is False

    def test_history_window_trimming(self):
        config = _make_config({"timing": {"window_size": 3, "warmup_epochs": 0}})
        controller = MetricBasedLR(config)

        for epoch in range(10):
            controller.update(
                _metrics(
                    epoch=epoch,
                    coverage=0.95 + 0.001 * epoch,
                    hierarchy_a=-0.8,
                    hierarchy_b=-0.8,
                    q_value=epoch / 10,
                    grad_norm_projections=0.4,
                )
            )

        assert len(controller._coverage_history) == 3
        assert len(controller._hierarchy_a_history) == 3
        assert len(controller._hierarchy_b_history) == 3
        assert len(controller._grad_norm_history) == 3
        assert len(controller._q_history) == 6

    def test_update_returns_event_list(self):
        config = _make_config({"timing": {"warmup_epochs": 0}})
        controller = MetricBasedLR(config)

        state = controller.update(_metrics(epoch=1, coverage=0.9))

        assert isinstance(state["events"], list)
        assert all(isinstance(event, str) for event in state["events"])


class TestOptimizerLrScaleUpdates:
    def test_updates_named_param_groups(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3, dtype=torch.float64),
            torch.nn.Linear(3, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(
            [
                {"params": list(model[0].parameters()), "lr": 0.1, "name": "encoder_a"},
                {"params": list(model[1].parameters()), "lr": 0.1, "name": "decoders"},
            ]
        )

        update_optimizer_lr_scales(
            optimizer,
            base_lr=0.1,
            lr_scales={"encoder_a": 0.0, "decoders": 0.5},
        )

        lrs = {group.get("name", ""): group["lr"] for group in optimizer.param_groups}
        assert lrs["encoder_a"] == 0.0
        assert lrs["decoders"] == 0.05

    def test_ignores_unnamed_param_groups(self):
        extra = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        model = torch.nn.Linear(2, 1, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.parameters()), "lr": 0.1, "name": "encoder_b"},
                {"params": [extra], "lr": 0.2},
            ]
        )

        update_optimizer_lr_scales(
            optimizer,
            base_lr=0.1,
            lr_scales={"encoder_b": 0.5},
        )

        named_group = next(
            group
            for group in optimizer.param_groups
            if group.get("name") == "encoder_b"
        )
        unnamed_group = next(
            group for group in optimizer.param_groups if "name" not in group
        )

        assert named_group["lr"] == 0.05
        assert unnamed_group["lr"] == 0.2


class TestOptimizerGradStats:
    def test_returns_zeros_when_state_empty(self):
        model = torch.nn.Linear(2, 1, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            [{"params": list(model.parameters()), "lr": 1e-3, "name": "encoder_b"}]
        )

        stats = get_optimizer_grad_stats(optimizer, "encoder_b")

        assert stats["n_params"] == 0
        assert stats["grad_norm_estimate"] == 0.0
        assert stats["exp_avg_norm"] == 0.0

    def test_returns_stats_after_training_step(self):
        model = torch.nn.Linear(4, 2, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            [{"params": list(model.parameters()), "lr": 1e-2, "name": "encoder_b"}]
        )

        x = torch.randn(16, 4, dtype=torch.float64)
        y = torch.randn(16, 2, dtype=torch.float64)

        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        stats = get_optimizer_grad_stats(optimizer, "encoder_b")

        assert stats["n_params"] > 0
        assert stats["grad_norm_estimate"] > 0.0
        assert stats["exp_avg_norm"] > 0.0
