from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.analysis.project_audit import (
    PROJECT_ROOT,
    build_model_kwargs,
    collect_run_results,
    monte_carlo_scenarios,
    representation_probe_suite,
    stratified_probe_indices,
    summarize_scalar_series,
)
import torch
import yaml

from src.core import TERNARY

PRESETS_DIR = PROJECT_ROOT / "src" / "presets"


def _load_preset(name: str) -> dict:
    with open(PRESETS_DIR / name) as handle:
        return yaml.safe_load(handle)


def test_build_model_kwargs_respects_legacy_factored_presets() -> None:
    legacy_v6 = _load_preset("v6.yaml")
    legacy_5124 = _load_preset("5.12.4.yaml")
    factored_v7 = _load_preset("v7.yaml")

    assert build_model_kwargs(legacy_v6)["factored"] is False
    assert build_model_kwargs(legacy_5124)["factored"] is False
    assert build_model_kwargs(factored_v7)["factored"] is True


def test_collect_run_results_sorts_best_q_descending(tmp_path: Path) -> None:
    run_a = tmp_path / "runs" / "run_a"
    run_b = tmp_path / "runs" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    (run_a / "results.json").write_text(
        json.dumps({"best_Q": 1.5, "best_hierarchy": 0.5, "best_coverage": 0.9, "epochs_trained": 10})
    )
    (run_b / "results.json").write_text(
        json.dumps({"best_Q": 2.1, "best_hierarchy": 0.8, "best_coverage": 0.99, "epochs_trained": 20})
    )

    rows = collect_run_results(tmp_path / "runs")

    assert [row.run_dir for row in rows] == ["runs/run_b", "runs/run_a"]
    assert [row.best_q for row in rows] == [2.1, 1.5]


def test_monte_carlo_scenarios_is_deterministic() -> None:
    baseline = monte_carlo_scenarios(2.1454, 0.9957, 0.1254, trials=1000, seed=123)
    repeat = monte_carlo_scenarios(2.1454, 0.9957, 0.1254, trials=1000, seed=123)

    assert baseline == repeat


def test_summarize_scalar_series_supports_max_and_min_modes() -> None:
    points = [(0, 0.5), (5, 0.7), (10, 0.6)]

    maximize = summarize_scalar_series("metric/max", points, optimize="max", tail_size=2)
    minimize = summarize_scalar_series("metric/min", points, optimize="min", tail_size=2)

    assert maximize is not None
    assert maximize.best_step == 5
    assert maximize.best_value == 0.7
    assert maximize.last_step == 10
    assert maximize.tail_mean == pytest.approx(0.65)

    assert minimize is not None
    assert minimize.best_step == 0
    assert minimize.best_value == 0.5


def test_stratified_probe_indices_is_deterministic_and_keeps_small_classes() -> None:
    labels = np.array([0] * 50 + [1] * 10 + [2] * 3)

    idx_a = stratified_probe_indices(labels, sample_budget=20, min_per_class=4, seed=42)
    idx_b = stratified_probe_indices(labels, sample_budget=20, min_per_class=4, seed=42)

    assert np.array_equal(idx_a, idx_b)
    sampled = labels[idx_a]
    assert (sampled == 0).sum() >= 4
    assert (sampled == 1).sum() >= 4
    assert (sampled == 2).sum() == 3


def test_representation_probe_suite_reports_tangent_and_hyperbolic_views() -> None:
    all_ops = TERNARY.all_ternary().to(torch.float64)
    all_indices = torch.arange(len(all_ops), dtype=torch.long)

    suite = representation_probe_suite(
        z_hyp=all_ops * 0.5,
        z_tangent=all_ops * 1.5,
        raw_inputs=all_ops,
        indices=all_indices,
        sample_budget=60,
        max_level=2,
        min_per_class=3,
        trustworthiness_size=20,
    )

    assert suite["levels_included"] == [0, 1, 2]
    assert suite["levels_excluded_due_to_sparse_support"] == [3, 4, 5, 6, 7, 8, 9]
    assert "hyperbolic_embedding" in suite
    assert "tangent_euclidean" in suite
    assert "raw_input" in suite
    assert suite["tangent_euclidean"]["trustworthiness_k15"] >= 0.0
    assert any("Euclidean tangent baselines" in note for note in suite["notes"])
