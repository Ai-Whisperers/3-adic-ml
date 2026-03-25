from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.analysis.project_audit import (
    PROJECT_ROOT,
    build_model_kwargs,
    collect_run_results,
    monte_carlo_scenarios,
)


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
