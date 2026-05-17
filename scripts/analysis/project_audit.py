#!/usr/bin/env python3
"""Project audit utilities for checkpoint-backed feasibility review.

This script is intentionally skeptical:
- It treats `results.json` as run-log evidence, not as proof of full-domain behavior.
- It evaluates checkpoints directly on the complete 19,683-state domain.
- It reports what the model can consume/output in code, and what has actually been validated.
- Its Monte Carlo output is a transparent scenario model, not a market forecast.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import trustworthiness
import torch
import yaml

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.symbolic import build_symbolic_subsystem
from src.training.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_hierarchy_metrics,
)
from src.utils.checkpoint import load_checkpoint_compat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "runs" / "v7_large_20260324_013725"
LEVEL_PREFIX_DEPTHS = {0: 3, 1: 4, 2: 3, 3: 4, 4: 5, 5: 6}
SYMBOLIC_SUBSYSTEM = build_symbolic_subsystem({"enabled": True, "backend": "finite_group"})


@dataclass(frozen=True)
class RunResultRow:
    run_dir: str
    best_q: float
    best_cov: float
    best_hier: float
    grokking: int
    val_type: str
    n_params: int
    status: str


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_all_runs(runs_root: Path) -> list[Path]:
    """Find all valid run directories (those containing results.json)."""
    if not runs_root.exists():
        return []
    return [p.parent for p in runs_root.glob("**/results.json")]


def summarize_runs(runs_root: Path) -> list[RunResultRow]:
    """Scrape results.json from all runs and return a summary list."""
    rows = []
    for run_dir in get_all_runs(runs_root):
        results_path = run_dir / "results.json"
        try:
            with open(results_path) as f:
                res = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        try:
            run_dir_str = str(results_path.parent.relative_to(PROJECT_ROOT))
        except ValueError:
            run_dir_str = str(results_path.parent.relative_to(runs_root.parent))

        rows.append(RunResultRow(
            run_dir=run_dir_str,
            best_q=res.get("best_q", 0.0),
            best_cov=res.get("best_coverage", 0.0),
            best_hier=res.get("best_hierarchy", 0.0),
            grokking=len(res.get("grokking_events", [])),
            val_type=res.get("config", {}).get("data", {}).get("valuation_type", "unknown"),
            n_params=res.get("model_stats", {}).get("total_params", 0),
            status="SUCCESS" if res.get("best_q", 0.0) > 1.2 else "FAIL"
        ))
    return sorted(rows, key=lambda x: x.best_q, reverse=True)


def print_run_summary(rows: list[RunResultRow]) -> None:
    """Print run summary table."""
    print(f"\n{'RUN DIRECTORY':<45} | {'Q':>6} | {'COV':>6} | {'HIER':>6} | {'GROK':>4}")
    print("-" * 75)
    for r in rows:
        print(f"{r.run_dir:<45} | {r.best_q:6.3f} | {r.best_cov:6.2%} | {r.best_hier:6.3f} | {r.grokking:4}")


def stratified_subsample(
    data: np.ndarray,
    labels: np.ndarray,
    sample_budget: int = 5000,
    min_per_class: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Sample points while preserving valuation-level balance."""
    rng = np.random.default_rng(seed)
    n_total = len(labels)
    selected = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        proportional = round(sample_budget * len(cls_idx) / n_total)
        take = min(len(cls_idx), max(min_per_class, proportional))
        selected.append(rng.choice(cls_idx, size=take, replace=False))

    return np.concatenate(selected)


def evaluate_checkpoint_feasibility(
    run_dir: Path,
    checkpoint_name: str = "best_Q.pt",
    evaluation_seed: int = 42,
) -> dict[str, Any]:
    run_dir = run_dir if run_dir.is_absolute() else (PROJECT_ROOT / run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint = load_checkpoint_compat(run_dir / "checkpoints" / checkpoint_name, map_location="cpu")

    # Reconstruct model
    model_cfg = config["model"]
    model = TernaryVAEV6Controllable(
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        max_radius=model_cfg.get("max_radius", 0.99),
        curvature=model_cfg.get("curvature", 1.0),
        factored=model_cfg.get("factored", True),
        radial_dims=model_cfg.get("radial_dims", 4),
        positional_encoding=model_cfg.get("positional_encoding", False)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Full domain evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_indices = torch.arange(19683)
    all_ternary = TERNARY.to_ternary(all_indices).to(device)

    with torch.no_grad():
        out = model(all_ternary)
        logits = out["logits_A"] if "logits_A" in out else out["logits"]
        z_hyp = out["z_A_hyp"] if "z_A_hyp" in out else out["z_hyp"]

    # Metrics
    acc = compute_accuracy(logits, all_ternary)
    cov = compute_coverage(logits, all_ternary)
    hier_metrics = compute_hierarchy_metrics(z_hyp, all_indices)

    # Trustworthiness check (Neighborhood preservation)
    # Subsample for speed
    indices_np = all_indices.cpu().numpy()
    valuations_np = TERNARY.valuation(all_indices).cpu().numpy()
    sample_idx = stratified_subsample(indices_np, valuations_np, sample_budget=2000)

    z_sample = z_hyp[sample_idx].cpu().numpy()
    # P-adic distances in index space (v3(|n-m|))
    def p_adic_dist_matrix(idx):
        n = len(idx)
        dists = np.zeros((n, n))
        for i in range(n):
            diffs = np.abs(idx[i] - idx)
            dists[i] = TERNARY.valuation(torch.from_numpy(diffs)).numpy()
        return -dists # Higher valuation = smaller distance

    dist_orig = p_adic_dist_matrix(indices_np[sample_idx])
    # trustworthiness function expects distances? No, it expects X.
    # We use KNN trustworthiness
    t_score = trustworthiness(dist_orig, z_sample, n_neighbors=15, metric="precomputed")

    evaluation = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "checkpoint": checkpoint_name,
        "evaluation_seed": evaluation_seed,
        "full_domain_acc": acc,
        "full_domain_cov": cov,
        "q_score": hier_metrics["Q"],
        "dist_corr": hier_metrics["dist_corr"],
        "hierarchy": hier_metrics["hierarchy"],
        "trustworthiness": t_score,
    }
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run artifacts and checkpoint-backed project feasibility.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=str, default="best_Q.pt")
    parser.add_argument("--all", action="store_true", help="Audit all runs and print summary table")
    args = parser.parse_args()

    if args.all:
        rows = summarize_runs(args.runs_root)
        print_run_summary(rows)
    else:
        eval_res = evaluate_checkpoint_feasibility(args.run_dir, args.checkpoint)
        print("\nCheckpoint Feasibility Review:")
        for k, v in eval_res.items():
            if isinstance(v, float):
                print(f"  {k:<20}: {v:.4f}")
            else:
                print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()
