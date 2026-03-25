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
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
import yaml

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.train import compute_accuracy, compute_coverage, compute_hierarchy_metrics
from src.utils.checkpoint import load_checkpoint_compat


DEFAULT_RUN = PROJECT_ROOT / "runs" / "v7_large_20260324_013725"
LEVEL_PREFIX_DEPTHS = {0: 3, 1: 4, 2: 3, 3: 4, 4: 5, 5: 6}


@dataclass(frozen=True)
class RunResultRow:
    run_dir: str
    best_q: float
    best_hierarchy: float
    best_coverage: float
    epochs_trained: int


@dataclass(frozen=True)
class SurfaceSummary:
    input_shape: str
    training_input_domain: str
    runtime_input_validation: str
    output_logits_shape: str
    decoded_symbol_shape: str
    hyperbolic_shape: str
    explicit_radius: bool
    factored: bool
    parameter_count: int


@dataclass(frozen=True)
class DirectionClusteringRow:
    level: int
    prefix_depth: int
    n_samples: int
    n_classes: int
    ari: float


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def build_model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    option_c_cfg = config.get("option_c", {})
    initial_cfg = config.get("statenet", {}).get("initial", {})
    return {
        "latent_dim": model_cfg.get("latent_dim", 16),
        "hidden_dim": model_cfg.get("hidden_dim", 64),
        "max_radius": model_cfg.get("max_radius", 0.95),
        "curvature": model_cfg.get("curvature", 1.0),
        "encoder_type": model_cfg.get("encoder_type", "improved"),
        "decoder_type": model_cfg.get("decoder_type", "improved"),
        "n_projection_layers": model_cfg.get("projection_layers", 1),
        "projection_dropout": model_cfg.get("projection_dropout", 0.0),
        "learnable_curvature": model_cfg.get("learnable_curvature", False),
        "init_identity": model_cfg.get("init_identity", True),
        "tangent_scale_init": model_cfg.get("tangent_scale", 0.1),
        "factored": model_cfg.get("factored", False),
        "radial_dims": model_cfg.get("radial_dims", 4),
        "encoder_a_lr_scale": option_c_cfg.get("encoder_a_lr_scale", 0.05),
        "encoder_b_lr_scale": option_c_cfg.get("encoder_b_lr_scale", 0.1),
        "projections_lr_scale": option_c_cfg.get("projections_lr_scale", 1.0),
        "encoder_a_trainable": initial_cfg.get("encoder_a_trainable", False),
        "encoder_b_trainable": initial_cfg.get("encoder_b_trainable", True),
        "projections_trainable": initial_cfg.get("projections_trainable", True),
    }


def build_model_from_config(config: dict[str, Any]) -> TernaryVAEV6Controllable:
    model = TernaryVAEV6Controllable(**build_model_kwargs(config)).to(torch.float64)
    return model


def collect_run_results(runs_root: Path) -> list[RunResultRow]:
    rows: list[RunResultRow] = []
    for results_path in sorted(runs_root.rglob("results.json")):
        try:
            payload = json.loads(results_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            run_dir = str(results_path.parent.relative_to(PROJECT_ROOT))
        except ValueError:
            run_dir = str(results_path.parent.relative_to(runs_root.parent))
        rows.append(
            RunResultRow(
                run_dir=run_dir,
                best_q=float(payload.get("best_Q", float("-inf"))),
                best_hierarchy=float(payload.get("best_hierarchy", 0.0)),
                best_coverage=float(payload.get("best_coverage", 0.0)),
                epochs_trained=int(payload.get("epochs_trained", 0)),
            )
        )
    rows.sort(key=lambda row: row.best_q, reverse=True)
    return rows


def summarize_surface(model: TernaryVAEV6Controllable) -> SurfaceSummary:
    factored = bool(model.factored)
    latent_dim = model.latent_dim
    radial_dims = model.projections.proj_A.radial_dims if factored else 0
    hyp_dim = latent_dim - radial_dims if factored else latent_dim
    return SurfaceSummary(
        input_shape="(B, 9)",
        training_input_domain="exact ternary vectors in {-1, 0, 1}^9",
        runtime_input_validation="none at encoder boundary; any float64 tensor of shape (B, 9) will run",
        output_logits_shape="(B, 27)",
        decoded_symbol_shape="(B, 9) via argmax over 9x3 logits",
        hyperbolic_shape=f"(B, {hyp_dim})",
        explicit_radius=factored,
        factored=factored,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def evaluate_direction_clustering(
    z_hyp: torch.Tensor,
    r: torch.Tensor | None,
    indices: torch.Tensor,
) -> list[DirectionClusteringRow]:
    if r is None:
        return []

    eps = torch.tensor(1e-10, dtype=z_hyp.dtype, device=z_hyp.device)
    direction = z_hyp / r.unsqueeze(-1).clamp(min=eps)
    valuations = TERNARY.valuation(indices)
    rows: list[DirectionClusteringRow] = []

    for level, prefix_depth in LEVEL_PREFIX_DEPTHS.items():
        mask = valuations == level
        n_samples = int(mask.sum().item())
        if n_samples < 2:
            continue

        direction_np = direction[mask].cpu().numpy()
        idx_level = indices[mask].cpu()
        prefix_labels = TERNARY.digit_prefix_class(idx_level, prefix_depth).numpy()
        n_classes = len(np.unique(prefix_labels))
        if n_classes < 2:
            continue

        labels = KMeans(n_clusters=n_classes, n_init=5, random_state=42).fit_predict(direction_np)
        rows.append(
            DirectionClusteringRow(
                level=level,
                prefix_depth=prefix_depth,
                n_samples=n_samples,
                n_classes=n_classes,
                ari=float(adjusted_rand_score(prefix_labels, labels)),
            )
        )

    return rows


def sample_decoder_prior(
    model: TernaryVAEV6Controllable,
    latent_dim: int,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.no_grad():
        z_tangent = torch.randn((n_samples, latent_dim), generator=generator, dtype=torch.float64)
        logits = model.decoder_A(z_tangent)
        decoded = logits.view(n_samples, 9, 3).argmax(dim=-1) - 1

    indices = TERNARY.from_ternary(decoded)
    valuations = TERNARY.valuation(indices)
    unique_indices = torch.unique(indices)
    hist = {
        str(level): int((valuations == level).sum().item())
        for level in range(TERNARY.MAX_VALUATION + 1)
        if int((valuations == level).sum().item()) > 0
    }

    samples = []
    for row in range(min(10, n_samples)):
        samples.append(
            {
                "index": int(indices[row].item()),
                "valuation": int(valuations[row].item()),
                "ternary": decoded[row].tolist(),
            }
        )

    return {
        "n_samples": n_samples,
        "unique_indices": int(unique_indices.numel()),
        "fraction_unique": float(unique_indices.numel() / n_samples),
        "space_coverage_fraction": float(unique_indices.numel() / TERNARY.N_OPERATIONS),
        "valuation_histogram": hist,
        "samples": samples,
    }


def monte_carlo_scenarios(
    observed_q: float,
    observed_coverage: float,
    observed_direction_ari_v0: float,
    trials: int = 50000,
    seed: int = 123,
) -> dict[str, Any]:
    """Transparent scenario model, not a claim of predictive validity.

    Assumptions:
    - Closed-domain value depends mainly on reconstruction quality + radial geometry.
    - External transfer depends on direction structure, external data evidence, and productization.
    - Commercial disruption additionally depends on integration and market execution.
    """
    rng = np.random.default_rng(seed)
    q = np.clip(rng.normal(observed_q, 0.02, size=trials), 0.0, 3.0)
    coverage = np.clip(rng.normal(observed_coverage, 0.003, size=trials), 0.0, 1.0)
    direction = np.clip(rng.normal(observed_direction_ari_v0, 0.05, size=trials), 0.0, 1.0)

    # These two factors are deliberately conservative because the repository
    # contains no external benchmark or serving stack proving them today.
    external_task_evidence = rng.beta(1.5, 9.5, size=trials)
    delivery_readiness = rng.beta(2.0, 6.0, size=trials)

    closed_domain_engine = (q > 2.10) & (coverage > 0.992)
    research_tooling = closed_domain_engine & (direction > 0.08)
    external_prediction_pilot = research_tooling & (direction > 0.20) & (external_task_evidence > 0.35)
    commercial_disruption = external_prediction_pilot & (delivery_readiness > 0.65)

    return {
        "trials": trials,
        "seed": seed,
        "assumptions": {
            "q_distribution": f"Normal(mean={observed_q:.4f}, sd=0.02)",
            "coverage_distribution": f"Normal(mean={observed_coverage:.4f}, sd=0.003)",
            "direction_ari_distribution": f"Normal(mean={observed_direction_ari_v0:.4f}, sd=0.05, clipped)",
            "external_task_evidence_distribution": "Beta(1.5, 9.5)",
            "delivery_readiness_distribution": "Beta(2.0, 6.0)",
            "note": "These last two are conservative subjective priors derived from the absence of external-task and deployment evidence in the repository.",
        },
        "probabilities": {
            "closed_domain_engine_value": float(closed_domain_engine.mean()),
            "research_tooling_value": float(research_tooling.mean()),
            "external_prediction_pilot": float(external_prediction_pilot.mean()),
            "commercial_disruption": float(commercial_disruption.mean()),
        },
    }


def evaluate_run(
    run_dir: Path,
    checkpoint_name: str = "best_Q.pt",
    generation_samples: int = 5000,
    scenario_trials: int = 50000,
) -> dict[str, Any]:
    run_dir = run_dir if run_dir.is_absolute() else (PROJECT_ROOT / run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint = load_checkpoint_compat(run_dir / "checkpoints" / checkpoint_name, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    all_ops = TERNARY.all_ternary().to(torch.float64)
    all_indices = torch.arange(len(all_ops), dtype=torch.long)
    with torch.no_grad():
        output = model(all_ops)

    curvature = model.projections.get_curvature()
    hierarchy = compute_hierarchy_metrics(output["z_A_hyp"], all_indices, curvature=curvature, seed=42)
    per_level_radii = []
    explicit_radius = output.get("r_A")
    if explicit_radius is not None:
        valuations = TERNARY.valuation(all_indices)
        for level in range(TERNARY.MAX_VALUATION + 1):
            mask = valuations == level
            if mask.any():
                per_level_radii.append(
                    {
                        "level": level,
                        "count": int(mask.sum().item()),
                        "mean": float(explicit_radius[mask].mean().item()),
                        "std": float(explicit_radius[mask].std(unbiased=False).item()),
                    }
                )

    direction_rows = evaluate_direction_clustering(output["z_A_hyp"], output.get("r_A"), all_indices)
    direction_ari_v0 = next((row.ari for row in direction_rows if row.level == 0), 0.0)

    evaluation = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "checkpoint": checkpoint_name,
        "surface": asdict(summarize_surface(model)),
        "full_domain_metrics": {
            "per_digit_accuracy": float(compute_accuracy(output["logits_A"], all_ops)),
            "perfect_reconstruction_coverage": float(compute_coverage(output["logits_A"], all_ops)),
            "curvature": float(curvature),
            "hierarchy": float(hierarchy["hierarchy"]),
            "dist_corr": float(hierarchy["dist_corr"]),
            "Q": float(hierarchy["Q"]),
        },
        "per_level_radii": per_level_radii,
        "direction_clustering": [asdict(row) for row in direction_rows],
        "generation_from_decoder_prior": sample_decoder_prior(model, model.latent_dim, generation_samples, seed=123),
    }
    evaluation["scenario_monte_carlo"] = monte_carlo_scenarios(
        observed_q=evaluation["full_domain_metrics"]["Q"],
        observed_coverage=evaluation["full_domain_metrics"]["perfect_reconstruction_coverage"],
        observed_direction_ari_v0=direction_ari_v0,
        trials=scenario_trials,
        seed=123,
    )
    return evaluation


def format_text_report(run_rows: list[RunResultRow], evaluation: dict[str, Any]) -> str:
    lines = []
    lines.append("Run leaderboard (results.json)")
    for row in run_rows[:10]:
        lines.append(
            f"- {row.run_dir}: best_Q={row.best_q:.6f}, "
            f"best_hierarchy={row.best_hierarchy:.6f}, "
            f"best_coverage={row.best_coverage:.6f}, epochs={row.epochs_trained}"
        )

    full = evaluation["full_domain_metrics"]
    lines.append("")
    lines.append(f"Checkpoint evaluation: {evaluation['run_dir']} / {evaluation['checkpoint']}")
    lines.append(
        f"- full-domain accuracy={full['per_digit_accuracy']:.6f}, "
        f"coverage={full['perfect_reconstruction_coverage']:.6f}, "
        f"Q={full['Q']:.6f}, hierarchy={full['hierarchy']:.6f}, dist_corr={full['dist_corr']:.6f}"
    )

    gen = evaluation["generation_from_decoder_prior"]
    lines.append(
        f"- decoder-prior generation: unique={gen['unique_indices']} / {gen['n_samples']}, "
        f"fraction_unique={gen['fraction_unique']:.4f}, "
        f"space_coverage={gen['space_coverage_fraction']:.4f}"
    )

    scenario = evaluation["scenario_monte_carlo"]["probabilities"]
    lines.append(
        "- scenario Monte Carlo: "
        f"closed-domain={scenario['closed_domain_engine_value']:.3f}, "
        f"research={scenario['research_tooling_value']:.3f}, "
        f"external-pilot={scenario['external_prediction_pilot']:.3f}, "
        f"commercial={scenario['commercial_disruption']:.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run artifacts and checkpoint-backed project feasibility.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=str, default="best_Q.pt")
    parser.add_argument("--generation-samples", type=int, default=5000)
    parser.add_argument("--scenario-trials", type=int, default=50000)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--text-out", type=Path, default=None)
    args = parser.parse_args()

    run_rows = collect_run_results(args.runs_root)
    evaluation = evaluate_run(
        run_dir=args.run_dir,
        checkpoint_name=args.checkpoint,
        generation_samples=args.generation_samples,
        scenario_trials=args.scenario_trials,
    )

    payload = {
        "run_leaderboard": [asdict(row) for row in run_rows[:20]],
        "evaluation": evaluation,
    }

    rendered = format_text_report(run_rows, evaluation)
    print(rendered)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.text_out is not None:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
