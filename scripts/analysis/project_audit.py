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
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import torch.nn.functional as F
import yaml

from src.core import TERNARY
from src.geometry.poincare import poincare_distance
from src.models.vae import TernaryVAEV6Controllable
from src.symbolic import build_symbolic_subsystem
from src.train import compute_accuracy, compute_coverage, compute_hierarchy_metrics
from src.utils.checkpoint import load_checkpoint_compat

DEFAULT_RUN = PROJECT_ROOT / "runs" / "v7_large_20260324_013725"
LEVEL_PREFIX_DEPTHS = {0: 3, 1: 4, 2: 3, 3: 4, 4: 5, 5: 6}
SYMBOLIC_SUBSYSTEM = build_symbolic_subsystem({"enabled": True, "backend": "finite_group"})


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


@dataclass(frozen=True)
class ScalarTrajectorySummary:
    tag: str
    optimize: str
    best_step: int
    best_value: float
    last_step: int
    last_value: float
    tail_mean: float
    tail_std: float


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


def summarize_scalar_series(
    tag: str,
    points: list[tuple[int, float]],
    optimize: str = "max",
    tail_size: int = 20,
) -> ScalarTrajectorySummary | None:
    if not points:
        return None

    clean = [(step, value) for step, value in points if not np.isnan(value)]
    if not clean:
        return None

    if optimize == "min":
        best_step, best_value = min(clean, key=lambda item: item[1])
    else:
        best_step, best_value = max(clean, key=lambda item: item[1])

    last_step, last_value = clean[-1]
    tail = [value for _, value in clean[-tail_size:]]
    return ScalarTrajectorySummary(
        tag=tag,
        optimize=optimize,
        best_step=int(best_step),
        best_value=float(best_value),
        last_step=int(last_step),
        last_value=float(last_value),
        tail_mean=float(np.mean(tail)),
        tail_std=float(np.std(tail)),
    )


def summarize_training_curves(run_dir: Path) -> dict[str, Any]:
    tb_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("ternary_vae_"))
    if not tb_dirs:
        return {"available": False, "summaries": {}}

    acc = EventAccumulator(str(tb_dirs[0]))
    acc.Reload()
    tags = {
        "Accuracy/val": "max",
        "Coverage": "max",
        "Hierarchy/Q_VAE_A": "max",
        "Hierarchy/corr_VAE_A": "max",
        "Hierarchy/dist_corr": "max",
        "Direction/ARI_v0": "max",
        "Direction/ARI_v1": "max",
        "Direction/ARI_composite": "max",
        "TreeCoherence/VAE_A": "min",
    }

    summaries: dict[str, Any] = {}
    for tag, optimize in tags.items():
        scalars = acc.Scalars(tag)
        series = [(event.step, event.value) for event in scalars]
        summary = summarize_scalar_series(tag, series, optimize=optimize)
        if summary is not None:
            summaries[tag] = asdict(summary)

    return {"available": True, "summaries": summaries}


def stratified_probe_indices(
    labels: np.ndarray,
    sample_budget: int = 2000,
    min_per_class: int = 20,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    n_total = max(1, len(labels))
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        proportional = int(round(sample_budget * len(cls_idx) / n_total))
        take = min(len(cls_idx), max(min_per_class, proportional))
        selected.append(rng.choice(cls_idx, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def _fit_linear_probe(
    features_train: np.ndarray,
    features_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features_train)
    test_scaled = scaler.transform(features_test)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(train_scaled, labels_train)
    predicted = classifier.predict(test_scaled)
    return {
        "accuracy": float(accuracy_score(labels_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_test, predicted)),
        "macro_f1": float(f1_score(labels_test, predicted, average="macro")),
    }


def _fit_knn_probe(
    features_train: np.ndarray,
    features_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features_train)
    test_scaled = scaler.transform(features_test)
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_scaled, labels_train)
    predicted = classifier.predict(test_scaled)
    return {
        "accuracy": float(accuracy_score(labels_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_test, predicted)),
        "macro_f1": float(f1_score(labels_test, predicted, average="macro")),
    }


def _probe_feature_family(
    features_train: np.ndarray,
    features_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    raw_train: np.ndarray,
    knn_neighbors: int,
    trustworthiness_size: int,
) -> dict[str, Any]:
    trust_n = min(trustworthiness_size, len(raw_train))
    trust_k = min(knn_neighbors, max(1, (trust_n - 1) // 2))
    trust = float("nan")
    if trust_n >= 3:
        trust = float(
            trustworthiness(raw_train[:trust_n], features_train[:trust_n], n_neighbors=trust_k)
        )
    return {
        "linear_probe": _fit_linear_probe(features_train, features_test, labels_train, labels_test),
        "knn_probe": _fit_knn_probe(
            features_train,
            features_test,
            labels_train,
            labels_test,
            n_neighbors=knn_neighbors,
        ),
        "trustworthiness_k15": trust,
    }


def representation_probe_suite(
    z_hyp: torch.Tensor,
    z_tangent: torch.Tensor,
    raw_inputs: torch.Tensor,
    indices: torch.Tensor,
    sample_budget: int = 2000,
    max_level: int = 6,
    min_per_class: int = 20,
    knn_neighbors: int = 15,
    trustworthiness_size: int = 800,
    seed: int = 42,
) -> dict[str, Any]:
    valuations = TERNARY.valuation(indices).cpu().numpy()
    raw_np = raw_inputs.cpu().numpy()
    z_np = z_hyp.detach().cpu().numpy()
    z_tangent_np = z_tangent.detach().cpu().numpy()

    mask = valuations <= max_level
    filtered_labels = valuations[mask]
    filtered_raw = raw_np[mask]
    filtered_z = z_np[mask]
    filtered_tangent = z_tangent_np[mask]
    sampled_idx = stratified_probe_indices(
        filtered_labels,
        sample_budget=sample_budget,
        min_per_class=min_per_class,
        seed=seed,
    )
    sampled_labels = filtered_labels[sampled_idx]
    sampled_raw = filtered_raw[sampled_idx]
    sampled_z = filtered_z[sampled_idx]
    sampled_tangent = filtered_tangent[sampled_idx]
    split_idx = np.arange(len(sampled_labels))
    train_idx, test_idx, labels_train, labels_test = train_test_split(
        split_idx,
        sampled_labels,
        test_size=0.2,
        random_state=seed,
        stratify=sampled_labels,
    )
    raw_train = sampled_raw[train_idx]
    raw_test = sampled_raw[test_idx]
    emb_train = sampled_z[train_idx]
    emb_test = sampled_z[test_idx]
    tangent_train = sampled_tangent[train_idx]
    tangent_test = sampled_tangent[test_idx]
    return {
        "levels_included": list(range(max_level + 1)),
        "levels_excluded_due_to_sparse_support": list(range(max_level + 1, TERNARY.MAX_VALUATION + 1)),
        "sample_size": len(sampled_labels),
        "class_counts": {
            str(level): int((sampled_labels == level).sum())
            for level in np.unique(sampled_labels)
        },
        "raw_input": {
            "linear_probe": _fit_linear_probe(raw_train, raw_test, labels_train, labels_test),
            "knn_probe": _fit_knn_probe(
                raw_train, raw_test, labels_train, labels_test, n_neighbors=knn_neighbors
            ),
        },
        "tangent_euclidean": _probe_feature_family(
            tangent_train,
            tangent_test,
            labels_train,
            labels_test,
            raw_train,
            knn_neighbors=knn_neighbors,
            trustworthiness_size=trustworthiness_size,
        ),
        "hyperbolic_embedding": _probe_feature_family(
            emb_train,
            emb_test,
            labels_train,
            labels_test,
            raw_train,
            knn_neighbors=knn_neighbors,
            trustworthiness_size=trustworthiness_size,
        ),
        "notes": [
            "Probe labels are valuation levels, which are internally derived rather than external semantic labels.",
            "Levels 7-9 are excluded by default because they are too sparse for a defensible stratified train/test split.",
            "Raw-input baselines are reported to prevent overstating learned-feature value when the original digits already encode the label strongly.",
            "Euclidean tangent baselines are reported to test whether hyperbolic projection adds value beyond the sampled latent fed directly to the decoder.",
        ],
    }


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


def reconstruction_quality(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> dict[str, float]:
    logits_3 = logits.view(-1, 9, 3)
    targets_3 = (targets.long() + 1)
    flat_logits = logits_3.reshape(-1, 3)
    flat_targets = targets_3.reshape(-1)

    cross_entropy = F.cross_entropy(logits_3.permute(0, 2, 1), targets_3).item()
    probs = flat_logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == flat_targets).double()
    bins = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece = torch.tensor(0.0, dtype=torch.float64)
    for idx in range(n_bins):
        lo = bins[idx]
        hi = bins[idx + 1]
        mask = (conf > lo) & (conf <= hi) if idx > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            frac = mask.double().mean()
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece = ece + frac * (acc - avg_conf).abs()

    one_hot = F.one_hot(flat_targets, num_classes=3).double()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()
    return {
        "cross_entropy": float(cross_entropy),
        "ece_15bin": float(ece.item()),
        "brier": float(brier),
    }


def _retrieval_metrics_from_distances(
    distances: torch.Tensor,
    idx_eval: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    valuations = TERNARY.valuation(idx_eval)
    parents = TERNARY.parent(idx_eval)
    distances = distances.clone()
    distances.fill_diagonal_(float("inf"))
    nn = distances.topk(k, largest=False).indices
    nn_vals = valuations[nn]
    nn1 = nn[:, 0]

    present = {int(value.item()): pos for pos, value in enumerate(idx_eval)}
    parent_hits: list[int] = []
    for row, parent in enumerate(parents):
        parent_index = int(parent.item())
        if parent_index < 0 or parent_index not in present:
            continue
        parent_hits.append(int(present[parent_index] in nn[row].tolist()))

    return {
        "sample_size": len(idx_eval),
        "k": k,
        "valuation_nn1_accuracy": float((valuations[nn1] == valuations).double().mean().item()),
        "same_valuation_precision_at_k": float((nn_vals == valuations[:, None]).double().mean().item()),
        "parent_hit_at_k": (
            float(sum(parent_hits) / len(parent_hits)) if parent_hits else float("nan")
        ),
    }


def retrieval_ablation_suite(
    z_hyp: torch.Tensor,
    z_tangent: torch.Tensor,
    indices: torch.Tensor,
    sample_size: int = 2000,
    k: int = 10,
    seed: int = 123,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if len(indices) > sample_size:
        selected = np.sort(rng.choice(len(indices), sample_size, replace=False))
        z_hyp_eval = z_hyp[selected]
        z_tangent_eval = z_tangent[selected]
        idx_eval = indices[selected]
    else:
        z_hyp_eval = z_hyp
        z_tangent_eval = z_tangent
        idx_eval = indices

    with torch.no_grad():
        hyp_distances = poincare_distance(z_hyp_eval[:, None, :], z_hyp_eval[None, :, :], c=1.0)
        tangent_distances = torch.cdist(z_tangent_eval, z_tangent_eval, p=2)
    return {
        "sample_size": len(idx_eval),
        "k": k,
        "hyperbolic_embedding": _retrieval_metrics_from_distances(hyp_distances, idx_eval, k),
        "tangent_euclidean": _retrieval_metrics_from_distances(tangent_distances, idx_eval, k),
    }


def _pair_distances(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    with torch.no_grad():
        if metric == "poincare":
            return poincare_distance(lhs, rhs, c=1.0)
        return torch.norm(lhs - rhs, dim=-1)


def _cross_distances(
    queries: torch.Tensor,
    bank: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    with torch.no_grad():
        if metric == "poincare":
            return poincare_distance(queries[:, None, :], bank[None, :, :], c=1.0)
        return torch.cdist(queries, bank, p=2)


def _ranking_metrics(distance_matrix: torch.Tensor, target_positions: torch.Tensor, k: int = 10) -> dict[str, float]:
    order = distance_matrix.argsort(dim=1)
    target_positions = target_positions[:, None]
    ranks = (order == target_positions).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "recall_at_1": float((ranks <= 1).double().mean().item()),
        "recall_at_10": float((ranks <= k).double().mean().item()),
        "mrr": float((1.0 / ranks.double()).mean().item()),
    }


def symbolic_orbit_retrieval_benchmark(
    z_hyp: torch.Tensor,
    z_tangent: torch.Tensor,
    raw_inputs: torch.Tensor,
    indices: torch.Tensor,
    orbit_sample_size: int = 512,
    seed: int = 321,
) -> dict[str, Any]:
    canon = SYMBOLIC_SUBSYSTEM.canonicalize(indices.cpu())
    orbit_reps = torch.unique(canon)
    rng = np.random.default_rng(seed)
    if len(orbit_reps) > orbit_sample_size:
        selected = np.sort(rng.choice(len(orbit_reps), size=orbit_sample_size, replace=False))
        orbit_reps = orbit_reps[selected]

    queries = SYMBOLIC_SUBSYSTEM.choose_non_identity_partner(orbit_reps, seed=seed).to(indices.device)
    bank_positions = orbit_reps.long()
    query_positions = queries.long()
    target_positions = torch.arange(len(bank_positions), dtype=torch.long)

    banks = {
        "hyperbolic_embedding": z_hyp[bank_positions],
        "tangent_euclidean": z_tangent[bank_positions],
        "raw_input": raw_inputs[bank_positions],
    }
    query_reps = {
        "hyperbolic_embedding": z_hyp[query_positions],
        "tangent_euclidean": z_tangent[query_positions],
        "raw_input": raw_inputs[query_positions],
    }
    metrics = {}
    for key, metric in (
        ("hyperbolic_embedding", "poincare"),
        ("tangent_euclidean", "euclidean"),
        ("raw_input", "euclidean"),
    ):
        distances = _cross_distances(query_reps[key], banks[key], metric=metric)
        metrics[key] = _ranking_metrics(distances, target_positions)
    return {
        "sample_size": len(bank_positions),
        "backend": SYMBOLIC_SUBSYSTEM.name,
        "description": SYMBOLIC_SUBSYSTEM.describe(),
        "group_size": len(SYMBOLIC_SUBSYSTEM.engine.elements),
        "notes": [
            "Each query is a non-identity symbolic transform of one orbit representative.",
            "Candidate banks contain one canonical representative per sampled symbolic orbit.",
        ],
        **metrics,
    }


def symbolic_pair_verification_benchmark(
    z_hyp: torch.Tensor,
    z_tangent: torch.Tensor,
    raw_inputs: torch.Tensor,
    indices: torch.Tensor,
    pair_sample_size: int = 2048,
    seed: int = 321,
) -> dict[str, Any]:
    pairs = SYMBOLIC_SUBSYSTEM.sample_feedback_pairs(indices.cpu(), sample_size=pair_sample_size)
    anchors = pairs["anchors"].long()
    positives = pairs["positives"].long()
    negatives = pairs["negatives"].long()

    labels = np.concatenate([np.ones(len(anchors)), np.zeros(len(anchors))])
    metrics = {}
    for key, metric in (
        ("hyperbolic_embedding", "poincare"),
        ("tangent_euclidean", "euclidean"),
        ("raw_input", "euclidean"),
    ):
        features = {
            "hyperbolic_embedding": z_hyp,
            "tangent_euclidean": z_tangent,
            "raw_input": raw_inputs,
        }[key]
        positive_scores = -_pair_distances(features[anchors], features[positives], metric=metric).cpu().numpy()
        negative_scores = -_pair_distances(features[anchors], features[negatives], metric=metric).cpu().numpy()
        scores = np.concatenate([positive_scores, negative_scores])
        metrics[key] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    return {
        "sample_size": len(anchors),
        "notes": pairs["notes"],
        **metrics,
    }


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
    true_counts = torch.tensor(
        [TERNARY.level_count(level) for level in range(TERNARY.MAX_VALUATION + 1)],
        dtype=torch.float64,
    )
    true_distribution = true_counts / true_counts.sum()
    generated_counts = torch.tensor(
        [(valuations == level).sum().item() for level in range(TERNARY.MAX_VALUATION + 1)],
        dtype=torch.float64,
    )
    generated_distribution = generated_counts / generated_counts.sum().clamp(min=1.0)
    midpoint = 0.5 * (true_distribution + generated_distribution)
    js_divergence = 0.5 * (
        (true_distribution * (true_distribution / midpoint).log()).sum()
        + (generated_distribution * (generated_distribution / midpoint).log()).sum()
    )

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
        "valuation_js_divergence": float(js_divergence.item()),
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
    evaluation_seed: int = 42,
) -> dict[str, Any]:
    run_dir = run_dir if run_dir.is_absolute() else (PROJECT_ROOT / run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint = load_checkpoint_compat(run_dir / "checkpoints" / checkpoint_name, map_location="cpu")
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    all_ops = TERNARY.all_ternary().to(torch.float64)
    all_indices = torch.arange(len(all_ops), dtype=torch.long)
    torch.manual_seed(evaluation_seed)
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
        "evaluation_seed": evaluation_seed,
        "surface": asdict(summarize_surface(model)),
        "full_domain_metrics": {
            "per_digit_accuracy": float(compute_accuracy(output["logits_A"], all_ops)),
            "perfect_reconstruction_coverage": float(compute_coverage(output["logits_A"], all_ops)),
            "curvature": float(curvature),
            "hierarchy": float(hierarchy["hierarchy"]),
            "dist_corr": float(hierarchy["dist_corr"]),
            "Q": float(hierarchy["Q"]),
        },
        "reconstruction_quality": reconstruction_quality(output["logits_A"], all_ops),
        "retrieval_ablation": retrieval_ablation_suite(
            output["z_A_hyp"],
            output["z_A_tangent"],
            all_indices,
        ),
        "symbolic_orbit_retrieval": symbolic_orbit_retrieval_benchmark(
            output["z_A_hyp"],
            output["z_A_tangent"],
            all_ops,
            all_indices,
        ),
        "symbolic_pair_verification": symbolic_pair_verification_benchmark(
            output["z_A_hyp"],
            output["z_A_tangent"],
            all_ops,
            all_indices,
        ),
        "representation_probe_suite": representation_probe_suite(
            output["z_A_hyp"],
            output["z_A_tangent"],
            all_ops,
            all_indices,
        ),
        "training_curves": summarize_training_curves(run_dir),
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
        f"space_coverage={gen['space_coverage_fraction']:.4f}, "
        f"valuation_jsd={gen['valuation_js_divergence']:.4f}"
    )
    recon = evaluation["reconstruction_quality"]
    lines.append(
        f"- reconstruction quality: ce={recon['cross_entropy']:.6f}, "
        f"ece15={recon['ece_15bin']:.6f}, brier={recon['brier']:.6f}"
    )
    retrieval = evaluation["retrieval_ablation"]
    retrieval_hyp = retrieval["hyperbolic_embedding"]
    retrieval_tangent = retrieval["tangent_euclidean"]
    lines.append(
        "- retrieval ablation: "
        f"hyp_val-nn1={retrieval_hyp['valuation_nn1_accuracy']:.4f}, "
        f"euc_val-nn1={retrieval_tangent['valuation_nn1_accuracy']:.4f}, "
        f"hyp_parent-hit@{retrieval['k']}={retrieval_hyp['parent_hit_at_k']:.4f}, "
        f"euc_parent-hit@{retrieval['k']}={retrieval_tangent['parent_hit_at_k']:.4f}"
    )
    probe = evaluation["representation_probe_suite"]
    probe_hyp = probe["hyperbolic_embedding"]
    probe_tangent = probe["tangent_euclidean"]
    probe_raw = probe["raw_input"]
    lines.append(
        "- probes (levels 0-6): "
        f"linear_hyp_acc={probe_hyp['linear_probe']['accuracy']:.4f}, "
        f"linear_euc_acc={probe_tangent['linear_probe']['accuracy']:.4f}, "
        f"linear_raw_acc={probe_raw['linear_probe']['accuracy']:.4f}, "
        f"knn_hyp_acc={probe_hyp['knn_probe']['accuracy']:.4f}, "
        f"knn_euc_acc={probe_tangent['knn_probe']['accuracy']:.4f}, "
        f"knn_raw_acc={probe_raw['knn_probe']['accuracy']:.4f}, "
        f"trust_hyp@15={probe_hyp['trustworthiness_k15']:.4f}, "
        f"trust_euc@15={probe_tangent['trustworthiness_k15']:.4f}"
    )
    symbolic_retrieval = evaluation["symbolic_orbit_retrieval"]
    sym_pair = evaluation["symbolic_pair_verification"]
    lines.append(
        "- symbolic orbit retrieval: "
        f"hyp_r1={symbolic_retrieval['hyperbolic_embedding']['recall_at_1']:.4f}, "
        f"euc_r1={symbolic_retrieval['tangent_euclidean']['recall_at_1']:.4f}, "
        f"raw_r1={symbolic_retrieval['raw_input']['recall_at_1']:.4f}, "
        f"hyp_mrr={symbolic_retrieval['hyperbolic_embedding']['mrr']:.4f}"
    )
    lines.append(
        "- symbolic pair verification: "
        f"hyp_auc={sym_pair['hyperbolic_embedding']['roc_auc']:.4f}, "
        f"euc_auc={sym_pair['tangent_euclidean']['roc_auc']:.4f}, "
        f"raw_auc={sym_pair['raw_input']['roc_auc']:.4f}, "
        f"hyp_ap={sym_pair['hyperbolic_embedding']['average_precision']:.4f}"
    )
    curve_q = evaluation["training_curves"]["summaries"].get("Hierarchy/Q_VAE_A")
    if curve_q is not None:
        lines.append(
            "- training curve Q: "
            f"peak={curve_q['best_value']:.6f}@{curve_q['best_step']}, "
            f"last={curve_q['last_value']:.6f}@{curve_q['last_step']}, "
            f"tail_mean={curve_q['tail_mean']:.6f}"
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
