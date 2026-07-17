"""Fase 4 of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md: does embedding
distance between cytochrome c orthologs recover real taxonomic distance,
for Conditions A (Euclidean) / B (generic hyperbolic) / C (p-adic)?

Per-species embedding = mean over that species' aligned windows (arithmetic
mean of mu for A; mean in tangent space via log_map_zero, then exp_map_zero
back to the ball, for B/C -- consistent with how the rest of the codebase
aggregates hyperbolic points). Distance matrices: Euclidean cdist for A,
Poincare geodesic for B/C. Correlated against taxonomic_distance.npy
(Fase 1) via a Mantel permutation test, since distance-matrix entries share
species and are not independent -- a naive Spearman p-value would be
inflated. Bootstrap over species gives a correlation confidence interval.

Known limitations (stated here rather than glossed over in the results):
- No species-level train/eval holdout exists yet. Fase 3's training scripts
  split by *window row*, not by species, so every species that has any
  window in indices.pt was seen during training. A genuine held-out-species
  evaluation requires regenerating indices.pt with some species' windows
  excluded before training -- deferred to a follow-up run, not faked here.
- 62% of window indices collide across species (161 unique / 429 windows),
  a direct consequence of the coarse 3-symbol hydropathy encoding collapsing
  conserved regions to the same digit pattern. This script reports the
  collision rate explicitly (see `index_collision` in the output) so a high
  correlation isn't misread as proof of learned structure when it may partly
  reflect encoding coarseness shared verbatim between species.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.project_audit import build_model_from_config, load_yaml
from src.geometry.poincare import exp_map_zero, log_map_zero, poincare_distance_matrix
from src.models.vae_baseline import TernaryVAEEuclideanBaseline
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat

Condition = dict[str, Any]


def load_species_data(
    window_map_path: Path, indices_path: Path, taxonomy_dir: Path,
) -> tuple[list[dict], torch.Tensor, list[str], np.ndarray]:
    window_map = json.loads(window_map_path.read_text())
    indices = torch.load(indices_path, weights_only=True).long()
    if len(window_map) != len(indices):
        raise ValueError(f"window_map ({len(window_map)}) and indices ({len(indices)}) length mismatch")

    species_order_all = json.loads((taxonomy_dir / "species_order.json").read_text())
    tax_dist_all = np.load(taxonomy_dir / "taxonomic_distance.npy")

    species_order = sorted({w["species"] for w in window_map})
    missing = [s for s in species_order if s not in species_order_all]
    if missing:
        raise ValueError(f"Species in window_map but not in taxonomy: {missing}")
    keep_idx = [species_order_all.index(s) for s in species_order]
    tax_dist = tax_dist_all[np.ix_(keep_idx, keep_idx)]

    return window_map, indices, species_order, tax_dist


def index_collision_report(window_map: list[dict], indices: torch.Tensor) -> dict[str, Any]:
    idx_to_species: dict[int, set] = defaultdict(set)
    for w, idx in zip(window_map, indices.tolist()):
        idx_to_species[idx].add(w["species"])
    cross_species = sum(1 for sp in idx_to_species.values() if len(sp) > 1)
    n_unique = len(idx_to_species)
    return {
        "n_windows": len(window_map),
        "n_unique_indices": n_unique,
        "n_indices_shared_across_species": cross_species,
        "collision_rate": round(1.0 - n_unique / len(window_map), 4),
    }


def _species_masks(window_map: list[dict], species_order: list[str]) -> list[list[int]]:
    return [[i for i, w in enumerate(window_map) if w["species"] == sp] for sp in species_order]


def embed_condition_a(
    checkpoint_path: Path, indices: torch.Tensor, window_map: list[dict],
    species_order: list[str], device: torch.device,
) -> np.ndarray:
    ckpt = load_checkpoint_compat(checkpoint_path, map_location=device)
    model = TernaryVAEEuclideanBaseline(
        latent_dim=ckpt["latent_dim"], hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(get_model_state_dict(ckpt))
    model.eval()
    with torch.no_grad():
        mu = model.get_mu_representations(indices, device)

    masks = _species_masks(window_map, species_order)
    points = torch.stack([mu[m].mean(dim=0) for m in masks])
    return torch.cdist(points, points).cpu().numpy()


def embed_condition_hyperbolic(
    run_dir: Path, indices: torch.Tensor, window_map: list[dict],
    species_order: list[str], device: torch.device,
) -> np.ndarray:
    config = load_yaml(run_dir / "config.yaml")
    model = build_model_from_config(config).to(device)
    ckpt = load_checkpoint_compat(run_dir / "checkpoints" / "best_Q.pt", map_location=device)
    model.load_state_dict(get_model_state_dict(ckpt))
    model.eval()
    c = model.projections.get_curvature()
    with torch.no_grad():
        z_hyp = model.get_hyperbolic_representations(indices, device)
        tangent = log_map_zero(z_hyp, c=c)

    masks = _species_masks(window_map, species_order)
    mean_tangent = torch.stack([tangent[m].mean(dim=0) for m in masks])
    hyp_points = exp_map_zero(mean_tangent, c=c)
    return poincare_distance_matrix(hyp_points, c=c).cpu().numpy()


def mantel_test(
    dist_model: np.ndarray, dist_taxonomy: np.ndarray, n_permutations: int, seed: int,
) -> dict[str, Any]:
    n = dist_model.shape[0]
    iu = np.triu_indices(n, k=1)
    model_flat = dist_model[iu]
    tax_flat = dist_taxonomy[iu]
    observed, _ = spearmanr(model_flat, tax_flat)

    rng = np.random.default_rng(seed)
    null_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        permuted = dist_taxonomy[np.ix_(perm, perm)][iu]
        null_corrs[i], _ = spearmanr(model_flat, permuted)

    # One-sided: alternative hypothesis is a positive correlation (embedding
    # distance grows with real taxonomic distance).
    p_value = float((np.sum(null_corrs >= observed) + 1) / (n_permutations + 1))
    return {
        "observed_spearman": float(observed),
        "p_value_one_sided": p_value,
        "null_mean": float(null_corrs.mean()),
        "null_std": float(null_corrs.std()),
        "n_permutations": n_permutations,
        "n_species_pairs": int(len(model_flat)),
    }


def bootstrap_ci(
    dist_model: np.ndarray, dist_taxonomy: np.ndarray, n_boot: int, seed: int,
) -> dict[str, Any]:
    n = dist_model.shape[0]
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        sample = rng.choice(n, size=n, replace=True)
        iu = np.triu_indices(n, k=1)
        # Bootstrap-resampled species pairs where sample[i] == sample[j] are
        # self-distances (0 by construction) and carry no signal; drop them.
        keep = sample[iu[0]] != sample[iu[1]]
        if keep.sum() < 3:
            continue
        sub_model = dist_model[np.ix_(sample, sample)][iu][keep]
        sub_tax = dist_taxonomy[np.ix_(sample, sample)][iu][keep]
        boots[i], _ = spearmanr(sub_model, sub_tax)
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_valid_bootstrap": int(len(boots))}


def evaluate_condition(
    name: str, dist_model: np.ndarray, dist_taxonomy: np.ndarray,
    n_permutations: int, n_bootstrap: int, seed: int,
) -> dict[str, Any]:
    mantel = mantel_test(dist_model, dist_taxonomy, n_permutations, seed)
    ci = bootstrap_ci(dist_model, dist_taxonomy, n_bootstrap, seed)
    print(
        f"[{name}] Spearman={mantel['observed_spearman']:.4f} "
        f"(Mantel p={mantel['p_value_one_sided']:.4f}, "
        f"bootstrap 95% CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}], "
        f"n_pairs={mantel['n_species_pairs']})"
    )
    return {**mantel, **ci}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-map", default="data/cytochrome_c/window_map.json")
    parser.add_argument("--indices", default="data/cytochrome_c/indices.pt")
    parser.add_argument("--taxonomy-dir", default="data/cytochrome_c")
    parser.add_argument("--run-a-checkpoint", default=None, help="Condition A checkpoint .pt (train_euclidean_baseline.py output)")
    parser.add_argument("--run-b-dir", default=None, help="Condition B run dir (src/train.py output, contains config.yaml + checkpoints/best_Q.pt)")
    parser.add_argument("--run-c-dir", default=None, help="Condition C run dir")
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/cytochrome_c/phylogeny_recovery_results.json")
    args = parser.parse_args()

    if not any([args.run_a_checkpoint, args.run_b_dir, args.run_c_dir]):
        raise SystemExit("Provide at least one of --run-a-checkpoint / --run-b-dir / --run-c-dir")

    device = torch.device("cpu")
    window_map, indices, species_order, tax_dist = load_species_data(
        Path(args.window_map), Path(args.indices), Path(args.taxonomy_dir),
    )
    collisions = index_collision_report(window_map, indices)
    print(
        f"[data] {len(species_order)} species, {collisions['n_windows']} windows, "
        f"{collisions['n_unique_indices']} unique indices "
        f"(collision_rate={collisions['collision_rate']:.2%})"
    )

    results: dict[str, Any] = {"species_order": species_order, "index_collision": collisions, "conditions": {}}

    if args.run_a_checkpoint:
        dist_a = embed_condition_a(Path(args.run_a_checkpoint), indices, window_map, species_order, device)
        results["conditions"]["A_euclidean"] = evaluate_condition(
            "A_euclidean", dist_a, tax_dist, args.n_permutations, args.n_bootstrap, args.seed,
        )

    if args.run_b_dir:
        dist_b = embed_condition_hyperbolic(Path(args.run_b_dir), indices, window_map, species_order, device)
        results["conditions"]["B_hyperbolic_generic"] = evaluate_condition(
            "B_hyperbolic_generic", dist_b, tax_dist, args.n_permutations, args.n_bootstrap, args.seed,
        )

    if args.run_c_dir:
        dist_c = embed_condition_hyperbolic(Path(args.run_c_dir), indices, window_map, species_order, device)
        results["conditions"]["C_padic"] = evaluate_condition(
            "C_padic", dist_c, tax_dist, args.n_permutations, args.n_bootstrap, args.seed,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
