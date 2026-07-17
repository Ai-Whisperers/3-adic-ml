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

Condition 0 (raw_encoding_baseline): Euclidean distance between species'
raw hydropathy-encoded aligned sequences (window_map.json residues,
concatenated in window_idx order) -- zero model, zero training. Always
computed. This exists because conserved-sequence-implies-close-relative is
textbook molecular phylogenetics: a 2026-07-17 check found that this trivial
signal alone (fraction of aligned positions with an identical ternary digit
between two species) already gives Spearman~0.72 against real taxonomic
distance (n=741 species pairs, p~1e-117). Without this baseline, a positive
Mantel result for A/B/C is uninterpretable -- it could mean the architecture
learned something, or it could mean the encoding alone already carries most
of the signal. The bar for "the architecture helped" is beating this number,
not beating zero.

Known limitations (also recorded in `caveats` in the output JSON, not just
here, so a reader of the JSON alone still sees them):
- No species-level train/eval holdout exists yet. Fase 3's training scripts
  split by *window row*, not by species, so every species that has any
  window in indices.pt was seen during training. A genuine held-out-species
  evaluation requires regenerating indices.pt with some species' windows
  excluded before training -- deferred to a follow-up run, not faked here.
- Window indices collide across species (see `index_collision` in the
  output), a direct consequence of the coarse 3-symbol hydropathy encoding
  collapsing conserved regions to the same digit pattern. This is exactly
  what raw_encoding_baseline quantifies directly -- see above.
- Conditions B/C's distance matrix is built from VAE-A only (the primary
  coverage pathway); VAE-B is checked separately for directional collapse
  (`vae_b_health` per condition) but does not otherwise contribute to the
  reported distances.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.project_audit import build_model_from_config, load_yaml
from scripts.data.peptide_encoding import AA_MAP
from scripts.data.prepare_cytochrome_c_dataset import encode_window_to_index
from src.geometry.poincare import exp_map_zero, log_map_zero, poincare_distance_matrix
from src.models.vae_baseline import TernaryVAEEuclideanBaseline
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAVEATS = [
    "No species-level train/eval holdout: every species with a window in "
    "indices.pt was seen during training (Fase 3's split is by window row, "
    "not species).",
    "Index collisions across species exist (see index_collision below) due "
    "to the coarse 3-symbol hydropathy encoding; correlation may partly "
    "reflect encoding coarseness rather than learned structure. "
    "raw_encoding_baseline quantifies this directly -- A/B/C must beat it, "
    "not beat zero, to demonstrate the architecture learned anything.",
    "B/C distance matrices are built from VAE-A only; see vae_b_health per "
    "condition for a directional-collapse check on VAE-B.",
]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def load_species_data(
    window_map_path: Path, indices_path: Path, taxonomy_dir: Path,
) -> tuple[list[dict], torch.Tensor, list[str], np.ndarray]:
    window_map = json.loads(window_map_path.read_text())
    indices = torch.load(indices_path, weights_only=True).long()
    if len(window_map) != len(indices):
        raise ValueError(f"window_map ({len(window_map)}) and indices ({len(indices)}) length mismatch")

    # Length matching alone doesn't prove the two files were produced
    # together -- recompute each index from window_map's own recorded
    # residues (same encoding prepare_cytochrome_c_dataset.py used) and
    # compare, so a stale/mismatched pair of coincidentally equal length
    # is caught instead of silently mispairing species with embeddings.
    recomputed = torch.tensor(
        [encode_window_to_index(w["residues"]) for w in window_map], dtype=torch.long,
    )
    mismatches = int((recomputed != indices).sum().item())
    if mismatches > 0:
        raise ValueError(
            f"{mismatches}/{len(indices)} indices don't match the residues recorded in window_map.json "
            "-- window_map.json and indices.pt appear to come from different prepare_cytochrome_c_dataset.py runs"
        )

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
    idx_counts: dict[int, int] = defaultdict(int)
    for w, idx in zip(window_map, indices.tolist()):
        idx_to_species[idx].add(w["species"])
        idx_counts[idx] += 1
    n_unique = len(idx_to_species)
    cross_species_indices = [idx for idx, sp in idx_to_species.items() if len(sp) > 1]
    windows_in_cross_species_collision = sum(idx_counts[idx] for idx in cross_species_indices)
    return {
        "n_windows": len(window_map),
        "n_unique_indices": n_unique,
        "n_indices_shared_across_species": len(cross_species_indices),
        # Any repeated index, including two windows of the *same* species
        # landing on the same digit pattern -- not species-comparison noise.
        "any_duplicate_rate": round(1.0 - n_unique / len(window_map), 4),
        # The rate that actually matters for "did the encoding make distinct
        # species look identical": only windows whose index recurs under a
        # *different* species.
        "cross_species_collision_rate": round(windows_in_cross_species_collision / len(window_map), 4),
    }


def _species_masks(window_map: list[dict], species_order: list[str]) -> list[list[int]]:
    return [[i for i, w in enumerate(window_map) if w["species"] == sp] for sp in species_order]


def _vae_b_collapse_check(z_b_hyp: torch.Tensor, collapse_threshold: float = 0.999) -> dict[str, Any]:
    """Directional-collapse check for VAE-B, independent of the distance
    matrix above (which is VAE-A only). CLAUDE.md's V24.0 section documents
    a real prior case: tangent_scale_B collapsed to ~6e-8 and all z_B_hyp
    directions converged to pairwise cosine similarity 1.000000, while
    VAE-A stayed healthy throughout -- a failure this script would
    otherwise have no way to notice."""
    norm = z_b_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    direction = (z_b_hyp / norm).cpu().numpy()
    cos_sim = direction @ direction.T
    n = cos_sim.shape[0]
    iu = np.triu_indices(n, k=1)
    mean_cos_sim = float(cos_sim[iu].mean())
    return {
        "mean_pairwise_cosine_similarity": mean_cos_sim,
        "collapsed": mean_cos_sim > collapse_threshold,
    }


def embed_condition_raw(window_map: list[dict], species_order: list[str]) -> np.ndarray:
    """Condition 0: zero-model baseline. Per-species raw hydropathy-encoded
    aligned sequence (windows concatenated in window_idx order -- the same
    reference-coordinate alignment every other condition uses), Euclidean
    distance between species. No VAE, no training, bypasses indices.pt
    entirely (works straight from window_map.json residues) so it isolates
    how much of any correlation is attributable to the alignment + encoding
    alone versus anything a model learned."""
    by_species: dict[str, dict[int, str]] = defaultdict(dict)
    for w in window_map:
        by_species[w["species"]][w["window_idx"]] = w["residues"]

    vectors = []
    for sp in species_order:
        windows = by_species[sp]
        full_seq = "".join(windows[i] for i in range(len(windows)))
        vectors.append([AA_MAP.get(aa.upper(), 0) for aa in full_seq])
    points = torch.tensor(vectors, dtype=torch.float64)
    return torch.cdist(points, points).cpu().numpy()


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
) -> tuple[np.ndarray, dict[str, Any]]:
    config = load_yaml(run_dir / "config.yaml")
    model = build_model_from_config(config).to(device)
    ckpt = load_checkpoint_compat(run_dir / "checkpoints" / "best_Q.pt", map_location=device)
    model.load_state_dict(get_model_state_dict(ckpt))
    model.eval()
    c = model.projections.get_curvature()
    with torch.no_grad():
        z_a_hyp = model.get_hyperbolic_representations(indices, device, head="A")
        z_b_hyp = model.get_hyperbolic_representations(indices, device, head="B")
        tangent = log_map_zero(z_a_hyp, c=c)

    masks = _species_masks(window_map, species_order)
    mean_tangent = torch.stack([tangent[m].mean(dim=0) for m in masks])
    hyp_points = exp_map_zero(mean_tangent, c=c)
    dist = poincare_distance_matrix(hyp_points, c=c).cpu().numpy()
    vae_b_health = _vae_b_collapse_check(z_b_hyp)
    return dist, vae_b_health


def mantel_test(
    dist_model: np.ndarray, dist_taxonomy: np.ndarray, n_permutations: int, seed: int,
) -> dict[str, Any]:
    n = dist_model.shape[0]
    iu = np.triu_indices(n, k=1)
    model_flat = dist_model[iu]
    tax_flat = dist_taxonomy[iu]
    observed, _ = spearmanr(model_flat, tax_flat)

    if np.isnan(observed):
        # Degenerate embedding distances (e.g. posterior collapse -> every
        # species point identical) make Spearman undefined. Reporting a
        # p-value here would be meaningless: NaN comparisons are always
        # False in numpy, so `null_corrs >= observed` would silently give a
        # spuriously "significant" p-value instead of surfacing the collapse.
        return {
            "observed_spearman": float("nan"),
            "p_value_one_sided": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "n_permutations": 0,
            "n_species_pairs": int(len(model_flat)),
            "note": "observed_spearman is NaN (degenerate/constant model distances); Mantel test skipped.",
        }

    rng = np.random.default_rng(seed)
    null_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        permuted = dist_taxonomy[np.ix_(perm, perm)][iu]
        null_corrs[i], _ = spearmanr(model_flat, permuted)
    null_corrs = null_corrs[~np.isnan(null_corrs)]

    # One-sided: alternative hypothesis is a positive correlation (embedding
    # distance grows with real taxonomic distance).
    if len(null_corrs) == 0:
        p_value = float("nan")
    else:
        p_value = float((np.sum(null_corrs >= observed) + 1) / (len(null_corrs) + 1))
    return {
        "observed_spearman": float(observed),
        "p_value_one_sided": p_value,
        "null_mean": float(null_corrs.mean()) if len(null_corrs) else float("nan"),
        "null_std": float(null_corrs.std()) if len(null_corrs) else float("nan"),
        "n_permutations": n_permutations,
        "n_species_pairs": int(len(model_flat)),
    }


def bootstrap_ci(
    dist_model: np.ndarray, dist_taxonomy: np.ndarray, n_boot: int, seed: int,
) -> dict[str, Any]:
    n = dist_model.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, k=1)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        sample = rng.choice(n, size=n, replace=True)
        # Bootstrap-resampled species pairs where sample[i] == sample[j] are
        # self-distances (0 by construction) and carry no signal; drop them.
        keep = sample[iu[0]] != sample[iu[1]]
        if keep.sum() < 3:
            continue
        sub_model = dist_model[np.ix_(sample, sample)][iu][keep]
        sub_tax = dist_taxonomy[np.ix_(sample, sample)][iu][keep]
        boots[i], _ = spearmanr(sub_model, sub_tax)
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_valid_bootstrap": 0}
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


def _load_existing_conditions(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {}
    try:
        return json.loads(out_path.read_text()).get("conditions", {})
    except (json.JSONDecodeError, OSError):
        return {}


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

    # No guard requiring a checkpoint: raw_encoding_baseline (Condition 0)
    # needs none of them and is worth computing on its own, e.g. before any
    # A/B/C run exists.

    device = torch.device("cpu")
    window_map, indices, species_order, tax_dist = load_species_data(
        Path(args.window_map), Path(args.indices), Path(args.taxonomy_dir),
    )
    collisions = index_collision_report(window_map, indices)
    print(
        f"[data] {len(species_order)} species, {collisions['n_windows']} windows, "
        f"{collisions['n_unique_indices']} unique indices "
        f"(cross_species_collision_rate={collisions['cross_species_collision_rate']:.2%})"
    )

    common_meta = {"seed": args.seed, "git_commit": get_git_commit()}
    new_conditions: dict[str, Any] = {}

    dist_raw = embed_condition_raw(window_map, species_order)
    new_conditions["raw_encoding_baseline"] = {
        **evaluate_condition("raw_encoding_baseline", dist_raw, tax_dist, args.n_permutations, args.n_bootstrap, args.seed),
        "source": "none (zero-model baseline)",
        **common_meta,
    }

    if args.run_a_checkpoint:
        dist_a = embed_condition_a(Path(args.run_a_checkpoint), indices, window_map, species_order, device)
        new_conditions["A_euclidean"] = {
            **evaluate_condition("A_euclidean", dist_a, tax_dist, args.n_permutations, args.n_bootstrap, args.seed),
            "source": str(args.run_a_checkpoint),
            **common_meta,
        }

    if args.run_b_dir:
        dist_b, vae_b_health = embed_condition_hyperbolic(Path(args.run_b_dir), indices, window_map, species_order, device)
        new_conditions["B_hyperbolic_generic"] = {
            **evaluate_condition("B_hyperbolic_generic", dist_b, tax_dist, args.n_permutations, args.n_bootstrap, args.seed),
            "source": str(args.run_b_dir),
            "vae_b_health": vae_b_health,
            **common_meta,
        }

    if args.run_c_dir:
        dist_c, vae_b_health = embed_condition_hyperbolic(Path(args.run_c_dir), indices, window_map, species_order, device)
        new_conditions["C_padic"] = {
            **evaluate_condition("C_padic", dist_c, tax_dist, args.n_permutations, args.n_bootstrap, args.seed),
            "source": str(args.run_c_dir),
            "vae_b_health": vae_b_health,
            **common_meta,
        }

    out_path = Path(args.out)
    existing_conditions = _load_existing_conditions(out_path)
    conditions = {**existing_conditions, **new_conditions}
    if existing_conditions:
        print(f"[merge] {len(existing_conditions)} condition(s) already in {out_path}, "
              f"{len(conditions)} total after this run")

    raw_rho = conditions.get("raw_encoding_baseline", {}).get("observed_spearman")
    if raw_rho is not None and not np.isnan(raw_rho):
        print(f"\n[verdict] raw_encoding_baseline Spearman={raw_rho:.4f} -- "
              f"any condition below must beat this to claim learned structure, "
              f"not just conserved-sequence-implies-close-relative:")
        for name, c in conditions.items():
            if name == "raw_encoding_baseline":
                continue
            rho = c.get("observed_spearman")
            if rho is None or np.isnan(rho):
                continue
            verdict = "BEATS baseline" if rho > raw_rho else "does NOT beat baseline"
            print(f"           {name}: Spearman={rho:.4f} -- {verdict}")

    results: dict[str, Any] = {
        "species_order": species_order,
        "index_collision": collisions,
        "caveats": CAVEATS,
        "conditions": conditions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
