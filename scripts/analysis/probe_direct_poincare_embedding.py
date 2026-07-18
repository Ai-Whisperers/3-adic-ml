"""Fase 0 of docs/plans/TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md: can hyperbolic
geometry fit this project's real taxonomic distance matrix at all, with no
VAE, no encoder, no dataset bottleneck -- just 39 free points on the
Poincare ball directly optimized against taxonomic_distance.npy (Nickel &
Kiela 2017 style)?

This isolates "can hyperbolic geometry represent this specific 39-species
distance matrix well" from every VAE/encoder/window-encoding confound in
Conditions A/B/C. If this can't beat raw_encoding_baseline either, the
ceiling is in the taxonomy data/Mantel-test noise at n=39, not in anything
a taxonomy-conditioned VAE loss (Fase 1-3 of the plan) could fix -- and
that plan should be shelved, not built.

Reuses evaluate_phylogeny_recovery.py's data loading, Mantel test, and
bootstrap CI so the comparison against raw_encoding_baseline is exact, not
a hardcoded number that could drift from a re-run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.evaluate_phylogeny_recovery import (
    embed_condition_raw,
    evaluate_condition,
    load_species_data,
)
from src.geometry.poincare import exp_map_zero, poincare_distance_matrix, project_to_poincare


def fit_direct_embedding(
    tax_dist: np.ndarray, dim: int, epochs: int, lr: float,
    max_target_distance: float, max_norm: float, seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    n = tax_dist.shape[0]
    tax_t = torch.tensor(tax_dist, dtype=torch.float64)
    target = max_target_distance * tax_t / tax_t.max()
    iu = torch.triu_indices(n, n, offset=1)
    target_flat = target[iu[0], iu[1]]

    tangent = torch.nn.Parameter(torch.randn(n, dim, dtype=torch.float64) * 0.01)
    optimizer = torch.optim.Adam([tangent], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = project_to_poincare(exp_map_zero(tangent, c=1.0), max_norm=max_norm, c=1.0)
        dist = poincare_distance_matrix(z, c=1.0)
        dist_flat = dist[iu[0], iu[1]]
        loss = torch.nn.functional.smooth_l1_loss(dist_flat, target_flat)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  [epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

    with torch.no_grad():
        z = project_to_poincare(exp_map_zero(tangent, c=1.0), max_norm=max_norm, c=1.0)
        dist = poincare_distance_matrix(z, c=1.0)
    return dist.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-map", default="data/cytochrome_c/window_map.json")
    parser.add_argument("--indices", default="data/cytochrome_c/indices.pt")
    parser.add_argument("--taxonomy-dir", default="data/cytochrome_c")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-target-distance", type=float, default=4.0)
    parser.add_argument("--max-norm", type=float, default=0.99)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reuse load_species_data (not a hand-rolled species_order.json read) so
    # species_order/tax_dist are aligned to window_map.json's actual 39
    # species, not the unfiltered 41 in Fase 1's species_order.json (which
    # still includes E. coli/B. subtilis, excluded from Fase 2 onward).
    window_map, _, species_order, tax_dist = load_species_data(
        Path(args.window_map), Path(args.indices), Path(args.taxonomy_dir),
    )
    print(f"[data] {len(species_order)} species, taxonomic_distance range=[{tax_dist.min():.0f}, {tax_dist.max():.0f}]")

    print(f"[fit] direct Poincare embedding, dim={args.dim}, epochs={args.epochs}")
    dist_direct = fit_direct_embedding(
        tax_dist, args.dim, args.epochs, args.lr,
        args.max_target_distance, args.max_norm, args.seed,
    )
    direct_result = evaluate_condition(
        "direct_poincare_embedding", dist_direct, tax_dist,
        args.n_permutations, args.n_bootstrap, args.seed,
    )

    dist_raw = embed_condition_raw(window_map, species_order)
    raw_result = evaluate_condition(
        "raw_encoding_baseline", dist_raw, tax_dist,
        args.n_permutations, args.n_bootstrap, args.seed,
    )
    raw_rho = raw_result["observed_spearman"]
    direct_rho = direct_result["observed_spearman"]
    verdict = "BEATS" if direct_rho > raw_rho else "does NOT beat"
    print(f"\n[verdict] direct_poincare_embedding Spearman={direct_rho:.4f} "
          f"{verdict} raw_encoding_baseline Spearman={raw_rho:.4f}")
    if direct_rho <= raw_rho:
        print("[verdict] Fase 0 gate: FAILED -- hyperbolic geometry alone (no VAE, no "
              "encoder bottleneck) cannot beat trivial sequence identity on this "
              "39-species matrix. Shelve TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md Fase 1-3; "
              "the ceiling is in the data/Mantel-test noise at n=39, not in the loss target.")
    else:
        print("[verdict] Fase 0 gate: PASSED -- proceed to Fase 1 "
              "(TaxonomyGeodesicLoss) with real prospect of a positive held-out result.")


if __name__ == "__main__":
    main()
