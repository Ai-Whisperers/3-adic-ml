"""Fase 2 of docs/plans/TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md: select a
species-level holdout set and produce a training-only window_map/indices
pair that never includes any window from a held-out species.

Every prior condition (A/B/C, Fase 1-4 of PHYLOGENY-VALIDATION-PIPELINE.md)
trained on windows from all 39 species -- the "no species-level holdout"
caveat has been open since Fase 4. Condition D needs new data-loading code
anyway, so this closes it: held-out species' windows never appear in
indices_train.pt, and evaluate_phylogeny_recovery.py-style Mantel scoring
can be reported separately for held-in vs. held-out-only species pairs.

Stratified by kingdom (from taxonomy_lineage.json), not plain random -- 8
species picked uniformly at random from 39 could by chance land entirely
within one clade (e.g. 8 birds) and say nothing about cross-kingdom
generalization, which is the case that actually matters here.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data.prepare_cytochrome_c_dataset import encode_window_to_index


def select_holdout(
    species_order: list[str], lineage: dict[str, dict], holdout_frac: float, seed: int,
) -> tuple[list[str], dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for sp in species_order:
        kingdom = lineage.get(sp, {}).get("kingdom") or "unranked"
        groups[kingdom].append(sp)

    holdout = []
    for kingdom in sorted(groups):
        members = list(groups[kingdom])
        rng.shuffle(members)
        n_holdout = max(1, round(len(members) * holdout_frac)) if len(members) > 1 else 0
        holdout.extend(members[:n_holdout])
    return sorted(holdout), {k: sorted(v) for k, v in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-map", default="data/cytochrome_c/window_map.json")
    parser.add_argument("--taxonomy-dir", default="data/cytochrome_c")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-heldout", default="data/cytochrome_c/heldout_species.json")
    parser.add_argument("--out-window-map-train", default="data/cytochrome_c/window_map_train.json")
    parser.add_argument("--out-indices-train", default="data/cytochrome_c/indices_train.pt")
    args = parser.parse_args()

    window_map = json.loads(Path(args.window_map).read_text())
    lineage = json.loads((Path(args.taxonomy_dir) / "taxonomy_lineage.json").read_text())
    species_order = sorted({w["species"] for w in window_map})

    holdout, groups = select_holdout(species_order, lineage, args.holdout_frac, args.seed)
    train_species = [s for s in species_order if s not in holdout]

    print(f"[groups] {len(groups)} kingdom groups: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items())))
    print(f"[holdout] {len(holdout)}/{len(species_order)} species "
          f"({len(holdout) / len(species_order):.0%}): {holdout}")
    print(f"[train] {len(train_species)} species remain")

    window_map_train = [w for w in window_map if w["species"] in train_species]
    indices_train = torch.tensor(
        [encode_window_to_index(w["residues"]) for w in window_map_train], dtype=torch.long,
    )

    Path(args.out_heldout).write_text(json.dumps({
        "heldout_species": holdout,
        "train_species": train_species,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "kingdom_groups": groups,
    }, indent=2))
    Path(args.out_window_map_train).write_text(json.dumps(window_map_train, indent=2))
    torch.save(indices_train, args.out_indices_train)

    print(f"\n[OK] {len(window_map_train)} training windows "
          f"(down from {len(window_map)}). Saved:")
    print(f"     {args.out_heldout}")
    print(f"     {args.out_window_map_train}")
    print(f"     {args.out_indices_train}")


if __name__ == "__main__":
    main()
