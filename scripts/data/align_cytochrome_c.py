"""Align cytochrome c orthologs against the human reference and cut aligned
windows.

Phase 2 of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md. Raw per-species
9-residue tiling isn't comparable across species -- position i in species X
isn't homologous to position i in species Y without alignment first. This
script aligns every species to the human reference (Bio.Align.PairwiseAligner,
global, BLOSUM62) and defines window boundaries on the *reference* coordinate
system, so "window k" is the same structural position for every species.

Species flagged length_plausible=False in the manifest (E. coli, B. subtilis
as of the last fetch_cytochrome_c.py run -- see manifest.csv) are excluded:
those UniProt hits are confirmed non-homologous proteins (cytochrome c
peroxidase, cytochrome c oxidase subunit 2), and aligning a 465/356 aa
catalytic enzyme against the ~105 aa reference would produce a meaningless
alignment, not a comparable window set.
"""

import argparse
import csv
import json
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices

GAP = "-"


def read_fasta(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def build_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def map_reference_to_target(alignment, ref_len: int) -> list[int | None]:
    """Returns ref_to_target[i] = target index aligned to reference index i,
    or None if reference position i is deleted (gapped) in the target."""
    ref_blocks, tgt_blocks = alignment.aligned
    mapping: list[int | None] = [None] * ref_len
    for (r0, r1), (t0, t1) in zip(ref_blocks, tgt_blocks):
        for offset in range(r1 - r0):
            mapping[r0 + offset] = t0 + offset
    return mapping


def cut_windows(target_seq: str, ref_to_target: list[int | None], window: int, stride: int) -> list[str]:
    windows = []
    for start in range(0, len(ref_to_target) - window + 1, stride):
        residues = []
        for p in range(start, start + window):
            t = ref_to_target[p]
            residues.append(target_seq[t] if t is not None else GAP)
        windows.append("".join(residues))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cytochrome_c/manifest.csv")
    parser.add_argument("--fasta-dir", default="data/cytochrome_c/fasta")
    parser.add_argument("--reference", default="Homo sapiens")
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument("--out", default="data/cytochrome_c/aligned_windows.json")
    args = parser.parse_args()

    fasta_dir = Path(args.fasta_dir)
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    included = [r for r in rows if r["status"].startswith("found") and r["length_plausible"] == "True"]
    excluded = [r for r in rows if r not in included]
    for r in excluded:
        print(f"[SKIP] {r['species']}: status={r['status']} length_plausible={r['length_plausible']}")

    ref_row = next((r for r in included if r["species"] == args.reference), None)
    if ref_row is None:
        raise SystemExit(f"Reference species '{args.reference}' not in included manifest rows.")
    ref_seq = read_fasta(fasta_dir / f"{args.reference.replace(' ', '_')}.fasta")
    print(f"Reference: {args.reference} ({len(ref_seq)} aa)")

    aligner = build_aligner()
    aligned_windows: dict[str, list[str]] = {}
    for i, row in enumerate(included):
        species = row["species"]
        target_seq = read_fasta(fasta_dir / f"{species.replace(' ', '_')}.fasta")
        if species == args.reference:
            ref_to_target = list(range(len(ref_seq)))
            score = None
        else:
            alignment = aligner.align(ref_seq, target_seq)[0]
            ref_to_target = map_reference_to_target(alignment, len(ref_seq))
            score = alignment.score
        windows = cut_windows(target_seq, ref_to_target, args.window, args.stride)
        aligned_windows[species] = windows
        n_gapped = sum(1 for w in windows if GAP in w)
        score_str = f"score={score:.0f}" if score is not None else "reference"
        print(f"[{i+1}/{len(included)}] {species}: {len(windows)} windows, "
              f"{n_gapped} with gaps ({score_str})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aligned_windows, indent=2))
    print(f"\n[OK] {len(aligned_windows)} species, "
          f"{len(next(iter(aligned_windows.values())))} windows/species. Saved: {out_path}")


if __name__ == "__main__":
    main()
