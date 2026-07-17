"""Convert aligned cytochrome c windows into ternary indices for training.

Phase 2 (final step) of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md. Reads
the per-species aligned windows from align_cytochrome_c.py, maps each
9-residue window to a ternary index via the same hydropathy encoding used
throughout the peptide pipeline (AA_MAP) and the canonical LSB-first digit
convention (TERNARY.from_ternary -- same as prepare_codon_data.py), and saves
a window_id -> (species, window_idx) map so embeddings can be re-aggregated
per species after training (Phase 4 of the pipeline plan).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core import TERNARY
from scripts.data.peptide_encoding import AA_MAP


def encode_window_to_index(window: str) -> int:
    digits = torch.tensor([AA_MAP.get(aa.upper(), 0) for aa in window], dtype=torch.float64)
    return int(TERNARY.from_ternary(digits).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-windows", default="data/cytochrome_c/aligned_windows.json")
    parser.add_argument("--indices-out", default="data/cytochrome_c/indices.pt")
    parser.add_argument("--window-map-out", default="data/cytochrome_c/window_map.json")
    args = parser.parse_args()

    aligned_windows = json.loads(Path(args.aligned_windows).read_text())

    indices = []
    window_map = []
    for species, windows in aligned_windows.items():
        for window_idx, window in enumerate(windows):
            indices.append(encode_window_to_index(window))
            window_map.append({"species": species, "window_idx": window_idx, "residues": window})

    indices_tensor = torch.tensor(indices, dtype=torch.long)
    Path(args.indices_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(indices_tensor, args.indices_out)
    Path(args.window_map_out).write_text(json.dumps(window_map, indent=2))

    print(f"[OK] {len(aligned_windows)} species, {len(indices)} windows total.")
    print(f"     Index range: [{indices_tensor.min().item()}, {indices_tensor.max().item()}] "
          f"(valid: [0, {TERNARY.N_OPERATIONS - 1}])")
    print(f"     Saved: {args.indices_out}, {args.window_map_out}")


if __name__ == "__main__":
    main()
