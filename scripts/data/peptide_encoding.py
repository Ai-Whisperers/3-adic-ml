"""Hydropathy-based amino-acid to ternary digit encoding.

Shared by the peptide/protein bioactivity analysis scripts
(scan_human_proteome.py, probe_bioactive_topology.py,
qspr_bioactivity_scoring.py, visualize_hotspots.py, prepare_rosetta_dataset.py)
so the acidic/neutral/hydrophobic mapping can't silently drift between them.
"""

AA_MAP = {
    'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1,
    'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0,
    'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1,
}


def encode_peptide_window(seq: str, window: int = 9) -> list[int]:
    """Pad/truncate a peptide sequence to `window` residues, map via hydropathy."""
    seq_padded = seq[:window].ljust(window, 'G')
    return [AA_MAP.get(aa.upper(), 0) for aa in seq_padded]
