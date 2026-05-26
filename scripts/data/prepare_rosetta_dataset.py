import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core import TERNARY

# AA Mapping (Hydropathy-based)
AA_MAP = {
    'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1,
    'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0,
    'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1
}

def encode_sequence_to_indices(seq, window=9):
    indices = []
    for i in range(len(seq) - window + 1):
        win = seq[i:i+window]
        digits = [AA_MAP.get(aa.upper(), 0) for aa in win]
        # TERNARY.from_ternary expects (..., 9) tensor
        idx = TERNARY.from_ternary(torch.tensor(digits, dtype=torch.float64).unsqueeze(0))
        indices.append(idx.item())
    return torch.tensor(indices, dtype=torch.long)

def read_fasta(file_path):
    seqs = []
    current_seq = ""
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    seqs.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
    if current_seq:
        seqs.append(current_seq)
    return seqs

def main():
    print("Preparing Rosetta Dataset (Phase 17)...")
    
    # 1. Synthetic Master
    synthetic_indices = torch.arange(19683, dtype=torch.long)
    print(f"  Added {len(synthetic_indices)} synthetic indices.")
    
    # 2. Human TP53
    tp53_fasta = "data/human/tp53_ref.fasta"
    if os.path.exists(tp53_fasta):
        tp53_seqs = read_fasta(tp53_fasta)
        tp53_indices = torch.cat([encode_sequence_to_indices(s) for s in tp53_seqs])
        print(f"  Added {len(tp53_indices)} Human TP53 indices.")
    else:
        tp53_indices = torch.empty((0,), dtype=torch.long)
        print("  [WARN] TP53 Fasta not found.")

    # 3. Curcumin/Ginger Peptides
    peptide_fasta = "curcumin-and-other-peptides/data/sequences.fasta"
    if os.path.exists(peptide_fasta):
        pep_seqs = read_fasta(peptide_fasta)
        pep_indices = torch.cat([encode_sequence_to_indices(s) for s in pep_seqs])
        print(f"  Added {len(pep_indices)} Peptide indices.")
    else:
        pep_indices = torch.empty((0,), dtype=torch.long)
        print("  [WARN] Peptide Fasta not found.")

    # Combine
    rosetta_indices = torch.cat([synthetic_indices, tp53_indices, pep_indices], dim=0)
    
    # Shuffle
    perm = torch.randperm(len(rosetta_indices))
    rosetta_indices = rosetta_indices[perm]
    
    output_path = "data/rosetta_indices.pt"
    torch.save(rosetta_indices, output_path)
    print(f"\n[OK] Rosetta Dataset saved to {output_path}")
    print(f"     Total Samples: {len(rosetta_indices)}")

if __name__ == "__main__":
    main()
