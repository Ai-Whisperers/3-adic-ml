import torch

from src.core import TERNARY

# Mapping strategy: A=0, C=1, G=2, T=2 (Merging G/T to fit 3-state logic)
NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 2}

def seq_to_ternary_index(sequence: str) -> int:
    """Converts a 9-nucleotide sequence to a ternary index.

    Delegates to TERNARY.from_ternary (canonical LSB-first digit convention:
    sequence position 0 has weight 3^0) instead of hand-rolling the base-3
    conversion. A previous hand-rolled version used the opposite digit order
    (position 0 weighted 3^8), so indices it produced decoded backwards via
    TERNARY.to_ternary() -- the exact function the model uses internally in
    get_mu_representations()/get_hyperbolic_representations(). That silently
    scrambled nucleotide-position semantics for every downstream consumer.
    """
    if len(sequence) != 9:
        raise ValueError("Sequence must be 9 nucleotides long (3 codons).")

    digits = torch.tensor([NUC_MAP[nuc] - 1 for nuc in sequence], dtype=torch.float64)
    return int(TERNARY.from_ternary(digits).item())

def prepare_codon_data(sequences: list[str]):
    """Converts a list of 9-nuc sequences to ternary indices."""
    return [seq_to_ternary_index(seq) for seq in sequences]
