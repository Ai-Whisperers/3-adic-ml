# Mapping strategy: A=0, C=1, G=2, T=2 (Merging G/T to fit 3-state logic)
NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 2}

def seq_to_ternary_index(sequence: str) -> int:
    """Converts a 9-nucleotide sequence to a ternary index."""
    if len(sequence) != 9:
        raise ValueError("Sequence must be 9 nucleotides long (3 codons).")
    
    digits = [NUC_MAP[nuc] for nuc in sequence]
    
    # Convert ternary list to integer index (base 3)
    index = 0
    for i, digit in enumerate(digits):
        index += digit * (3 ** (8 - i))
    return index

def prepare_codon_data(sequences: list[str]):
    """Converts a list of 9-nuc sequences to ternary indices."""
    return [seq_to_ternary_index(seq) for seq in sequences]
