import torch
import numpy as np
from scripts.data.prepare_codon_data import seq_to_ternary_index


def compute_attribution(sequence: str, model, detector):
    """Importance score per nucleotide position via perturbation analysis."""
    def get_dist(seq):
        idx = torch.tensor([seq_to_ternary_index(seq)])
        with torch.no_grad():
            mu = model.get_mu_representations(idx, torch.device("cpu"))
            # proj_A returns (z_hyp, r) in factored mode, plain z_hyp otherwise.
            result = model.projections.proj_A(mu)
            z = result[0] if isinstance(result, tuple) else result
        res = detector.detect(z)
        return float(res["min_dist"][0])

    orig_dist = get_dist(sequence)
    attribution = []
    
    for i in range(len(sequence)):
        original_nuc = sequence[i]
        pos_sens = 0.0
        
        # Perturb this position
        for nuc in 'ACGT':
            if nuc == original_nuc: continue
            
            perturbed_seq = sequence[:i] + nuc + sequence[i+1:]
            new_dist = get_dist(perturbed_seq)
            
            # Sensitivity = magnitude of change
            pos_sens += abs(new_dist - orig_dist)
            
        attribution.append(pos_sens / 3.0)
        
    return attribution, orig_dist
