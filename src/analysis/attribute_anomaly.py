import torch
import numpy as np
from scripts.data.prepare_codon_data import seq_to_ternary_index
from src.core import TERNARY

def compute_attribution(sequence: str, model, detector):
    """
    Computes importance scores for each nucleotide position using perturbation.
    """
    NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 2} # Consistent with trainer
    
    def get_dist(seq):
        idx = torch.tensor([seq_to_ternary_index(seq)])
        with torch.no_grad():
            mu = model.get_mu_representations(idx, torch.device("cpu"))
            z, _ = model.projections.proj_A(mu)
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
