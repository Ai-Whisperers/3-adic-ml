import torch
from pathlib import Path
from src.core import TERNARY
from scripts.data.prepare_codon_data import seq_to_ternary_index
from scripts.analysis.test_hierarchical_search import setup_model
from src.geometry import poincare_distance

def probe_codon_geometry(checkpoint_path, sequences):
    model = setup_model(checkpoint_path)
    device = torch.device("cpu")
    
    indices = torch.tensor([seq_to_ternary_index(s) for s in sequences], device=device)
    
    with torch.no_grad():
        mu = model.get_mu_representations(indices, device)
        z_hyp, _ = model.projections.proj_A(mu)
        z_hyp = z_hyp.cpu()
        
    print("\nProbing Codon Geometry (Poincaré Distances):")
    for i, seq in enumerate(sequences):
        dist = poincare_distance(torch.zeros(1, z_hyp.shape[1]), z_hyp[i].unsqueeze(0), c=1.0)
        val = TERNARY.valuation(indices[i].unsqueeze(0)).item()
        print(f"Sequence: {seq} | Valuation: {val} | Dist to Root: {dist.item():.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    # Examples: Diverse sequences to test radial spread
    test_seqs = ["AAAAAAAAA", "ACGTACGTA", "GGGGGGGGG", "TTTTTTTTT"]
    probe_codon_geometry(ckpt_path, test_seqs)
