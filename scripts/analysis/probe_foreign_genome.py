import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from scripts.analysis.test_hierarchical_search import setup_model
from src.geometry import poincare_distance
from prepare_codon_data import seq_to_ternary_index

def probe_foreign_genome(checkpoint_path, normal_indices_path, foreign_sequences):
    model = setup_model(checkpoint_path)
    device = torch.device("cpu")
    
    # 1. Load normal embeddings
    normal_indices = torch.load(normal_indices_path, weights_only=True)
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, device)
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    # 2. Compute Threshold (Baseline from E. coli)
    nn_dists = []
    for i in range(len(z_norm)):
        dists = [poincare_distance(z_norm[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item() for j in range(len(z_norm)) if i != j]
        nn_dists.append(min(dists))
    threshold = np.mean(nn_dists) + 2 * np.std(nn_dists)
    
    # 3. Project Foreign sequences
    foreign_indices = torch.tensor([seq_to_ternary_index(s) for s in foreign_sequences], device=device)
    with torch.no_grad():
        mu_foreign = model.get_mu_representations(foreign_indices, device)
        z_foreign, _ = model.projections.proj_A(mu_foreign)
        
    # 4. Classify
    print(f"\nForeign Genome Detection (Threshold={threshold:.4f}):")
    for i in range(len(foreign_sequences)):
        dists = [poincare_distance(z_foreign[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item() for j in range(len(z_norm))]
        min_dist = min(dists)
        is_anomaly = min_dist > threshold
        label = "ANOMALY (Foreign)" if is_anomaly else "NORMAL (Recognized)"
        print(f"[{label}] {foreign_sequences[i]} | Min Dist to Normal: {min_dist:.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    # Example yeast-like sequences (higher GC content or different patterns than E. coli)
    foreign_seqs = [
        "ATATATATA", "GCGCGCGCG", "TTTCCCCCC", "AAAAAGGGG"
    ]
    probe_foreign_genome(ckpt_path, "data/ecoli_indices.pt", foreign_seqs)
