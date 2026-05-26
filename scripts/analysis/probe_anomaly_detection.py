import torch
import numpy as np
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.test_hierarchical_search import setup_model
from src.geometry import poincare_distance

def generate_anomaly():
    """Generates a random 9-nt sequence (high entropy)."""
    return "".join(random.choice("ACGT") for _ in range(9))

def probe_anomaly_detection(checkpoint_path, normal_indices_path, k=5):
    model = setup_model(checkpoint_path)
    device = torch.device("cpu")
    
    # 1. Load normal embeddings
    normal_indices = torch.load(normal_indices_path, weights_only=True)
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, device)
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    # 2. Compute Normal Distance Baseline (k-NN distances)
    print(f"Computing normal distance baseline (k={k})...")
    n_norm = z_norm.shape[0]
    nn_dists = []
    
    # Precompute pairwise distance matrix
    dist_matrix = torch.zeros(n_norm, n_norm)
    for i in range(n_norm):
        for j in range(i + 1, n_norm):
            d = poincare_distance(z_norm[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    for i in range(n_norm):
        dists = sorted(dist_matrix[i].tolist())
        # Use top k neighbors (skipping self-distance at index 0)
        nn_dists.append(np.mean(dists[1:k+1]))

    norm_mean = np.mean(nn_dists)
    norm_std = np.std(nn_dists)
    threshold = norm_mean + 3 * norm_std # Balanced threshold
    print(f"k-NN dists: Mean={norm_mean:.4f}, Std={norm_std:.4f}, Threshold={threshold:.4f}")
        
    # 3. Generate and project anomalies
    anomalies = [generate_anomaly() for _ in range(20)]
    NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 2}
    def seq_to_ternary_index(sequence):
        digits = [NUC_MAP[nuc] for nuc in sequence]
        index = 0
        for i, digit in enumerate(digits):
            index += digit * (3 ** (8 - i))
        return index

    anomaly_indices = torch.tensor([seq_to_ternary_index(s) for s in anomalies], device=device)
    
    with torch.no_grad():
        mu_anom = model.get_mu_representations(anomaly_indices, device)
        z_anom, _ = model.projections.proj_A(mu_anom)
        
    # 4. Calculate distance to nearest normal neighbor (k-NN)
    print(f"\nAnomaly Detection Results (k={k}, Threshold={threshold:.4f}):")
    for i in range(len(anomalies)):
        dists = [poincare_distance(z_anom[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item() for j in range(n_norm)]
        # Use mean of k smallest
        min_dist = np.mean(sorted(dists)[:k])
        is_anomaly = min_dist > threshold
        label = "ANOMALY" if is_anomaly else "NORMAL"
        print(f"[{label}] {anomalies[i]} | k-NN Dist to Normal: {min_dist:.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    probe_anomaly_detection(ckpt_path, "data/ecoli_indices.pt", k=5)
