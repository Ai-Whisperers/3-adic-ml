import torch
from src.geometry import poincare_distance

def analyze_projections(projections_path):
    z_hyp = torch.load(projections_path)
    print(f"Loaded {z_hyp.shape[0]} embeddings of dim {z_hyp.shape[1]}")
    
    # Calculate pairwise distances (all-to-all)
    # Using a loop for clarity on small batch (34 samples)
    n = z_hyp.shape[0]
    dist_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = poincare_distance(z_hyp[i].unsqueeze(0), z_hyp[j].unsqueeze(0), c=1.0)
            
    # Test: Find neighbors for index 0
    query_idx = 0
    dists = dist_matrix[query_idx]
    
    # Sort distances (skip index 0 itself)
    sorted_dists, sorted_indices = torch.sort(dists)
    
    print("\nNearest neighbors to sequence 0:")
    for i in range(1, 6): # Top 5 neighbors
        idx = sorted_indices[i].item()
        print(f"  Neighbor index {idx}: Dist = {sorted_dists[i].item():.4f}")

if __name__ == "__main__":
    analyze_projections("data/ecoli_projections.pt")
