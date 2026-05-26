import torch
import numpy as np
from typing import Optional, List, Dict
from src.geometry import poincare_distance

class AnomalyDetector:
    """Production-grade anomaly detection using latent space density."""
    
    def __init__(self, model, device: torch.device = torch.device("cpu")):
        self.model = model
        self.device = device
        self.z_norm = None
        self.threshold = None
        
    def fit(self, normal_embeddings: torch.Tensor, k: int = 5, sigma_factor: float = 3.0):
        """Calibrates threshold based on k-NN distances in normal dataset."""
        self.z_norm = normal_embeddings.to(self.device)
        n_norm = self.z_norm.shape[0]
        
        nn_dists = []
        # Pairwise distance matrix computation (optimized)
        # Using a subset if N is large for fitting
        dists = torch.cdist(self.z_norm, self.z_norm, p=2) # Placeholder for Poincare distance matrix logic
        
        # We need actual Poincare distances. For production, consider using geoopt vectorized dists
        for i in range(n_norm):
            d_i = [poincare_distance(self.z_norm[i].unsqueeze(0), self.z_norm[j].unsqueeze(0), c=1.0).item() 
                   for j in range(n_norm) if i != j]
            nn_dists.append(np.mean(sorted(d_i)[:k]))
            
        self.threshold = np.mean(nn_dists) + sigma_factor * np.std(nn_dists)
        print(f"Detector fitted: Threshold={self.threshold:.4f}")
        
    def detect(self, query_embeddings: torch.Tensor, k: int = 5) -> Dict[str, np.ndarray]:
        """Classifies queries as Normal or Anomaly."""
        if self.z_norm is None or self.threshold is None:
            raise ValueError("Detector must be fitted first.")
            
        queries = query_embeddings.to(self.device)
        dists_to_normal = []
        for q in queries:
            d = [poincare_distance(q.unsqueeze(0), n.unsqueeze(0), c=1.0).item() for n in self.z_norm]
            dists_to_normal.append(np.mean(sorted(d)[:k]))
            
        dists_to_normal = np.array(dists_to_normal)
        is_anomaly = dists_to_normal > self.threshold
        
        return {
            "is_anomaly": is_anomaly,
            "min_dist": dists_to_normal
        }
