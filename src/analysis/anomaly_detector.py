# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Anomaly detection using Poincaré-ball k-NN density estimation."""

from typing import Dict

import numpy as np
import torch

from src.geometry import get_manifold, poincare_distance_matrix


class AnomalyDetector:
    """Anomaly detection using k-NN distances in hyperbolic (Poincaré-ball) space.

    Calibration: fit() computes mean k-NN Poincaré distance over normal embeddings
    and sets a threshold at mean + sigma_factor * std.

    Detection: detect() flags query embeddings whose mean k-NN distance to the
    normal set exceeds the threshold.

    Both fit() and detect() are fully vectorized — no Python loops over points.
    """

    def __init__(self, model=None, device: torch.device = torch.device("cpu"), curvature: float = 1.0):
        self.model = model
        self.device = device
        self.curvature = curvature
        self.z_norm: torch.Tensor | None = None
        self.threshold: float | None = None

    def fit(self, normal_embeddings: torch.Tensor, k: int = 5, sigma_factor: float = 3.0) -> None:
        """Calibrate threshold = mean + sigma_factor * std of mean-kNN distances over normal_embeddings."""
        self.z_norm = normal_embeddings.to(self.device)
        n_norm = self.z_norm.shape[0]

        # Full N×N pairwise Poincaré distance matrix — one vectorized call.
        dist_matrix = poincare_distance_matrix(self.z_norm, c=self.curvature)  # (N, N)

        # Mask self-distances so they are never selected as a neighbour.
        eye = torch.eye(n_norm, dtype=torch.bool, device=self.device)
        dist_matrix = dist_matrix.masked_fill(eye, float("inf"))

        k_actual = min(k, n_norm - 1)
        knn_dists, _ = dist_matrix.topk(k_actual, dim=1, largest=False)  # (N, k)
        mean_knn = knn_dists.mean(dim=1).cpu().numpy()  # (N,)

        self.threshold = float(mean_knn.mean() + sigma_factor * mean_knn.std())
        print(f"[AnomalyDetector] Fitted: threshold={self.threshold:.4f} "
              f"(n={n_norm}, k={k_actual}, sigma_factor={sigma_factor})")

    def detect(self, query_embeddings: torch.Tensor, k: int = 5) -> Dict[str, np.ndarray]:
        """Return {'is_anomaly': (M,) bool, 'min_dist': (M,) float} for each query point."""
        if self.z_norm is None or self.threshold is None:
            raise ValueError("AnomalyDetector must be fitted before calling detect().")

        queries = query_embeddings.to(self.device)   # (M, D)
        n_norm = self.z_norm.shape[0]

        # Cross-distance matrix (M, N) via geoopt broadcasting.
        manifold = get_manifold(self.curvature, device=self.device)
        q_exp = queries.unsqueeze(1)          # (M, 1, D)
        n_exp = self.z_norm.unsqueeze(0)      # (1, N, D)
        cross_dists = manifold.dist(q_exp, n_exp, keepdim=False)  # (M, N)

        k_actual = min(k, n_norm)
        knn_dists, _ = cross_dists.topk(k_actual, dim=1, largest=False)  # (M, k)
        mean_knn = knn_dists.mean(dim=1).cpu().numpy()  # (M,)

        return {
            "is_anomaly": mean_knn > self.threshold,
            "min_dist": mean_knn,
        }
