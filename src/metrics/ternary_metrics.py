# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rich Ternary Metrics for Deep Q-Metric Feedback.

This module provides multi-dimensional metrics that leverage the structured
properties from src/core/ternary.py to provide StateNet with more nuanced
feedback than scalar hierarchy/coverage alone.

Architecture:
    TernaryMetrics acts as a bridge between:
    - ternary.py: Structured properties (valuation, digit_count, parent, level_rank)
    - geometry/poincare.py: Hyperbolic distances and radii
    - models/statenet.py: Trainability control via Q-feedback

Key Metrics:
    1. Level-Stratified Hierarchy: Per-valuation-level radial correlation
       - Detects which levels are underperforming
       - Enables targeted intervention by StateNet

    2. Tree Coherence: Parent-child geodesic distances
       - Measures preservation of 3-adic tree structure
       - Strong p-adic signal that's independent of radius

    3. Cohort Angular Spread: Distribution within same-valuation cohort
       - Ensures VAE isn't collapsing same-level operations
       - Angular diversity at each radial shell

    4. Digit Pattern Clustering: Similarity of same-digit-count operations
       - Secondary structure beyond valuation
       - digit_count=1 vs digit_count=9 have different semantics

Usage:
    from src.metrics import TernaryMetrics, compute_Q_extended

    metrics = TernaryMetrics(curvature=1.0)
    feedback = metrics.compute_all(z_hyp, indices)
    Q_ext = compute_Q_extended(feedback)

Reference:
    These metrics extend the Q-metric from StateNet (V5.11.8) with
    property-aware multi-scale feedback.
"""

from typing import Dict, Optional

import torch

from src.core.ternary import TERNARY
from src.geometry import hyperbolic_radius, poincare_distance


def level_stratified_hierarchy(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
) -> Dict[int, float]:
    """Compute radial-valuation correlation per valuation level.

    Instead of one global hierarchy correlation, this computes correlation
    at each level, enabling StateNet to detect underperforming levels.

    Args:
        z_hyp: Hyperbolic embeddings on Poincaré ball, shape (N, latent_dim)
        indices: Operation indices, shape (N,)
        curvature: Poincaré ball curvature

    Returns:
        Dict mapping level (0-9) to Spearman correlation at that level.
        Empty levels return NaN.

    Example:
        >>> correlations = level_stratified_hierarchy(z_hyp, indices)
        >>> # correlations = {0: -0.72, 1: -0.81, 2: -0.65, ...}
        >>> # Level 2 is underperforming (-0.65 vs -0.81)
    """
    device = z_hyp.device
    radii = hyperbolic_radius(z_hyp, c=curvature)
    valuations = TERNARY.valuation(indices)

    correlations = {}
    for level in range(TERNARY.MAX_VALUATION + 1):
        mask = valuations == level
        count = mask.sum().item()

        if count < 2:
            correlations[level] = float('nan')
            continue

        # Get radii at this level
        level_radii = radii[mask]

        # Get level_rank (position within cohort) for correlation
        level_ranks = TERNARY.level_rank(indices[mask]).float()

        # Spearman correlation: correlation of ranks
        # Higher rank should correlate with angular position, not radius
        # For hierarchy, we want radius to be consistent within level
        # So compute variance of radii at this level (lower = better)
        radius_std = level_radii.std().item()

        # Convert to correlation-like metric: -1 / (1 + std)
        # Perfect = all same radius = std=0 = -1 (best)
        # Spread = high std = close to 0 (worst)
        correlations[level] = -1.0 / (1.0 + radius_std)

    return correlations


def tree_coherence(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
    sample_size: int = 1000,
) -> float:
    """Compute tree coherence: mean geodesic distance between parent-child pairs.

    In a well-structured embedding, children should be close to their parent
    in hyperbolic space (preserving 3-adic tree structure).

    Args:
        z_hyp: Hyperbolic embeddings on Poincaré ball, shape (N, latent_dim)
        indices: Operation indices, shape (N,)
        curvature: Poincaré ball curvature
        sample_size: Max pairs to sample (for efficiency)

    Returns:
        Mean parent-child geodesic distance (lower = better tree structure)
        Returns 0.0 if no valid parent-child pairs found.
    """
    device = z_hyp.device
    n = z_hyp.shape[0]

    # Get parent indices
    parent_indices = TERNARY.parent(indices)

    # Find pairs where parent is also in batch
    # Create index mapping: index_value -> position in z_hyp
    index_to_pos = {idx.item(): pos for pos, idx in enumerate(indices)}

    # Collect valid parent-child pairs
    child_positions = []
    parent_positions = []

    for pos, (idx, parent_idx) in enumerate(zip(indices, parent_indices)):
        idx_val = idx.item()
        parent_val = parent_idx.item()

        # Skip root (parent=-1) and invalid parents
        if parent_val < 0:
            continue

        # Check if parent is in batch
        if parent_val in index_to_pos:
            child_positions.append(pos)
            parent_positions.append(index_to_pos[parent_val])

    if len(child_positions) == 0:
        return 0.0

    # Sample if too many pairs
    if len(child_positions) > sample_size:
        perm = torch.randperm(len(child_positions), device=device)[:sample_size]
        child_positions = [child_positions[i] for i in perm.tolist()]
        parent_positions = [parent_positions[i] for i in perm.tolist()]

    # Convert to tensors
    child_pos_t = torch.tensor(child_positions, device=device, dtype=torch.long)
    parent_pos_t = torch.tensor(parent_positions, device=device, dtype=torch.long)

    # Get embeddings
    z_children = z_hyp[child_pos_t]
    z_parents = z_hyp[parent_pos_t]

    # Compute geodesic distances
    distances = poincare_distance(z_children, z_parents, c=curvature)

    return distances.mean().item()


def cohort_angular_spread(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
) -> Dict[int, float]:
    """Compute angular spread within each valuation cohort.

    For each level, measure how spread out the embeddings are angularly.
    Uses pairwise distances within cohort normalized by cohort size.

    Good = spread out (high diversity), Bad = collapsed (low diversity)

    Args:
        z_hyp: Hyperbolic embeddings on Poincaré ball, shape (N, latent_dim)
        indices: Operation indices, shape (N,)
        curvature: Poincaré ball curvature

    Returns:
        Dict mapping level to angular spread metric (higher = better)
    """
    device = z_hyp.device
    valuations = TERNARY.valuation(indices)

    spreads = {}
    for level in range(TERNARY.MAX_VALUATION + 1):
        mask = valuations == level
        count = mask.sum().item()

        if count < 2:
            spreads[level] = 0.0
            continue

        # Get embeddings at this level
        z_level = z_hyp[mask]

        # Compute pairwise distances
        # Use a sample if cohort is large
        if count > 100:
            perm = torch.randperm(count, device=device)[:100]
            z_level = z_level[perm]
            count = 100

        # Compute mean pairwise distance (angular/geodesic spread)
        # Expand for pairwise: (n, 1, d) vs (1, n, d)
        z_i = z_level.unsqueeze(1)
        z_j = z_level.unsqueeze(0)
        pairwise = poincare_distance(z_i, z_j, c=curvature)

        # Mean of upper triangle (excluding diagonal)
        n = z_level.shape[0]
        triu_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        mean_dist = pairwise[triu_mask].mean().item()

        spreads[level] = mean_dist

    return spreads


def digit_pattern_similarity(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
) -> Dict[str, float]:
    """Compute clustering metrics based on digit patterns.

    Operations with same digit_count might have similar structure.
    This measures whether the embedding respects this secondary structure.

    Args:
        z_hyp: Hyperbolic embeddings on Poincaré ball, shape (N, latent_dim)
        indices: Operation indices, shape (N,)
        curvature: Poincaré ball curvature

    Returns:
        Dict with:
        - intra_cluster_dist: Mean distance within same digit_count
        - inter_cluster_dist: Mean distance between different digit_counts
        - silhouette: (inter - intra) / max(inter, intra)
    """
    device = z_hyp.device
    digit_counts = TERNARY.digit_count(indices)

    # Sample pairs for efficiency
    n = z_hyp.shape[0]
    n_pairs = min(2000, n * (n - 1) // 2)

    # Random pair indices
    i_idx = torch.randint(0, n, (n_pairs,), device=device)
    j_idx = torch.randint(0, n, (n_pairs,), device=device)

    # Ensure i != j
    same_mask = i_idx == j_idx
    j_idx[same_mask] = (j_idx[same_mask] + 1) % n

    # Compute distances
    distances = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], c=curvature)

    # Check if same digit_count
    same_dc = digit_counts[i_idx] == digit_counts[j_idx]

    intra = distances[same_dc].mean().item() if same_dc.any() else 0.0
    inter = distances[~same_dc].mean().item() if (~same_dc).any() else 0.0

    silhouette = (inter - intra) / max(inter, intra, 1e-8)

    return {
        "intra_cluster_dist": intra,
        "inter_cluster_dist": inter,
        "silhouette": silhouette,
    }


class TernaryMetrics:
    """Rich metrics computer leveraging ternary properties for StateNet feedback.

    Computes multi-dimensional Q-feedback that goes beyond scalar hierarchy:
    - Level-stratified hierarchy (per-level performance)
    - Tree coherence (parent-child structure preservation)
    - Cohort angular spread (diversity within levels)
    - Digit pattern clustering (secondary structure)

    Usage:
        metrics = TernaryMetrics(curvature=1.0)
        feedback = metrics.compute_all(z_hyp, indices)
        # feedback contains: level_hierarchy, tree_coherence, cohort_spread, ...
    """

    def __init__(
        self,
        curvature: float = 1.0,
        compute_level_hierarchy: bool = True,
        compute_tree_coherence: bool = True,
        compute_cohort_spread: bool = True,
        compute_digit_pattern: bool = False,  # Expensive, disabled by default
    ):
        """Initialize metrics computer.

        Args:
            curvature: Poincaré ball curvature
            compute_level_hierarchy: Whether to compute per-level hierarchy
            compute_tree_coherence: Whether to compute parent-child distances
            compute_cohort_spread: Whether to compute angular spread
            compute_digit_pattern: Whether to compute digit pattern clustering
        """
        self.curvature = curvature
        self.compute_level_hierarchy = compute_level_hierarchy
        self.compute_tree_coherence = compute_tree_coherence
        self.compute_cohort_spread = compute_cohort_spread
        self.compute_digit_pattern = compute_digit_pattern

    def compute_all(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
    ) -> Dict[str, any]:
        """Compute all enabled metrics.

        Args:
            z_hyp: Hyperbolic embeddings, shape (N, latent_dim)
            indices: Operation indices, shape (N,)

        Returns:
            Dict with all computed metrics
        """
        results = {}

        if self.compute_level_hierarchy:
            level_corrs = level_stratified_hierarchy(z_hyp, indices, self.curvature)
            results["level_hierarchy"] = level_corrs

            # Aggregate: mean of valid levels
            valid = [v for v in level_corrs.values() if not (v != v)]  # filter NaN
            results["mean_level_hierarchy"] = sum(valid) / len(valid) if valid else 0.0

            # Worst level (least negative = worst)
            results["worst_level_idx"] = max(
                level_corrs.keys(),
                key=lambda k: level_corrs[k] if level_corrs[k] == level_corrs[k] else -float('inf')
            )
            results["worst_level_value"] = level_corrs[results["worst_level_idx"]]

        if self.compute_tree_coherence:
            results["tree_coherence"] = tree_coherence(z_hyp, indices, self.curvature)

        if self.compute_cohort_spread:
            spreads = cohort_angular_spread(z_hyp, indices, self.curvature)
            results["cohort_spread"] = spreads
            results["mean_cohort_spread"] = sum(spreads.values()) / len(spreads)

        if self.compute_digit_pattern:
            digit_metrics = digit_pattern_similarity(z_hyp, indices, self.curvature)
            results["digit_pattern"] = digit_metrics

        return results

    def compute_global_hierarchy(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
    ) -> float:
        """Compute global radial-valuation correlation (original hierarchy metric).

        Args:
            z_hyp: Hyperbolic embeddings, shape (N, latent_dim)
            indices: Operation indices, shape (N,)

        Returns:
            Spearman correlation between radius and valuation (negative = good)
        """
        radii = hyperbolic_radius(z_hyp, c=self.curvature)
        valuations = TERNARY.valuation(indices).float()

        # Spearman correlation via rank correlation
        r_rank = radii.argsort().argsort().float()
        v_rank = valuations.argsort().argsort().float()

        # Pearson correlation of ranks
        r_mean = r_rank.mean()
        v_mean = v_rank.mean()
        numerator = ((r_rank - r_mean) * (v_rank - v_mean)).sum()
        denominator = torch.sqrt(
            ((r_rank - r_mean) ** 2).sum() * ((v_rank - v_mean) ** 2).sum()
        )

        if denominator < 1e-8:
            return 0.0

        return (numerator / denominator).item()


def compute_Q_extended(
    feedback: Dict[str, any],
    dist_corr: float = 0.0,
    global_hierarchy: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute extended Q-metric from rich feedback.

    Q_extended = w_dist * dist_corr
               + w_hier * |global_hierarchy|
               + w_level * mean_level_hierarchy
               + w_tree * (1 - tree_coherence)  # Lower distance = better
               + w_spread * mean_cohort_spread

    Args:
        feedback: Dict from TernaryMetrics.compute_all()
        dist_corr: Distance correlation (from existing metrics)
        global_hierarchy: Global hierarchy correlation (from existing)
        weights: Optional weight dict. Defaults to balanced weights.

    Returns:
        Extended Q metric (higher = better)
    """
    if weights is None:
        weights = {
            "dist_corr": 1.0,
            "hierarchy": 1.5,
            "level_hierarchy": 0.5,
            "tree_coherence": 0.3,
            "cohort_spread": 0.2,
        }

    Q = 0.0
    Q += weights.get("dist_corr", 0) * dist_corr
    Q += weights.get("hierarchy", 0) * abs(global_hierarchy)

    if "mean_level_hierarchy" in feedback:
        # Level hierarchy is already negative for good, so use abs
        Q += weights.get("level_hierarchy", 0) * abs(feedback["mean_level_hierarchy"])

    if "tree_coherence" in feedback:
        # Tree coherence: lower distance = better, so use 1/(1+d)
        tc = feedback["tree_coherence"]
        Q += weights.get("tree_coherence", 0) * (1.0 / (1.0 + tc))

    if "mean_cohort_spread" in feedback:
        # Spread: higher = better (more diverse)
        Q += weights.get("cohort_spread", 0) * feedback["mean_cohort_spread"]

    return Q


__all__ = [
    "TernaryMetrics",
    "compute_Q_extended",
    "level_stratified_hierarchy",
    "tree_coherence",
    "cohort_angular_spread",
    "digit_pattern_similarity",
]
