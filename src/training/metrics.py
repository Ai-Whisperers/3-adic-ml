# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Metrics computation for p-adic VAE training."""

from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Union

import numpy as np
from scipy.stats import spearmanr
import torch

from ..core import TERNARY
from ..core.contracts import GrokkingState
from ..geometry import hyperbolic_radius, poincare_distance
from ..models import compute_Q
from ..utils.scatter_utils import level_scatter_std


def _decode_preds(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """Decode logits to discrete ternary predictions in {-1, 0, 1}.

    Supports (B, 9, 3) and (B, 27) logit layouts; returns None for anything else.
    """
    if logits.shape[-1] == 3:
        return torch.argmax(logits, dim=-1) - 1
    if logits.shape[-1] == 27:
        return torch.argmax(logits.view(-1, 9, 3), dim=-1) - 1
    return None


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute reconstruction accuracy.

    Args:
        logits: Model output logits (B, 27) or (B, 9, 3)
        targets: Target ternary values (B, 9) in {-1, 0, 1}

    Returns:
        Accuracy as fraction [0, 1]
    """
    with torch.no_grad():
        preds = _decode_preds(logits)
        if preds is None:
            return 0.0
        return (preds == targets.long()).float().mean().item()


def compute_coverage(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perfect reconstruction coverage.

    Args:
        logits: Model output logits
        targets: Target ternary values

    Returns:
        Coverage as fraction of samples with perfect reconstruction
    """
    with torch.no_grad():
        preds = _decode_preds(logits)
        if preds is None:
            return 0.0
        correct_per_sample = (preds == targets.long()).float().mean(dim=1)
        return (correct_per_sample == 1.0).float().mean().item()


def compute_tree_coherence(
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
    """
    with torch.no_grad():
        device = z_hyp.device
        parent_indices = TERNARY.parent(indices)

        # Create index mapping: index_value -> position in z_hyp
        index_to_pos = {idx.item(): pos for pos, idx in enumerate(indices)}

        # Collect valid parent-child pairs
        child_positions = []
        parent_positions = []

        for pos, (_idx_val_t, p_idx_val_t) in enumerate(zip(indices, parent_indices, strict=False)):
            parent_val = p_idx_val_t.item()
            if parent_val < 0:  # Skip root
                continue
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

        child_pos_t = torch.tensor(child_positions, device=device, dtype=torch.long)
        parent_pos_t = torch.tensor(parent_positions, device=device, dtype=torch.long)

        z_children = z_hyp[child_pos_t]
        z_parents = z_hyp[parent_pos_t]

        distances = poincare_distance(z_children, z_parents, c=curvature)
        return float(distances.mean().item())


def compute_level_stratified_hierarchy(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
    valuation_fn=None,
) -> Dict[int, float]:
    """Compute radial consistency per valuation level.

    For each level, measures how consistent the radii are (lower std = better).
    Returns a correlation-like metric: -1/(1+std) where -1 is perfect.

    Args:
        z_hyp: Hyperbolic embeddings on Poincaré ball, shape (N, latent_dim)
        indices: Operation indices, shape (N,)
        curvature: Poincaré ball curvature
        valuation_fn: Optional custom valuation function

    Returns:
        Dict mapping level (0-9) to consistency metric (more negative = better)
    """
    _val_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
    with torch.no_grad():
        radii = hyperbolic_radius(z_hyp, c=curvature)
        valuations = _val_fn(indices)
        dim_size = TERNARY.MAX_VALUATION + 1
        vals_long = valuations.long()

        # Vectorized per-level std via scatter (single pass, no Python loop)
        stds_all = level_scatter_std(radii, cast(torch.LongTensor, vals_long), dim_size=dim_size)  # (dim_size,)
        counts = torch.zeros(dim_size, dtype=torch.long, device=radii.device)
        counts.scatter_add_(0, vals_long, torch.ones_like(vals_long))

        correlations = {}
        for level in range(dim_size):
            if counts[level].item() < 2:
                correlations[level] = float("nan")
            else:
                correlations[level] = -1.0 / (1.0 + stds_all[level].item())

        return correlations


def compute_hierarchy_metrics(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
    seed: int = 42,
    valuation_fn=None,
) -> Dict[str, Union[float, bool, Dict[int, float]]]:
    """Compute hierarchy and Q metrics.

    Args:
        z_hyp: Hyperbolic embeddings (N, latent_dim)
        indices: Operation indices (N,)
        curvature: Poincaré ball curvature
        seed: Random seed for sampling
        valuation_fn: Optional custom valuation function

    Returns:
        Dict with hierarchy, dist_corr, Q, and diagnostic metrics
    """
    _val_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
    with torch.no_grad():
        n = z_hyp.size(0)
        if n < 2:
            # Same key set as the full path below (engine.py reads e.g.
            # "mean_radius" via direct bracket access) so a validation split
            # with 0-1 samples degrades gracefully instead of raising KeyError.
            return {
                "hierarchy": 0.0,
                "dist_corr": 0.0,
                "Q": 0.0,
                "mean_radius": 0.0,
                "std_radius": 0.0,
                "tree_coherence": 0.0,
                "level_hierarchy": {},
                "level_radii": {},
                "worst_level": 0,
                "mean_level_hierarchy": 0.0,
                "hierarchy_collapsed": True,
                "dist_corr_collapsed": True,
            }

        # 1. Radius Hierarchy (Spearman correlation between valuation and -radius)
        # Using hyperbolic radius for consistency
        radii = hyperbolic_radius(z_hyp, c=curvature).cpu().numpy()
        valuations = _val_fn(indices).cpu().numpy()

        # spearmanr can return NaN if all radii or valuations are identical (collapse)
        res = spearmanr(valuations, -radii)
        hierarchy_raw = res.correlation
        hierarchy_collapsed = bool(np.isnan(hierarchy_raw))
        hierarchy = 0.0 if hierarchy_collapsed else float(hierarchy_raw)

        # 2. Distance Correlation (Spearman correlation between pairwise hyperbolic
        # distances and pairwise p-adic distances)
        # Sample pairs for efficiency if N is large
        max_pairs = 2000
        if n > 100:  # ~5000 pairs if all used; sample if larger
            rng = np.random.default_rng(seed)
            i_idx = rng.integers(0, n, size=max_pairs)
            j_idx = rng.integers(0, n, size=max_pairs)
            # Avoid self-pairs
            mask = i_idx == j_idx
            j_idx[mask] = (j_idx[mask] + 1) % n
        else:
            # Use all pairs
            i_idx, j_idx = np.triu_indices(n, k=1)

        # Compute hyperbolic distances
        z1 = z_hyp[i_idx]
        z2 = z_hyp[j_idx]
        r_dists = poincare_distance(z1, z2, c=curvature).cpu().numpy()

        # Compute p-adic distances
        idx1 = indices[i_idx].cpu().numpy()
        idx2 = indices[j_idx].cpu().numpy()
        # v_3(|n-m|)
        diffs = np.abs(idx1 - idx2)
        v_dists = TERNARY.valuation(torch.from_numpy(diffs)).numpy()

        # Spearman correlation: high valuation -> small hyperbolic distance
        # So we correlate (hyperbolic distance) with (-valuation)
        # Or (hyperbolic distance) with (p-adic distance) where p-adic dist = 3^-v
        res_corr = spearmanr(r_dists, -v_dists)
        dist_corr_raw = res_corr.correlation
        dist_corr_collapsed = bool(np.isnan(dist_corr_raw))
        dist_corr = 0.0 if dist_corr_collapsed else float(dist_corr_raw)

        Q = compute_Q(dist_corr, hierarchy)

        # Additional metrics: tree coherence and per-level hierarchy
        tree_coh = compute_tree_coherence(z_hyp, indices, curvature)
        level_hier = compute_level_stratified_hierarchy(z_hyp, indices, curvature,
                                                        valuation_fn=_val_fn)

        # Compute per-level mean radii
        level_radii = {}
        for level in range(TERNARY.MAX_VALUATION + 1):
            mask = (valuations == level)
            if mask.any():
                level_radii[level] = float(radii[mask].mean())
            else:
                level_radii[level] = float("nan")

        # Compute worst level (least negative = worst performing)
        valid_levels = {k: v for k, v in level_hier.items() if not np.isnan(v)}
        worst_level = (
            max(valid_levels.keys(), key=lambda k: valid_levels[k])
            if valid_levels
            else 0
        )
        mean_level_hier = np.mean(list(valid_levels.values())) if valid_levels else 0.0

        return {
            "hierarchy": float(hierarchy),
            "dist_corr": float(dist_corr),
            "Q": float(Q),
            "mean_radius": float(radii.mean()),
            "std_radius": float(radii.std()),
            "tree_coherence": float(tree_coh),
            "level_hierarchy": level_hier,
            "level_radii": level_radii,
            "worst_level": int(worst_level),
            "mean_level_hierarchy": float(mean_level_hier),
            "hierarchy_collapsed": bool(hierarchy_collapsed),
            "dist_corr_collapsed": bool(dist_corr_collapsed),
        }


def plot_algebraic_consistency(
    model: torch.nn.Module,
    indices: torch.Tensor,
    mu_A: torch.Tensor,
    device: torch.device,
    n_samples: int = 100,
) -> Optional[Any]:
    """Generate a figure comparing predicted vs actual algebraic results in latent space."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    model.eval()
    B = indices.size(0)
    if B < 2:
        return None

    n_samples = min(n_samples, B // 2)
    perm = torch.randperm(B, device=device)[: 2 * n_samples]
    idx_a = indices[perm[:n_samples]]
    idx_b = indices[perm[n_samples:]]

    with torch.no_grad():
        mu_a = mu_A[perm[:n_samples]]
        mu_b = mu_A[perm[n_samples:]]
        
        # Predicted mu sum
        mu_sum_pred = mu_a + mu_b
        
        # Actual mu sum
        idx_sum_gt = TERNARY.ternary_add(idx_a, idx_b)
        mu_sum_gt = model.get_mu_representations(idx_sum_gt, device)
        
        # Project both to Poincaré ball.
        # proj_A returns (z_hyp, r) in factored mode, plain z_hyp otherwise.
        def _project(v):
            result = model.projections.proj_A(v)
            return result[0] if isinstance(result, tuple) else result

        z_pred = _project(mu_sum_pred)
        z_gt = _project(mu_sum_gt)
        
        z_pred = z_pred.cpu().numpy()
        z_gt = z_gt.cpu().numpy()

    # Use first 2 dimensions for simple 2D visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_gt[:, 0], z_gt[:, 1], c='blue', alpha=0.5, label='Actual z(a+b)', s=20)
    ax.scatter(z_pred[:, 0], z_pred[:, 1], c='red', alpha=0.5, label='Predicted z(a)+z(b)', s=20)
    
    # Draw lines between pairs
    for i in range(n_samples):
        ax.plot([z_gt[i, 0], z_pred[i, 0]], [z_gt[i, 1], z_pred[i, 1]], 'gray', alpha=0.2, lw=0.5)

    # Draw boundary
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Algebraic Addition Consistency (Latent Space)")
    ax.legend(fontsize='small')
    
    return fig


@dataclass
class GrokkingEvent:
    """Record of a grokking event."""
    epoch: int
    plateau_duration: int
    val_lift: float
    gap_collapse: float


class GrokkingDetector:
    """Detects grokking via: Plateau -> Lift -> Gap Collapse."""

    def __init__(
        self,
        window: int = 20,
        slope_eps: float = 1e-4,
        sustain_k: int = 6,
        val_lift_min: float = 0.02,
        gap_collapse_min: float = 0.02,
    ):
        self.window = window
        self.slope_eps = slope_eps
        self.sustain_k = sustain_k
        self.val_lift_min = val_lift_min
        self.gap_collapse_min = gap_collapse_min

        self.history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_acc": []}
        self.plateau_start: Optional[int] = None
        self.events: List[GrokkingEvent] = []

    def _slope(self, ys: List[float]) -> float:
        if len(ys) < 2:
            return 0.0
        x = np.arange(len(ys))
        return float(np.polyfit(x, ys, 1)[0])

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_acc: float,
    ) -> GrokkingState:
        """Update detector with new epoch metrics.

        Returns dict with:
            - plateau: bool, whether in plateau
            - potential: bool, whether grokking potential detected
            - event: bool, whether grokking event just occurred
        """
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        out: GrokkingState = {"plateau": False, "potential": False, "event": False, "val_lift": 0.0}

        if len(self.history["train_loss"]) < self.window:
            return out

        # Recent history windows
        recent_loss = self.history["train_loss"][-self.window:]
        recent_val = self.history["val_acc"][-self.window:]
        recent_train = self.history["train_acc"][-self.window:]

        # Plateau check: low slope on training loss
        loss_slope = self._slope(recent_loss)
        in_plateau = abs(loss_slope) < self.slope_eps

        if in_plateau:
            if self.plateau_start is None:
                self.plateau_start = epoch
            out["plateau"] = True
        # else: plateau_start is intentionally NOT reset here, to allow "lift" detection.

        # Grokking potential: train accuracy near 1.0, val accuracy still low/flat
        train_near_one = np.mean(recent_train[-self.sustain_k:]) > 0.98
        val_flat = abs(self._slope(recent_val)) < self.slope_eps
        if train_near_one and val_flat:
            out["potential"] = True

        # Grokking event: significant lift in val accuracy + closing gap
        if self.plateau_start is not None:
            # Measure lift from plateau start
            # Using median to avoid outlier noise
            hist_len = len(self.history["val_acc"])
            plateau_idx = max(0, hist_len - (epoch - self.plateau_start + 1))
            val_at_plateau = np.median(self.history["val_acc"][plateau_idx : plateau_idx + 5])
            val_lift = val_acc - val_at_plateau

            if val_lift > self.val_lift_min:
                # Potential event! Check gap collapse
                gap_then = np.median(self.history["train_acc"][plateau_idx:plateau_idx+5]) - val_at_plateau
                gap_now = train_acc - val_acc
                gap_collapse = gap_then - gap_now

                if gap_collapse > self.gap_collapse_min:
                    # GROKKING!
                    event = GrokkingEvent(
                        epoch=epoch,
                        plateau_duration=epoch - self.plateau_start,
                        val_lift=val_lift,
                        gap_collapse=gap_collapse,
                    )
                    self.events.append(event)
                    out["event"] = True
                    out["val_lift"] = float(val_lift)
                    self.plateau_start = None  # Reset for next potential event

        return out

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state."""
        return {
            "history": self.history,
            "plateau_start": self.plateau_start,
            "events": [
                {
                    "epoch": e.epoch,
                    "plateau_duration": e.plateau_duration,
                    "val_lift": e.val_lift,
                    "gap_collapse": e.gap_collapse,
                }
                for e in self.events
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from dict."""
        self.history = state.get("history", {"train_loss": [], "train_acc": [], "val_acc": []})
        self.plateau_start = state.get("plateau_start")
        events_raw = state.get("events", [])
        self.events = [GrokkingEvent(**e) for e in events_raw]
