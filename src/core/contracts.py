# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Type-safe contracts for p-adic VAE components."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

import torch


# Metrics dicts primarily contain int or float scalars for logging.
# Any is allowed so that Lagrangian dual code can store per-level violation
# tensors (in-graph) alongside the usual float logging values.
# Convention: keys ending in '_tensor' hold torch.Tensor; all others are
# int/float and safe for .item() / logging.
MetricsDict = Dict[str, Any]


class CombinedLossOutput(TypedDict, total=False):
    """Typed return value of ``CombinedLoss.forward()``.

    All keys except ``total`` are optional — they are only present when the
    corresponding loss component is enabled in the config.  Keys ending in
    ``_metrics`` are MetricsDict sub-dicts (safe to log, no gradients).
    Keys ending in ``_tensor`` inside the sub-dicts are in-graph tensors
    used by the Lagrangian dual; all other values are float/int.
    """

    total: Any          # torch.Tensor — always present
    kl: Any
    rich_hierarchy: Any
    radial: Any
    geodesic: Any
    rank: Any
    monotonic: Any
    angular_coherence: Any
    valuation_prior: Any
    within_level_contrastive: Any
    coverage: Any
    lagrangian_margin: Any
    lagrangian_scatter: Any
    lagrangian_prior: Any
    rich_hierarchy_detail: MetricsDict
    radial_metrics: MetricsDict
    geodesic_metrics: MetricsDict
    rank_metrics: MetricsDict
    monotonic_metrics: MetricsDict
    angular_coherence_metrics: MetricsDict
    valuation_prior_metrics: MetricsDict
    wlc_metrics: MetricsDict
    alg_coherence_metrics: MetricsDict
    alg_addition_metrics: MetricsDict
    algebraic_coherence: Any
    algebraic_addition: Any
    ac_skipped_no_r: int
    alg_skipped_no_r: int


class VAEOutput(TypedDict):
    """Output contract for TernaryVAE forward pass."""
    logits: torch.Tensor
    logits_A: Optional[torch.Tensor]
    logits_B: Optional[torch.Tensor]
    mu_A: torch.Tensor
    logvar_A: torch.Tensor
    mu_B: torch.Tensor
    logvar_B: torch.Tensor
    z_A_tangent: torch.Tensor
    z_B_tangent: torch.Tensor
    z_A_hyp: torch.Tensor
    z_B_hyp: torch.Tensor
    r_A: Optional[torch.Tensor]
    r_B: Optional[torch.Tensor]


class TrainingResults(TypedDict):
    """Final results of a training run."""
    epochs_trained: int
    best_Q: float
    best_hierarchy: float
    best_coverage: float
    grokking_events: List[Dict[str, Any]]


@dataclass
class EpochMetrics:
    """Metrics for a single training/validation epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float
    val_coverage: float
    hierarchy_A: float
    hierarchy_B: float
    Q_A: float
    Q_B: float
    dist_corr: float
    mean_radius_A: float
    tree_coherence_A: float
    tree_coherence_B: float
    # Optional detailed metrics
    level_hierarchy: Optional[Dict[int, float]] = None
    ari_per_level: Optional[Dict[int, float]] = None
    ari_composite: Optional[float] = None
    aq_value: Optional[float] = None
    intra_sim: Optional[float] = None
    inter_sim: Optional[float] = None
