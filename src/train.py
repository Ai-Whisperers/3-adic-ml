#!/usr/bin/env python3
# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Unified Training Script for p-adic VAE.

This is the canonical training entry point for all p-adic VAE experiments.
It combines scientific rigor (audit, reproducibility) with config-driven
flexibility (any config from src/presets/).

Features:
    - Config-driven: Loads any YAML config from src/presets/
    - Reproducible: Deterministic seeding for all random sources
    - Audited: Data integrity and model health checks before training
    - StateNet: Adaptive trainability controller
    - Combined Loss: Config-driven loss composition
    - Metrics: Coverage, hierarchy, Q metric, grokking detection

Usage:
    # Production training
    python src/train.py --config src/presets/production_rich_hierarchy.yaml

    # Quick test
    python src/train.py --config src/presets/minimal_smoke_test.yaml

    # Validate config only (no training)
    python src/train.py --config src/presets/production_rich_hierarchy.yaml --validate-only

    # Force training even if audit fails
    python src/train.py --config src/presets/production_rich_hierarchy.yaml --force
"""

import argparse
import atexit
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

# tqdm for progress bars (optional)
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Internal imports
from src.config import StateNetConfig
from src.config.paths import RUNS_DIR
from src.config.schema import normalize_config
from src.core import TERNARY
from src.core.ternary import get_valuation_fn
from src.geometry import get_riemannian_optimizer, hyperbolic_radius, poincare_distance
from src.losses import CombinedLoss
from src.losses.lagrangian import LagrangianDualState
from src.models import (
    MetricBasedLR,
    TernaryVAEV6Controllable,
    TrainingMetrics,
    compute_Q,
    get_optimizer_grad_stats,
    update_optimizer_lr_scales,
)
from src.utils import HardwareMonitor, TensorBoardLogger
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat
from src.utils.scatter_utils import level_scatter_std
from src.utils.visualization import VisualizationPipeline

# =============================================================================
# DETERMINISM
# =============================================================================


def set_determinism(
    seed: int, deterministic: bool = True, use_float64: bool = True
) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but exact)
        use_float64: If True, use float64 as default dtype (required for geoopt stability)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set default dtype to float64 for geoopt compatibility
    if use_float64:
        torch.set_default_dtype(torch.float64)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUDA >= 10.2 requires this for cuBLAS determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            # Use warn_only=True because some ops (like nll_loss) lack deterministic implementations
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't have warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def get_timestamp() -> str:
    """Get formatted timestamp for run naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# DATA AUDITOR
# =============================================================================


class DataAuditor:
    """Validates data integrity before training."""

    def __init__(self, seed: int):
        self.seed = seed
        self.audit_log = {}

    def prepare_data(
        self,
        val_frac: float = 0.1,
        device: torch.device = None,
    ) -> Tuple[TensorDataset, TensorDataset, torch.Tensor]:
        """Generate data, split, and validate.

        Args:
            val_frac: Fraction of data for validation
            device: Device for data tensors

        Returns:
            Tuple of (train_dataset, val_dataset, all_indices)
        """
        if device is None:
            device = torch.device("cpu")
        print("\n[AUDIT] Data Integrity Check...")

        # Generate all operations (uses cached LUT from TERNARY singleton)
        all_ops = TERNARY.all_ternary()
        n = len(all_ops)
        all_indices = torch.arange(n, dtype=torch.long)

        # Deterministic split
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, round(n * val_frac))

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train = all_ops[train_idx]
        X_val = all_ops[val_idx]
        idx_train = all_indices[train_idx]
        idx_val = all_indices[val_idx]

        # Leakage check
        train_set = {tuple(x.tolist()) for x in X_train}
        val_set = {tuple(x.tolist()) for x in X_val}
        overlap = train_set.intersection(val_set)

        self.audit_log["n_train"] = len(X_train)
        self.audit_log["n_val"] = len(X_val)
        self.audit_log["data_leakage_count"] = len(overlap)

        if len(overlap) > 0:
            print(
                f"  [WARN] Data leakage detected: {len(overlap)} samples in both sets"
            )
        else:
            print(f"  [OK] No data leakage (train={len(X_train)}, val={len(X_val)})")

        # Value distribution check
        vals, counts = torch.unique(X_train, return_counts=True)
        dist = counts.float() / counts.sum()
        self.audit_log["value_distribution"] = dict(zip(vals.tolist(), dist.tolist(), strict=False))
        print(
            f"  [OK] Value distribution: {dict(zip(vals.tolist(), [f'{d:.2%}' for d in dist.tolist()], strict=False))}"
        )

        # Create datasets
        train_ds = TensorDataset(X_train, idx_train)
        val_ds = TensorDataset(X_val, idx_val)

        return train_ds, val_ds, all_indices


# =============================================================================
# MODEL AUDITOR
# =============================================================================


class ModelAuditor:
    """Validates model health before training."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.audit_log = {}

    def create_and_validate_model(self, force: bool = False) -> nn.Module:
        """Create model, load checkpoint, validate gradients.

        Args:
            force: If True, continue even if validation fails

        Returns:
            Validated model ready for training
        """
        print("\n[AUDIT] Model Health Check...")

        model_cfg = self.config.get("model", {})
        option_c_cfg = self.config.get("option_c", {})
        model_name = model_cfg.get("name", "TernaryVAEV6Controllable")
        encoder_type = model_cfg.get("encoder_type", "improved")
        decoder_type = model_cfg.get("decoder_type", "improved")

        # LR scales from option_c config (differential learning rates per component)
        encoder_a_lr_scale = option_c_cfg.get("encoder_a_lr_scale", 0.05)
        encoder_b_lr_scale = option_c_cfg.get("encoder_b_lr_scale", 0.1)
        projections_lr_scale = option_c_cfg.get("projections_lr_scale", 1.0)

        # Initial trainability from statenet config (nested structure)
        statenet_cfg = self.config.get("statenet", {})
        initial_cfg = statenet_cfg.get("initial", {})
        encoder_a_trainable = initial_cfg.get("encoder_a_trainable", False)
        encoder_b_trainable = initial_cfg.get("encoder_b_trainable", True)
        projections_trainable = initial_cfg.get("projections_trainable", True)

        # Instantiate model
        model = TernaryVAEV6Controllable(
            latent_dim=model_cfg.get("latent_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            max_radius=model_cfg.get("max_radius", 0.95),
            curvature=model_cfg.get("curvature", 1.0),
            n_projection_layers=model_cfg.get("projection_layers", 2),
            projection_dropout=model_cfg.get("projection_dropout", 0.1),
            learnable_curvature=model_cfg.get("learnable_curvature", False),
            init_identity=model_cfg.get("init_identity", True),
            tangent_scale_init=model_cfg.get("tangent_scale", 0.1),
            factored=model_cfg.get("factored", False),
            radial_dims=model_cfg.get("radial_dims", 4),
            encoder_a_trainable=encoder_a_trainable,
            encoder_b_trainable=encoder_b_trainable,
            projections_trainable=projections_trainable,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            encoder_a_lr_scale=encoder_a_lr_scale,
            encoder_b_lr_scale=encoder_b_lr_scale,
            projections_lr_scale=projections_lr_scale,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] Model created: {model_name}")
        print(f"       Parameters: {n_params:,} total, {n_trainable:,} trainable")

        # Load anchor checkpoint if specified
        anchor_cfg = self.config.get("anchor_checkpoint") or {}
        ckpt_path_str = anchor_cfg.get("path")

        if ckpt_path_str and ckpt_path_str != "null":
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            if ckpt_path.exists():
                try:
                    ckpt = load_checkpoint_compat(ckpt_path, map_location=self.device)
                    state_dict = get_model_state_dict(ckpt)

                    # Load with strict=False (projections may not match)
                    missing, unexpected = model.load_state_dict(
                        state_dict, strict=False
                    )
                    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
                    print(
                        f"       Missing keys: {len(missing)}, Unexpected: {len(unexpected)}"
                    )

                    self.audit_log["checkpoint_loaded"] = True
                    self.audit_log["checkpoint_path"] = str(ckpt_path)

                except Exception as e:
                    print(f"  [WARN] Checkpoint load failed: {e}")
                    self.audit_log["checkpoint_loaded"] = False
                    self.audit_log["checkpoint_error"] = str(e)
                    if not force:
                        raise
                    print("  [WARN] --force active: proceeding with RANDOM INIT (checkpoint ignored)")
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt_path}")
                self.audit_log["checkpoint_loaded"] = False
                self.audit_log["checkpoint_error"] = f"not found: {ckpt_path}"
                if not force:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                print("  [WARN] --force active: proceeding with RANDOM INIT (checkpoint not found)")
        else:
            print("  [INFO] No anchor checkpoint specified (training from scratch)")
            self.audit_log["checkpoint_loaded"] = False

        # Gradient flow check
        model.train()
        dummy_input = torch.randint(-1, 2, (32, 9)).double().to(self.device)

        try:
            out = model(dummy_input)
            logits = out.get("logits_A", out.get("logits"))

            # Simple forward check
            if logits is None:
                raise ValueError("Model output missing logits")

            # Compute dummy loss and backward
            loss = logits.mean()
            loss.backward()

            # Check gradient norm
            total_norm = 0.0
            dead_params = 0
            total_params = 0

            for p in model.parameters():
                if p.requires_grad:
                    total_params += 1
                    if p.grad is not None:
                        total_norm += p.grad.norm(2).item()
                        if p.grad.norm(2).item() == 0:
                            dead_params += 1
                    else:
                        dead_params += 1

            self.audit_log["initial_grad_norm"] = total_norm
            self.audit_log["dead_params"] = dead_params

            if total_norm == 0:
                print("  [ERROR] Vanishing gradients - model is dead")
                if not force:
                    raise RuntimeError("Dead model detected")
            elif total_norm > 1000:
                print("  [WARN] Large gradient norm - may explode")
            else:
                print(f"  [OK] Gradient flow active (norm={total_norm:.4f})")

        except Exception as e:
            print(f"  [ERROR] Model health check failed: {e}")
            if not force:
                raise
        finally:
            model.zero_grad()

        return model


# =============================================================================
# METRICS
# =============================================================================


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute reconstruction accuracy.

    Args:
        logits: Model output logits (B, 27) or (B, 9, 3)
        targets: Target ternary values (B, 9) in {-1, 0, 1}

    Returns:
        Accuracy as fraction [0, 1]
    """
    with torch.no_grad():
        if logits.shape[-1] == 3:
            # (B, 9, 3) format
            preds = torch.argmax(logits, dim=-1) - 1
            return (preds == targets.long()).float().mean().item()
        elif logits.shape[-1] == 27:
            # (B, 27) format
            logits_3 = logits.view(-1, 9, 3)
            preds = torch.argmax(logits_3, dim=-1) - 1
            return (preds == targets.long()).float().mean().item()
        return 0.0


def compute_coverage(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perfect reconstruction coverage.

    Args:
        logits: Model output logits
        targets: Target ternary values

    Returns:
        Coverage as fraction of samples with perfect reconstruction
    """
    with torch.no_grad():
        if logits.shape[-1] == 3:
            preds = torch.argmax(logits, dim=-1) - 1
        elif logits.shape[-1] == 27:
            logits_3 = logits.view(-1, 9, 3)
            preds = torch.argmax(logits_3, dim=-1) - 1
        else:
            return 0.0

        correct_per_sample = (preds == targets.long()).float().mean(dim=1)
        perfect = (correct_per_sample == 1.0).float().mean().item()
        return perfect


def compute_hyperbolic_coverage(z_hyp: torch.Tensor, curvature: float = 1.0) -> float:
    """Compute hyperbolic coverage via radial entropy/spread."""
    with torch.no_grad():
        # Normalize hyperbolic radius [0, inf) -> [0, 1) using 1 - exp(-r)
        radii_raw = hyperbolic_radius(z_hyp, c=curvature)
        radii = 1.0 - torch.exp(-radii_raw)  # Maps [0, inf) -> [0, 1)
        radii = radii.clamp(min=0.001, max=0.999)
        hist = torch.histc(radii, bins=10, min=0.0, max=1.0)
        probs = (hist / hist.sum().clamp(min=1)) + 1e-10
        probs = probs / probs.sum()
        entropy = -(probs * torch.log(probs)).sum()
        max_entropy = torch.log(torch.tensor(10, dtype=torch.float64))
        return (entropy / max_entropy).item()


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

        for pos, (_idx, parent_idx) in enumerate(zip(indices, parent_indices, strict=False)):
            parent_val = parent_idx.item()
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
        return distances.mean().item()


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
        stds_all = level_scatter_std(radii, vals_long, dim_size=dim_size)  # (dim_size,)
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
) -> Dict[str, float]:
    """Compute hierarchy and Q metrics.

    Args:
        z_hyp: Hyperbolic embeddings (B, latent_dim)
        indices: Operation indices (B,)
        curvature: Poincaré ball curvature
        seed: Random seed for reproducible sampling
        valuation_fn: Callable(indices) -> valuations. Defaults to TERNARY.valuation.

    Returns:
        Dict with hierarchy, dist_corr, Q, tree_coherence, level metrics
    """
    _val_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
    with torch.no_grad():
        # Compute radii using canonical hyperbolic_radius
        radii = hyperbolic_radius(z_hyp, c=curvature).cpu().numpy()
        valuations = _val_fn(indices).cpu().numpy()

        # Hierarchy: negated Spearman correlation between valuation and radius.
        # High valuation → low radius (near origin), so raw correlation is negative.
        # We negate so positive values indicate good hierarchy (range -1..1, 1 = perfect).
        hierarchy_raw = -spearmanr(valuations, radii).correlation
        hierarchy_collapsed = bool(np.isnan(hierarchy_raw))
        hierarchy = 0.0 if hierarchy_collapsed else float(hierarchy_raw)

        # Distance correlation (sample-based for efficiency)
        n = min(1000, len(z_hyp))
        if n < 2:
            return {"hierarchy": hierarchy, "dist_corr": 0.0, "Q": 0.0,
                    "hierarchy_collapsed": hierarchy_collapsed, "dist_corr_collapsed": True}

        # Use explicit RNG for reproducibility
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z_hyp), n, replace=False)
        r_sample = radii[idx]
        v_sample = valuations[idx]

        # Pairwise distances in radius and valuation
        r_dists = np.abs(r_sample[:, None] - r_sample[None, :])
        v_dists = np.abs(v_sample[:, None] - v_sample[None, :])

        triu_idx = np.triu_indices(n, k=1)
        dist_corr_raw = spearmanr(r_dists[triu_idx], v_dists[triu_idx]).correlation
        dist_corr_collapsed = bool(np.isnan(dist_corr_raw))
        dist_corr = 0.0 if dist_corr_collapsed else float(dist_corr_raw)

        Q = compute_Q(dist_corr, hierarchy)

        # Additional metrics: tree coherence and per-level hierarchy
        tree_coh = compute_tree_coherence(z_hyp, indices, curvature)
        level_hier = compute_level_stratified_hierarchy(z_hyp, indices, curvature,
                                                        valuation_fn=_val_fn)

        # Compute worst level (least negative = worst performing)
        valid_levels = {k: v for k, v in level_hier.items() if not np.isnan(v)}
        worst_level = (
            max(valid_levels.keys(), key=lambda k: valid_levels[k])
            if valid_levels
            else 0
        )
        mean_level_hier = np.mean(list(valid_levels.values())) if valid_levels else 0.0

        return {
            "hierarchy": hierarchy,
            "dist_corr": dist_corr,
            "Q": Q,
            "mean_radius": float(radii.mean()),
            "std_radius": float(radii.std()),
            "tree_coherence": tree_coh,
            "level_hierarchy": level_hier,
            "worst_level": worst_level,
            "mean_level_hierarchy": mean_level_hier,
            "hierarchy_collapsed": hierarchy_collapsed,
            "dist_corr_collapsed": dist_corr_collapsed,
        }


# =============================================================================
# GROKKING DETECTOR
# =============================================================================


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

        self.history = {"train_loss": [], "train_acc": [], "val_acc": []}
        self.plateau_start = None
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
    ) -> Dict[str, Any]:
        """Update detector with new epoch metrics.

        Returns dict with:
            - plateau: bool, whether in plateau
            - potential: bool, whether grokking potential detected
            - event: bool, whether grokking event just occurred
        """
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        out = {"plateau": False, "potential": False, "event": False, "val_lift": 0.0}

        if len(self.history["train_loss"]) < self.window:
            return out

        # Plateau check
        recent_loss = self.history["train_loss"][-self.window :]
        slope = abs(self._slope(recent_loss))
        out["plateau"] = slope < self.slope_eps

        if out["plateau"] and self.plateau_start is None:
            self.plateau_start = epoch - self.window
        elif not out["plateau"] and self.plateau_start is not None:
            if epoch - self.plateau_start > self.window * 3:
                self.plateau_start = None

        # Grokking check
        if (
            self.plateau_start is not None
            and len(self.history["val_acc"]) > self.window + self.sustain_k
        ):
            baseline_val = np.mean(
                self.history["val_acc"][
                    -(self.window + self.sustain_k) : -self.sustain_k
                ]
            )
            recent_val = np.mean(self.history["val_acc"][-self.sustain_k :])
            val_lift = recent_val - baseline_val

            baseline_gap = (
                np.mean(
                    self.history["train_acc"][
                        -(self.window + self.sustain_k) : -self.sustain_k
                    ]
                )
                - baseline_val
            )
            recent_gap = (
                np.mean(self.history["train_acc"][-self.sustain_k :]) - recent_val
            )
            gap_collapse = baseline_gap - recent_gap

            out["val_lift"] = val_lift

            if val_lift > self.val_lift_min and gap_collapse > self.gap_collapse_min:
                out["potential"] = True
                self.events.append(
                    GrokkingEvent(
                        epoch=epoch,
                        plateau_duration=epoch - self.plateau_start,
                        val_lift=val_lift,
                        gap_collapse=gap_collapse,
                    )
                )
                out["event"] = True
                self.plateau_start = None

        return out


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================


def _build_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    lr_controller: Any = None,
    dual_state: Any = None,
    loss_cfg: Optional[Dict[str, Any]] = None,
    loss_fn: Optional[nn.Module] = None,
    loss_fn_b: Optional[nn.Module] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a checkpoint payload dict with all resumable state.

    Centralises the checkpoint construction that was previously duplicated
    across best-Q, periodic, and final checkpoint sites.

    Args:
        epoch: Current epoch number.
        model: The VAE model.
        optimizer: Optimizer with state dict.
        scheduler: LR scheduler with state dict.
        lr_controller: Optional MetricBasedLR controller.
        dual_state: Optional LagrangianDualState.
        loss_cfg: Loss config dict (checked for learnable_weights flag).
        loss_fn: Primary CombinedLoss (VAE-A). Saved when learnable_weights=True.
        loss_fn_b: Secondary CombinedLoss (VAE-B). Saved when learnable_weights=True.
        extra: Optional dict of additional keys merged into the payload
               (e.g. best_Q, best_hierarchy, best_coverage).

    Returns:
        Dict ready for ``torch.save()``.
    """
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if lr_controller is not None:
        payload["controller_state"] = {
            "active": dict(lr_controller._active),
            "best_q": lr_controller._best_q,
            "last_epoch": lr_controller._last_epoch,
            "last_change": dict(lr_controller._last_change),
            "hierarchy_b_plateau_count": lr_controller._hierarchy_b_plateau_count,
            "grad_low_count": lr_controller._grad_low_count,
            "hierarchy_a_stall_count": lr_controller._hierarchy_a_stall_count,
            "coverage_history": list(lr_controller._coverage_history),
            "hierarchy_a_history": list(lr_controller._hierarchy_a_history),
            "hierarchy_b_history": list(lr_controller._hierarchy_b_history),
            "grad_norm_history": list(lr_controller._grad_norm_history),
            "q_history": list(lr_controller._q_history),
        }
    if dual_state is not None:
        payload["lagrangian_state"] = dual_state.state_dict()
    if loss_cfg is not None and loss_cfg.get("learnable_weights", False):
        if loss_fn is not None:
            payload["loss_fn_state_dict"] = loss_fn.state_dict()
        if loss_fn_b is not None:
            payload["loss_fn_b_state_dict"] = loss_fn_b.state_dict()
    if grokking_detector is not None:
        payload["grokking_state"] = {
            "history": {k: list(v) for k, v in grokking_detector.history.items()},
            "plateau_start": grokking_detector.plateau_start,
            "events": [e.__dict__ for e in grokking_detector.events],
        }
    return payload


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train(
    config: Dict[str, Any],
    device: torch.device,
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    seed: int,
    log_dir: Path,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Main training loop.

    Args:
        config: Full configuration dictionary
        device: Training device
        model: Initialized model
        train_ds: Training dataset
        val_ds: Validation dataset
        seed: Random seed
        log_dir: Directory for logs and checkpoints
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dict with training results and metrics
    """
    # Extract config sections
    train_cfg = config.get("training", {})
    statenet_cfg = config.get("statenet", {})
    loss_cfg = config.get("loss", {})
    riemannian_cfg = config.get("riemannian", {})
    option_c_cfg = config.get("option_c", {})
    memory_cfg = config.get("memory", {})

    # Resolve valuation function early — used by sampler, losses, and metrics
    valuation_type = config.get("data", {}).get("valuation_type", "index")
    valuation_fn = get_valuation_fn(valuation_type)

    # Hyperparameters
    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 512)
    base_lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    eval_every = train_cfg.get("eval_every", 5)
    save_every = train_cfg.get("save_every", 25)
    print_every = train_cfg.get("print_every", 5)

    # Worker init function for reproducible multi-worker data loading
    def worker_init_fn(worker_id: int) -> None:
        """Seed each DataLoader worker deterministically."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Create data loaders with reproducible seeding
    num_workers = train_cfg.get("num_workers", 4)

    # Stratified valuation sampler: up-weights high-valuation samples (v=9 has
    # only 1/19683 natural frequency; without stratification these levels are
    # nearly absent from every batch, starving the hierarchy loss of rare examples).
    #
    # Weight formula: 1 / count^0.5  (square-root inverse frequency)
    # Rationale: 3-adic level counts follow a geometric series (count_v ≈ 3^(9-v)).
    # Full inverse (1/count = 3^v) gives ratio v=9:v=0 of 13122x — v=9 appears
    # ~1848x/epoch from 1 unique sample, causing severe overfitting.
    # log(count) collapses to linear-in-v for geometric series, giving only ~13x
    # ratio — v=9 still nearly absent per batch.
    # sqrt(1/count) = 3^(v/2) preserves geometric structure at half the exponent:
    # ratio ~114x, ~50 samples/level/batch, v=0 each seen ~0.64x/epoch (not 0.15x).
    train_indices = train_ds.tensors[1]  # second element is the index tensor
    train_valuations = valuation_fn(train_indices)
    level_counts = torch.bincount(train_valuations, minlength=TERNARY.MAX_VALUATION + 1)
    # Vectorized: compute per-level weight then index into it (no Python loop)
    level_weights = torch.zeros(TERNARY.MAX_VALUATION + 1, dtype=torch.float64)
    nonzero = level_counts > 0
    level_weights[nonzero] = 1.0 / level_counts[nonzero].to(torch.float64).sqrt()
    sample_weights = level_weights[train_valuations.long()]
    # Warn if highest valuation level is absent from train split (landed in val)
    absent = (~nonzero).nonzero(as_tuple=False).squeeze(-1).tolist()
    if absent:
        print(f"  [WARN] Stratified sampler: valuation levels {absent} absent from "
              f"train split (in val set). Those levels will not appear in batches.")
    stratified_generator = torch.Generator()
    stratified_generator.manual_seed(seed)
    stratified_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=stratified_generator,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=stratified_sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Loss function
    curvature = config.get("model", {}).get("curvature", 1.0)
    if valuation_type != "index":
        print(f"  [Hierarchy] valuation_type={valuation_type!r} (content-based Option B)")
    loss_fn = CombinedLoss(loss_cfg, curvature=curvature, device=device,
                           valuation_type=valuation_type)
    print(f"  Loss functions: {loss_fn.get_enabled_losses()}")

    # VAE-B loss: hierarchy-only (no coverage/reconstruction).
    # VAE-B's z_B_hyp is shaped for hierarchy; applying coverage loss creates a
    # gradient conflict: decoder_B must simultaneously reconstruct AND encode hierarchy.
    # Fix: zero out coverage in rich_hierarchy for VAE-B.
    loss_cfg_b = {k: dict(v) if isinstance(v, dict) else v for k, v in loss_cfg.items()}
    if "rich_hierarchy" in loss_cfg_b and isinstance(loss_cfg_b["rich_hierarchy"], dict):
        loss_cfg_b["rich_hierarchy"] = dict(loss_cfg_b["rich_hierarchy"])
        loss_cfg_b["rich_hierarchy"]["coverage_weight"] = 0.0
    loss_fn_b = CombinedLoss(loss_cfg_b, curvature=curvature, device=device,
                             valuation_type=valuation_type)
    print(f"  Loss functions (VAE-B, no coverage): {loss_fn_b.get_enabled_losses()}")

    # Training controller (LR=0 for frozen components)
    lr_controller = None

    if statenet_cfg.get("enabled", True):
        sn_config = StateNetConfig.from_dict(statenet_cfg)

        # Merge option_c LR scales if present — go through LRScales constructor
        # to preserve __post_init__ validation (direct field assignment bypasses it)
        if option_c_cfg.get("enabled", True):
            from src.config.statenet_config import LRScales
            sn_config.lr_scales = LRScales(
                encoder_a=option_c_cfg.get("encoder_a_lr_scale", sn_config.lr_scales.encoder_a),
                encoder_b=option_c_cfg.get("encoder_b_lr_scale", sn_config.lr_scales.encoder_b),
                projections=option_c_cfg.get("projections_lr_scale", sn_config.lr_scales.projections),
                decoders=sn_config.lr_scales.decoders,
            )

        lr_controller = MetricBasedLR(sn_config)
        print(f"  LRController: MetricBasedLR ({sn_config.summary()})")
    else:
        print("  Training controller: Disabled")

    # Optimizer
    # Include loss_fn parameters when learnable weights are enabled
    loss_params = (
        list(loss_fn.parameters()) + list(loss_fn_b.parameters())
        if loss_cfg.get("learnable_weights", False) else []
    )
    if loss_params:
        print(f"  Learnable loss weights: {len(loss_params)} parameters")

    if riemannian_cfg.get("enabled", False):
        param_groups = model.get_param_groups(base_lr)
        if loss_params:
            param_groups.append(
                {"params": loss_params, "lr": base_lr, "name": "loss_weights"}
            )
        stabilize = riemannian_cfg.get(
            "stabilize", 10
        )  # Re-project to manifold every N steps
        optimizer = get_riemannian_optimizer(
            param_groups,
            lr=base_lr,
            stabilize=stabilize,
            weight_decay=weight_decay,
        )
        print(f"  Optimizer: RiemannianAdam (stabilize={stabilize}, wd={weight_decay})")
    else:
        param_groups = model.get_param_groups(base_lr)
        if loss_params:
            param_groups.append(
                {"params": loss_params, "lr": base_lr, "name": "loss_weights"}
            )
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )
        print("  Optimizer: AdamW")

    # Scheduler
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "cosine")
    if scheduler_type == "cosine_warmup_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get("T_0", 25),
            T_mult=scheduler_cfg.get("T_mult", 2),
        )
    elif scheduler_type == "multi_phase_cosine":
        # Multi-phase cosine: use first phase's T_0/T_mult for CosineAnnealingWarmRestarts,
        # and apply phase-specific LR scaling via a LambdaLR wrapper.
        phases = scheduler_cfg.get("phases", [])
        if phases:
            # Extract T_0/T_mult from first phase (or top-level fallback)
            first_phase = phases[0]
            T_0 = first_phase.get("T_0", scheduler_cfg.get("T_0", 25))
            T_mult = first_phase.get("T_mult", scheduler_cfg.get("T_mult", 2))
            # Build epoch → lr_scale mapping from phases
            phase_schedule = []
            for p in phases:
                epoch_range = p.get("epoch_range", [0, epochs])
                scale = p.get("base_lr_scale", 1.0)
                phase_schedule.append((epoch_range[0], epoch_range[1], scale))
            # Sort by start epoch
            phase_schedule.sort(key=lambda x: x[0])

            def multi_phase_lr_lambda(epoch):
                """Return LR multiplier based on current phase."""
                current_scale = phase_schedule[-1][2]  # default to last phase
                for start, end, scale in phase_schedule:
                    if start <= epoch < end:
                        current_scale = scale
                        break
                return current_scale

            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
            )
            phase_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=multi_phase_lr_lambda,
            )
            # Use ChainedScheduler to compose cosine restarts with phase scaling
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [base_scheduler, phase_scheduler]
            )
            print(f"  Scheduler: multi_phase_cosine ({len(phases)} phases)")
        else:
            # No phases defined — fall back to plain cosine
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            print("  Scheduler: cosine (multi_phase_cosine with no phases)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # TensorBoard logger (uses TensorBoardLogger from utils)
    logging_cfg = config.get("logging", {})
    run_name = log_dir.name
    tb_logger = TensorBoardLogger(
        tensorboard_dir=str(log_dir),
        experiment_name=run_name,
        log_callback=lambda msg: (
            print(f"  {msg}") if logging_cfg.get("verbose", False) else None
        ),
    )
    # Safety: flush + close TB writer on crash or unclean exit
    atexit.register(tb_logger.close)

    if not tb_logger.is_available:
        print("  [WARN] TensorBoard is NOT available — metrics will NOT be logged.")
        print("         Install with: pip install tensorboard>=2.13.0")

    # Register custom dashboard layout (called once; defines multi-chart views in TB UI).
    # Uses add_custom_scalars — no phantom sub-run dirs, just layout metadata.
    if tb_logger.is_available:
        _layout = {
            "Q Metric": {
                "Q (VAE-A vs VAE-B)": ["Multiline", ["Hierarchy/Q_VAE_A", "Hierarchy/Q_VAE_B"]],
                "dist_corr vs hierarchy": ["Multiline", ["Hierarchy/dist_corr", "Hierarchy/corr_VAE_A", "Hierarchy/corr_VAE_B"]],
            },
            "Direction Geometry": {
                "ARI per level (v=0..5)": ["Multiline", [f"Direction/ARI_v{v}" for v in range(6)]],
                "ARI composite vs prefix3": ["Multiline", ["Direction/ARI_composite", "Direction/ARI_prefix3"]],
                "intra vs inter sim": ["Multiline", ["Direction/intra_level_sim", "Direction/inter_level_sim"]],
            },
            "Radial Hierarchy": {
                "r_v per level (A)": ["Multiline", [f"Radius/r_v{v}_A" for v in range(10)]],
                "std per level (A)": ["Multiline", [f"Radius/std_v{v}_A" for v in range(10)]],
                "level gaps (A)": ["Multiline", [f"Geometry/level_gap_v{v}" for v in range(9)]],
                "mean radius A vs B": ["Multiline", ["Radius/mean_VAE_A", "Radius/mean_VAE_B"]],
            },
            "Lagrangian": {
                "margin λ per level": ["Multiline", [f"Lagrangian/margin_v{v}" for v in range(9)]],
                "scatter λ per level": ["Multiline", [f"Lagrangian/scatter_v{v}" for v in range(10)]],
                "n_active λ": ["Multiline", ["Lagrangian/n_active"]],
            },
            "Topology": {
                "Betti Numbers": ["Multiline", ["Topology/betti_0", "Topology/betti_1"]],
                "Persistence Entropy": ["Multiline", ["Topology/persistence_entropy"]],
                "Betti Ratio (H1/H0)": ["Multiline", ["Topology/betti_ratio"]],
            },
        }
        tb_logger.writer.add_custom_scalars(_layout)
        # Log full config YAML as text for run reproducibility.
        tb_logger.writer.add_text("Config/yaml", f"```yaml\n{yaml.dump(config, default_flow_style=False)}```", 0)

    # Phase 2 Visualization Pipeline — hyperbolic geometry projections + topology
    vis_cfg = config.get("visualization", {})
    vis_pipeline = VisualizationPipeline(
        config=vis_cfg,
        writer=tb_logger.writer,
        log_callback=lambda msg: print(f"  {msg}"),
    )

    # Hardware monitor for memory tracking
    hw_monitor = HardwareMonitor(device, warn_threshold=0.9)

    # Config for enhanced logging
    histogram_every = logging_cfg.get("histogram_every", 10)

    enhanced_cfg = logging_cfg.get("enhanced_metrics", {})
    log_batch_metrics = enhanced_cfg.get("enabled", False)
    log_gradients = enhanced_cfg.get("log_gradients", False)

    # Memory management
    empty_cache_freq = memory_cfg.get("empty_cache_freq", 25)

    # Checkpoints directory
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Grokking detector — map YAML keys to constructor params
    grok_cfg = train_cfg.get("grokking_detection", {})
    if grok_cfg:
        # Map legacy/YAML key names to GrokkingDetector constructor params
        _grok_key_map = {
            "monitor_window": "window",
            "plateau_threshold": "slope_eps",
            "plateau_patience": "sustain_k",
            "accuracy_jump_threshold": "val_lift_min",
        }
        # Known constructor params (pass through directly)
        _grok_known = {
            "window",
            "slope_eps",
            "sustain_k",
            "val_lift_min",
            "gap_collapse_min",
        }
        grok_params = {}
        for k, v in grok_cfg.items():
            if k in ("enabled",):  # skip meta-keys
                continue
            mapped = _grok_key_map.get(k, k)
            if mapped in _grok_known:
                grok_params[mapped] = v
            # else: silently ignore unknown keys (gradient_norm_track, representation_analysis, etc.)
        grokking_detector = GrokkingDetector(**grok_params)
    else:
        grokking_detector = GrokkingDetector()

    # Lagrangian dual adaptive weighting (Phase 3B)
    # Plain Python object: NOT in optimizer.parameters()
    lagrangian_cfg = config.get('loss', {}).get('lagrangian', {})
    if lagrangian_cfg.get('enabled', False):
        dual_state: Optional[LagrangianDualState] = LagrangianDualState.from_config(lagrangian_cfg)
        print(f"  [Lagrangian] Dual weighting enabled: lr={dual_state.lr}, "
              f"max_lambda={dual_state.max_lambda}, warmup={dual_state.warmup_epochs} epochs")
    else:
        dual_state = None

    # Per-epoch violation accumulators (float only, for dual update)
    dual_violation_acc: Dict[str, float] = {}
    dual_violation_count: int = 0

    # Current dual weights (updated at eval cadence, applied every training step)
    current_dual_weights = None  # None = no Lagrangian penalties

    # Training state
    best_Q = -1.0
    best_hierarchy = 0.0
    best_coverage = 0.0
    start_epoch = 0  # overwritten when resume_checkpoint is used
    results = {
        "epochs_trained": 0,
        "best_Q": 0.0,
        "best_hierarchy": 0.0,
        "best_coverage": 0.0,
        "grokking_events": [],
    }

    # -------------------------------------------------------------------------
    # Resume checkpoint — full exact-state restore for interrupted runs.
    #
    # Distinct from anchor_checkpoint (transfer learning, model weights only,
    # strict=False, fresh optimizer). resume_checkpoint restores everything:
    # model (strict=True), optimizer, scheduler, loss_fn, lagrangian, and
    # lr_controller history, then resumes from the next epoch.
    # -------------------------------------------------------------------------
    resume_cfg = config.get("resume_checkpoint", {})
    resume_path_str = resume_cfg.get("path") if isinstance(resume_cfg, dict) else resume_cfg
    if resume_path_str and str(resume_path_str) not in ("null", ""):
        resume_path = PROJECT_ROOT / resume_path_str
        if not resume_path.exists():
            raise FileNotFoundError(f"[resume] Checkpoint not found: {resume_path}")
        print(f"\n  [Resume] Restoring from: {resume_path.name}")
        ckpt = load_checkpoint_compat(resume_path, map_location=device)

        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"[resume] State dict mismatch — missing: {missing}, unexpected: {unexpected}. "
                "Architecture changed. Use anchor_checkpoint instead."
            )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if loss_cfg.get("learnable_weights", False):
            if "loss_fn_state_dict" in ckpt:
                loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
                loss_fn_b.load_state_dict(ckpt["loss_fn_b_state_dict"])
                print("  [Resume] Learnable loss weights restored.")
            else:
                print(
                    "  [Resume][WARN] learnable_weights=true but checkpoint has no "
                    "loss_fn_state_dict — log-sigma params reset to initial values."
                )

        if dual_state is not None and "lagrangian_state" in ckpt:
            dual_state.load_state_dict(ckpt["lagrangian_state"])
            print("  [Resume] Lagrangian dual state restored.")

        if lr_controller is not None and "controller_state" in ckpt:
            cs = ckpt["controller_state"]
            # Restore active states and metadata
            lr_controller._active.update(cs.get("active", {}))
            lr_controller._best_q = cs.get("best_q", -1.0)
            lr_controller._last_epoch = cs.get("last_epoch", start_epoch - 1)
            
            # Restore hysteresis timers
            if "last_change" in cs:
                lr_controller._last_change.update(cs["last_change"])
            
            # Restore plateau and stall counters
            lr_controller._hierarchy_b_plateau_count = cs.get("hierarchy_b_plateau_count", 0)
            lr_controller._grad_low_count = cs.get("grad_low_count", 0)
            lr_controller._hierarchy_a_stall_count = cs.get("hierarchy_a_stall_count", 0)

            # Restore rolling histories (deques)
            history_keys = {
                "coverage_history": "_coverage_history",
                "hierarchy_a_history": "_hierarchy_a_history",
                "hierarchy_b_history": "_hierarchy_b_history",
                "grad_norm_history": "_grad_norm_history",
                "q_history": "_q_history"
            }
            for ckpt_key, attr_name in history_keys.items():
                if ckpt_key in cs:
                    history_list = cs[ckpt_key]
                    target_deque = getattr(lr_controller, attr_name)
                    setattr(lr_controller, attr_name, deque(history_list, maxlen=target_deque.maxlen))
            
            print("  [Resume] LR controller state restored (including histories and counters).")

        if "grokking_state" in ckpt and grokking_detector is not None:
            gs = ckpt["grokking_state"]
            grokking_detector.history = gs.get("history", grokking_detector.history)
            grokking_detector.plateau_start = gs.get("plateau_start")
            # Restore events if any (re-wrap in NamedTuple if needed, or just list of dicts)
            saved_events = gs.get("events", [])
            grokking_detector.events = [GrokkingEvent(**e) for e in saved_events]
            print("  [Resume] Grokking detector state restored.")

        best_Q = float(ckpt["Q"]) if ckpt.get("Q") is not None else float(ckpt.get("best_Q", -1.0))
        best_hierarchy = float(ckpt["hierarchy_A"]) if ckpt.get("hierarchy_A") is not None else float(ckpt.get("best_hierarchy", 0.0))
        best_coverage = float(ckpt["coverage"]) if ckpt.get("coverage") is not None else float(ckpt.get("best_coverage", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  [Resume] Resuming from epoch {start_epoch} / {epochs}.")
        if start_epoch >= epochs:
            print(f"  [Resume][WARN] start_epoch={start_epoch} >= epochs={epochs}. "
                  "Nothing to train — increase training.epochs in config.")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: epochs {start_epoch}-{epochs}, batch_size={batch_size}, lr={base_lr}")
    if TQDM_AVAILABLE:
        print("  Progress: tqdm enabled")
    print(f"{'=' * 60}\n")

    # Global batch counter for TensorBoard
    global_step = 0

    # Epoch iterator (with or without tqdm)
    epoch_iter = (
        tqdm(range(start_epoch, epochs), desc="Training", unit="epoch")
        if TQDM_AVAILABLE
        else range(start_epoch, epochs)
    )

    for epoch in epoch_iter:
        t0 = time.time()
        model.train()

        # Reset per-epoch Lagrangian violation accumulators
        if dual_state is not None:
            dual_violation_acc = {}
            dual_violation_count = 0
            dual_state.step_epoch(epoch)

        # Training epoch
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n_batches = 0
        ac_loss_per_level_sum: dict = {}  # v -> accumulated loss sum
        r_std_per_level_sum: dict = {}   # v -> accumulated per-level radius std sum

        # Batch iterator (with or without tqdm)
        if TQDM_AVAILABLE:
            batch_iter = tqdm(
                train_loader, desc=f"Ep {epoch:03d}", leave=False, unit="batch"
            )
        else:
            batch_iter = train_loader

        for batch_ops, batch_idx in batch_iter:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            optimizer.zero_grad()

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(batch_ops, decode_b=False)
                    z_A_hyp = out["z_A_hyp"]
                    z_B_hyp = out["z_B_hyp"]
                    logits_A = out.get("logits_A", out.get("logits"))
                    logits_B = out.get("logits_B")  # None when decode_b=False

                    # Get current curvature for ValuationPriorLoss (learnable c)
                    current_curvature = model.projections.get_curvature()

                    # VAE-A: full loss (coverage + hierarchy)
                    losses = loss_fn(
                        z_A_hyp, batch_idx, logits_A, batch_ops, epoch=epoch,
                        mu=out.get("mu_A"), logvar=out.get("logvar_A"),
                        curvature=current_curvature,
                        dual_weights=current_dual_weights,
                        r=out.get("r_A"),
                        model=model,
                    )
                    # VAE-B: hierarchy-only (no coverage/reconstruction).
                    # coverage_weight=0.0 in loss_fn_b prevents gradient conflict
                    # between reconstruction objective and hierarchy shaping.
                    losses_B = loss_fn_b(
                        z_B_hyp, batch_idx, logits_B, batch_ops, epoch=epoch,
                        mu=out.get("mu_B"), logvar=out.get("logvar_B"),
                        curvature=current_curvature,
                        dual_weights=current_dual_weights,
                        r=out.get("r_B"),
                        model=model,
                    )
                    # Combine: both VAEs contribute to total loss
                    loss = losses["total"] + losses_B["total"]

                    # Accumulate per-level violation floats for Lagrangian dual update.
                    # Only floats — tensors are already consumed in the backward pass.
                    if dual_state is not None:
                        for src_losses in (losses, losses_B):
                            for m_key, pfx_map in (
                                ('monotonic_metrics', [('gap_viol_v', 9)]),
                                ('rank_metrics', [('scatter_v', 10)]),
                                ('valuation_prior_metrics', [('vp_gap_v', 10)]),
                            ):
                                m = src_losses.get(m_key, {})
                                for pfx, n in pfx_map:
                                    for v in range(n):
                                        k = f'{pfx}{v}'
                                        val = m.get(k)
                                        if isinstance(val, (int, float)) and val > 0.0:
                                            dual_violation_acc[k] = dual_violation_acc.get(k, 0.0) + val
                        # Count each VAE branch as a separate "instance" to avoid double-averaging bias
                        dual_violation_count += 2

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Clip model + loss_fn params together so learnable log_sigma
                # weights are subject to the same norm bound as model parameters.
                _params_to_clip = list(model.parameters()) + loss_params
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _params_to_clip, max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

            except torch.cuda.OutOfMemoryError:
                # OOM handling
                print(f"\n[OOM] CUDA Out of Memory at epoch {epoch}, batch {n_batches}")
                print(f"[OOM] {hw_monitor.get_oom_diagnostic(batch_size)}")

                # Clear cache
                torch.cuda.empty_cache()

                # Save emergency checkpoint
                emergency_path = ckpt_dir / f"emergency_oom_epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    emergency_path,
                )
                print(f"[OOM] Emergency checkpoint saved: {emergency_path}")

                # Re-raise to exit training
                raise

            train_loss_sum += loss.item()
            train_acc_sum += compute_accuracy(logits_A, batch_ops)
            n_batches += 1

            # Accumulate per-level AC loss (for TB logging at eval cadence)
            _ac_m = losses.get("angular_coherence_metrics", {})
            for _v in range(10):
                _k = f"ac_loss_v{_v}"
                if _k in _ac_m:
                    ac_loss_per_level_sum[_v] = ac_loss_per_level_sum.get(_v, 0.0) + _ac_m[_k]

            # Accumulate per-level radius std from RichHierarchyLoss (scatter refactor bonus)
            _rh_m = losses.get("rich_hierarchy_detail", {})
            for _v in range(10):
                _k = f"r_std_v{_v}"
                if _k in _rh_m:
                    r_std_per_level_sum[_v] = r_std_per_level_sum.get(_v, 0.0) + _rh_m[_k]
            global_step += 1

            # Batch-level TensorBoard logging
            if log_batch_metrics and tb_logger.is_available:
                tb_logger.log_batch(global_step, loss.item())
                if hasattr(grad_norm, "item"):
                    tb_logger.writer.add_scalar(
                        "Batch/GradNorm", grad_norm.item(), global_step
                    )

            # Update tqdm postfix with loss and memory
            if TQDM_AVAILABLE and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "gpu": hw_monitor.get_gpu_status_string()
                        if device.type == "cuda"
                        else "cpu",
                    }
                )

            # Check memory warning periodically
            if n_batches % 10 == 0:
                warning = hw_monitor.check_memory_warning()
                if warning:
                    print(f"\n{warning}")

        scheduler.step()

        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        if tb_logger.is_available:
            tb_logger.writer.add_scalar("LR/scheduled", current_lr, epoch)

        assert n_batches > 0, f"No training batches at epoch {epoch} — DataLoader may be empty"
        avg_train_loss = train_loss_sum / n_batches
        avg_train_acc = train_acc_sum / n_batches

        # Validation (every eval_every epochs)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_acc_sum = 0.0
            val_coverage_sum = 0.0
            val_hyperbolic_coverage_sum = 0.0
            val_batches = 0
            z_A_all = []
            z_B_all = []
            idx_all = []
            r_A_all = []

            with torch.no_grad():
                for batch_ops, batch_idx in val_loader:
                    batch_ops = batch_ops.to(device)
                    batch_idx = batch_idx.to(device)

                    out = model(batch_ops)
                    logits = out.get("logits_A", out.get("logits"))

                    val_acc_sum += compute_accuracy(logits, batch_ops)
                    val_coverage_sum += compute_coverage(logits, batch_ops)
                    val_hyperbolic_coverage_sum += compute_hyperbolic_coverage(out["z_A_hyp"], curvature)
                    val_batches += 1

                    # Collect both VAE embeddings for separate hierarchy computation.
                    # .detach() releases the forward-graph reference: under no_grad()
                    # grad_fn is still present on tensors, keeping all intermediate
                    # activations alive. Detaching lets them be freed after each batch.
                    z_A_all.append(out["z_A_hyp"].detach())
                    z_B_all.append(out["z_B_hyp"].detach())
                    idx_all.append(batch_idx.detach())
                    if out.get("r_A") is not None:
                        r_A_all.append(out["r_A"].detach())

            assert val_batches > 0, f"No validation batches at epoch {epoch} — DataLoader may be empty"
            avg_val_acc = val_acc_sum / val_batches
            avg_val_coverage = val_coverage_sum / val_batches

            # Hierarchy metrics for both VAEs (seed varies per epoch, but reproducible)
            z_A_cat = torch.cat(z_A_all)
            z_B_cat = torch.cat(z_B_all)
            idx_cat = torch.cat(idx_all)
            hier_metrics_A = compute_hierarchy_metrics(
                z_A_cat, idx_cat, curvature, seed=seed + epoch,
                valuation_fn=valuation_fn,
            )
            hier_metrics_B = compute_hierarchy_metrics(
                z_B_cat, idx_cat, curvature, seed=seed + epoch + 1000,
                valuation_fn=valuation_fn,
            )

            # Angular Q metric (direction geometry quality)
            aq_value = 0.0
            intra_sim = 0.0
            inter_sim = 0.0
            if r_A_all:
                r_A_cat = torch.cat(r_A_all)
                eps = 1e-10
                dir_A = z_A_cat / r_A_cat.unsqueeze(-1).clamp(min=eps)
                dir_A = dir_A / dir_A.norm(dim=-1, keepdim=True).clamp(min=eps)
                # Get valuations for all indices
                vals = valuation_fn(idx_cat)
                # Sample pairs for efficiency
                n_sample = min(500, len(idx_cat))
                perm = torch.randperm(len(idx_cat), device=device)[:n_sample]
                perm2 = torch.randperm(len(idx_cat), device=device)[:n_sample]
                same_level = (vals[perm] == vals[perm2])
                cos_sims = (dir_A[perm] * dir_A[perm2]).sum(dim=-1)
                if same_level.sum() > 0:
                    intra_sim = cos_sims[same_level].mean().item()
                if (~same_level).sum() > 0:
                    inter_sim = cos_sims[~same_level].mean().item()
                aq_value = intra_sim - inter_sim

            # Per-level ARI: K-means on direction vectors vs digit_prefix_class
            # v=0-2: deeper prefix (k=3/4/5), k_means=18 (true class count 2x3x3)
            # v=3-8: prefix k=2, k_means=min(n_classes, n_samples//3) adaptive
            # v=9: single sample, skipped
            ari_per_level = {}  # {v: float}
            ari_prefix3 = float("nan")  # backward compat (v=0 ARI)
            _r_full_cat = None  # full-dataset radii (set inside if r_A_all block)
            _vals_full = None
            if r_A_all:
                # Bug 7 fix: use full-dataset embeddings for ARI so sparse levels
                # (v=5: 54 ops) are not dropped when val set has < 6 samples per level.
                # Quick no_grad forward pass on all 19683 ops in chunks (~5 batches).
                _all_ops = TERNARY.all_ternary().to(device)
                _all_idx = torch.arange(len(_all_ops), device=device)
                _z_full, _r_full, _idx_full = [], [], []
                model.eval()
                with torch.no_grad():
                    for _s in range(0, len(_all_ops), batch_size):
                        _co = _all_ops[_s:_s + batch_size]
                        _ci = _all_idx[_s:_s + batch_size]
                        _out = model(_co)
                        _z_full.append(_out["z_A_hyp"].detach().cpu())
                        _r_full.append(_out["r_A"].detach().cpu())
                        _idx_full.append(_ci.cpu())
                _z_full_cat = torch.cat(_z_full)
                _r_full_cat = torch.cat(_r_full)
                _idx_full_cat = torch.cat(_idx_full)
                _eps = 1e-10
                _dir_full = _z_full_cat / _r_full_cat.unsqueeze(-1).clamp(min=_eps)
                _dir_full = _dir_full / _dir_full.norm(dim=-1, keepdim=True).clamp(min=_eps)
                _vals_full = valuation_fn(_idx_full_cat)

                # prefix depth per level: matches AC loss level_prefix_k
                # v=0:k=3→18cls, v=1:k=4→18cls, v=2:k=3→2cls, v=3:k=4→2cls, v=4:k=5→2cls, v=5:k=6→2cls
                level_pfx = {0: 3, 1: 4, 2: 3, 3: 4, 4: 5, 5: 6, 6: 2, 7: 2, 8: 2}
                for v, pfx_k in level_pfx.items():
                    v_mask = (_vals_full == v)
                    n_v = v_mask.sum().item()
                    if n_v < 2:  # skip only if truly too few
                        continue
                    dir_v = _dir_full[v_mask].numpy()
                    idx_v = _idx_full_cat[v_mask]
                    if n_v > 5000:
                        sub = np.random.choice(n_v, 5000, replace=False)
                        dir_v = dir_v[sub]
                        idx_v = idx_v[sub]
                    # Determine true class count and cap k_means
                    pfx = TERNARY.digit_prefix_class(idx_v, pfx_k).numpy()
                    n_classes = len(np.unique(pfx))
                    km_k = min(n_classes, max(2, n_v // 3))
                    if km_k < 2:
                        continue
                    labels = KMeans(n_clusters=km_k, n_init=3, random_state=42).fit_predict(dir_v)
                    ari_per_level[v] = adjusted_rand_score(pfx, labels)
                ari_prefix3 = ari_per_level.get(0, float("nan"))

            # Get gradient statistics from optimizer (V6.0: avoids manual grad loop)
            controller_grad_norm = None
            if lr_controller is not None:
                grad_stats = get_optimizer_grad_stats(optimizer, "projections")
                controller_grad_norm = grad_stats.get("grad_norm_estimate", 0.0)

                # Fallback to manual computation if optimizer stats unavailable
                if controller_grad_norm == 0.0 and hasattr(model, "projections"):
                    proj_grads = [
                        p.grad
                        for p in model.projections.parameters()
                        if p.grad is not None
                    ]
                    if proj_grads:
                        controller_grad_norm = sum(g.norm(2).item() for g in proj_grads)

            # Training controller update
            controller_state = None
            train_summary = "N/A"

            if lr_controller is not None:
                # V6.0: LRController - unified control via optimizer LR
                metrics = TrainingMetrics(
                    epoch=epoch,
                    coverage=avg_val_acc,  # per-digit accuracy (not perfect-sample rate)
                    hierarchy_a=hier_metrics_A["hierarchy"],
                    hierarchy_b=hier_metrics_B["hierarchy"],
                    dist_corr_a=hier_metrics_A["dist_corr"],
                    q_value=hier_metrics_A["Q"],
                    grad_norm_projections=controller_grad_norm or 0.0,
                    hierarchy_a_collapsed=hier_metrics_A.get("hierarchy_collapsed", False),
                    hierarchy_b_collapsed=hier_metrics_B.get("hierarchy_collapsed", False),
                )
                controller_state = lr_controller.update(metrics)

                # Apply LR scales to optimizer (the key Option C mechanism).
                # scheduler.get_last_lr()[0] returns the already-scaled LR for
                # encoder_a (base_lr * encoder_a_scale * cosine_factor), not the
                # raw cosine-annealed base. Recover the cosine factor by dividing
                # by the scheduler's stored base for that group, then multiply by
                # the true base_lr so controller scales are applied to the right value.
                cosine_factor = scheduler.get_last_lr()[0] / scheduler.base_lrs[0]
                current_base_lr = base_lr * cosine_factor
                new_scales = controller_state["lr_scales"]
                update_optimizer_lr_scales(optimizer, current_base_lr, new_scales)

                # Sync requires_grad with controller decisions.
                # LR=0 prevents weight updates but gradients still compute through
                # frozen components, wasting GPU memory and compute each backward pass.
                # Toggling requires_grad stops gradient computation at the source.
                prev_a = any(p.requires_grad for p in model.head_A.parameters())
                prev_b = any(p.requires_grad for p in model.head_B.parameters())
                prev_p = any(p.requires_grad for p in model.projections.parameters())
                new_a = new_scales.get('encoder_a', 0.0) > 0
                new_b = new_scales.get('encoder_b', 0.0) > 0
                new_p = new_scales.get('projections', 0.0) > 0
                model.set_encoder_a_trainable(new_a)
                model.set_encoder_b_trainable(new_b)
                model.set_projections_trainable(new_p)
                # Zero stale momentum when a component unfreezes.
                # Accumulated exp_avg/exp_avg_sq from before freeze are stale
                # and cause a momentum spike on the first post-unfreeze update.
                if (not prev_a and new_a) or (not prev_b and new_b) or (not prev_p and new_p):
                    for group in optimizer.param_groups:
                        gname = group.get('name', '')
                        if ((gname == 'encoder_a' and not prev_a and new_a) or
                            (gname == 'encoder_b' and not prev_b and new_b) or
                            (gname == 'projections' and not prev_p and new_p)):
                            for p in group['params']:
                                if p in optimizer.state:
                                    st = optimizer.state[p]
                                    if 'exp_avg' in st:
                                        st['exp_avg'].zero_()
                                    if 'exp_avg_sq' in st:
                                        st['exp_avg_sq'].zero_()

                # Format summary from LR scales
                scales = new_scales
                train_summary = f"A:{scales.get('encoder_a', 0):.2f} B:{scales.get('encoder_b', 0):.2f} P:{scales.get('projections', 0):.2f}"

            # Lagrangian dual ascent update (after LR controller, at eval cadence)
            if dual_state is not None and dual_violation_count > 0:
                avg_violations = {
                    k: v / dual_violation_count
                    for k, v in dual_violation_acc.items()
                }
                dual_state.update(avg_violations)
                current_dual_weights = dual_state.get_dual_weights()
                if dual_state.is_active():
                    print(f"  [{dual_state.summary()}]")

            # Grokking detection
            grok_state = grokking_detector.update(
                epoch, avg_train_loss, avg_train_acc, avg_val_acc
            )
            if grok_state["event"]:
                print(f"  [GROKKING] Event detected at epoch {epoch}!")
                results["grokking_events"].append(grokking_detector.events[-1].__dict__)

            # TensorBoard logging (log both VAE metrics)
            # NOTE: Uses add_scalar (not add_scalars) to avoid phantom sub-run
            # directories. TensorBoard auto-groups by "/" prefix in the UI.
            if tb_logger.is_available:
                tb_logger.writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
                tb_logger.writer.add_scalar("Accuracy/val", avg_val_acc, epoch)
                tb_logger.writer.add_scalar("Loss/train", avg_train_loss, epoch)
                tb_logger.writer.add_scalar("Coverage", avg_val_coverage, epoch)
                tb_logger.writer.add_scalar("Hierarchy/corr_VAE_A", hier_metrics_A["hierarchy"], epoch)
                tb_logger.writer.add_scalar("Hierarchy/corr_VAE_B", hier_metrics_B["hierarchy"], epoch)
                tb_logger.writer.add_scalar("Hierarchy/Q_VAE_A", hier_metrics_A["Q"], epoch)
                tb_logger.writer.add_scalar("Hierarchy/Q_VAE_B", hier_metrics_B["Q"], epoch)
                tb_logger.writer.add_scalar("Hierarchy/dist_corr", hier_metrics_A["dist_corr"], epoch)
                tb_logger.writer.add_scalar("Radius/mean_VAE_A", hier_metrics_A["mean_radius"], epoch)
                if hier_metrics_A.get("hierarchy_collapsed"):
                    tb_logger.writer.add_scalar("Diagnostics/hierarchy_collapsed", 1.0, epoch)
                if hier_metrics_A.get("dist_corr_collapsed"):
                    tb_logger.writer.add_scalar("Diagnostics/dist_corr_collapsed", 1.0, epoch)
                tb_logger.writer.add_scalar("Radius/mean_VAE_B", hier_metrics_B["mean_radius"], epoch)

                # Per-level mean radii — full dataset (from full-dataset pass done for ARI)
                # Falls back to val-set norms if ARI pass wasn't triggered.
                if r_A_all and _r_full_cat is not None:
                    _r_norms_full = _r_full_cat  # r_A IS the Poincaré radius
                    for _v in range(10):
                        _mask = (_vals_full == _v)
                        if _mask.sum() >= 2:
                            tb_logger.writer.add_scalar(
                                f"Radius/r_v{_v}_A", _r_norms_full[_mask].mean().item(), epoch
                            )
                else:
                    with torch.no_grad():
                        r_norms = z_A_cat.norm(dim=-1)
                        vals_all = valuation_fn(idx_cat)
                        for _v in range(10):
                            _mask = (vals_all == _v)
                            if _mask.sum() >= 2:
                                tb_logger.writer.add_scalar(
                                    f"Radius/r_v{_v}_A", r_norms[_mask].mean().item(), epoch
                                )

                # Per-level radius std (from RichHierarchyLoss scatter refactor — batch-averaged)
                if r_std_per_level_sum and n_batches > 0:
                    for _v, _sum in r_std_per_level_sum.items():
                        tb_logger.writer.add_scalar(
                            f"Radius/std_v{_v}_A", _sum / n_batches, epoch
                        )

                # Per-level gaps between adjacent mean radii (from full-dataset pass)
                if r_A_all and _r_full_cat is not None:
                    _gaps = {}
                    for _v in range(9):
                        _m0 = ((_vals_full == _v).sum() >= 2)
                        _m1 = ((_vals_full == _v + 1).sum() >= 2)
                        if _m0 and _m1:
                            _g = _r_full_cat[_vals_full == _v].mean() - _r_full_cat[_vals_full == _v + 1].mean()
                            tb_logger.writer.add_scalar(f"Geometry/level_gap_v{_v}", _g.item(), epoch)

                # Angular Q metric (direction geometry)
                if r_A_all:
                    tb_logger.writer.add_scalar("Direction/AQ", aq_value, epoch)
                    tb_logger.writer.add_scalar("Direction/intra_level_sim", intra_sim, epoch)
                    tb_logger.writer.add_scalar("Direction/inter_level_sim", inter_sim, epoch)
                    if not np.isnan(ari_prefix3):
                        tb_logger.writer.add_scalar("Direction/ARI_prefix3", ari_prefix3, epoch)
                    for v, ari_v in ari_per_level.items():
                        tb_logger.writer.add_scalar(f"Direction/ARI_v{v}", ari_v, epoch)
                    if len(ari_per_level) >= 2:
                        # Composite: weighted by data fraction
                        # v=0:66.7%, v=1:22.2%, v=2:7.4%, v=3:2.5%, v=4+:<1%
                        w = {0: 0.60, 1: 0.20, 2: 0.10, 3: 0.05, 4: 0.02, 5: 0.01, 6: 0.01, 7: 0.005, 8: 0.005}
                        ari_composite = sum(w.get(v, 0) * a for v, a in ari_per_level.items())
                        tb_logger.writer.add_scalar("Direction/ARI_composite", ari_composite, epoch)

                # Per-level AC loss (Bug 5 fix: per-level breakdown, epoch-averaged from train batches)
                if ac_loss_per_level_sum and n_batches > 0:
                    for _v, _sum in ac_loss_per_level_sum.items():
                        tb_logger.writer.add_scalar(
                            f"Direction/AC_loss_v{_v}", _sum / n_batches, epoch
                        )

                # Tree coherence (lower = better tree structure)
                tb_logger.writer.add_scalar("TreeCoherence/VAE_A", hier_metrics_A["tree_coherence"], epoch)
                tb_logger.writer.add_scalar("TreeCoherence/VAE_B", hier_metrics_B["tree_coherence"], epoch)

                # Per-level hierarchy (log worst and mean)
                tb_logger.writer.add_scalar(
                    "LevelHierarchy/worst_level_A", hier_metrics_A["worst_level"], epoch
                )
                tb_logger.writer.add_scalar(
                    "LevelHierarchy/mean_A",
                    hier_metrics_A["mean_level_hierarchy"],
                    epoch,
                )

                # Training controller metrics
                if controller_state is not None:
                    # V6.0: Log LR scales (continuous, 0.0 = frozen)
                    scales = controller_state.get("lr_scales", {})
                    tb_logger.writer.add_scalar(
                        "LRController/encoder_a_lr_scale",
                        scales.get("encoder_a", 0),
                        epoch,
                    )
                    tb_logger.writer.add_scalar(
                        "LRController/encoder_b_lr_scale",
                        scales.get("encoder_b", 0),
                        epoch,
                    )
                    tb_logger.writer.add_scalar(
                        "LRController/projections_lr_scale",
                        scales.get("projections", 0),
                        epoch,
                    )
                    tb_logger.writer.add_scalar(
                        "LRController/best_Q", controller_state.get("best_q", 0), epoch
                    )

                    # Log events if any
                    events = controller_state.get("events", [])
                    if events and events != ["warmup"]:
                        print(f"    LRController events: {events}")

                    # Grad norm
                    if controller_grad_norm is not None:
                        tb_logger.writer.add_scalar(
                            "Controller/grad_norm", controller_grad_norm, epoch
                        )

                # Hardware metrics
                if device.type == "cuda":
                    gpu_mem = hw_monitor.get_gpu_memory_gb()
                    tb_logger.writer.add_scalar(
                        "Hardware/GPU_allocated_GB", gpu_mem["allocated"], epoch
                    )
                    tb_logger.writer.add_scalar(
                        "Hardware/GPU_peak_GB", gpu_mem["peak"], epoch
                    )

                # Lagrangian dual variable values (λ per constraint level).
                # Logs whether constraints are violated (λ growing) or satisfied
                # (λ at 0). valuation_prior is disabled → lambda_prior stays 0;
                # logged anyway so future activation is immediately visible.
                if dual_state is not None:
                    dw = dual_state.get_dual_weights()
                    for v, lam in enumerate(dw.get("lambda_margin", [])):
                        tb_logger.writer.add_scalar(
                            f"Lagrangian/margin_v{v}", lam, epoch
                        )
                    for v, lam in enumerate(dw.get("lambda_scatter", [])):
                        tb_logger.writer.add_scalar(
                            f"Lagrangian/scatter_v{v}", lam, epoch
                        )
                    all_lams = (
                        dw.get("lambda_margin", [])
                        + dw.get("lambda_scatter", [])
                        + dw.get("lambda_prior", [])
                    )
                    tb_logger.writer.add_scalar(
                        "Lagrangian/n_active",
                        sum(1 for x in all_lams if x > 0),
                        epoch,
                    )

                # Phase 2: Visualization pipeline (UMAP, PaCMAP, TriMAP, topology)
                # Uses z_A_cat (Poincaré ball) + idx_cat (for valuations).
                # All computation is CPU/numpy; runs under no_grad inside pipeline.
                _vis_vals = valuation_fn(idx_cat.cpu())
                if epoch > 0:
                    vis_pipeline.run(epoch, z_A_cat.cpu(), _vis_vals, idx_cat.cpu())

                # Flush for real-time updates
                tb_logger.flush()

            # Weight histograms (periodic)
            if histogram_every > 0 and epoch % histogram_every == 0 and epoch > 0:
                if tb_logger.is_available:
                    tb_logger.log_histograms(epoch, model)

                    # Gradient norms per layer (if enabled)
                    if log_gradients:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                tb_logger.writer.add_scalar(
                                    f"Gradients/{name}",
                                    param.grad.norm(2).item(),
                                    epoch,
                                )

            # Track best metrics (use VAE-A as primary)
            if hier_metrics_A["Q"] > best_Q:
                best_Q = hier_metrics_A["Q"]
                torch.save(
                    _build_checkpoint_payload(
                        epoch, model, optimizer, scheduler,
                        lr_controller=lr_controller, dual_state=dual_state,
                        loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
                        extra={
                            "Q": best_Q,
                            "hierarchy_A": hier_metrics_A["hierarchy"],
                            "hierarchy_B": hier_metrics_B["hierarchy"],
                            "coverage": avg_val_coverage,
                        },
                    ),
                    ckpt_dir / "best_Q.pt",
                )

            if avg_val_coverage > best_coverage:
                best_coverage = avg_val_coverage

            if hier_metrics_A["hierarchy"] > best_hierarchy:  # Higher is better (negated Spearman)
                best_hierarchy = hier_metrics_A["hierarchy"]

            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                dt = time.time() - t0
                # Main metrics line
                if (
                    not TQDM_AVAILABLE
                ):  # Only print if not using tqdm (tqdm handles its own output)
                    print(
                        f"Ep {epoch:03d} | "
                        f"Loss {avg_train_loss:.4f} | "
                        f"Acc T/V {avg_train_acc:.3f}/{avg_val_acc:.3f} | "
                        f"Cov {avg_val_coverage:.3f} | "
                        f"Hier A/B {hier_metrics_A['hierarchy']:.3f}/{hier_metrics_B['hierarchy']:.3f} | "
                        f"Q {hier_metrics_A['Q']:.3f} | "
                        f"State {train_summary} | "
                        f"{dt:.1f}s"
                    )
                else:
                    # Update tqdm with epoch stats
                    epoch_iter.set_postfix(
                        {
                            "loss": f"{avg_train_loss:.4f}",
                            "cov": f"{avg_val_coverage:.3f}",
                            "Q": f"{hier_metrics_A['Q']:.2f}",
                        }
                    )
                    # Print hardware stats on separate line
                    tqdm.write(
                        f"  Ep {epoch:03d} | Cov {avg_val_coverage:.3f} | "
                        f"Hier A/B {hier_metrics_A['hierarchy']:.3f}/{hier_metrics_B['hierarchy']:.3f} | "
                        f"Q {hier_metrics_A['Q']:.3f} | {hw_monitor.get_peak_status_string()} | {dt:.1f}s"
                    )

        # Periodic checkpoint
        if epoch % save_every == 0 and epoch > 0:
            torch.save(
                _build_checkpoint_payload(
                    epoch, model, optimizer, scheduler,
                    lr_controller=lr_controller, dual_state=dual_state,
                    loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
                ),
                ckpt_dir / f"epoch_{epoch}.pt",
            )

        # Periodic memory cleanup
        if device.type == "cuda" and epoch % empty_cache_freq == 0 and epoch > 0:
            torch.cuda.empty_cache()

    # Final checkpoint
    torch.save(
        _build_checkpoint_payload(
            epochs, model, optimizer, scheduler,
            lr_controller=lr_controller, dual_state=dual_state,
            loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
            extra={
                "best_Q": best_Q,
                "best_hierarchy": best_hierarchy,
                "best_coverage": best_coverage,
            },
        ),
        ckpt_dir / "final.pt",
    )

    # Log hparams + final metrics together (run_name='.' avoids phantom sub-run dirs).
    if tb_logger.is_available:
        _mc = config.get("model", {})
        _tc = config.get("training", {})
        _hparam_dict = {
            "latent_dim": int(_mc.get("latent_dim", 0)),
            "hidden_dim": int(_mc.get("hidden_dim", 0)),
            "radial_dims": int(_mc.get("radial_dims", 0)),
            "epochs": int(epochs),
            "batch_size": int(_tc.get("batch_size", 0)),
            "lr": float(_tc.get("lr", 0)),
            "ac_weight": float(loss_cfg.get("angular_coherence", {}).get("weight", 0)),
            "hierarchy_weight": float(loss_cfg.get("rich_hierarchy", {}).get("hierarchy_weight", 0)),
            "n_pairs_ac": int(loss_cfg.get("angular_coherence", {}).get("n_pairs", 0)),
        }
        _metric_dict = {
            "hparam/best_Q": best_Q,
            "hparam/best_hierarchy": best_hierarchy,
            "hparam/best_coverage": best_coverage,
        }
        tb_logger.writer.add_hparams(_hparam_dict, _metric_dict, run_name=".")

    # Close TensorBoard logger
    tb_logger.close()

    results.update(
        {
            "epochs_trained": epochs,
            "best_Q": best_Q,
            "best_hierarchy": best_hierarchy,
            "best_coverage": best_coverage,
        }
    )

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified p-adic VAE Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config and data, do not train",
    )
    parser.add_argument(
        "--force", action="store_true", help="Continue even if validation fails"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom run name (default: config name + timestamp)",
    )
    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # Set determinism
    set_determinism(args.seed)
    print(f"\n{'=' * 60}")
    print("  P-ADIC VAE UNIFIED TRAINING")
    print(f"{'=' * 60}")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}")
    config_path = Path(args.config)
    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Validate config against Pydantic schema
    try:
        config = normalize_config(raw_config)
        print("  Schema validation: PASSED")
    except Exception as e:
        print(f"[ERROR] Schema validation FAILED: {e}")
        if not args.force:
            sys.exit(1)
        print("  Continuing anyway due to --force flag")
        config = raw_config

    # Override device settings from YAML if not specified on command line
    device_cfg = config.get("device", {})
    if args.device == "cuda" and device_cfg.get("cuda_device") is not None:
        cuda_device = device_cfg.get("cuda_device", 0)
        device = torch.device(f"cuda:{cuda_device}")
        print(f"  Using CUDA device {cuda_device} from config")

    # Use AMP from config if not specified on command line
    if not args.amp and device_cfg.get("use_amp", False):
        args.amp = True
        print("  Using AMP from config")

    # Memory optimization settings
    memory_cfg = config.get("memory", {})
    if memory_cfg.get("cudnn_benchmark", True) and device.type == "cuda":
        if not torch.backends.cudnn.deterministic:
            torch.backends.cudnn.benchmark = True
            print("  cuDNN benchmark: enabled")
        else:
            print("  cuDNN benchmark: skipped (deterministic mode active)")

    # Create run directory
    config_name = config_path.stem
    run_name = args.name or f"{config_name}_{get_timestamp()}"
    log_dir = RUNS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Run directory: {log_dir}")

    # Data audit
    data_auditor = DataAuditor(args.seed)
    val_frac = config.get("training", {}).get("val_frac", 0.1)
    train_ds, val_ds, _ = data_auditor.prepare_data(val_frac, device)

    # Model audit
    model_auditor = ModelAuditor(config, device)
    try:
        model = model_auditor.create_and_validate_model(force=args.force)
    except Exception as e:
        print(f"\n[ERROR] Model validation failed: {e}")
        if not args.force:
            sys.exit(1)
        model = None

    if args.validate_only:
        print("\n[OK] Validation complete. Exiting (--validate-only)")
        # Save audit logs
        audit_log = {
            "data": data_auditor.audit_log,
            "model": model_auditor.audit_log,
        }
        with open(log_dir / "audit_log.json", "w") as f:
            json.dump(audit_log, f, indent=2)
        sys.exit(0)

    if model is None:
        print("[ERROR] Cannot train without valid model")
        sys.exit(1)

    # Train
    print("\n[TRAIN] Starting training...")
    results = train(
        config=config,
        device=device,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        seed=args.seed,
        log_dir=log_dir,
        use_amp=args.amp,
    )

    # Save final results
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Epochs: {results['epochs_trained']}")
    print(f"  Best Q: {results['best_Q']:.4f}")
    print(f"  Best Hierarchy: {results['best_hierarchy']:.4f}")
    print(f"  Best Coverage: {results['best_coverage']:.4f}")
    print(f"  Grokking Events: {len(results['grokking_events'])}")
    print(f"  Results: {results_path}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
