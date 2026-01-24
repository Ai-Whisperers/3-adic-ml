#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
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
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

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
from src.config.paths import RUNS_DIR, CHECKPOINTS_DIR
from src.config import StateNetConfig
from src.core import TERNARY
from src.geometry import poincare_distance, get_riemannian_optimizer
from src.losses import CombinedLoss
from src.models import (
    StateNet, compute_Q, TernaryVAEV6Controllable,
    # V6.0: LRController for unified training control
    MetricBasedLR, ScheduleBasedLR, TrainingMetrics,
    update_optimizer_lr_scales, get_optimizer_grad_stats,
)
from src.utils.checkpoint import load_checkpoint_compat, get_model_state_dict
from src.utils import TensorBoardLogger, TENSORBOARD_AVAILABLE, HardwareMonitor


# =============================================================================
# DETERMINISM
# =============================================================================

def set_determinism(seed: int, deterministic: bool = True, use_float64: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but exact)
        use_float64: If True, use float64 as default dtype (required for geoopt stability)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[TensorDataset, TensorDataset, torch.Tensor]:
        """Generate data, split, and validate.

        Args:
            val_frac: Fraction of data for validation
            device: Device for data tensors

        Returns:
            Tuple of (train_dataset, val_dataset, all_indices)
        """
        print("\n[AUDIT] Data Integrity Check...")

        # Generate all operations (uses cached LUT from TERNARY singleton)
        all_ops = TERNARY.all_ternary()
        n = len(all_ops)
        all_indices = torch.arange(n, dtype=torch.long)

        # Deterministic split
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train = all_ops[train_idx]
        X_val = all_ops[val_idx]
        idx_train = all_indices[train_idx]
        idx_val = all_indices[val_idx]

        # Leakage check
        train_set = set(tuple(x.tolist()) for x in X_train)
        val_set = set(tuple(x.tolist()) for x in X_val)
        overlap = train_set.intersection(val_set)

        self.audit_log['n_train'] = len(X_train)
        self.audit_log['n_val'] = len(X_val)
        self.audit_log['data_leakage_count'] = len(overlap)

        if len(overlap) > 0:
            print(f"  [WARN] Data leakage detected: {len(overlap)} samples in both sets")
        else:
            print(f"  [OK] No data leakage (train={len(X_train)}, val={len(X_val)})")

        # Value distribution check
        vals, counts = torch.unique(X_train, return_counts=True)
        dist = counts.float() / counts.sum()
        self.audit_log['value_distribution'] = dict(zip(vals.tolist(), dist.tolist()))
        print(f"  [OK] Value distribution: {dict(zip(vals.tolist(), [f'{d:.2%}' for d in dist.tolist()]))}")

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

        model_cfg = self.config.get('model', {})
        option_c_cfg = self.config.get('option_c', {})
        model_name = model_cfg.get('name', 'TernaryVAEV6Controllable')
        encoder_type = model_cfg.get('encoder_type', 'improved')
        decoder_type = model_cfg.get('decoder_type', 'improved')

        # LR scales from option_c config (differential learning rates per component)
        encoder_a_lr_scale = option_c_cfg.get('encoder_a_lr_scale', 0.05)
        encoder_b_lr_scale = option_c_cfg.get('encoder_b_lr_scale', 0.1)
        projections_lr_scale = option_c_cfg.get('projections_lr_scale', 1.0)

        # Initial trainability from statenet config
        statenet_cfg = self.config.get('statenet', {})
        encoder_a_trainable = statenet_cfg.get('encoder_a_trainable', False)
        encoder_b_trainable = statenet_cfg.get('encoder_b_trainable', True)
        projections_trainable = statenet_cfg.get('projections_trainable', True)

        # Instantiate model
        model = TernaryVAEV6Controllable(
            latent_dim=model_cfg.get('latent_dim', 16),
            hidden_dim=model_cfg.get('hidden_dim', 64),
            max_radius=model_cfg.get('max_radius', 0.95),
            curvature=model_cfg.get('curvature', 1.0),
            n_projection_layers=model_cfg.get('projection_layers', 2),
            projection_dropout=model_cfg.get('projection_dropout', 0.1),
            learnable_curvature=model_cfg.get('learnable_curvature', False),
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
        anchor_cfg = self.config.get('anchor_checkpoint', {})
        ckpt_path_str = anchor_cfg.get('path')

        if ckpt_path_str and ckpt_path_str != 'null':
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            if ckpt_path.exists():
                try:
                    ckpt = load_checkpoint_compat(ckpt_path, map_location=self.device)
                    state_dict = get_model_state_dict(ckpt)

                    # Load with strict=False (projections may not match)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
                    print(f"       Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

                    self.audit_log['checkpoint_loaded'] = True
                    self.audit_log['checkpoint_path'] = str(ckpt_path)

                except Exception as e:
                    print(f"  [WARN] Checkpoint load failed: {e}")
                    self.audit_log['checkpoint_loaded'] = False
                    if not force:
                        raise
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt_path}")
                self.audit_log['checkpoint_loaded'] = False
                if not force:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            print("  [INFO] No anchor checkpoint specified (training from scratch)")
            self.audit_log['checkpoint_loaded'] = False

        # Gradient flow check
        model.train()
        dummy_input = torch.randint(-1, 2, (32, 9)).double().to(self.device)

        try:
            out = model(dummy_input)
            logits = out.get('logits_A', out.get('logits'))

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

            self.audit_log['initial_grad_norm'] = total_norm
            self.audit_log['dead_params'] = dead_params

            if total_norm == 0:
                print("  [ERROR] Vanishing gradients - model is dead")
                if not force:
                    raise RuntimeError("Dead model detected")
            elif total_norm > 1000:
                print("  [WARN] Large gradient norm - may explode")
            else:
                print(f"  [OK] Gradient flow active (norm={total_norm:.4f})")

            # Clear gradients
            model.zero_grad()

        except Exception as e:
            print(f"  [ERROR] Model health check failed: {e}")
            if not force:
                raise

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


def compute_hierarchy_metrics(
    z_hyp: torch.Tensor,
    indices: torch.Tensor,
    curvature: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute hierarchy and Q metrics.

    Args:
        z_hyp: Hyperbolic embeddings (B, latent_dim)
        indices: Operation indices (B,)
        curvature: Poincaré ball curvature
        seed: Random seed for reproducible sampling

    Returns:
        Dict with hierarchy, dist_corr, Q metrics
    """
    with torch.no_grad():
        # Compute radii using hyperbolic distance
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=curvature).cpu().numpy()
        valuations = TERNARY.valuation(indices).cpu().numpy()

        # Hierarchy: Spearman correlation between valuation and radius
        # Negative correlation is good (high valuation = low radius = near origin)
        hierarchy = spearmanr(valuations, radii).correlation
        if np.isnan(hierarchy):
            hierarchy = 0.0

        # Distance correlation (sample-based for efficiency)
        n = min(1000, len(z_hyp))
        if n < 2:
            return {'hierarchy': hierarchy, 'dist_corr': 0.0, 'Q': 0.0}

        # Use explicit RNG for reproducibility
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z_hyp), n, replace=False)
        r_sample = radii[idx]
        v_sample = valuations[idx]

        # Pairwise distances in radius and valuation
        r_dists = np.abs(r_sample[:, None] - r_sample[None, :])
        v_dists = np.abs(v_sample[:, None] - v_sample[None, :])

        triu_idx = np.triu_indices(n, k=1)
        dist_corr = spearmanr(r_dists[triu_idx], v_dists[triu_idx]).correlation
        if np.isnan(dist_corr):
            dist_corr = 0.0

        Q = compute_Q(dist_corr, hierarchy)

        return {
            'hierarchy': hierarchy,
            'dist_corr': dist_corr,
            'Q': Q,
            'mean_radius': float(radii.mean()),
            'std_radius': float(radii.std()),
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

        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
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
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

        out = {'plateau': False, 'potential': False, 'event': False, 'val_lift': 0.0}

        if len(self.history['train_loss']) < self.window:
            return out

        # Plateau check
        recent_loss = self.history['train_loss'][-self.window:]
        slope = abs(self._slope(recent_loss))
        out['plateau'] = slope < self.slope_eps

        if out['plateau'] and self.plateau_start is None:
            self.plateau_start = epoch - self.window
        elif not out['plateau'] and self.plateau_start is not None:
            if epoch - self.plateau_start > self.window * 3:
                self.plateau_start = None

        # Grokking check
        if self.plateau_start is not None and len(self.history['val_acc']) > self.window + self.sustain_k:
            baseline_val = np.mean(self.history['val_acc'][-(self.window + self.sustain_k):-self.sustain_k])
            recent_val = np.mean(self.history['val_acc'][-self.sustain_k:])
            val_lift = recent_val - baseline_val

            baseline_gap = np.mean(self.history['train_acc'][-(self.window + self.sustain_k):-self.sustain_k]) - baseline_val
            recent_gap = np.mean(self.history['train_acc'][-self.sustain_k:]) - recent_val
            gap_collapse = baseline_gap - recent_gap

            out['val_lift'] = val_lift

            if val_lift > self.val_lift_min and gap_collapse > self.gap_collapse_min:
                out['potential'] = True
                self.events.append(GrokkingEvent(
                    epoch=epoch,
                    plateau_duration=epoch - self.plateau_start,
                    val_lift=val_lift,
                    gap_collapse=gap_collapse,
                ))
                out['event'] = True
                self.plateau_start = None

        return out


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
    train_cfg = config.get('training', {})
    statenet_cfg = config.get('statenet', {})
    loss_cfg = config.get('loss', {})
    riemannian_cfg = config.get('riemannian', {})
    option_c_cfg = config.get('option_c', {})

    # Hyperparameters
    epochs = train_cfg.get('epochs', 100)
    batch_size = train_cfg.get('batch_size', 512)
    base_lr = train_cfg.get('lr', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
    eval_every = train_cfg.get('eval_every', 5)
    save_every = train_cfg.get('save_every', 25)
    print_every = train_cfg.get('print_every', 5)

    # Worker init function for reproducible multi-worker data loading
    def worker_init_fn(worker_id: int) -> None:
        """Seed each DataLoader worker deterministically."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Create data loaders with reproducible seeding
    num_workers = train_cfg.get('num_workers', 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
    curvature = config.get('model', {}).get('curvature', 1.0)
    loss_fn = CombinedLoss(loss_cfg, curvature=curvature, device=device)
    print(f"  Loss functions: {loss_fn.get_enabled_losses()}")

    # Training controller (V6.0: LRController replaces StateNet)
    # LRController uses LR=0 for frozen components instead of requires_grad toggle
    lr_controller = None
    statenet = None  # Legacy compatibility
    use_lr_controller = statenet_cfg.get('use_lr_controller', True)  # Default to new system

    if statenet_cfg.get('enabled', False):
        if use_lr_controller:
            # V6.0: Use centralized StateNetConfig + MetricBasedLR
            sn_config = StateNetConfig.from_dict(statenet_cfg)

            # Also merge option_c LR scales if present
            if option_c_cfg.get('enabled', True):
                sn_config.lr_scales.encoder_a = option_c_cfg.get('encoder_a_lr_scale', 0.05)
                sn_config.lr_scales.encoder_b = option_c_cfg.get('encoder_b_lr_scale', 0.1)
                sn_config.lr_scales.projections = option_c_cfg.get('projections_lr_scale', 1.0)

            lr_controller = MetricBasedLR(sn_config)
            print(f"  LRController: MetricBasedLR ({sn_config.summary()})")
        else:
            # Legacy StateNet (for backwards compatibility)
            statenet = StateNet(
                coverage_fix_threshold=statenet_cfg.get('coverage_fix_threshold', 0.995),
                coverage_train_threshold=statenet_cfg.get('coverage_train_threshold', 0.999),
                coverage_floor=statenet_cfg.get('coverage_floor', 0.95),
                warmup_epochs=statenet_cfg.get('warmup_epochs', 10),
                hysteresis_epochs=statenet_cfg.get('hysteresis_epochs', 5),
                enable_annealing=statenet_cfg.get('enable_annealing', True),
                annealing_step=statenet_cfg.get('annealing_step', 0.005),
                hierarchy_plateau_threshold=statenet_cfg.get('hierarchy_plateau_threshold', 0.001),
                hierarchy_plateau_patience=statenet_cfg.get('hierarchy_plateau_patience', 15),
                hierarchy_patience_ceiling=statenet_cfg.get('hierarchy_patience_ceiling', 30),
                controller_grad_threshold=statenet_cfg.get('controller_grad_threshold', 0.01),
                controller_grad_patience=statenet_cfg.get('controller_grad_patience', 10),
                controller_patience_ceiling=statenet_cfg.get('controller_patience_ceiling', 20),
            )
            print("  StateNet: Enabled (legacy mode)")
    else:
        print("  Training controller: Disabled")

    # Optimizer
    if riemannian_cfg.get('enabled', False):
        param_groups = model.get_param_groups(base_lr)
        stabilize = riemannian_cfg.get('stabilize', 10)  # Re-project to manifold every N steps
        optimizer = get_riemannian_optimizer(param_groups, lr=base_lr, stabilize=stabilize)
        print(f"  Optimizer: RiemannianAdam (stabilize={stabilize})")
    else:
        optimizer = torch.optim.AdamW(
            model.get_param_groups(base_lr),
            weight_decay=weight_decay,
        )
        print("  Optimizer: AdamW")

    # Scheduler
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'cosine')
    if scheduler_type == 'cosine_warmup_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get('T_0', 25),
            T_mult=scheduler_cfg.get('T_mult', 2),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # TensorBoard logger (uses TensorBoardLogger from utils)
    logging_cfg = config.get('logging', {})
    run_name = log_dir.name
    tb_logger = TensorBoardLogger(
        tensorboard_dir=str(log_dir),
        experiment_name=run_name,
        log_callback=lambda msg: print(f"  {msg}") if logging_cfg.get('verbose', False) else None,
    )

    # Hardware monitor for memory tracking
    hw_monitor = HardwareMonitor(device, warn_threshold=0.9)

    # Config for enhanced logging
    histogram_every = logging_cfg.get('histogram_every', 10)
    embedding_every = logging_cfg.get('embedding_every', 50)
    log_batch_metrics = logging_cfg.get('enhanced_metrics', {}).get('enabled', False)

    # Checkpoints directory
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Grokking detector
    grok_cfg = train_cfg.get('grokking_detection', {})
    grokking_detector = GrokkingDetector(**grok_cfg) if grok_cfg else GrokkingDetector()

    # Training state
    best_Q = -1.0
    best_hierarchy = 0.0
    best_coverage = 0.0
    results = {
        'epochs_trained': 0,
        'best_Q': 0.0,
        'best_hierarchy': 0.0,
        'best_coverage': 0.0,
        'grokking_events': [],
    }

    print(f"\n{'='*60}")
    print(f"  TRAINING: {epochs} epochs, batch_size={batch_size}, lr={base_lr}")
    if TQDM_AVAILABLE:
        print("  Progress: tqdm enabled")
    print(f"{'='*60}\n")

    # Global batch counter for TensorBoard
    global_step = 0
    total_batches = len(train_loader)

    # Epoch iterator (with or without tqdm)
    epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if TQDM_AVAILABLE else range(epochs)

    for epoch in epoch_iter:
        t0 = time.time()
        model.train()

        # Training epoch
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n_batches = 0

        # Batch iterator (with or without tqdm)
        if TQDM_AVAILABLE:
            batch_iter = tqdm(train_loader, desc=f"Ep {epoch:03d}", leave=False, unit="batch")
        else:
            batch_iter = train_loader

        for batch_ops, batch_idx in batch_iter:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            optimizer.zero_grad()

            try:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(batch_ops)
                    z_hyp = out.get('z_A_hyp', out.get('z_B_hyp'))
                    logits = out.get('logits_A', out.get('logits'))

                    losses = loss_fn(z_hyp, batch_idx, logits, batch_ops, epoch=epoch)
                    loss = losses['total']

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            except torch.cuda.OutOfMemoryError:
                # OOM handling
                print(f"\n[OOM] CUDA Out of Memory at epoch {epoch}, batch {n_batches}")
                print(f"[OOM] {hw_monitor.get_oom_diagnostic(batch_size)}")

                # Clear cache
                torch.cuda.empty_cache()

                # Save emergency checkpoint
                emergency_path = ckpt_dir / f'emergency_oom_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, emergency_path)
                print(f"[OOM] Emergency checkpoint saved: {emergency_path}")

                # Re-raise to exit training
                raise

            train_loss_sum += loss.item()
            train_acc_sum += compute_accuracy(logits, batch_ops)
            n_batches += 1
            global_step += 1

            # Batch-level TensorBoard logging
            if log_batch_metrics and tb_logger.is_available:
                tb_logger.log_batch(global_step, loss.item())
                if hasattr(grad_norm, 'item'):
                    tb_logger.writer.add_scalar('Batch/GradNorm', grad_norm.item(), global_step)

            # Update tqdm postfix with loss and memory
            if TQDM_AVAILABLE and hasattr(batch_iter, 'set_postfix'):
                batch_iter.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gpu': hw_monitor.get_gpu_status_string() if device.type == 'cuda' else 'cpu',
                })

            # Check memory warning periodically
            if n_batches % 10 == 0:
                warning = hw_monitor.check_memory_warning()
                if warning:
                    print(f"\n{warning}")

        scheduler.step()

        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        if tb_logger.is_available:
            tb_logger.writer.add_scalar('LR/scheduled', current_lr, epoch)

        avg_train_loss = train_loss_sum / n_batches
        avg_train_acc = train_acc_sum / n_batches

        # Validation (every eval_every epochs)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_acc_sum = 0.0
            val_coverage_sum = 0.0
            val_batches = 0
            z_A_all = []
            z_B_all = []
            idx_all = []

            with torch.no_grad():
                for batch_ops, batch_idx in val_loader:
                    batch_ops = batch_ops.to(device)
                    batch_idx = batch_idx.to(device)

                    out = model(batch_ops)
                    logits = out.get('logits_A', out.get('logits'))

                    val_acc_sum += compute_accuracy(logits, batch_ops)
                    val_coverage_sum += compute_coverage(logits, batch_ops)
                    val_batches += 1

                    # Collect both VAE embeddings for separate hierarchy computation
                    z_A_all.append(out['z_A_hyp'])
                    z_B_all.append(out['z_B_hyp'])
                    idx_all.append(batch_idx)

            avg_val_acc = val_acc_sum / val_batches
            avg_val_coverage = val_coverage_sum / val_batches

            # Hierarchy metrics for both VAEs (seed varies per epoch, but reproducible)
            z_A_cat = torch.cat(z_A_all)
            z_B_cat = torch.cat(z_B_all)
            idx_cat = torch.cat(idx_all)
            hier_metrics_A = compute_hierarchy_metrics(z_A_cat, idx_cat, curvature, seed=seed + epoch)
            hier_metrics_B = compute_hierarchy_metrics(z_B_cat, idx_cat, curvature, seed=seed + epoch + 1000)

            # Get gradient statistics from optimizer (V6.0: avoids manual grad loop)
            controller_grad_norm = None
            if lr_controller is not None or statenet is not None:
                grad_stats = get_optimizer_grad_stats(optimizer, 'projections')
                controller_grad_norm = grad_stats.get('grad_norm_estimate', 0.0)

                # Fallback to manual computation if optimizer stats unavailable
                if controller_grad_norm == 0.0 and hasattr(model, 'projections'):
                    proj_grads = [p.grad for p in model.projections.parameters()
                                  if p.grad is not None]
                    if proj_grads:
                        controller_grad_norm = sum(g.norm(2).item() for g in proj_grads)

            # Training controller update
            controller_state = None
            train_summary = "N/A"

            if lr_controller is not None:
                # V6.0: LRController - unified control via optimizer LR
                metrics = TrainingMetrics(
                    epoch=epoch,
                    coverage=avg_val_coverage,
                    hierarchy_a=hier_metrics_A['hierarchy'],
                    hierarchy_b=hier_metrics_B['hierarchy'],
                    dist_corr_a=hier_metrics_A['dist_corr'],
                    q_value=hier_metrics_A['Q'],
                    grad_norm_projections=controller_grad_norm or 0.0,
                )
                controller_state = lr_controller.update(metrics)

                # Apply LR scales to optimizer (the key Option C mechanism)
                update_optimizer_lr_scales(optimizer, base_lr, controller_state['lr_scales'])

                # Format summary from LR scales
                scales = controller_state['lr_scales']
                train_summary = f"A:{scales.get('encoder_a', 0):.2f} B:{scales.get('encoder_b', 0):.2f} P:{scales.get('projections', 0):.2f}"

            elif statenet is not None:
                # Legacy StateNet path
                statenet_state = statenet.update(
                    epoch=epoch,
                    coverage=avg_val_coverage,
                    hierarchy_A=hier_metrics_A['hierarchy'],
                    hierarchy_B=hier_metrics_B['hierarchy'],
                    dist_corr_A=hier_metrics_A['dist_corr'],
                    controller_grad_norm=controller_grad_norm,
                )
                model.apply_statenet_state(statenet_state)
                train_summary = model.get_trainability_summary()
                controller_state = statenet_state  # For TensorBoard logging

            # Grokking detection
            grok_state = grokking_detector.update(epoch, avg_train_loss, avg_train_acc, avg_val_acc)
            if grok_state['event']:
                print(f"  [GROKKING] Event detected at epoch {epoch}!")
                results['grokking_events'].append(grokking_detector.events[-1].__dict__)

            # TensorBoard logging (log both VAE metrics)
            if tb_logger.is_available:
                tb_logger.writer.add_scalars('Accuracy', {'train': avg_train_acc, 'val': avg_val_acc}, epoch)
                tb_logger.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                tb_logger.writer.add_scalar('Coverage', avg_val_coverage, epoch)
                tb_logger.writer.add_scalars('Hierarchy/corr', {
                    'VAE_A': hier_metrics_A['hierarchy'],
                    'VAE_B': hier_metrics_B['hierarchy'],
                }, epoch)
                tb_logger.writer.add_scalars('Hierarchy/Q', {
                    'VAE_A': hier_metrics_A['Q'],
                    'VAE_B': hier_metrics_B['Q'],
                }, epoch)
                tb_logger.writer.add_scalar('Hierarchy/dist_corr', hier_metrics_A['dist_corr'], epoch)
                tb_logger.writer.add_scalars('Radius/mean', {
                    'VAE_A': hier_metrics_A['mean_radius'],
                    'VAE_B': hier_metrics_B['mean_radius'],
                }, epoch)

                # Training controller metrics
                if controller_state is not None:
                    if lr_controller is not None:
                        # V6.0: Log LR scales (continuous, 0.0 = frozen)
                        scales = controller_state.get('lr_scales', {})
                        tb_logger.writer.add_scalar('LRController/encoder_a_lr_scale',
                                                     scales.get('encoder_a', 0), epoch)
                        tb_logger.writer.add_scalar('LRController/encoder_b_lr_scale',
                                                     scales.get('encoder_b', 0), epoch)
                        tb_logger.writer.add_scalar('LRController/projections_lr_scale',
                                                     scales.get('projections', 0), epoch)
                        tb_logger.writer.add_scalar('LRController/best_Q',
                                                     controller_state.get('best_q', 0), epoch)

                        # Log events if any
                        events = controller_state.get('events', [])
                        if events and events != ['warmup']:
                            print(f"    LRController events: {events}")
                    else:
                        # Legacy StateNet logging
                        tb_logger.writer.add_scalar('StateNet/encoder_a_trainable',
                                                     int(controller_state['encoder_a_trainable']), epoch)
                        tb_logger.writer.add_scalar('StateNet/encoder_b_trainable',
                                                     int(controller_state['encoder_b_trainable']), epoch)
                        tb_logger.writer.add_scalar('StateNet/controller_trainable',
                                                     int(controller_state['controller_trainable']), epoch)
                        tb_logger.writer.add_scalar('StateNet/best_Q',
                                                     controller_state.get('best_Q', 0), epoch)

                    # Grad norm (common to both)
                    if controller_grad_norm is not None:
                        tb_logger.writer.add_scalar('Controller/grad_norm', controller_grad_norm, epoch)

                # Hardware metrics
                if device.type == 'cuda':
                    gpu_mem = hw_monitor.get_gpu_memory_gb()
                    tb_logger.writer.add_scalar('Hardware/GPU_allocated_GB', gpu_mem['allocated'], epoch)
                    tb_logger.writer.add_scalar('Hardware/GPU_peak_GB', gpu_mem['peak'], epoch)

                # Flush for real-time updates
                tb_logger.flush()

            # Weight histograms (periodic)
            if histogram_every > 0 and epoch % histogram_every == 0 and epoch > 0:
                if tb_logger.is_available:
                    tb_logger.log_histograms(epoch, model)

            # Track best metrics (use VAE-A as primary)
            if hier_metrics_A['Q'] > best_Q:
                best_Q = hier_metrics_A['Q']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'Q': best_Q,
                    'hierarchy_A': hier_metrics_A['hierarchy'],
                    'hierarchy_B': hier_metrics_B['hierarchy'],
                    'coverage': avg_val_coverage,
                }, ckpt_dir / 'best_Q.pt')

            if avg_val_coverage > best_coverage:
                best_coverage = avg_val_coverage

            if hier_metrics_A['hierarchy'] < best_hierarchy:  # More negative is better
                best_hierarchy = hier_metrics_A['hierarchy']

            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                dt = time.time() - t0
                # Main metrics line
                if not TQDM_AVAILABLE:  # Only print if not using tqdm (tqdm handles its own output)
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
                    epoch_iter.set_postfix({
                        'loss': f'{avg_train_loss:.4f}',
                        'cov': f'{avg_val_coverage:.3f}',
                        'Q': f'{hier_metrics_A["Q"]:.2f}',
                    })
                    # Print hardware stats on separate line
                    tqdm.write(
                        f"  Ep {epoch:03d} | Cov {avg_val_coverage:.3f} | "
                        f"Hier A/B {hier_metrics_A['hierarchy']:.3f}/{hier_metrics_B['hierarchy']:.3f} | "
                        f"Q {hier_metrics_A['Q']:.3f} | {hw_monitor.get_peak_status_string()} | {dt:.1f}s"
                    )

        # Periodic checkpoint
        if epoch % save_every == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch}.pt')

    # Final checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_Q': best_Q,
        'best_hierarchy': best_hierarchy,
        'best_coverage': best_coverage,
    }, ckpt_dir / 'final.pt')

    # Close TensorBoard logger
    tb_logger.close()

    results.update({
        'epochs_trained': epochs,
        'best_Q': best_Q,
        'best_hierarchy': best_hierarchy,
        'best_coverage': best_coverage,
    })

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified p-adic VAE Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate config and data, do not train')
    parser.add_argument('--force', action='store_true',
                        help='Continue even if validation fails')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom run name (default: config name + timestamp)')
    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Set determinism
    set_determinism(args.seed)
    print(f"\n{'='*60}")
    print("  P-ADIC VAE UNIFIED TRAINING")
    print(f"{'='*60}")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override device settings from YAML if not specified on command line
    device_cfg = config.get('device', {})
    if args.device == 'cuda' and device_cfg.get('cuda_device') is not None:
        cuda_device = device_cfg.get('cuda_device', 0)
        device = torch.device(f'cuda:{cuda_device}')
        print(f"  Using CUDA device {cuda_device} from config")

    # Use AMP from config if not specified on command line
    if not args.amp and device_cfg.get('use_amp', False):
        args.amp = True
        print("  Using AMP from config")

    # Create run directory
    config_name = config_path.stem
    run_name = args.name or f"{config_name}_{get_timestamp()}"
    log_dir = RUNS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Run directory: {log_dir}")

    # Data audit
    data_auditor = DataAuditor(args.seed)
    val_frac = config.get('training', {}).get('val_frac', 0.1)
    train_ds, val_ds, all_indices = data_auditor.prepare_data(val_frac, device)

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
            'data': data_auditor.audit_log,
            'model': model_auditor.audit_log,
        }
        with open(log_dir / 'audit_log.json', 'w') as f:
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
    results_path = log_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Epochs: {results['epochs_trained']}")
    print(f"  Best Q: {results['best_Q']:.4f}")
    print(f"  Best Hierarchy: {results['best_hierarchy']:.4f}")
    print(f"  Best Coverage: {results['best_coverage']:.4f}")
    print(f"  Grokking Events: {len(results['grokking_events'])}")
    print(f"  Results: {results_path}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
