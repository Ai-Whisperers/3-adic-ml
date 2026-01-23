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
    - StateNet: Adaptive freeze/unfreeze controller
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

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Internal imports
from src.config.paths import RUNS_DIR, CHECKPOINTS_DIR
from src.core import TERNARY
from src.geometry import poincare_distance, get_riemannian_optimizer
from src.losses import CombinedLoss
from src.models import StateNet, compute_Q, TernaryVAEV5_11_PartialFreeze
from src.models.vae import map_v5_5_keys
from src.utils.checkpoint import load_checkpoint_compat, get_model_state_dict


# =============================================================================
# DETERMINISM
# =============================================================================

def set_determinism(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but exact)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        model_name = model_cfg.get('name', 'TernaryVAEV5_11_PartialFreeze')
        encoder_type = model_cfg.get('encoder_type', 'improved')
        decoder_type = model_cfg.get('decoder_type', 'improved')

        # Validate: v5.5 checkpoints require standard architecture (no silent override)
        frozen_cfg = self.config.get('frozen_checkpoint', {})
        ckpt_path_str = frozen_cfg.get('path')

        if ckpt_path_str and ckpt_path_str != 'null':
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            is_v5_5_checkpoint = ckpt_path.exists() and 'v5_5' in str(ckpt_path)

            if is_v5_5_checkpoint and encoder_type != 'standard':
                raise ValueError(
                    f"Config mismatch: v5.5 checkpoint requires encoder_type='standard', "
                    f"but config specifies '{encoder_type}'. "
                    f"Update your preset to set encoder_type: standard and decoder_type: standard"
                )

        # Instantiate model
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=model_cfg.get('latent_dim', 16),
            hidden_dim=model_cfg.get('hidden_dim', 64),
            max_radius=model_cfg.get('max_radius', 0.95),
            curvature=model_cfg.get('curvature', 1.0),
            use_controller=model_cfg.get('use_controller', True),
            use_dual_projection=model_cfg.get('use_dual_projection', True),
            n_projection_layers=model_cfg.get('projection_layers', 2),
            projection_dropout=model_cfg.get('projection_dropout', 0.1),
            learnable_curvature=model_cfg.get('learnable_curvature', False),
            freeze_encoder_b=False,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] Model created: {model_name}")
        print(f"       Parameters: {n_params:,} total, {n_trainable:,} trainable")

        # Load frozen checkpoint if specified
        frozen_cfg = self.config.get('frozen_checkpoint', {})
        ckpt_path_str = frozen_cfg.get('path')

        if ckpt_path_str and ckpt_path_str != 'null':
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            if ckpt_path.exists():
                try:
                    ckpt = load_checkpoint_compat(ckpt_path, map_location=self.device)
                    state_dict = get_model_state_dict(ckpt)

                    # Detect v5.5 checkpoint (has encoder_A.encoder.X keys)
                    is_v5_5 = any(k.startswith('encoder_A.encoder.') for k in state_dict.keys())

                    if is_v5_5:
                        # Map v5.5 keys to V5.11 format
                        state_dict = map_v5_5_keys(state_dict)
                        print(f"  [OK] Detected v5.5 checkpoint, applied key mapping")

                    # Load with strict=False (projections may not match)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
                    print(f"       Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

                    self.audit_log['checkpoint_loaded'] = True
                    self.audit_log['checkpoint_path'] = str(ckpt_path)
                    self.audit_log['checkpoint_is_v5_5'] = is_v5_5

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
            print("  [INFO] No frozen checkpoint specified (training from scratch)")
            self.audit_log['checkpoint_loaded'] = False

        # Gradient flow check
        model.train()
        dummy_input = torch.randint(-1, 2, (32, 9)).float().to(self.device)

        try:
            out = model(dummy_input, compute_control=False)
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

    # Hyperparameters
    epochs = train_cfg.get('epochs', 100)
    batch_size = train_cfg.get('batch_size', 512)
    base_lr = train_cfg.get('lr', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
    eval_every = train_cfg.get('eval_every', 5)
    save_every = train_cfg.get('save_every', 25)
    print_every = train_cfg.get('print_every', 5)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=train_cfg.get('num_workers', 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Loss function
    curvature = config.get('model', {}).get('curvature', 1.0)
    loss_fn = CombinedLoss(loss_cfg, curvature=curvature, device=device)
    print(f"  Loss functions: {loss_fn.get_enabled_losses()}")

    # StateNet controller
    statenet = None
    if statenet_cfg.get('enabled', False):
        statenet = StateNet(
            coverage_freeze_threshold=statenet_cfg.get('coverage_freeze_threshold', 0.995),
            coverage_unfreeze_threshold=statenet_cfg.get('coverage_unfreeze_threshold', 0.999),
            coverage_floor=statenet_cfg.get('coverage_floor', 0.95),
            warmup_epochs=statenet_cfg.get('warmup_epochs', 10),
            hysteresis_epochs=statenet_cfg.get('hysteresis_epochs', 5),
            enable_annealing=statenet_cfg.get('enable_annealing', True),
            annealing_step=statenet_cfg.get('annealing_step', 0.005),
            hierarchy_plateau_threshold=statenet_cfg.get('hierarchy_plateau_threshold', 0.001),
            hierarchy_plateau_patience=statenet_cfg.get('hierarchy_plateau_patience', 15),
        )
        print("  StateNet: Enabled")
    else:
        print("  StateNet: Disabled")

    # Optimizer
    if riemannian_cfg.get('enabled', False):
        param_groups = model.get_param_groups(base_lr)
        optimizer = get_riemannian_optimizer(param_groups, lr=base_lr)
        print("  Optimizer: RiemannianAdam")
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

    # TensorBoard (optional)
    writer = SummaryWriter(str(log_dir)) if TENSORBOARD_AVAILABLE else None

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
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        # Training epoch
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n_batches = 0

        for batch_ops, batch_idx in train_loader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(batch_ops, compute_control=False)
                z_hyp = out.get('z_A_hyp', out.get('z_B_hyp'))
                logits = out.get('logits_A', out.get('logits'))

                losses = loss_fn(z_hyp, batch_idx, logits, batch_ops, epoch=epoch)
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_acc_sum += compute_accuracy(logits, batch_ops)
            n_batches += 1

        scheduler.step()

        avg_train_loss = train_loss_sum / n_batches
        avg_train_acc = train_acc_sum / n_batches

        # Validation (every eval_every epochs)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_acc_sum = 0.0
            val_coverage_sum = 0.0
            val_batches = 0
            z_all = []
            idx_all = []

            with torch.no_grad():
                for batch_ops, batch_idx in val_loader:
                    batch_ops = batch_ops.to(device)
                    batch_idx = batch_idx.to(device)

                    out = model(batch_ops, compute_control=False)
                    z_hyp = out.get('z_A_hyp', out.get('z_B_hyp'))
                    logits = out.get('logits_A', out.get('logits'))

                    val_acc_sum += compute_accuracy(logits, batch_ops)
                    val_coverage_sum += compute_coverage(logits, batch_ops)
                    val_batches += 1

                    z_all.append(z_hyp)
                    idx_all.append(batch_idx)

            avg_val_acc = val_acc_sum / val_batches
            avg_val_coverage = val_coverage_sum / val_batches

            # Hierarchy metrics
            z_cat = torch.cat(z_all)
            idx_cat = torch.cat(idx_all)
            hier_metrics = compute_hierarchy_metrics(z_cat, idx_cat, curvature)

            # StateNet update
            if statenet is not None:
                statenet_state = statenet.update(
                    epoch=epoch,
                    coverage=avg_val_coverage,
                    hierarchy_A=hier_metrics['hierarchy'],
                    hierarchy_B=hier_metrics['hierarchy'],
                    dist_corr_A=hier_metrics['dist_corr'],
                )
                model.apply_statenet_state(statenet_state)
                freeze_summary = model.get_freeze_state_summary()
            else:
                freeze_summary = "N/A"

            # Grokking detection
            grok_state = grokking_detector.update(epoch, avg_train_loss, avg_train_acc, avg_val_acc)
            if grok_state['event']:
                print(f"  [GROKKING] Event detected at epoch {epoch}!")
                results['grokking_events'].append(grokking_detector.events[-1].__dict__)

            # TensorBoard logging
            if writer is not None:
                writer.add_scalars('Accuracy', {'train': avg_train_acc, 'val': avg_val_acc}, epoch)
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                writer.add_scalar('Coverage', avg_val_coverage, epoch)
                writer.add_scalar('Hierarchy/corr', hier_metrics['hierarchy'], epoch)
                writer.add_scalar('Hierarchy/Q', hier_metrics['Q'], epoch)
                writer.add_scalar('Hierarchy/dist_corr', hier_metrics['dist_corr'], epoch)

            # Track best metrics
            if hier_metrics['Q'] > best_Q:
                best_Q = hier_metrics['Q']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'Q': best_Q,
                    'hierarchy': hier_metrics['hierarchy'],
                    'coverage': avg_val_coverage,
                }, ckpt_dir / 'best_Q.pt')

            if avg_val_coverage > best_coverage:
                best_coverage = avg_val_coverage

            if hier_metrics['hierarchy'] < best_hierarchy:  # More negative is better
                best_hierarchy = hier_metrics['hierarchy']

            # Print progress
            if epoch % print_every == 0 or epoch == epochs - 1:
                dt = time.time() - t0
                print(
                    f"Ep {epoch:03d} | "
                    f"Loss {avg_train_loss:.4f} | "
                    f"Acc T/V {avg_train_acc:.3f}/{avg_val_acc:.3f} | "
                    f"Cov {avg_val_coverage:.3f} | "
                    f"Hier {hier_metrics['hierarchy']:.4f} | "
                    f"Q {hier_metrics['Q']:.3f} | "
                    f"Freeze {freeze_summary} | "
                    f"{dt:.1f}s"
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

    if writer is not None:
        writer.close()

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
