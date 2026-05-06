# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Learning Rate Controller - Unified Training Control via Optimizer.

Replaces StateNet's if/else decision logic with:
1. Schedule-based control (predetermined epochs)
2. Metric-based control (soft gating via LR multipliers)
3. Learnable control (differentiable LR prediction)

Philosophy (Option C Aggressive):
    - ALL trainability control happens via LR multipliers
    - LR=0 means "frozen", no separate requires_grad flag
    - Decisions are continuous (soft), not binary (hard)
    - Optimizer param groups are the single source of truth

Usage:
    from src.models import LRController, ScheduleBasedLR, MetricBasedLR

    # Schedule-based (simplest, reproducible)
    controller = ScheduleBasedLR({
        'encoder_a': [(0, 0.0), (50, 0.05), (100, 0.0)],  # epoch -> lr_scale
        'encoder_b': [(0, 0.1), (80, 0.0)],
        'projections': [(0, 1.0)],  # always full
    })

    # Metric-based (replaces StateNet logic)
    controller = MetricBasedLR(config)

    # In training loop
    lr_scales = controller.get_lr_scales(epoch, metrics)
    update_param_group_lrs(optimizer, base_lr, lr_scales)
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.config.statenet_config import StateNetConfig


def compute_Q(dist_corr: float, hierarchy: float) -> float:
    """Compute conserved structure capacity Q.

    Q = dist_corr + 1.5 × hierarchy

    hierarchy is the negated Spearman correlation (positive = good ordering).
    Higher Q indicates better structure learning.
    """
    return dist_corr + 1.5 * hierarchy


@dataclass
class TrainingMetrics:
    """Snapshot of metrics for LR decisions.

    All bounded metrics must be in [-1, 1]:
      - coverage: per-digit reconstruction accuracy in [0, 1]
      - hierarchy_a/b: Spearman correlation in [-1, 1]
      - dist_corr_a: distance correlation in [-1, 1]
    """
    epoch: int
    coverage: float          # per-digit accuracy in [0, 1]
    hierarchy_a: float       # Spearman corr in [-1, 1]
    hierarchy_b: float       # Spearman corr in [-1, 1]
    dist_corr_a: float = 0.0
    q_value: float = 0.0
    grad_norm_projections: float = 0.0

    def __post_init__(self) -> None:
        for name in ("coverage", "hierarchy_a", "hierarchy_b", "dist_corr_a"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"TrainingMetrics.{name} must be a scalar float, got {type(v).__name__}"
                )
            if not (-1.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"TrainingMetrics.{name}={v} is out of range [-1, 1]. "
                    f"Check that the metric is computed correctly and matches "
                    f"the expected scale (per-digit accuracy or correlation)."
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], epoch: int = 0) -> "TrainingMetrics":
        return cls(
            epoch=epoch,
            coverage=d.get('coverage', 0.0),
            hierarchy_a=d.get('hierarchy_a', d.get('hierarchy_A', 0.0)),
            hierarchy_b=d.get('hierarchy_b', d.get('hierarchy_B', 0.0)),
            dist_corr_a=d.get('dist_corr_a', d.get('dist_corr_A', 0.0)),
            q_value=d.get('q_value', d.get('Q', 0.0)),
            grad_norm_projections=d.get('grad_norm_projections', 0.0),
        )


class LRController(ABC):
    """Abstract base for LR controllers."""

    @abstractmethod
    def get_lr_scales(self, metrics: TrainingMetrics) -> Dict[str, float]:
        """Return LR scale for each component.

        Args:
            metrics: Current training metrics

        Returns:
            Dict mapping component name to LR scale (0.0 = frozen, 1.0 = full)
        """
        pass

    @abstractmethod
    def update(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Update internal state and return status.

        Args:
            metrics: Current training metrics

        Returns:
            Dict with events, current state, etc.
        """
        pass



class MetricBasedLR(LRController):
    """Metric-based LR control (soft version of StateNet logic).

    Replaces hard if/else thresholds with continuous functions:
    - Coverage gate: sigmoid(coverage - threshold) * base_scale
    - Hierarchy gate: based on plateau detection
    - Projections gate: based on gradient norm stability

    Key difference from StateNet: outputs are CONTINUOUS, not binary.
    """

    def __init__(self, config: StateNetConfig) -> None:
        """Initialize metric-based controller.

        Args:
            config: Centralized StateNet configuration
        """
        self.config = config

        # Rolling statistics (GPU-compatible)
        window = config.timing.window_size
        self._coverage_history: deque[float] = deque(maxlen=window)
        self._hierarchy_a_history: deque[float] = deque(maxlen=window)
        self._hierarchy_b_history: deque[float] = deque(maxlen=window)
        self._grad_norm_history: deque[float] = deque(maxlen=window)
        self._q_history: deque[float] = deque(maxlen=window * 2)  # Longer for Q

        # Plateau counters
        self._hierarchy_b_plateau_count = 0
        self._grad_low_count = 0
        self._hierarchy_a_stall_count = 0

        # Last state change epoch (for hysteresis)
        self._last_change = {
            'encoder_a': -config.timing.hysteresis_epochs,
            'encoder_b': -config.timing.hysteresis_epochs,
            'projections': -config.timing.hysteresis_epochs,
        }

        # Current "active" state (for soft gating)
        self._active = {
            'encoder_a': config.initial.encoder_a_trainable,
            'encoder_b': config.initial.encoder_b_trainable,
            'projections': config.initial.projections_trainable,
        }

        # Best Q tracking
        self._best_q = 0.0

    def _can_change(self, component: str, epoch: int) -> bool:
        """Check hysteresis constraint."""
        return epoch - self._last_change[component] >= self.config.timing.hysteresis_epochs

    def _update_histories(self, metrics: TrainingMetrics) -> None:
        """Update rolling histories and advance plateau counters once per epoch."""
        self._coverage_history.append(metrics.coverage)
        self._hierarchy_a_history.append(metrics.hierarchy_a)
        self._hierarchy_b_history.append(metrics.hierarchy_b)
        self._q_history.append(metrics.q_value)

        if metrics.grad_norm_projections > 0:
            self._grad_norm_history.append(metrics.grad_norm_projections)

        # Advance stall counter here (single write path per epoch, not in gate methods)
        window = self.config.timing.window_size
        if len(self._hierarchy_a_history) >= window:
            recent_a = list(self._hierarchy_a_history)
            improvement_a = abs(recent_a[-1]) - abs(recent_a[0])
            if improvement_a < self.config.hierarchy.plateau_threshold:
                self._hierarchy_a_stall_count += 1
            else:
                self._hierarchy_a_stall_count = 0

    def _compute_coverage_gate(self, metrics: TrainingMetrics) -> Tuple[float, Optional[str]]:
        """Compute soft coverage gate for encoder_a.

        Returns (scale_multiplier, event_string)
        """
        event = None
        cfg = self.config.coverage

        if not self._can_change('encoder_a', metrics.epoch):
            return (self.config.lr_scales.encoder_a if self._active['encoder_a'] else 0.0, None)

        if self._active['encoder_a']:
            # Currently active - check if should deactivate
            if metrics.coverage < cfg.fix_threshold:
                self._active['encoder_a'] = False
                self._last_change['encoder_a'] = metrics.epoch
                event = "encoder_a frozen (coverage drop)"
                return (0.0, event)
        else:
            # Currently frozen - unfreeze when coverage recovers OR hierarchy stalls
            # (stall without coverage means encoder_a needs to train to break the stall)
            coverage_ok = metrics.coverage >= cfg.train_threshold
            hierarchy_stalled = self._is_hierarchy_a_stalled()
            if coverage_ok or hierarchy_stalled:
                self._active['encoder_a'] = True
                self._last_change['encoder_a'] = metrics.epoch
                reason = "coverage recovered" if coverage_ok else "hierarchy stall"
                event = f"encoder_a unfrozen ({reason})"
                return (self.config.lr_scales.encoder_a, event)

        return (self.config.lr_scales.encoder_a if self._active['encoder_a'] else 0.0, event)

    def _is_hierarchy_a_stalled(self) -> bool:
        """Check if VAE-A hierarchy has plateaued.

        Pure predicate — reads stall count only. Counter is advanced once per
        epoch in _update_histories to prevent double-increment when this method
        is called multiple times per update cycle.
        """
        if len(self._hierarchy_a_history) < self.config.timing.window_size:
            return False
        return self._hierarchy_a_stall_count >= self.config.hierarchy.stall_patience

    def _compute_hierarchy_gate(self, metrics: TrainingMetrics) -> Tuple[float, Optional[str]]:
        """Compute soft hierarchy gate for encoder_b."""
        event = None
        cfg = self.config.hierarchy

        if not self._can_change('encoder_b', metrics.epoch):
            return (self.config.lr_scales.encoder_b if self._active['encoder_b'] else 0.0, None)

        if len(self._hierarchy_b_history) < 2:
            return (self.config.lr_scales.encoder_b if self._active['encoder_b'] else 0.0, None)

        if self._active['encoder_b']:
            # Check for plateau
            window = self.config.timing.window_size
            if len(self._hierarchy_b_history) >= window:
                recent = list(self._hierarchy_b_history)
                improvement = abs(recent[-1]) - abs(recent[0])

                if improvement < cfg.plateau_threshold:
                    self._hierarchy_b_plateau_count += 1
                else:
                    self._hierarchy_b_plateau_count = 0

                if self._hierarchy_b_plateau_count >= cfg.plateau_patience:
                    self._active['encoder_b'] = False
                    self._last_change['encoder_b'] = metrics.epoch
                    event = f"encoder_b frozen (plateau {self._hierarchy_b_plateau_count} epochs)"
                    return (0.0, event)
        else:
            # Unfreeze only on sustained degradation: net decline > 2% over window/2 steps.
            # plateau_threshold is for stagnation (0.0005 scale), not decline — use a
            # fixed meaningful threshold (0.02 on correlation scale) to avoid noise oscillation.
            _DEGRADATION_THRESHOLD = 0.02
            half_win = max(2, self.config.timing.window_size // 2)
            if len(self._hierarchy_b_history) >= half_win:
                recent = list(self._hierarchy_b_history)[-half_win:]
                net_change = abs(recent[-1]) - abs(recent[0])
                if net_change < -_DEGRADATION_THRESHOLD:
                    self._active['encoder_b'] = True
                    self._hierarchy_b_plateau_count = 0
                    self._last_change['encoder_b'] = metrics.epoch
                    event = "encoder_b unfrozen (sustained hierarchy degradation)"
                    return (self.config.lr_scales.encoder_b, event)

        return (self.config.lr_scales.encoder_b if self._active['encoder_b'] else 0.0, event)

    def _compute_projections_gate(self, metrics: TrainingMetrics) -> Tuple[float, Optional[str]]:
        """Compute soft gradient gate for projections."""
        event = None
        cfg = self.config.controller

        if not self._can_change('projections', metrics.epoch):
            return (self.config.lr_scales.projections if self._active['projections'] else 0.0, None)

        if len(self._grad_norm_history) < 2:
            return (self.config.lr_scales.projections if self._active['projections'] else 0.0, None)

        current_grad = metrics.grad_norm_projections

        if self._active['projections']:
            if current_grad < cfg.grad_threshold:
                self._grad_low_count += 1
            else:
                self._grad_low_count = 0

            if self._grad_low_count >= cfg.grad_patience:
                self._active['projections'] = False
                self._last_change['projections'] = metrics.epoch
                event = "projections frozen (stable gradients)"
                return (0.0, event)
        else:
            # Check for gradient spike
            avg_grad = sum(self._grad_norm_history) / len(self._grad_norm_history)
            if current_grad > avg_grad * cfg.spike_multiplier:
                self._active['projections'] = True
                self._grad_low_count = 0
                self._last_change['projections'] = metrics.epoch
                event = "projections unfrozen (gradient spike)"
                return (self.config.lr_scales.projections, event)

        return (self.config.lr_scales.projections if self._active['projections'] else 0.0, event)

    def get_lr_scales(self, metrics: TrainingMetrics) -> Dict[str, float]:
        """Get LR scales based on current metrics."""
        # During warmup, use initial states
        if metrics.epoch < self.config.timing.warmup_epochs:
            return {
                'encoder_a': self.config.lr_scales.encoder_a if self.config.initial.encoder_a_trainable else 0.0,
                'encoder_b': self.config.lr_scales.encoder_b if self.config.initial.encoder_b_trainable else 0.0,
                'projections': self.config.lr_scales.projections if self.config.initial.projections_trainable else 0.0,
                'decoders': self.config.lr_scales.decoders,
            }

        cov_scale, _ = self._compute_coverage_gate(metrics)
        hier_scale, _ = self._compute_hierarchy_gate(metrics)
        proj_scale, _ = self._compute_projections_gate(metrics)

        return {
            'encoder_a': cov_scale,
            'encoder_b': hier_scale,
            'projections': proj_scale,
            'decoders': self.config.lr_scales.decoders,  # Always full
        }

    def update(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Update state and return status with events."""
        self._update_histories(metrics)

        # Track best Q
        if metrics.q_value > self._best_q:
            self._best_q = metrics.q_value

        events = []

        # During warmup, just return current state
        if metrics.epoch < self.config.timing.warmup_epochs:
            return {
                "lr_scales": self.get_lr_scales(metrics),
                "events": ["warmup"],
                "type": "metric_based",
                "best_q": self._best_q,
                "active_states": dict(self._active),
            }

        # Compute gates (which may update state and generate events)
        cov_scale, cov_event = self._compute_coverage_gate(metrics)
        hier_scale, hier_event = self._compute_hierarchy_gate(metrics)
        proj_scale, proj_event = self._compute_projections_gate(metrics)

        if cov_event:
            events.append(cov_event)
        if hier_event:
            events.append(hier_event)
        if proj_event:
            events.append(proj_event)

        # Use scales from the gate calls above instead of calling
        # get_lr_scales() again, which would re-invoke gates with side effects
        lr_scales = {
            'encoder_a': cov_scale,
            'encoder_b': hier_scale,
            'projections': proj_scale,
            'decoders': self.config.lr_scales.decoders,
        }

        return {
            "lr_scales": lr_scales,
            "events": events,
            "type": "metric_based",
            "best_q": self._best_q,
            "active_states": dict(self._active),
        }

    def reset(self):
        """Reset all state."""
        self._coverage_history.clear()
        self._hierarchy_a_history.clear()
        self._hierarchy_b_history.clear()
        self._grad_norm_history.clear()
        self._q_history.clear()

        self._hierarchy_b_plateau_count = 0
        self._grad_low_count = 0
        self._hierarchy_a_stall_count = 0

        self._last_change = {
            'encoder_a': -self.config.timing.hysteresis_epochs,
            'encoder_b': -self.config.timing.hysteresis_epochs,
            'projections': -self.config.timing.hysteresis_epochs,
        }

        self._active = {
            'encoder_a': self.config.initial.encoder_a_trainable,
            'encoder_b': self.config.initial.encoder_b_trainable,
            'projections': self.config.initial.projections_trainable,
        }

        self._best_q = 0.0



def update_optimizer_lr_scales(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    lr_scales: Dict[str, float],
) -> None:
    """Update optimizer param group learning rates.

    This is the key function for Option C: all control happens here.

    Args:
        optimizer: PyTorch optimizer with named param groups
        base_lr: Base learning rate
        lr_scales: Dict mapping group name to LR scale
    """
    matched = set()
    for group in optimizer.param_groups:
        name = group.get('name', '')
        if name in lr_scales:
            group['lr'] = base_lr * lr_scales[name]
            matched.add(name)

    unmatched_scales = set(lr_scales) - matched
    if unmatched_scales:
        import warnings
        warnings.warn(
            f"update_optimizer_lr_scales: lr_scales keys {unmatched_scales} matched no "
            f"optimizer param group. LR control is silently inactive for these components. "
            f"Group names present: {[g.get('name','<unnamed>') for g in optimizer.param_groups]}",
            UserWarning,
            stacklevel=2,
        )


def get_optimizer_grad_stats(
    optimizer: torch.optim.Optimizer,
    group_name: str,
) -> Dict[str, float]:
    """Get gradient statistics from optimizer state (Adam/AdamW).

    Uses optimizer's internal exp_avg_sq for grad norm estimation.
    Avoids recomputing gradients manually.

    Args:
        optimizer: Adam/AdamW optimizer
        group_name: Name of param group to query

    Returns:
        Dict with grad_norm_estimate, exp_avg_norm, etc.
    """
    stats = {
        'grad_norm_estimate': 0.0,
        'exp_avg_norm': 0.0,
        'n_params': 0,
    }

    for group in optimizer.param_groups:
        if group.get('name', '') != group_name:
            continue

        total_sq = 0.0
        total_avg = 0.0
        n_params = 0

        for p in group['params']:
            if p not in optimizer.state:
                continue

            state = optimizer.state[p]

            # Adam stores exp_avg_sq (EMA of squared gradients)
            if 'exp_avg_sq' in state:
                # RMS of sqrt(exp_avg_sq) ≈ grad norm
                sq = state['exp_avg_sq']
                total_sq += sq.sum().item()
                n_params += sq.numel()

            if 'exp_avg' in state:
                avg = state['exp_avg']
                total_avg += avg.abs().sum().item()

        if n_params > 0:
            stats['grad_norm_estimate'] = (total_sq / n_params) ** 0.5
            stats['exp_avg_norm'] = total_avg / n_params
            stats['n_params'] = n_params

    return stats


__all__ = [
    "LRController",
    "MetricBasedLR",
    "TrainingMetrics",
    "update_optimizer_lr_scales",
    "get_optimizer_grad_stats",
    "compute_Q",
]
