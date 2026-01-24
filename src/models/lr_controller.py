# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.config.statenet_config import StateNetConfig


def compute_Q(dist_corr: float, hierarchy: float) -> float:
    """Compute conserved structure capacity Q.

    Q = dist_corr + 1.5 × |hierarchy|

    Higher Q indicates better structure learning.
    """
    return dist_corr + 1.5 * abs(hierarchy)


@dataclass
class TrainingMetrics:
    """Snapshot of metrics for LR decisions."""
    epoch: int
    coverage: float
    hierarchy_a: float
    hierarchy_b: float
    dist_corr_a: float = 0.0
    q_value: float = 0.0
    grad_norm_projections: float = 0.0

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


class ScheduleBasedLR(LRController):
    """Predetermined schedule for LR scales.

    Simplest approach: define (epoch, lr_scale) pairs for each component.
    Interpolates linearly between points.

    Example:
        controller = ScheduleBasedLR({
            'encoder_a': [(0, 0.0), (50, 0.05), (100, 0.0)],
            'encoder_b': [(0, 0.1), (80, 0.0)],
            'projections': [(0, 1.0)],
            'decoders': [(0, 1.0)],
        })
    """

    def __init__(
        self,
        schedules: Dict[str, List[Tuple[int, float]]],
        interpolate: bool = True,
    ):
        """Initialize schedule-based controller.

        Args:
            schedules: Dict mapping component name to list of (epoch, lr_scale)
            interpolate: If True, linearly interpolate between points
        """
        self.schedules = schedules
        self.interpolate = interpolate
        self._validate_schedules()

    def _validate_schedules(self):
        """Ensure schedules are sorted by epoch."""
        for name, schedule in self.schedules.items():
            if not schedule:
                raise ValueError(f"Schedule for {name} is empty")
            sorted_schedule = sorted(schedule, key=lambda x: x[0])
            self.schedules[name] = sorted_schedule

    def _get_scale_at_epoch(self, schedule: List[Tuple[int, float]], epoch: int) -> float:
        """Get LR scale at given epoch from schedule."""
        if epoch <= schedule[0][0]:
            return schedule[0][1]
        if epoch >= schedule[-1][0]:
            return schedule[-1][1]

        # Find surrounding points
        for i in range(len(schedule) - 1):
            e1, s1 = schedule[i]
            e2, s2 = schedule[i + 1]
            if e1 <= epoch < e2:
                if self.interpolate:
                    t = (epoch - e1) / (e2 - e1)
                    return s1 + t * (s2 - s1)
                return s1

        return schedule[-1][1]

    def get_lr_scales(self, metrics: TrainingMetrics) -> Dict[str, float]:
        """Get LR scales for current epoch."""
        return {
            name: self._get_scale_at_epoch(schedule, metrics.epoch)
            for name, schedule in self.schedules.items()
        }

    def update(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Return current scales (no state to update)."""
        scales = self.get_lr_scales(metrics)
        return {
            "lr_scales": scales,
            "events": [],
            "type": "schedule",
        }


class MetricBasedLR(LRController):
    """Metric-based LR control (soft version of StateNet logic).

    Replaces hard if/else thresholds with continuous functions:
    - Coverage gate: sigmoid(coverage - threshold) * base_scale
    - Hierarchy gate: based on plateau detection
    - Projections gate: based on gradient norm stability

    Key difference from StateNet: outputs are CONTINUOUS, not binary.
    """

    def __init__(self, config: StateNetConfig):
        """Initialize metric-based controller.

        Args:
            config: Centralized StateNet configuration
        """
        self.config = config

        # Rolling statistics (GPU-compatible)
        self._coverage_history: List[float] = []
        self._hierarchy_a_history: List[float] = []
        self._hierarchy_b_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._q_history: List[float] = []

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

    def _update_histories(self, metrics: TrainingMetrics):
        """Update rolling histories."""
        window = self.config.timing.window_size

        self._coverage_history.append(metrics.coverage)
        self._hierarchy_a_history.append(metrics.hierarchy_a)
        self._hierarchy_b_history.append(metrics.hierarchy_b)
        self._q_history.append(metrics.q_value)

        if metrics.grad_norm_projections > 0:
            self._grad_norm_history.append(metrics.grad_norm_projections)

        # Trim to window
        self._coverage_history = self._coverage_history[-window:]
        self._hierarchy_a_history = self._hierarchy_a_history[-window:]
        self._hierarchy_b_history = self._hierarchy_b_history[-window:]
        self._grad_norm_history = self._grad_norm_history[-window:]
        self._q_history = self._q_history[-(window * 2):]  # Longer for Q

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
            # Currently frozen - check if should activate
            if metrics.coverage >= cfg.train_threshold:
                # Also check hierarchy stall
                if self._is_hierarchy_a_stalled():
                    self._active['encoder_a'] = True
                    self._last_change['encoder_a'] = metrics.epoch
                    event = "encoder_a unfrozen (hierarchy stall)"
                    return (self.config.lr_scales.encoder_a, event)

        return (self.config.lr_scales.encoder_a if self._active['encoder_a'] else 0.0, event)

    def _is_hierarchy_a_stalled(self) -> bool:
        """Check if VAE-A hierarchy has plateaued."""
        window = self.config.timing.window_size
        if len(self._hierarchy_a_history) < window:
            return False

        recent = self._hierarchy_a_history[-window:]
        improvement = abs(recent[-1]) - abs(recent[0])

        if improvement < self.config.hierarchy.plateau_threshold:
            self._hierarchy_a_stall_count += 1
        else:
            self._hierarchy_a_stall_count = 0

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
                recent = self._hierarchy_b_history[-window:]
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
            # Check if hierarchy degraded
            if abs(self._hierarchy_b_history[-1]) < abs(self._hierarchy_b_history[-2]) - 0.01:
                self._active['encoder_b'] = True
                self._hierarchy_b_plateau_count = 0
                self._last_change['encoder_b'] = metrics.epoch
                event = "encoder_b unfrozen (hierarchy degraded)"
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
        _, cov_event = self._compute_coverage_gate(metrics)
        _, hier_event = self._compute_hierarchy_gate(metrics)
        _, proj_event = self._compute_projections_gate(metrics)

        if cov_event:
            events.append(cov_event)
        if hier_event:
            events.append(hier_event)
        if proj_event:
            events.append(proj_event)

        return {
            "lr_scales": self.get_lr_scales(metrics),
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


class LearnableLRController(nn.Module, LRController):
    """Learnable LR controller (differentiable, experimental).

    A small MLP that takes metrics and outputs LR multipliers.
    Can be trained end-to-end with the VAE using meta-learning.

    Warning: This is experimental and may destabilize training.
    """

    def __init__(
        self,
        n_metrics: int = 6,
        hidden_dim: int = 32,
        n_components: int = 4,
        temperature: float = 1.0,
    ):
        """Initialize learnable controller.

        Args:
            n_metrics: Number of input metrics
            hidden_dim: Hidden dimension for MLP
            n_components: Number of output LR scales
            temperature: Softmax temperature (higher = softer)
        """
        super().__init__()
        self.n_components = n_components
        self.temperature = temperature

        self.mlp = nn.Sequential(
            nn.Linear(n_metrics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_components),
        )

        # Initialize to output near-1.0 scales
        with torch.no_grad():
            self.mlp[-1].bias.fill_(2.0)  # sigmoid(2) ≈ 0.88

        self.component_names = ['encoder_a', 'encoder_b', 'projections', 'decoders']

    def forward(self, metrics_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass: metrics -> LR scales.

        Args:
            metrics_tensor: (batch, n_metrics) or (n_metrics,)

        Returns:
            LR scales in [0, 1], shape (batch, n_components) or (n_components,)
        """
        logits = self.mlp(metrics_tensor)
        return torch.sigmoid(logits / self.temperature)

    def get_lr_scales(self, metrics: TrainingMetrics) -> Dict[str, float]:
        """Get LR scales from metrics."""
        # Convert metrics to tensor
        metrics_tensor = torch.tensor([
            metrics.coverage,
            metrics.hierarchy_a,
            metrics.hierarchy_b,
            metrics.dist_corr_a,
            metrics.q_value,
            metrics.grad_norm_projections,
        ], dtype=torch.float32)

        with torch.no_grad():
            scales = self.forward(metrics_tensor)

        return {
            name: scales[i].item()
            for i, name in enumerate(self.component_names[:self.n_components])
        }

    def update(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Return scales (state is implicit in weights)."""
        scales = self.get_lr_scales(metrics)
        return {
            "lr_scales": scales,
            "events": [],
            "type": "learnable",
        }


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
    for group in optimizer.param_groups:
        name = group.get('name', '')
        if name in lr_scales:
            group['lr'] = base_lr * lr_scales[name]


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
    "ScheduleBasedLR",
    "MetricBasedLR",
    "LearnableLRController",
    "TrainingMetrics",
    "update_optimizer_lr_scales",
    "get_optimizer_grad_stats",
]
