# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Learning Rate Controller - Unified Training Control via Optimizer.

Replaces StateNet's if/else decision logic with:
Metric-based control (soft gating via LR multipliers).

Philosophy (Option C Aggressive):
    - ALL trainability control happens via LR multipliers
    - LR=0 means "frozen", no separate requires_grad flag
    - Decisions are continuous (soft), not binary (hard)
    - Optimizer param groups are the single source of truth

Usage:
    from src.models import MetricBasedLR, TrainingMetrics

    # Metric-based (replaces StateNet logic)
    controller = MetricBasedLR(config)

    # In training loop
    metrics = TrainingMetrics(epoch, coverage, hierarchy_a, ...)
    state = controller.update(metrics)
    lr_scales = state['lr_scales']
    update_optimizer_lr_scales(optimizer, current_base_lr, lr_scales)
"""

import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    # Collapse flags (from Spearman NaN check)
    hierarchy_a_collapsed: bool = False
    hierarchy_b_collapsed: bool = False

    def __post_init__(self) -> None:
        # coverage is accuracy ∈ [0, 1]; correlations are ∈ [-1, 1]
        _ranges = {
            "coverage": (0.0, 1.0),
            "hierarchy_a": (-1.0, 1.0),
            "hierarchy_b": (-1.0, 1.0),
            "dist_corr_a": (-1.0, 1.0),
        }
        for name, (lo, hi) in _ranges.items():
            v = getattr(self, name)
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"TrainingMetrics.{name} must be a scalar float, got {type(v).__name__}"
                )
            if not (lo <= float(v) <= hi):
                raise ValueError(
                    f"TrainingMetrics.{name}={v} is out of range [{lo}, {hi}]. "
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
            hierarchy_a_collapsed=d.get('hierarchy_a_collapsed', d.get('hierarchy_collapsed', False)),
            hierarchy_b_collapsed=d.get('hierarchy_b_collapsed', False),
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
        window = config.timing.window_size
        self._coverage_history: deque[float] = deque(maxlen=window)
        self._hierarchy_a_history: deque[float] = deque(maxlen=window)
        self._hierarchy_b_history: deque[float] = deque(maxlen=window)
        self._grad_norm_history: deque[float] = deque(maxlen=window)
        self._q_history: deque[float] = deque(maxlen=window * 2)
        self._init_state()

    def _init_state(self) -> None:
        """Initialize (or reset) all mutable controller state."""
        cfg = self.config
        self._hierarchy_b_plateau_count = 0
        self._grad_low_count = 0
        self._hierarchy_a_stall_count = 0
        self._last_change = {
            'encoder_a': -cfg.timing.hysteresis_epochs,
            'encoder_b': -cfg.timing.hysteresis_epochs,
            'projections': -cfg.timing.hysteresis_epochs,
        }
        self._active = {
            'encoder_a': cfg.initial.encoder_a_trainable,
            'encoder_b': cfg.initial.encoder_b_trainable,
            'projections': cfg.initial.projections_trainable,
        }
        self._best_q = 0.0
        self._last_epoch = -1

    def _can_change(self, component: str, epoch: int) -> bool:
        """Check hysteresis constraint."""
        return epoch - self._last_change[component] >= self.config.timing.hysteresis_epochs

    def _update_histories(self, metrics: TrainingMetrics, delta: int) -> None:
        """Update rolling histories and advance plateau counters once per epoch."""
        self._coverage_history.append(metrics.coverage)
        self._hierarchy_a_history.append(metrics.hierarchy_a)
        self._hierarchy_b_history.append(metrics.hierarchy_b)
        self._q_history.append(metrics.q_value)

        if metrics.grad_norm_projections > 0:
            self._grad_norm_history.append(metrics.grad_norm_projections)

        # Advance stall counter here (single write path per epoch, not in gate methods)
        window = self.config.timing.window_size
        if len(self._hierarchy_a_history) >= window and not metrics.hierarchy_a_collapsed:
            recent_a = list(self._hierarchy_a_history)
            improvement_a = abs(recent_a[-1]) - abs(recent_a[0])
            if improvement_a < self.config.hierarchy.plateau_threshold:
                self._hierarchy_a_stall_count += delta
            else:
                self._hierarchy_a_stall_count = 0
        elif metrics.hierarchy_a_collapsed:
            # Collapse reset - don't count a collapse as a 'stall'
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
            # Never unfreeze based on hierarchy if the hierarchy is currently collapsed.
            coverage_ok = metrics.coverage >= cfg.train_threshold
            hierarchy_stalled = self._is_hierarchy_a_stalled() and not metrics.hierarchy_a_collapsed
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

    # Minimum net hierarchy decline over half-window to trigger encoder_b unfreeze.
    # plateau_threshold is ~0.0005 (stagnation scale); 0.02 is a meaningful drop on
    # the Spearman correlation scale that won't fire on noise.
    _HIERARCHY_DEGRADATION_THRESHOLD: float = 0.02

    def _compute_hierarchy_gate(self, metrics: TrainingMetrics, delta: int) -> Tuple[float, Optional[str]]:
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

                # Don't increment plateau count if collapsed - collapse is not a plateau
                if metrics.hierarchy_b_collapsed:
                    self._hierarchy_b_plateau_count = 0
                elif improvement < cfg.plateau_threshold:
                    self._hierarchy_b_plateau_count += delta
                else:
                    self._hierarchy_b_plateau_count = 0

                if self._hierarchy_b_plateau_count >= cfg.plateau_patience:
                    self._active['encoder_b'] = False
                    self._last_change['encoder_b'] = metrics.epoch
                    event = f"encoder_b frozen (plateau {self._hierarchy_b_plateau_count} epochs)"
                    return (0.0, event)
        else:
            # Unfreeze only on sustained degradation: net decline > threshold over window/2 steps.
            # Also unfreeze IF collapsed (collapse is the ultimate degradation).
            half_win = max(2, self.config.timing.window_size // 2)

            if metrics.hierarchy_b_collapsed:
                self._active['encoder_b'] = True
                self._hierarchy_b_plateau_count = 0
                self._last_change['encoder_b'] = metrics.epoch
                event = "encoder_b unfrozen (hierarchy collapsed)"
                return (self.config.lr_scales.encoder_b, event)

            if len(self._hierarchy_b_history) >= half_win:
                recent = list(self._hierarchy_b_history)[-half_win:]
                net_change = abs(recent[-1]) - abs(recent[0])
                if net_change < -self._HIERARCHY_DEGRADATION_THRESHOLD:
                    self._active['encoder_b'] = True
                    self._hierarchy_b_plateau_count = 0
                    self._last_change['encoder_b'] = metrics.epoch
                    event = "encoder_b unfrozen (sustained hierarchy degradation)"
                    return (self.config.lr_scales.encoder_b, event)

        return (self.config.lr_scales.encoder_b if self._active['encoder_b'] else 0.0, event)

    def _compute_projections_gate(self, metrics: TrainingMetrics, delta: int) -> Tuple[float, Optional[str]]:
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
                self._grad_low_count += delta
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
        """Get LR scales based on current metrics.

        WARNING: This method is NOT side-effect-free. The gate helpers it calls
        (_compute_coverage_gate, _compute_hierarchy_gate, _compute_projections_gate)
        may mutate self._active and self._last_change when threshold conditions are
        met.  Call update() instead in the training loop; reserve get_lr_scales()
        for read-only inspection only when you know the gates won't trigger.
        """
        # During warmup, use initial states
        if metrics.epoch < self.config.timing.warmup_epochs:
            return {
                'encoder_a': self.config.lr_scales.encoder_a if self.config.initial.encoder_a_trainable else 0.0,
                'encoder_b': self.config.lr_scales.encoder_b if self.config.initial.encoder_b_trainable else 0.0,
                'projections': self.config.lr_scales.projections if self.config.initial.projections_trainable else 0.0,
                'decoders': self.config.lr_scales.decoders,
            }

        cov_scale, _ = self._compute_coverage_gate(metrics)
        hier_scale, _ = self._compute_hierarchy_gate(metrics, delta=0)
        proj_scale, _ = self._compute_projections_gate(metrics, delta=0)

        return {
            'encoder_a': cov_scale,
            'encoder_b': hier_scale,
            'projections': proj_scale,
            'decoders': self.config.lr_scales.decoders,  # Always full
        }

    def update(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Update state and return status with events."""
        # Calculate delta (epochs since last update)
        if self._last_epoch == -1:
            delta = 1
        else:
            delta = metrics.epoch - self._last_epoch
        self._last_epoch = metrics.epoch

        self._update_histories(metrics, delta)

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
        hier_scale, hier_event = self._compute_hierarchy_gate(metrics, delta)
        proj_scale, proj_event = self._compute_projections_gate(metrics, delta)

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

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state."""
        return {
            "coverage_history": list(self._coverage_history),
            "hierarchy_a_history": list(self._hierarchy_a_history),
            "hierarchy_b_history": list(self._hierarchy_b_history),
            "grad_norm_history": list(self._grad_norm_history),
            "q_history": list(self._q_history),
            "hierarchy_b_plateau_count": self._hierarchy_b_plateau_count,
            "grad_low_count": self._grad_low_count,
            "hierarchy_a_stall_count": self._hierarchy_a_stall_count,
            "last_change": dict(self._last_change),
            "active": dict(self._active),
            "best_q": self._best_q,
            "last_epoch": self._last_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from dict."""
        for attr, key in [
            ("_coverage_history", "coverage_history"),
            ("_hierarchy_a_history", "hierarchy_a_history"),
            ("_hierarchy_b_history", "hierarchy_b_history"),
            ("_grad_norm_history", "grad_norm_history"),
            ("_q_history", "q_history"),
        ]:
            buf = getattr(self, attr)
            buf.clear()
            buf.extend(state.get(key, []))

        self._hierarchy_b_plateau_count = state.get("hierarchy_b_plateau_count", 0)
        self._grad_low_count = state.get("grad_low_count", 0)
        self._hierarchy_a_stall_count = state.get("hierarchy_a_stall_count", 0)

        self._last_change = dict(state.get("last_change", self._last_change))
        self._active = dict(state.get("active", self._active))
        self._best_q = state.get("best_q", 0.0)
        self._last_epoch = state.get("last_epoch", -1)

    def reset(self) -> None:
        """Reset all state to initial values."""
        for buf in (
            self._coverage_history, self._hierarchy_a_history,
            self._hierarchy_b_history, self._grad_norm_history, self._q_history,
        ):
            buf.clear()
        self._init_state()


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
    matched_scales = set()
    unmatched_groups = []
    for group in optimizer.param_groups:
        name = group.get('name', '')
        if name in lr_scales:
            group['lr'] = base_lr * lr_scales[name]
            matched_scales.add(name)
            # Sync requires_grad atomically with the LR update.
            # PyTorch computes gradients for all requires_grad=True params regardless
            # of the optimizer LR; setting requires_grad=False prevents wasted compute
            # when a component is frozen (LR=0).
            is_active = group['lr'] > 0
            for p in group['params']:
                p.requires_grad = is_active
        elif name:
            unmatched_groups.append(name)

    unmatched_scales = set(lr_scales) - matched_scales
    if unmatched_scales:
        warnings.warn(
            f"update_optimizer_lr_scales: lr_scales keys {unmatched_scales} matched no "
            f"optimizer param group. LR control is silently inactive for these components. "
            f"Group names present in optimizer: {[g.get('name','<unnamed>') for g in optimizer.param_groups]}",
            UserWarning,
            stacklevel=2,
        )

    if unmatched_groups:
        warnings.warn(
            f"update_optimizer_lr_scales: Optimizer param groups {unmatched_groups} "
            f"have no corresponding scale in lr_scales {list(lr_scales.keys())}. "
            f"These groups will stay at the base learning rate (cosine-annealed).",
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
