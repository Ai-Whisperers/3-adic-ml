# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized StateNet Configuration.

Single source of truth for all training control parameters.
Replaces scattered constants across config files, statenet.py, and train.py.

Usage:
    from src.config import StateNetConfig

    # From YAML config dict
    config = StateNetConfig.from_dict(yaml_config.get('statenet', {}))

    # Direct construction with defaults
    config = StateNetConfig()

    # Access via dot notation
    print(config.coverage_fix_threshold)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CoverageThresholds:
    """Thresholds for encoder_A (coverage-gated) control."""
    fix_threshold: float = 0.995      # Below this: freeze encoder_A
    train_threshold: float = 1.0      # Above this + stalled: unfreeze
    floor: float = 0.95               # Minimum allowed (annealing limit)


@dataclass
class HierarchyThresholds:
    """Thresholds for encoder_B (hierarchy-gated) control."""
    plateau_threshold: float = 0.0005  # Improvement below this = plateau
    plateau_patience: int = 10         # Epochs of plateau before freeze
    patience_ceiling: int = 25         # Max patience (annealing limit)
    stall_patience: int = 5            # For encoder_A hierarchy stall detection


@dataclass
class ControllerThresholds:
    """Thresholds for projections (gradient-gated) control."""
    grad_threshold: float = 0.005     # Grad norm below this = stable
    grad_patience: int = 5            # Epochs of low grad before freeze
    patience_ceiling: int = 20        # Max patience (annealing limit)
    spike_multiplier: float = 2.0     # Grad spike = current > avg * this


@dataclass
class AnnealingConfig:
    """Q-gated annealing configuration."""
    enabled: bool = True
    step: float = 0.002               # Threshold adjustment step
    q_decrease_threshold: float = -0.05  # Q drop triggering tightening


@dataclass
class TimingConfig:
    """Timing and window configuration."""
    warmup_epochs: int = 10           # Skip StateNet decisions during warmup
    hysteresis_epochs: int = 5        # Min epochs between state changes
    window_size: int = 10             # Moving window for metric history


@dataclass
class LRScales:
    """Learning rate multipliers per component."""
    encoder_a: float = 0.05           # Slow learner (coverage anchor)
    encoder_b: float = 0.10           # Medium learner (hierarchy)
    projections: float = 1.0          # Fast adapter
    decoders: float = 1.0             # Always full LR


@dataclass
class InitialStates:
    """Initial trainability states."""
    encoder_a_trainable: bool = False  # Starts frozen (coverage anchor)
    encoder_b_trainable: bool = True   # Starts trainable
    projections_trainable: bool = True # Starts trainable


@dataclass
class StateNetConfig:
    """Centralized configuration for StateNet controller.

    Consolidates all 17+ scattered thresholds into a single dataclass.
    Provides type safety, defaults, and easy YAML serialization.

    Example:
        config = StateNetConfig.from_dict({
            'enabled': True,
            'coverage_fix_threshold': 0.99,
            'warmup_epochs': 15,
        })
    """
    # Enable/disable StateNet
    enabled: bool = True

    # Component thresholds (grouped)
    coverage: CoverageThresholds = field(default_factory=CoverageThresholds)
    hierarchy: HierarchyThresholds = field(default_factory=HierarchyThresholds)
    controller: ControllerThresholds = field(default_factory=ControllerThresholds)

    # Annealing
    annealing: AnnealingConfig = field(default_factory=AnnealingConfig)

    # Timing
    timing: TimingConfig = field(default_factory=TimingConfig)

    # Learning rates
    lr_scales: LRScales = field(default_factory=LRScales)

    # Initial states
    initial: InitialStates = field(default_factory=InitialStates)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateNetConfig":
        """Create config from dictionary (e.g., from YAML).

        Supports both flat keys (legacy) and nested structure.

        Flat keys (legacy compatibility):
            coverage_fix_threshold, coverage_train_threshold, coverage_floor,
            hierarchy_plateau_threshold, hierarchy_plateau_patience, etc.

        Nested structure (preferred):
            coverage:
              fix_threshold: 0.995
            hierarchy:
              plateau_patience: 10
        """
        config = cls()

        # Enable flag
        config.enabled = d.get('enabled', config.enabled)

        # Coverage thresholds (flat or nested)
        if 'coverage' in d and isinstance(d['coverage'], dict):
            cov = d['coverage']
            config.coverage.fix_threshold = cov.get('fix_threshold', config.coverage.fix_threshold)
            config.coverage.train_threshold = cov.get('train_threshold', config.coverage.train_threshold)
            config.coverage.floor = cov.get('floor', config.coverage.floor)
        else:
            # Legacy flat keys
            config.coverage.fix_threshold = d.get('coverage_fix_threshold', config.coverage.fix_threshold)
            config.coverage.train_threshold = d.get('coverage_train_threshold', config.coverage.train_threshold)
            config.coverage.floor = d.get('coverage_floor', config.coverage.floor)

        # Hierarchy thresholds
        if 'hierarchy' in d and isinstance(d['hierarchy'], dict):
            hier = d['hierarchy']
            config.hierarchy.plateau_threshold = hier.get('plateau_threshold', config.hierarchy.plateau_threshold)
            config.hierarchy.plateau_patience = hier.get('plateau_patience', config.hierarchy.plateau_patience)
            config.hierarchy.patience_ceiling = hier.get('patience_ceiling', config.hierarchy.patience_ceiling)
            config.hierarchy.stall_patience = hier.get('stall_patience', config.hierarchy.stall_patience)
        else:
            config.hierarchy.plateau_threshold = d.get('hierarchy_plateau_threshold', config.hierarchy.plateau_threshold)
            config.hierarchy.plateau_patience = d.get('hierarchy_plateau_patience', config.hierarchy.plateau_patience)
            config.hierarchy.patience_ceiling = d.get('hierarchy_patience_ceiling', config.hierarchy.patience_ceiling)

        # Controller thresholds
        if 'controller' in d and isinstance(d['controller'], dict):
            ctrl = d['controller']
            config.controller.grad_threshold = ctrl.get('grad_threshold', config.controller.grad_threshold)
            config.controller.grad_patience = ctrl.get('grad_patience', config.controller.grad_patience)
            config.controller.patience_ceiling = ctrl.get('patience_ceiling', config.controller.patience_ceiling)
            config.controller.spike_multiplier = ctrl.get('spike_multiplier', config.controller.spike_multiplier)
        else:
            config.controller.grad_threshold = d.get('controller_grad_threshold', config.controller.grad_threshold)
            config.controller.grad_patience = d.get('controller_grad_patience', config.controller.grad_patience)
            config.controller.patience_ceiling = d.get('controller_patience_ceiling', config.controller.patience_ceiling)

        # Annealing
        if 'annealing' in d and isinstance(d['annealing'], dict):
            ann = d['annealing']
            config.annealing.enabled = ann.get('enabled', config.annealing.enabled)
            config.annealing.step = ann.get('step', config.annealing.step)
            config.annealing.q_decrease_threshold = ann.get('q_decrease_threshold', config.annealing.q_decrease_threshold)
        else:
            config.annealing.enabled = d.get('enable_annealing', config.annealing.enabled)
            config.annealing.step = d.get('annealing_step', config.annealing.step)

        # Timing
        if 'timing' in d and isinstance(d['timing'], dict):
            tim = d['timing']
            config.timing.warmup_epochs = tim.get('warmup_epochs', config.timing.warmup_epochs)
            config.timing.hysteresis_epochs = tim.get('hysteresis_epochs', config.timing.hysteresis_epochs)
            config.timing.window_size = tim.get('window_size', config.timing.window_size)
        else:
            config.timing.warmup_epochs = d.get('warmup_epochs', config.timing.warmup_epochs)
            config.timing.hysteresis_epochs = d.get('hysteresis_epochs', config.timing.hysteresis_epochs)

        # LR scales (from option_c or nested)
        if 'lr_scales' in d and isinstance(d['lr_scales'], dict):
            lr = d['lr_scales']
            config.lr_scales.encoder_a = lr.get('encoder_a', config.lr_scales.encoder_a)
            config.lr_scales.encoder_b = lr.get('encoder_b', config.lr_scales.encoder_b)
            config.lr_scales.projections = lr.get('projections', config.lr_scales.projections)
            config.lr_scales.decoders = lr.get('decoders', config.lr_scales.decoders)
        else:
            config.lr_scales.encoder_a = d.get('encoder_a_lr_scale', config.lr_scales.encoder_a)
            config.lr_scales.encoder_b = d.get('encoder_b_lr_scale', config.lr_scales.encoder_b)
            config.lr_scales.projections = d.get('projections_lr_scale', config.lr_scales.projections)

        # Initial states
        if 'initial' in d and isinstance(d['initial'], dict):
            init = d['initial']
            config.initial.encoder_a_trainable = init.get('encoder_a_trainable', config.initial.encoder_a_trainable)
            config.initial.encoder_b_trainable = init.get('encoder_b_trainable', config.initial.encoder_b_trainable)
            config.initial.projections_trainable = init.get('projections_trainable', config.initial.projections_trainable)
        else:
            config.initial.encoder_a_trainable = d.get('encoder_a_trainable', config.initial.encoder_a_trainable)
            config.initial.encoder_b_trainable = d.get('encoder_b_trainable', config.initial.encoder_b_trainable)
            config.initial.projections_trainable = d.get('projections_trainable', config.initial.projections_trainable)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML/JSON export."""
        return {
            'enabled': self.enabled,
            'coverage': {
                'fix_threshold': self.coverage.fix_threshold,
                'train_threshold': self.coverage.train_threshold,
                'floor': self.coverage.floor,
            },
            'hierarchy': {
                'plateau_threshold': self.hierarchy.plateau_threshold,
                'plateau_patience': self.hierarchy.plateau_patience,
                'patience_ceiling': self.hierarchy.patience_ceiling,
                'stall_patience': self.hierarchy.stall_patience,
            },
            'controller': {
                'grad_threshold': self.controller.grad_threshold,
                'grad_patience': self.controller.grad_patience,
                'patience_ceiling': self.controller.patience_ceiling,
                'spike_multiplier': self.controller.spike_multiplier,
            },
            'annealing': {
                'enabled': self.annealing.enabled,
                'step': self.annealing.step,
                'q_decrease_threshold': self.annealing.q_decrease_threshold,
            },
            'timing': {
                'warmup_epochs': self.timing.warmup_epochs,
                'hysteresis_epochs': self.timing.hysteresis_epochs,
                'window_size': self.timing.window_size,
            },
            'lr_scales': {
                'encoder_a': self.lr_scales.encoder_a,
                'encoder_b': self.lr_scales.encoder_b,
                'projections': self.lr_scales.projections,
                'decoders': self.lr_scales.decoders,
            },
            'initial': {
                'encoder_a_trainable': self.initial.encoder_a_trainable,
                'encoder_b_trainable': self.initial.encoder_b_trainable,
                'projections_trainable': self.initial.projections_trainable,
            },
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"StateNetConfig(enabled={self.enabled}, "
            f"cov_fix={self.coverage.fix_threshold:.3f}, "
            f"hier_pat={self.hierarchy.plateau_patience}, "
            f"ctrl_pat={self.controller.grad_patience}, "
            f"warmup={self.timing.warmup_epochs})"
        )


__all__ = [
    "StateNetConfig",
    "CoverageThresholds",
    "HierarchyThresholds",
    "ControllerThresholds",
    "AnnealingConfig",
    "TimingConfig",
    "LRScales",
    "InitialStates",
]
