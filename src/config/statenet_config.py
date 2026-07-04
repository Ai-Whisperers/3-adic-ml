# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Centralized StateNet Configuration.

Single source of truth for all training control parameters.

Usage:
    from src.config import StateNetConfig

    # From YAML config dict
    config = StateNetConfig.from_dict(yaml_config.get('statenet', {}))

    # Direct construction with defaults
    config = StateNetConfig()

    # Access via dot notation
    print(config.coverage.fix_threshold)
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict


def _apply_section(obj: Any, data: Dict[str, Any]) -> None:
    """Apply values from data dict to a dataclass object (existing values as defaults)."""
    for f in fields(obj):
        if f.name in data:
            setattr(obj, f.name, data[f.name])


@dataclass
class CoverageThresholds:
    """Thresholds for encoder_A (coverage-gated) control.

    All thresholds are compared against per-digit reconstruction accuracy in [0, 1].
    Invariant: 0 <= fix_threshold < train_threshold <= 1.0
    """
    fix_threshold: float = 0.35       # Below this: freeze encoder_A (near random=0.33)
    train_threshold: float = 0.45     # Above this + stalled: unfreeze
    floor: float = 0.3                # Minimum allowed (annealing limit)

    def __post_init__(self) -> None:
        if not (0.0 <= self.fix_threshold < self.train_threshold <= 1.0):
            raise ValueError(
                f"CoverageThresholds: fix_threshold={self.fix_threshold} must be < "
                f"train_threshold={self.train_threshold}, both in [0, 1]. "
                f"These compare against per-digit reconstruction accuracy."
            )


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
class TimingConfig:
    """Timing and window configuration."""
    warmup_epochs: int = 10           # Skip StateNet decisions during warmup
    hysteresis_epochs: int = 5        # Min epochs between state changes
    window_size: int = 10             # Moving window for metric history


@dataclass
class LRScales:
    """Learning rate multipliers per component. All must be in (0, 1]."""
    encoder_a: float = 0.05           # Slow learner (coverage anchor)
    encoder_b: float = 0.10           # Medium learner (hierarchy)
    projections: float = 1.0          # Fast adapter
    decoders: float = 1.0             # Always full LR

    def __post_init__(self) -> None:
        for name in ("encoder_a", "encoder_b", "projections", "decoders"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError(
                    f"LRScales.{name}={v} must be in (0, 1]. "
                    f"Use statenet.enabled=false to disable the controller entirely."
                )


@dataclass
class InitialStates:
    """Initial trainability states."""
    encoder_a_trainable: bool = False  # Starts frozen (coverage anchor)
    encoder_b_trainable: bool = True   # Starts trainable
    projections_trainable: bool = True # Starts trainable


@dataclass
class StateNetConfig:
    """Centralized configuration for LR Controller.

    Consolidates all training control thresholds into a single dataclass.
    Provides type safety, defaults, and easy YAML serialization.

    YAML format (nested structure required):
        statenet:
          enabled: true
          coverage:
            fix_threshold: 0.35
            train_threshold: 0.45
          hierarchy:
            plateau_patience: 10
          timing:
            warmup_epochs: 10
    """
    # Enable/disable training controller
    enabled: bool = True

    # Component thresholds (grouped)
    coverage: CoverageThresholds = field(default_factory=CoverageThresholds)
    hierarchy: HierarchyThresholds = field(default_factory=HierarchyThresholds)
    controller: ControllerThresholds = field(default_factory=ControllerThresholds)

    # Timing
    timing: TimingConfig = field(default_factory=TimingConfig)

    # Learning rates
    lr_scales: LRScales = field(default_factory=LRScales)

    # Initial states
    initial: InitialStates = field(default_factory=InitialStates)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateNetConfig":
        """Create StateNetConfig from a nested YAML config dict."""
        _known_keys = {'enabled', 'coverage', 'hierarchy', 'controller', 'timing', 'lr_scales', 'initial'}
        unknown = set(d.keys()) - _known_keys
        if unknown:
            raise ValueError(
                f"StateNetConfig.from_dict: unknown config keys {unknown}. "
                f"Valid keys: {_known_keys}. Check for typos in your YAML."
            )

        _sub_keys: Dict[str, set] = {
            'coverage':   {'fix_threshold', 'train_threshold', 'floor'},
            'hierarchy':  {'plateau_threshold', 'plateau_patience', 'patience_ceiling', 'stall_patience'},
            'controller': {'grad_threshold', 'grad_patience', 'patience_ceiling', 'spike_multiplier'},
            'timing':     {'warmup_epochs', 'hysteresis_epochs', 'window_size'},
            'lr_scales':  {'encoder_a', 'encoder_b', 'projections', 'decoders'},
            'initial':    {'encoder_a_trainable', 'encoder_b_trainable', 'projections_trainable', 'decoders_trainable'},
        }
        for section, known in _sub_keys.items():
            if section in d and isinstance(d[section], dict):
                unknown_sub = set(d[section].keys()) - known
                if unknown_sub:
                    raise ValueError(
                        f"StateNetConfig.from_dict: unknown keys in '{section}': {unknown_sub}. "
                        f"Valid keys: {known}. Check for typos in your YAML."
                    )

        config = cls()
        config.enabled = d.get('enabled', config.enabled)
        for attr in ('coverage', 'hierarchy', 'controller', 'timing', 'lr_scales', 'initial'):
            if attr in d and isinstance(d[attr], dict):
                _apply_section(getattr(config, attr), d[attr])
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
    "TimingConfig",
    "LRScales",
    "InitialStates",
]
