# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project."""

from .constants import N_TERNARY_OPERATIONS
from .paths import CHECKPOINTS_DIR, MODELS_DIR, PROJECT_ROOT, RUNS_DIR, SRC_PRESETS_DIR
from .schema import (
    AngularCoherenceLossConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    TrainingConfigSchema,
    VisualizationConfig,
    normalize_config,
    validate,
    validate_and_normalize,
    validate_config,
)
from .statenet_config import (
    ControllerThresholds,
    CoverageThresholds,
    HierarchyThresholds,
    InitialStates,
    LRScales,
    StateNetConfig,
    TimingConfig,
)

__all__ = [
    "N_TERNARY_OPERATIONS",
    "PROJECT_ROOT",
    "RUNS_DIR",
    "CHECKPOINTS_DIR",
    "MODELS_DIR",
    "SRC_PRESETS_DIR",
    "StateNetConfig",
    "CoverageThresholds",
    "HierarchyThresholds",
    "ControllerThresholds",
    "TimingConfig",
    "LRScales",
    "InitialStates",
    "TrainingConfigSchema",
    "validate_config",
    "validate",
    "normalize_config",
    "validate_and_normalize",
    "AngularCoherenceLossConfig",
    "LossConfig",
    "ModelConfig",
    "TrainingConfig",
    "VisualizationConfig",
]
