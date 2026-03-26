# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project.

Exports project paths and centralized StateNet configuration.
"""

from .constants import N_TERNARY_OPERATIONS
from .paths import CHECKPOINTS_DIR, MODELS_DIR, PROJECT_ROOT, RUNS_DIR, SRC_PRESETS_DIR

# Centralized configuration (single source of truth)
from .statenet_config import (
    ControllerThresholds,
    CoverageThresholds,
    HierarchyThresholds,
    InitialStates,
    LRScales,
    StateNetConfig,
    TimingConfig,
)

# Pydantic schema validation (V7.2+)
from .schema import (
    TrainingConfigSchema,
    validate_config,
    validate,
    normalize_config,
    validate_and_normalize,
    # Sub-schemas for granular validation
    AngularCoherenceLossConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    VisualizationConfig,
)

__all__ = [
    # Constants
    "N_TERNARY_OPERATIONS",
    # Paths
    "PROJECT_ROOT",
    "RUNS_DIR",
    "CHECKPOINTS_DIR",
    "MODELS_DIR",
    "SRC_PRESETS_DIR",
    # StateNet Configuration (dataclass)
    "StateNetConfig",
    "CoverageThresholds",
    "HierarchyThresholds",
    "ControllerThresholds",
    "TimingConfig",
    "LRScales",
    "InitialStates",
    # Pydantic Schema Validation
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
