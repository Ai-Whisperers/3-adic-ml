# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project.

Exports project paths, StateNet constants, and centralized configuration.

V6.0: Added StateNetConfig dataclass for centralized threshold management.
"""

from .constants import (
    # General
    N_TERNARY_OPERATIONS,
    # StateNet thresholds (legacy - prefer StateNetConfig)
    STATENET_COVERAGE_FIX_THRESHOLD,
    STATENET_COVERAGE_TRAIN_THRESHOLD,
    STATENET_COVERAGE_FLOOR,
    STATENET_WARMUP_EPOCHS,
    STATENET_HYSTERESIS_EPOCHS,
    STATENET_WINDOW_SIZE,
    # StateNet annealing (legacy - prefer StateNetConfig)
    STATENET_ANNEALING_STEP,
    STATENET_HIERARCHY_PLATEAU_THRESHOLD,
    STATENET_HIERARCHY_PLATEAU_PATIENCE,
    STATENET_HIERARCHY_PATIENCE_CEILING,
    STATENET_CONTROLLER_GRAD_THRESHOLD,
    STATENET_CONTROLLER_GRAD_PATIENCE,
    STATENET_CONTROLLER_PATIENCE_CEILING,
)
from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR, MODELS_DIR, SRC_PRESETS_DIR

# V6.0: Centralized configuration (preferred over individual constants)
from .statenet_config import (
    StateNetConfig,
    CoverageThresholds,
    HierarchyThresholds,
    ControllerThresholds,
    AnnealingConfig,
    TimingConfig,
    LRScales,
    InitialStates,
)
