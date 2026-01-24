# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project.

Exports project paths and StateNet constants.
"""

from .constants import (
    # General
    N_TERNARY_OPERATIONS,
    # StateNet thresholds
    STATENET_COVERAGE_FIX_THRESHOLD,
    STATENET_COVERAGE_TRAIN_THRESHOLD,
    STATENET_COVERAGE_FLOOR,
    STATENET_WARMUP_EPOCHS,
    STATENET_HYSTERESIS_EPOCHS,
    STATENET_WINDOW_SIZE,
    # StateNet annealing
    STATENET_ANNEALING_STEP,
    STATENET_HIERARCHY_PLATEAU_THRESHOLD,
    STATENET_HIERARCHY_PLATEAU_PATIENCE,
    STATENET_HIERARCHY_PATIENCE_CEILING,
    STATENET_CONTROLLER_GRAD_THRESHOLD,
    STATENET_CONTROLLER_GRAD_PATIENCE,
    STATENET_CONTROLLER_PATIENCE_CEILING,
)
from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR, MODELS_DIR, SRC_PRESETS_DIR
