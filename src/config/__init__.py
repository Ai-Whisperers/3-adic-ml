# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project.

Exports project paths and centralized StateNet configuration.
"""

from .constants import N_TERNARY_OPERATIONS
from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR, MODELS_DIR, SRC_PRESETS_DIR

# Centralized configuration (single source of truth)
from .statenet_config import (
    StateNetConfig,
    CoverageThresholds,
    HierarchyThresholds,
    ControllerThresholds,
    TimingConfig,
    LRScales,
    InitialStates,
)
