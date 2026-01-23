# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration module for TernaryVAE project.

Exports project paths and StateNet constants.
"""

from .constants import (
    STATENET_COVERAGE_FREEZE_THRESHOLD,
    STATENET_COVERAGE_UNFREEZE_THRESHOLD,
    N_TERNARY_OPERATIONS,
)
from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR
