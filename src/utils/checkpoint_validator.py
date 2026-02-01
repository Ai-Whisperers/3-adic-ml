# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training configuration validation utilities.

Provides validation functions to catch common configuration issues
before training starts.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate training configuration for common issues.

    Checks for:
    - Missing required configuration sections
    - Invalid hyperparameter values
    - Null checkpoint paths for V6+ architectures
    - Invalid StateNet threshold relationships

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check for required sections
    if "model" not in config:
        errors.append("Missing 'model' section in configuration")
        return False, errors

    model_config = config.get("model", {})
    model_name = model_config.get("name", "")

    # V6+ architecture checks
    v6_models = ["TernaryVAEV6", "TernaryVAEV6Controllable"]

    if model_name in v6_models:
        # Check anchor checkpoint configuration
        anchor_cfg = config.get("anchor_checkpoint", {})
        checkpoint_path = anchor_cfg.get("path")

        if checkpoint_path is None or str(checkpoint_path).lower() == "null":
            errors.append(
                f"Model '{model_name}' requires an anchor checkpoint for proper initialization. "
                f"Set 'anchor_checkpoint.path' to a valid checkpoint."
            )
        elif checkpoint_path:
            # Validate path exists
            path = Path(checkpoint_path)
            if not path.exists():
                errors.append(f"Anchor checkpoint not found: {checkpoint_path}")

    # Check training configuration
    training_cfg = config.get("training", {})

    if "epochs" in training_cfg:
        epochs = training_cfg["epochs"]
        if not isinstance(epochs, int) or epochs < 1:
            errors.append(f"Invalid 'training.epochs': must be a positive integer, got {epochs}")

    if "learning_rate" in training_cfg:
        lr = training_cfg["learning_rate"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"Invalid 'training.learning_rate': must be positive, got {lr}")

    if "batch_size" in training_cfg:
        batch_size = training_cfg["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append(f"Invalid 'training.batch_size': must be a positive integer, got {batch_size}")

    # StateNet configuration checks
    statenet_cfg = config.get("statenet", {})
    if statenet_cfg.get("enabled", False):
        coverage_cfg = statenet_cfg.get("coverage", {})
        coverage_fix = coverage_cfg.get("fix_threshold", 0.995)
        coverage_train = coverage_cfg.get("train_threshold", 1.0)

        if coverage_fix >= coverage_train:
            errors.append(
                f"StateNet thresholds invalid: coverage.fix_threshold ({coverage_fix}) "
                f"must be < coverage.train_threshold ({coverage_train})"
            )

    return len(errors) == 0, errors


__all__ = ["validate_training_config"]
