# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Checkpoint validation utilities.

Provides validation functions to prevent common training pipeline issues,
particularly the 0% coverage problem with V6+ architectures that require
anchor checkpoints for proper initialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class CheckpointCompatibilityError(Exception):
    """Raised when checkpoint is incompatible with model architecture."""
    pass


class CheckpointValidator:
    """Utility class for validating checkpoints and configurations."""

    # Known good checkpoints for V6+ architectures
    # These paths are relative to PROJECT_ROOT
    RECOMMENDED_CHECKPOINTS = {
        "TernaryVAEV6": "models/checkpoints/v5_12/v5_12_4/best_Q.pt",
        "TernaryVAEV6Controllable": "models/checkpoints/v5_12/v5_12_4/best_Q.pt",
    }

    @classmethod
    def validate_checkpoint_exists(cls, path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Check if checkpoint file exists.

        Args:
            path: Path to checkpoint file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if path is None or str(path).lower() == 'null':
            return False, "Checkpoint path is null or None"

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return False, f"Checkpoint file not found: {checkpoint_path}"

        return True, None

    @classmethod
    def validate_checkpoint_dimensions(
        cls,
        checkpoint_path: Union[str, Path],
        model: nn.Module
    ) -> Tuple[bool, List[str]]:
        """Validate checkpoint dimensions against model architecture.

        Args:
            checkpoint_path: Path to the checkpoint file
            model: Target model to load into

        Returns:
            Tuple of (is_compatible, list of error messages)
        """
        errors = []
        path = Path(checkpoint_path)

        if not path.exists():
            return False, [f"Checkpoint not found: {path}"]

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Extract state dict
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Get model state dict for comparison
            model_state = model.state_dict()

            # Check for dimension mismatches in overlapping keys
            for key in state_dict:
                if key in model_state:
                    ckpt_shape = state_dict[key].shape
                    model_shape = model_state[key].shape
                    if ckpt_shape != model_shape:
                        errors.append(
                            f"Dimension mismatch for '{key}': "
                            f"checkpoint={ckpt_shape}, model={model_shape}"
                        )

        except Exception as e:
            errors.append(f"Failed to load checkpoint: {e}")

        return len(errors) == 0, errors

    @classmethod
    def fix_null_checkpoint_config(
        cls,
        config: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Attempt to fix null checkpoint configuration.

        For V6+ architectures, provides a default checkpoint path
        to prevent the 0% coverage issue.

        Args:
            config: Training configuration dictionary
            model_name: Optional model name override

        Returns:
            Modified configuration with fixed checkpoint path
        """
        config = config.copy()

        # Determine model name
        if model_name is None:
            model_name = config.get("model", {}).get("name", "")

        # Check if this model needs an anchor checkpoint
        if model_name in cls.RECOMMENDED_CHECKPOINTS:
            anchor_cfg = config.get("anchor_checkpoint", {})
            current_path = anchor_cfg.get("path")

            if current_path is None or str(current_path).lower() == "null":
                if "anchor_checkpoint" not in config:
                    config["anchor_checkpoint"] = {}
                config["anchor_checkpoint"]["path"] = cls.RECOMMENDED_CHECKPOINTS[model_name]
                config["anchor_checkpoint"]["auto_fixed"] = True

        return config

    @classmethod
    def get_checkpoint_info(cls, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata
        """
        path = Path(checkpoint_path)
        info = {"path": str(path), "exists": path.exists()}

        if not path.exists():
            return info

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Extract available metadata
            info["keys"] = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["raw_state_dict"]
            info["has_model_state_dict"] = "model_state_dict" in checkpoint
            info["has_optimizer"] = "optimizer_state_dict" in checkpoint

            if "epoch" in checkpoint:
                info["epoch"] = checkpoint["epoch"]
            if "metrics" in checkpoint:
                info["metrics"] = checkpoint["metrics"]

            # Get state dict size info
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                state_dict = {}

            info["num_parameters"] = len(state_dict)
            info["total_params"] = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))

        except Exception as e:
            info["error"] = str(e)

        return info


def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate training configuration for common issues.

    Checks for:
    - Null checkpoint paths for V6+ architectures
    - Missing required configuration sections
    - Invalid hyperparameter values

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

        # Validate if path exists
        elif checkpoint_path:
            is_valid, error = CheckpointValidator.validate_checkpoint_exists(checkpoint_path)
            if not is_valid:
                errors.append(error)

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
        coverage_fix = statenet_cfg.get("coverage_fix_threshold", 0.99)
        coverage_train = statenet_cfg.get("coverage_train_threshold", 0.999)

        if coverage_fix >= coverage_train:
            errors.append(
                f"StateNet thresholds invalid: coverage_fix_threshold ({coverage_fix}) "
                f"must be < coverage_train_threshold ({coverage_train})"
            )

    return len(errors) == 0, errors
