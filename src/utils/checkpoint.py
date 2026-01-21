# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Checkpoint utility functions.

Provides helper functions for loading checkpoints safely and
extracting model state dictionaries.
"""

from typing import Any, Dict, Union
from pathlib import Path
import torch


def load_checkpoint_compat(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """Load checkpoint with compatibility for different formats.

    Args:
        checkpoint_path: Path to the .pt file
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint dictionary
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Use weights_only=False if needed for older PyTorch versions or complex dicts,
    # but prefer True for security if possible. For now, assume we need full pickle.
    return torch.load(path, map_location=map_location, weights_only=False)


def get_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model state dictionary from checkpoint.

    Handles different checkpoint structures (plain dict, 'model', 'model_state_dict').

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Model state dictionary
    """
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    else:
        # Assume the checkpoint itself is the state dict
        return checkpoint
