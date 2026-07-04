# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Checkpoint utility functions.

Provides helper functions for loading checkpoints safely and
extracting model state dictionaries.
"""

from pathlib import Path
from typing import Any, Dict, Union

import torch


def load_checkpoint_compat(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """Load a .pt checkpoint file, raising FileNotFoundError if missing."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Use weights_only=False if needed for older PyTorch versions or complex dicts,
    # but prefer True for security if possible. For now, assume we need full pickle.
    return torch.load(path, map_location=map_location, weights_only=False)


def get_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model state dict from checkpoint (handles 'model_state_dict', 'model', or plain dict)."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    else:
        # Assume the checkpoint itself is the state dict
        return checkpoint
