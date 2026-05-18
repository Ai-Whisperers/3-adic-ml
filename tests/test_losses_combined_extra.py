# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Extra CombinedLoss Tests for edge cases and configuration validation."""

import pytest
import torch
from src.losses.combined import CombinedLoss

def test_combined_loss_all_disabled_raises_error():
    """Verify CombinedLoss raises ValueError if no loss is enabled."""
    config = {
        "rich_hierarchy": {"enabled": False},
        "radial": {"enabled": False},
        "geodesic": {"enabled": False},
        "rank": {"enabled": False},
        "monotonic": {"enabled": False},
        "angular_coherence": {"enabled": False},
        "algebraic_coherence": {"enabled": False},
        "algebraic_addition": {"enabled": False},
        "algebraic_multiplication": {"enabled": False},
        "hyperbolic_kl": {"enabled": False},
    }
    
    with pytest.raises(ValueError, match="CombinedLoss: all losses are disabled"):
        CombinedLoss(config, device=torch.device("cpu"))

def test_combined_loss_negative_weight_raises_error():
    """Verify CombinedLoss raises ValueError for negative weights."""
    config = {
        "geodesic": {"enabled": True, "weight": -1.0}
    }
    
    with pytest.raises(ValueError, match="CombinedLoss: geodesic.weight is negative"):
        CombinedLoss(config, device=torch.device("cpu"))

def test_combined_loss_learnable_weights_init():
    """Verify learnable weights initialization."""
    config = {
        "learnable_weights": True,
        "rich_hierarchy": {
            "enabled": True,
            "hierarchy_weight": 1.0,
            "coverage_weight": 1.0,
            "separation_weight": 1.0
        }
    }
    
    loss_fn = CombinedLoss(config, device=torch.device("cpu"))
    loss_fn._init_learnable_weights()
    
    assert hasattr(loss_fn, "log_sigma_hierarchy")
    assert isinstance(loss_fn.log_sigma_hierarchy, torch.nn.Parameter)
