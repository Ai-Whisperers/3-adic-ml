# Copyright 2024-2025 AI Whisperers
# Shared pytest fixtures and configuration

from pathlib import Path
import sys

import pytest
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set deterministic seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Return available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_indices():
    """Sample of ternary operation indices for testing."""
    return torch.arange(100)


@pytest.fixture
def sample_z_hyp():
    """Sample hyperbolic embeddings (inside Poincaré ball)."""
    z = torch.randn(50, 16, dtype=torch.float64) * 0.5
    return z


@pytest.fixture
def sample_tangent():
    """Sample tangent vectors at origin."""
    return torch.randn(50, 16, dtype=torch.float64) * 0.5


# Markers for conditional tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
