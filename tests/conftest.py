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
# YAML Validation Tests
import subprocess

# Get all YAML files in presets
PRESETS_DIR = Path(__file__).parent.parent / "src" / "presets"
YAML_FILES = list(PRESETS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_file", YAML_FILES)
def test_yaml_syntax(yaml_file):
    """Test that all preset YAML files have valid syntax."""
    import yaml
    with open(yaml_file) as f:
        # This will raise if syntax is invalid
        yaml.safe_load(f)


@pytest.mark.parametrize("yaml_file", YAML_FILES)
def test_yaml_linting(yaml_file):
    """Test that all preset YAML files pass yamllint."""
    result = subprocess.run(
        ["yamllint", "-c", ".yamllint", str(yaml_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"YAML linting failed for {yaml_file}:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("yaml_file", YAML_FILES)
def test_yaml_schema(yaml_file):
    """Test that all preset YAML files pass Pydantic schema validation."""
    import yaml

    from src.config.schema import validate_config

    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    # Should not raise
    validated = validate_config(config)
    assert validated is not None
