# Copyright 2024-2025 AI Whisperers

"""Focused tests for configuration validation and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.schema import normalize_config, validate_config


PRESETS_DIR = Path(__file__).parent.parent / "src" / "presets"


def _load_preset(name: str) -> dict:
    with open(PRESETS_DIR / name) as f:
        return yaml.safe_load(f)


def test_normalize_config_preserves_visualization_block() -> None:
    config = _load_preset("v7_large.yaml")

    normalized = normalize_config(config)

    assert normalized["visualization"]["max_per_level"] == 500
    assert normalized["visualization"]["curvature"] == normalized["model"]["curvature"]


def test_validate_config_rejects_visualization_curvature_mismatch() -> None:
    config = _load_preset("v7_large.yaml")
    config["visualization"]["curvature"] = 0.5

    with pytest.raises(ValueError, match="visualization\\.curvature must match model\\.curvature"):
        validate_config(config)


def test_validate_config_rejects_unknown_visualization_key() -> None:
    config = _load_preset("v7_large.yaml")
    config["visualization"]["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        validate_config(config)


def test_normalize_config_maps_legacy_radial_weight_alias() -> None:
    config = _load_preset("5.12.4.yaml")

    normalized = normalize_config(config)

    assert normalized["loss"]["radial"]["weight"] == 1.0


def test_normalize_config_preserves_legacy_non_factored_presets() -> None:
    for preset_name in ("5.12.4.yaml", "v6.yaml"):
        normalized = normalize_config(_load_preset(preset_name))
        assert normalized["model"]["factored"] is False
