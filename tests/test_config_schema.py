# Copyright 2024-2025 AI Whisperers

"""Focused tests for configuration validation and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import N_TERNARY_OPERATIONS
from src.config.schema import normalize_config, validate_config
from src.core import TERNARY

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


def test_config_constant_matches_canonical_ternary_space_size() -> None:
    assert N_TERNARY_OPERATIONS == TERNARY.N_OPERATIONS


# ---------------------------------------------------------------------------
# AngularCoherenceLossConfig — validators guarding known ARI regressions
# ---------------------------------------------------------------------------

def test_validate_rejects_target_sim_v0_not_1() -> None:
    """target_sim[0] != 1.0 caused ARI regression 0.844→0.716 in V7.
    The schema must catch this at config load time, not silently pass."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [0.90] + [0.85] * 9
    with pytest.raises(ValueError, match="target_sim\\[0\\]"):
        validate_config(config)


def test_validate_rejects_target_sim_wrong_length() -> None:
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [1.0, 0.85, 0.70]  # only 3
    with pytest.raises(ValueError, match="target_sim must have 10 elements"):
        validate_config(config)


def test_validate_rejects_level_prefix_k_wrong_length() -> None:
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["level_prefix_k"] = [3, 4, 5]  # only 3
    with pytest.raises(ValueError, match="level_prefix_k must have 10 elements"):
        validate_config(config)


def test_validate_accepts_valid_target_sim() -> None:
    """A correctly formed target_sim must pass — guards against over-strict validation."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [1.0, 0.85, 0.70] + [0.0] * 7
    validated = validate_config(config)
    assert validated.loss.angular_coherence is not None
    assert validated.loss.angular_coherence.target_sim is not None
    assert validated.loss.angular_coherence.target_sim[0] == 1.0


# ---------------------------------------------------------------------------
# ModelConfig cross-field validation
# ---------------------------------------------------------------------------

def test_validate_rejects_radial_dims_ge_latent_dim_when_factored() -> None:
    """radial_dims must be < latent_dim when factored=True — prevents degenerate factoring."""
    config = _load_preset("v7_large.yaml")
    assert config["model"]["factored"] is True
    config["model"]["radial_dims"] = config["model"]["latent_dim"]  # equal → invalid
    with pytest.raises(ValueError, match="radial_dims"):
        validate_config(config)


# ---------------------------------------------------------------------------
# TrainingSchedulerConfig phase validation
# ---------------------------------------------------------------------------

def test_validate_rejects_multi_phase_cosine_without_phases() -> None:
    config = _load_preset("v7_large.yaml")
    config["training"]["scheduler"]["type"] = "multi_phase_cosine"
    config["training"]["scheduler"]["phases"] = []
    with pytest.raises(ValueError, match="phases must be provided"):
        validate_config(config)


# ---------------------------------------------------------------------------
# LossConfig bounds
# ---------------------------------------------------------------------------

def test_validate_rejects_negative_loss_weight() -> None:
    config = _load_preset("v7_large.yaml")
    config["loss"]["rich_hierarchy"]["hierarchy_weight"] = -1.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_validate_rejects_curvature_zero_or_negative() -> None:
    config = _load_preset("v7_large.yaml")
    config["model"]["curvature"] = 0.0
    config["visualization"]["curvature"] = 0.0  # keep cross-check consistent
    with pytest.raises(ValueError, match="curvature"):
        validate_config(config)


# ---------------------------------------------------------------------------
# Extra keys at top-level (extra="forbid" coverage)
# ---------------------------------------------------------------------------

def test_validate_rejects_unknown_top_level_key() -> None:
    config = _load_preset("v7_large.yaml")
    config["unknown_block"] = {"foo": 1}
    with pytest.raises(ValueError, match="unknown_block"):
        validate_config(config)


def test_validate_rejects_unknown_model_key() -> None:
    config = _load_preset("v7_large.yaml")
    config["model"]["nonexistent_field"] = True
    with pytest.raises(ValueError, match="nonexistent_field"):
        validate_config(config)


# ---------------------------------------------------------------------------
# Schema-constructor default alignment — catch silent behavioral divergence
# ---------------------------------------------------------------------------


def test_schema_defaults_geodesic_max_target_distance() -> None:
    """Schema default must equal PAdicGeodesicLoss constructor default (3.0).

    A mismatch causes silent divergence: omitting max_target_distance from YAML
    gives 0.8 from schema but 3.0 from the constructor — target distances 3.75x wrong.
    """
    from src.config.schema import GeodesicLossConfig
    cfg = GeodesicLossConfig()
    assert cfg.max_target_distance == 3.0


def test_schema_defaults_geodesic_valuation_scale() -> None:
    """Schema must expose valuation_scale so users can tune it; default=3.0 matches constructor."""
    from src.config.schema import GeodesicLossConfig
    cfg = GeodesicLossConfig()
    assert cfg.valuation_scale == 3.0


def test_schema_defaults_monotonic_target_loss_weight() -> None:
    """Schema default for target_loss_weight must match MonotonicRadialLoss constructor (0.5)."""
    from src.config.schema import MonotonicLossConfig
    cfg = MonotonicLossConfig()
    assert cfg.target_loss_weight == 0.5


def test_schema_defaults_monotonic_new_fields() -> None:
    """Schema must expose margin_scale, use_soft_margin, temperature with correct defaults."""
    from src.config.schema import MonotonicLossConfig
    cfg = MonotonicLossConfig()
    assert cfg.margin_scale == 1.0
    assert cfg.use_soft_margin is True
    assert cfg.temperature == 0.05


def test_schema_defaults_angular_coherence_prefix_k() -> None:
    """Schema default for prefix_k must match AngularCoherenceLoss constructor (2)."""
    from src.config.schema import AngularCoherenceLossConfig
    cfg = AngularCoherenceLossConfig()
    assert cfg.prefix_k == 2


def test_schema_defaults_angular_coherence_weight() -> None:
    """Schema default for weight must match AngularCoherenceLoss constructor (0.3)."""
    from src.config.schema import AngularCoherenceLossConfig
    cfg = AngularCoherenceLossConfig()
    assert cfg.weight == 0.3


def test_schema_defaults_lagrangian_lr() -> None:
    """Schema lr default must match LagrangianDualState constructor default (0.01)."""
    from src.config.schema import LagrangianConfig
    cfg = LagrangianConfig()
    assert cfg.lr == 0.01


def test_schema_defaults_lagrangian_warmup_epochs() -> None:
    """Schema warmup_epochs default must match LagrangianDualState constructor default (20)."""
    from src.config.schema import LagrangianConfig
    cfg = LagrangianConfig()
    assert cfg.warmup_epochs == 20


def test_schema_defaults_lagrangian_n_levels() -> None:
    """Schema must expose n_levels so users can tune it; default=10 matches constructor."""
    from src.config.schema import LagrangianConfig
    cfg = LagrangianConfig()
    assert cfg.n_levels == 10


def test_normalize_config_propagates_geodesic_valuation_scale() -> None:
    """valuation_scale written into schema must survive round-trip through normalize_config."""
    config = _load_preset("v7_large.yaml")
    del config["loss"]["geodesic"]["max_target_distance"]  # force schema default
    normalized = normalize_config(config)
    # Schema should fill max_target_distance=3.0 and valuation_scale=3.0
    assert normalized["loss"]["geodesic"]["max_target_distance"] == 3.0
    assert normalized["loss"]["geodesic"]["valuation_scale"] == 3.0


def test_normalize_config_propagates_monotonic_new_fields() -> None:
    """margin_scale/use_soft_margin/temperature must be present after normalization."""
    config = _load_preset("v7_large.yaml")
    normalized = normalize_config(config)
    mono = normalized["loss"]["monotonic"]
    assert mono["margin_scale"] == 1.0
    assert mono["use_soft_margin"] is True
    assert mono["temperature"] == 0.05


# ---------------------------------------------------------------------------
# Radius ordering validators — inner_radius < outer_radius
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_key", ["rich_hierarchy", "radial", "monotonic"])
def test_validate_rejects_inner_radius_ge_outer_radius(loss_key: str) -> None:
    """All loss configs with both radius fields must reject inner >= outer."""
    config = _load_preset("v7_large.yaml")
    config["loss"][loss_key]["inner_radius"] = 0.9
    config["loss"][loss_key]["outer_radius"] = 0.5
    with pytest.raises(ValueError, match="inner_radius"):
        validate_config(config)


@pytest.mark.parametrize("loss_key", ["rich_hierarchy", "radial", "monotonic"])
def test_validate_rejects_equal_radii(loss_key: str) -> None:
    """inner_radius == outer_radius is also invalid (degenerate interval)."""
    config = _load_preset("v7_large.yaml")
    config["loss"][loss_key]["inner_radius"] = 0.5
    config["loss"][loss_key]["outer_radius"] = 0.5
    with pytest.raises(ValueError, match="inner_radius"):
        validate_config(config)


def test_validate_rejects_valuation_prior_inverted_radii() -> None:
    """ValuationPriorConfig must reject inner >= outer when both are set."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["valuation_prior"]["inner_radius"] = 0.8
    config["loss"]["valuation_prior"]["outer_radius"] = 0.2
    with pytest.raises(ValueError, match="inner_radius"):
        validate_config(config)


def test_validate_allows_valuation_prior_partial_radii() -> None:
    """ValuationPriorConfig must accept when only one radius is set (uses fallback logic)."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["valuation_prior"]["inner_radius"] = 0.1
    config["loss"]["valuation_prior"]["outer_radius"] = None
    validated = validate_config(config)
    assert validated.loss.valuation_prior.inner_radius == 0.1


# ---------------------------------------------------------------------------
# target_sim element range validation
# ---------------------------------------------------------------------------


def test_validate_rejects_target_sim_value_above_1() -> None:
    """target_sim elements must be in [0, 1] — cosine similarity targets."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [1.0, 1.5] + [0.0] * 8
    with pytest.raises(ValueError, match="out of range"):
        validate_config(config)


def test_validate_rejects_target_sim_negative_value() -> None:
    """target_sim elements must be non-negative."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [1.0, -0.1] + [0.0] * 8
    with pytest.raises(ValueError, match="out of range"):
        validate_config(config)


def test_validate_accepts_target_sim_boundary_values() -> None:
    """target_sim values of exactly 0.0 and 1.0 must be valid."""
    config = _load_preset("v7_large.yaml")
    config["loss"]["angular_coherence"]["target_sim"] = [1.0, 0.0] + [0.0] * 8
    validated = validate_config(config)
    assert validated.loss.angular_coherence is not None
    assert validated.loss.angular_coherence.target_sim is not None
    assert validated.loss.angular_coherence.target_sim[1] == 0.0


# ---------------------------------------------------------------------------
# combined.py fallback alignment — direct construction without schema
# ---------------------------------------------------------------------------


def test_combined_loss_separation_margin_default() -> None:
    """CombinedLoss constructed directly must use separation_margin=0.1, not stale 0.01."""
    from src.losses.combined import CombinedLoss
    config = {"rich_hierarchy": {"enabled": True}}
    loss_fn = CombinedLoss(config, curvature=1.0)
    # Access the underlying loss object to verify the margin it was built with
    assert loss_fn.rich_hierarchy is not None
    assert loss_fn.rich_hierarchy.separation_margin == pytest.approx(0.1)


def test_combined_loss_margin_step_factor_default() -> None:
    """CombinedLoss constructed directly must use margin_step_factor=0.01, not stale 0.5."""
    from src.losses.combined import CombinedLoss
    config = {"radial": {"enabled": True}, "rich_hierarchy": {"enabled": False}}
    loss_fn = CombinedLoss(config, curvature=1.0)
    assert loss_fn.radial_loss is not None
    assert loss_fn.radial_loss.margin_step_factor == pytest.approx(0.01)


def test_combined_loss_geodesic_weight_default() -> None:
    """Geodesic weight fallback must be 0.4, matching GeodesicLossConfig default."""
    from src.losses.combined import CombinedLoss
    config = {"geodesic": {"enabled": True}, "rich_hierarchy": {"enabled": False}}
    loss_fn = CombinedLoss(config, curvature=1.0)
    assert loss_fn.geodesic_weight == pytest.approx(0.4)
