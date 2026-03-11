"""Centralized radius hyperparameter defaults for 3-adic VAE losses.

This module provides a single source of truth for inner_radius and outer_radius
hyperparameters used across multiple loss functions. Previously, each loss class
had its own defaults (0.1/0.85), and CombinedLoss read them separately from config
blocks, leading to redundant declarations and potential inconsistencies.

Key features:
- Single DEFAULT_INNER_RADIUS and DEFAULT_OUTER_RADIUS constants
- Automatic sharing mechanism when hyperparameters aren't explicitly specified
- Cross-block validation to catch configuration inconsistencies
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


# ============================================================================
# CENTRALIZED DEFAULTS
# ============================================================================
# These are the canonical defaults for radial target hyperparameters.
# All loss classes should use these unless explicitly overridden in config.
# ============================================================================
DEFAULT_INNER_RADIUS = 0.08  # Target for highest valuation (v=9, near origin)
DEFAULT_OUTER_RADIUS = 0.70  # Target for lowest valuation (v=0, near boundary)


@dataclass
class RadiusConfig:
    """Centralized configuration for radial target hyperparameters.

    Used by:
    - RichHierarchyLoss
    - RadialHierarchyLoss
    - MonotonicRadialLoss

    Example:
        config = RadiusConfig.from_dict({'inner_radius': 0.08, 'outer_radius': 0.70})
        # Or use defaults:
        config = RadiusConfig()  # Uses DEFAULT_INNER_RADIUS, DEFAULT_OUTER_RADIUS
    """

    inner_radius: float = DEFAULT_INNER_RADIUS
    outer_radius: float = DEFAULT_OUTER_RADIUS

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RadiusConfig":
        """Create from config dict with fallback to defaults."""
        return cls(
            inner_radius=cfg.get("inner_radius", DEFAULT_INNER_RADIUS),
            outer_radius=cfg.get("outer_radius", DEFAULT_OUTER_RADIUS),
        )

    def to_dict(self) -> Dict[str, float]:
        """Export to dictionary format."""
        return {"inner_radius": self.inner_radius, "outer_radius": self.outer_radius}

    def validate(self) -> Tuple[bool, str]:
        """Validate the radius configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.inner_radius <= 0:
            return False, f"inner_radius must be positive, got {self.inner_radius}"
        if self.outer_radius <= 0:
            return False, f"outer_radius must be positive, got {self.outer_radius}"
        if self.inner_radius >= self.outer_radius:
            return (
                False,
                f"inner_radius ({self.inner_radius}) must be < outer_radius ({self.outer_radius})",
            )
        if self.inner_radius >= 1.0 or self.outer_radius >= 1.0:
            return False, f"Radii must be < 1.0 for valid Poincaré ball coordinates"
        return True, "Valid"


def auto_share_radius_config(
    loss_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, RadiusConfig]:
    """Automatically share radius hyperparameters across loss blocks.

    Strategy:
    1. If only one block specifies radii, use those for all blocks
    2. If multiple blocks specify radii, use the first one found (with warning)
    3. If no blocks specify radii, use defaults

    This ensures consistent radial targets across all hierarchy-related losses,
    which is critical for proper p-adic structure learning.

    Args:
        loss_configs: Dictionary mapping loss block names to their config dicts
                     (e.g., {'rich_hierarchy': {...}, 'radial': {...}, ...})

    Returns:
        Dictionary mapping loss block names to RadiusConfig objects
    """
    # Collect all radius configs from enabled blocks
    radius_sources = {}
    for block_name, block_cfg in loss_configs.items():
        if "inner_radius" in block_cfg or "outer_radius" in block_cfg:
            radius_sources[block_name] = RadiusConfig.from_dict(block_cfg)

    # Determine the shared radius config
    if len(radius_sources) == 0:
        # No explicit radii specified - use defaults for all
        shared_config = RadiusConfig()
        print(
            f"[RadiusDefaults] No explicit radii found. Using defaults: "
            f"inner={shared_config.inner_radius}, outer={shared_config.outer_radius}"
        )

    elif len(radius_sources) == 1:
        # One block specified radii - use it for all
        block_name, shared_config = next(iter(radius_sources.items()))
        print(
            f"[RadiusDefaults] Using radii from '{block_name}': "
            f"inner={shared_config.inner_radius}, outer={shared_config.outer_radius}"
        )

    else:
        # Multiple blocks specified radii - use first one with warning
        block_name, shared_config = next(iter(radius_sources.items()))
        other_blocks = [b for b in radius_sources.keys() if b != block_name]
        print(
            f"[RadiusDefaults] WARNING: Multiple blocks specified radii. "
            f"Using first from '{block_name}': inner={shared_config.inner_radius}, "
            f"outer={shared_config.outer_radius}. "
            f"Ignoring radii from: {other_blocks}"
        )

    # Apply shared config to all blocks
    result = {}
    for block_name in loss_configs.keys():
        result[block_name] = shared_config

    return result


def compare_radius_configs(configs: Dict[str, RadiusConfig]) -> Tuple[bool, str]:
    """Compare radius configs across blocks and flag inconsistencies.

    Returns:
        Tuple of (all_equal, description_of_differences)
    """
    if len(configs) <= 1:
        return True, "Single config or no configs to compare"

    configs_list = list(configs.values())
    first = configs_list[0]

    for block_name, config in configs.items():
        if (
            config.inner_radius != first.inner_radius
            or config.outer_radius != first.outer_radius
        ):
            differences = []
            for b_name, b_config in configs.items():
                differences.append(
                    f"  {b_name}: inner={b_config.inner_radius}, outer={b_config.outer_radius}"
                )
            return False, "Inconsistent radius configs across blocks:\n" + "\n".join(
                differences
            )

    return True, "All configs have consistent radii"
