"""Optional symbolic subsystem abstraction.

This module keeps symbolic capabilities isolated from the rest of the codebase.
The baseline model path remains unchanged unless a caller explicitly enables a
symbolic subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .engine import TERNARY_GROUP_ENGINE


@dataclass(frozen=True)
class SymbolicRuntimeConfig:
    """Runtime config for optional symbolic modules."""

    enabled: bool = False
    backend: str = "disabled"
    orbit_sample_size: int = 512
    pair_sample_size: int = 2048
    match_valuation_negatives: bool = True
    seed: int = 321


class DisabledSymbolicSubsystem:
    """No-op symbolic subsystem used by default."""

    enabled = False
    name = "disabled"

    def __init__(self, config: SymbolicRuntimeConfig | None = None):
        self.config = config or SymbolicRuntimeConfig()

    def describe(self) -> str:
        return "disabled (baseline path unchanged)"

    def canonicalize(self, indices: torch.Tensor) -> torch.Tensor:
        return indices

    def choose_non_identity_partner(self, indices: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        return indices

    def sample_feedback_pairs(self, indices: torch.Tensor, sample_size: int | None = None) -> dict[str, Any]:
        empty = indices[:0]
        return {
            "anchors": empty,
            "positives": empty,
            "negatives": empty,
            "notes": ["Symbolic subsystem disabled."],
        }


class GroupOrbitSymbolicSubsystem:
    """Finite-group symbolic subsystem for exact orbit generation."""

    enabled = True
    name = "finite_group"

    def __init__(self, config: SymbolicRuntimeConfig):
        self.config = config
        self.engine = TERNARY_GROUP_ENGINE

    def describe(self) -> str:
        return (
            f"finite_group orbit engine (group_size={len(self.engine.elements)}, "
            f"orbit_sample_size={self.config.orbit_sample_size}, "
            f"pair_sample_size={self.config.pair_sample_size})"
        )

    def canonicalize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.engine.canonicalize(indices)

    def choose_non_identity_partner(self, indices: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        return self.engine.choose_non_identity_partner(indices, seed=seed if seed is not None else self.config.seed)

    def sample_feedback_pairs(self, indices: torch.Tensor, sample_size: int | None = None) -> dict[str, Any]:
        return self.engine.sample_feedback_pairs(
            indices,
            sample_size=sample_size or self.config.pair_sample_size,
            seed=self.config.seed,
            match_valuation_negatives=self.config.match_valuation_negatives,
        )


def normalize_symbolic_config(config: Mapping[str, Any] | None = None) -> SymbolicRuntimeConfig:
    """Normalize a loose config mapping into a strict runtime config."""
    config = dict(config or {})
    enabled = bool(config.get("enabled", False))
    backend = str(config.get("backend", "finite_group" if enabled else "disabled"))
    if not enabled:
        backend = "disabled"
    return SymbolicRuntimeConfig(
        enabled=enabled,
        backend=backend,
        orbit_sample_size=int(config.get("orbit_sample_size", 512)),
        pair_sample_size=int(config.get("pair_sample_size", 2048)),
        match_valuation_negatives=bool(config.get("match_valuation_negatives", True)),
        seed=int(config.get("seed", 321)),
    )


def build_symbolic_subsystem(config: Mapping[str, Any] | None = None) -> DisabledSymbolicSubsystem | GroupOrbitSymbolicSubsystem:
    """Build an optional symbolic subsystem with a disabled default."""
    normalized = normalize_symbolic_config(config)
    if not normalized.enabled:
        return DisabledSymbolicSubsystem(normalized)
    if normalized.backend != "finite_group":
        raise ValueError(
            f"Unsupported symbolic backend '{normalized.backend}'. "
            "Supported backends: disabled, finite_group."
        )
    return GroupOrbitSymbolicSubsystem(normalized)
