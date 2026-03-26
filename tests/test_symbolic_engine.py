from __future__ import annotations

import torch

from src.core import TERNARY
from src.symbolic import TERNARY_GROUP_ENGINE, build_symbolic_subsystem


def test_group_engine_preserves_valid_ternary_domain() -> None:
    indices = torch.tensor([0, 1, 17, 243, 19682], dtype=torch.long)

    for element in TERNARY_GROUP_ENGINE.elements:
        transformed = TERNARY_GROUP_ENGINE.apply_element_to_indices(indices, element)
        ternary = TERNARY.to_ternary(transformed)
        assert TERNARY.is_valid_index(transformed).all()
        assert TERNARY.is_valid_ternary(ternary).all()


def test_canonicalize_is_orbit_invariant() -> None:
    indices = torch.tensor([5, 17, 901, 4096], dtype=torch.long)
    canonical = TERNARY_GROUP_ENGINE.canonicalize(indices)
    transformed = TERNARY_GROUP_ENGINE.choose_non_identity_partner(indices, seed=123)

    assert torch.equal(canonical, TERNARY_GROUP_ENGINE.canonicalize(transformed))


def test_feedback_pairs_match_orbit_and_separate_negatives() -> None:
    indices = torch.arange(TERNARY.N_OPERATIONS, dtype=torch.long)
    pairs = TERNARY_GROUP_ENGINE.sample_feedback_pairs(indices, sample_size=64, seed=123)

    positive_canon = TERNARY_GROUP_ENGINE.canonicalize(pairs["positives"])
    anchor_canon = TERNARY_GROUP_ENGINE.canonicalize(pairs["anchors"])
    negative_canon = TERNARY_GROUP_ENGINE.canonicalize(pairs["negatives"])

    assert torch.equal(anchor_canon, positive_canon)
    assert (anchor_canon != negative_canon).all()


def test_build_symbolic_subsystem_defaults_to_disabled() -> None:
    subsystem = build_symbolic_subsystem()

    assert subsystem.enabled is False
    assert subsystem.name == "disabled"
    assert "baseline path unchanged" in subsystem.describe()


def test_build_symbolic_subsystem_enables_finite_group_backend() -> None:
    subsystem = build_symbolic_subsystem({"enabled": True, "backend": "finite_group", "seed": 123})
    indices = torch.tensor([5, 17, 901], dtype=torch.long)

    assert subsystem.enabled is True
    assert subsystem.name == "finite_group"
    assert subsystem.describe().startswith("finite_group orbit engine")
    assert torch.equal(subsystem.canonicalize(indices), TERNARY_GROUP_ENGINE.canonicalize(indices))
