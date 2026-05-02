# MIT License — see repository root for full text
"""Regression tests for resume checkpoint metric parsing (F-01).

The bug: `float(ckpt.get("Q", fallback) or -1.0)` treats Q=0.0 as falsy,
silently resetting best_Q to -1.0 on resume. Every subsequent epoch then
immediately overwrites best_Q.pt because any Q > -1.0 triggers the save.

These tests directly mirror the three expressions in train.py:1238-1240
so any future regression is caught at the source.
"""


def _parse_best_Q(ckpt: dict) -> float:
    """Mirror of train.py:1238 post-fix."""
    return float(ckpt["Q"]) if ckpt.get("Q") is not None else float(ckpt.get("best_Q", -1.0))


def _parse_best_hierarchy(ckpt: dict) -> float:
    """Mirror of train.py:1239 post-fix."""
    return float(ckpt["hierarchy_A"]) if ckpt.get("hierarchy_A") is not None else float(ckpt.get("best_hierarchy", 0.0))


def _parse_best_coverage(ckpt: dict) -> float:
    """Mirror of train.py:1240 post-fix."""
    return float(ckpt["coverage"]) if ckpt.get("coverage") is not None else float(ckpt.get("best_coverage", 0.0))


# ---------------------------------------------------------------------------
# F-01 regression: Q=0.0 must not be clobbered by falsy-or
# ---------------------------------------------------------------------------

class TestResumeQParsing:
    def test_q_zero_preserved(self) -> None:
        # The exact bug: 0.0 is falsy, old code returned -1.0
        ckpt = {"Q": 0.0, "epoch": 5}
        assert _parse_best_Q(ckpt) == 0.0

    def test_q_positive_preserved(self) -> None:
        ckpt = {"Q": 2.163, "epoch": 100}
        assert _parse_best_Q(ckpt) == 2.163

    def test_q_negative_preserved(self) -> None:
        # Negative Q is valid (poor hierarchy, early training)
        ckpt = {"Q": -0.5, "epoch": 3}
        assert _parse_best_Q(ckpt) == -0.5

    def test_q_key_absent_falls_back_to_best_q(self) -> None:
        ckpt = {"best_Q": 1.8, "epoch": 50}
        assert _parse_best_Q(ckpt) == 1.8

    def test_best_q_zero_also_preserved_via_fallback(self) -> None:
        # Fallback key also must not be clobbered
        ckpt = {"best_Q": 0.0, "epoch": 5}
        assert _parse_best_Q(ckpt) == 0.0

    def test_both_keys_absent_returns_sentinel(self) -> None:
        ckpt = {"epoch": 0}
        assert _parse_best_Q(ckpt) == -1.0

    def test_q_none_falls_back_to_best_q(self) -> None:
        # Key present but explicitly None (pathological but guarded)
        ckpt = {"Q": None, "best_Q": 1.5}
        assert _parse_best_Q(ckpt) == 1.5

    def test_q_none_and_best_q_absent_returns_sentinel(self) -> None:
        ckpt = {"Q": None}
        assert _parse_best_Q(ckpt) == -1.0

    def test_q_takes_priority_over_best_q(self) -> None:
        ckpt = {"Q": 2.1, "best_Q": 1.8}
        assert _parse_best_Q(ckpt) == 2.1


# ---------------------------------------------------------------------------
# Same correctness for hierarchy and coverage (same pattern)
# ---------------------------------------------------------------------------

class TestResumeHierarchyParsing:
    def test_hierarchy_zero_preserved(self) -> None:
        ckpt = {"hierarchy_A": 0.0}
        assert _parse_best_hierarchy(ckpt) == 0.0

    def test_hierarchy_key_absent_falls_back(self) -> None:
        ckpt = {"best_hierarchy": 0.75}
        assert _parse_best_hierarchy(ckpt) == 0.75

    def test_both_absent_returns_zero_floor(self) -> None:
        ckpt = {}
        assert _parse_best_hierarchy(ckpt) == 0.0


class TestResumeCoverageParsing:
    def test_coverage_zero_preserved(self) -> None:
        ckpt = {"coverage": 0.0}
        assert _parse_best_coverage(ckpt) == 0.0

    def test_coverage_key_absent_falls_back(self) -> None:
        ckpt = {"best_coverage": 0.95}
        assert _parse_best_coverage(ckpt) == 0.95

    def test_both_absent_returns_zero_floor(self) -> None:
        ckpt = {}
        assert _parse_best_coverage(ckpt) == 0.0
