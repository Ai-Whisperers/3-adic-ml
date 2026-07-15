# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Tests for src/training/metrics.py.

Written incrementally, part by part, in the order the file's functions
appear. See conversation history for the per-part analysis notes.

Historical note: compute_hyperbolic_coverage was dead code (no caller
anywhere in the codebase) and has since been removed. compute_hierarchy_metrics
used to have an inconsistent return-dict contract (n<2 early-exit returned only
5 keys vs. 12 on the normal path, missing "mean_radius" which
src/training/engine.py reads via direct bracket access) -- fixed so both paths
return the same key set; see
TestComputeHierarchyMetrics.test_early_exit_dict_has_same_keys_as_full_path.
"""

from __future__ import annotations

import pytest
import torch

from src.geometry import hyperbolic_radius, poincare_distance
from src.training.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_hierarchy_metrics,
    compute_level_stratified_hierarchy,
    compute_tree_coherence,
)


def _one_hot_logits(targets: torch.Tensor, wrong_positions: dict[int, int] | None = None) -> torch.Tensor:
    """Build (B, 9, 3) logits that argmax to `targets` exactly, except at
    the given {batch_index: position} overrides, which are forced wrong."""
    wrong_positions = wrong_positions or {}
    B, D = targets.shape
    logits = torch.zeros(B, D, 3)
    for b in range(B):
        for d in range(D):
            correct_class = int(targets[b, d].item()) + 1
            if wrong_positions.get(b) == d:
                # Force a different class to win argmax.
                wrong_class = (correct_class + 1) % 3
                logits[b, d, wrong_class] = 10.0
            else:
                logits[b, d, correct_class] = 10.0
    return logits


class TestComputeAccuracy:
    def test_perfect_match_is_1(self):
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits = _one_hot_logits(targets)
        assert compute_accuracy(logits, targets) == 1.0

    def test_one_wrong_position_out_of_nine(self):
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits = _one_hot_logits(targets, wrong_positions={0: 0})
        assert compute_accuracy(logits, targets) == pytest.approx(8 / 9)

    def test_flat_27_logits_equivalent_to_9x3(self):
        """The (B, 27) code path must decode identically to (B, 9, 3)."""
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits_3 = _one_hot_logits(targets, wrong_positions={0: 3})
        logits_27 = logits_3.view(1, 27)
        assert compute_accuracy(logits_27, targets) == compute_accuracy(logits_3, targets)

    def test_unsupported_last_dim_returns_zero(self):
        weird = torch.randn(4, 9, 5)
        assert compute_accuracy(weird, torch.zeros(4, 9)) == 0.0

    def test_averages_across_batch(self):
        """Batch of 2: one perfect, one all-wrong -> accuracy should be the
        mean of per-sample accuracies, not per-position across the batch."""
        targets = torch.tensor([
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        ])
        logits = _one_hot_logits(targets)
        # Corrupt every position of sample 1.
        for d in range(9):
            correct_class = int(targets[1, d].item()) + 1
            wrong_class = (correct_class + 1) % 3
            logits[1, d] = 0.0
            logits[1, d, wrong_class] = 10.0
        assert compute_accuracy(logits, targets) == 0.5


class TestComputeCoverage:
    def test_perfect_reconstruction_is_1(self):
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits = _one_hot_logits(targets)
        assert compute_coverage(logits, targets) == 1.0

    def test_single_wrong_position_breaks_coverage_to_zero(self):
        """Coverage is per-SAMPLE all-or-nothing: even one wrong digit out
        of nine must drop that sample's contribution to 0, unlike accuracy
        which averages per-digit."""
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits = _one_hot_logits(targets, wrong_positions={0: 4})
        assert compute_coverage(logits, targets) == 0.0

    def test_mixed_batch_fraction(self):
        """3 samples: 2 perfect, 1 imperfect -> coverage = 2/3."""
        targets = torch.tensor([
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        ])
        logits = _one_hot_logits(targets, wrong_positions={2: 0})
        assert compute_coverage(logits, targets) == pytest.approx(2 / 3)

    def test_flat_27_logits_equivalent_to_9x3(self):
        targets = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        logits_3 = _one_hot_logits(targets, wrong_positions={0: 1})
        logits_27 = logits_3.view(1, 27)
        assert compute_coverage(logits_27, targets) == compute_coverage(logits_3, targets)

    def test_unsupported_last_dim_returns_zero(self):
        weird = torch.randn(4, 9, 5)
        assert compute_coverage(weird, torch.zeros(4, 9)) == 0.0


# ---------------------------------------------------------------------------
# compute_tree_coherence
# ---------------------------------------------------------------------------

class TestComputeTreeCoherence:
    def test_mean_matches_hand_computed_pairwise_distances(self):
        """indices=[0(root), 1(child of 0), 4(child of 1)]: two valid
        parent-child pairs. The returned value must equal the exact mean of
        their individually-computed Poincaré distances."""
        idx = torch.tensor([0, 1, 4])
        z = torch.tensor([
            [0.0, 0.0],   # pos0 -> idx 0 (root, parent=-1, excluded as a child)
            [0.3, 0.0],   # pos1 -> idx 1 (parent=0 -> pos0)
            [0.3, 0.2],   # pos2 -> idx 4 (parent=1 -> pos1)
        ], dtype=torch.float64)

        result = compute_tree_coherence(z, idx, curvature=1.0)

        d_child1_parent0 = poincare_distance(z[1:2], z[0:1], c=1.0).item()
        d_child2_parent1 = poincare_distance(z[2:3], z[1:2], c=1.0).item()
        expected = (d_child1_parent0 + d_child2_parent1) / 2
        assert result == pytest.approx(expected, rel=1e-9)

    def test_root_excluded_as_a_child(self):
        """Index 0's parent is -1 (root sentinel) — it must never be treated
        as a child, even though -1 could otherwise collide with a real
        index if looked up carelessly."""
        idx = torch.tensor([0])
        z = torch.zeros(1, 2, dtype=torch.float64)
        assert compute_tree_coherence(z, idx, curvature=1.0) == 0.0

    def test_pair_skipped_when_parent_not_in_batch(self):
        """If a child's true parent index isn't present in this batch, that
        pair must be silently skipped rather than matched to some other
        position or raising."""
        idx = torch.tensor([0, 4])  # index 4's parent (1) is absent
        z = torch.tensor([[0.0, 0.0], [0.3, 0.2]], dtype=torch.float64)
        assert compute_tree_coherence(z, idx, curvature=1.0) == 0.0

    def test_no_valid_pairs_returns_zero_not_nan(self):
        idx = torch.tensor([0, 3, 6])  # none of these are parents of each other
        # (parent(3)=1, parent(6)=2 -- neither 1 nor 2 is in idx)
        z = torch.randn(3, 2, dtype=torch.float64) * 0.1
        result = compute_tree_coherence(z, idx, curvature=1.0)
        assert result == 0.0

    def test_sampling_path_does_not_crash_and_stays_in_range(self):
        """With more valid pairs than sample_size, the function must
        subsample rather than crash, and the result must still be a finite,
        non-negative distance."""
        idx = torch.arange(0, 40)
        torch.manual_seed(0)
        z = torch.randn(40, 4, dtype=torch.float64) * 0.05

        result = compute_tree_coherence(z, idx, curvature=1.0, sample_size=5)
        assert result >= 0.0
        assert result == result  # not NaN


# ---------------------------------------------------------------------------
# compute_level_stratified_hierarchy
# ---------------------------------------------------------------------------

class TestComputeLevelStratifiedHierarchy:
    def test_matches_hand_computed_formula_for_populated_level(self):
        """indices 1,2,4 all have valuation 0 (v_3(n)=0 for n not divisible
        by 3). The metric for that level must equal -1/(1+std) computed
        from the *hyperbolic* radii (not raw Euclidean norms), using
        unbiased (N-1) std to match level_scatter_std's convention."""
        idx = torch.tensor([1, 2, 4])
        z = torch.tensor([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], dtype=torch.float64)

        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)

        r = hyperbolic_radius(z, c=1.0).numpy()
        expected = -1.0 / (1.0 + r.std(ddof=1))
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_level_with_zero_samples_is_nan(self):
        idx = torch.tensor([1, 2])  # both valuation 0; every other level empty
        z = torch.tensor([[0.3, 0.0], [0.5, 0.0]], dtype=torch.float64)
        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)
        for level in range(1, 10):
            assert result[level] != result[level]  # NaN

    def test_level_with_exactly_one_sample_is_nan(self):
        """Boundary: count==1 must be NaN (std of a single point is
        undefined for the unbiased estimator), not a spurious finite value."""
        idx = torch.tensor([1, 9])  # v=0 (count 1) and v=2 (count 1)
        z = torch.tensor([[0.3, 0.0], [0.1, 0.0]], dtype=torch.float64)
        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)
        assert result[0] != result[0]  # NaN
        assert result[2] != result[2]  # NaN

    def test_level_with_exactly_two_samples_is_finite(self):
        """Boundary on the other side: count==2 is the minimum for a
        defined (non-NaN) value."""
        idx = torch.tensor([1, 2])
        z = torch.tensor([[0.3, 0.0], [0.5, 0.0]], dtype=torch.float64)
        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)
        assert result[0] == result[0]  # not NaN
        assert -1.0 <= result[0] < 0.0

    def test_identical_radii_within_level_gives_near_perfect_score(self):
        """Zero spread (all points at the same radius) must score close to
        the -1.0 'perfect consistency' bound the docstring describes."""
        idx = torch.tensor([1, 2, 4])
        z = torch.tensor([[0.4, 0.0]] * 3, dtype=torch.float64)
        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)
        assert result[0] == pytest.approx(-1.0, abs=1e-4)

    def test_all_ten_levels_present_in_output_regardless_of_data(self):
        idx = torch.tensor([1, 2])
        z = torch.tensor([[0.3, 0.0], [0.5, 0.0]], dtype=torch.float64)
        result = compute_level_stratified_hierarchy(z, idx, curvature=1.0)
        assert set(result.keys()) == set(range(10))


# ---------------------------------------------------------------------------
# compute_hierarchy_metrics
# ---------------------------------------------------------------------------

class TestComputeHierarchyMetrics:
    def test_perfect_monotonic_radius_gives_hierarchy_one(self):
        """Radius strictly decreasing with valuation (the target structure
        every hierarchy loss in this codebase optimizes toward) must yield
        hierarchy == 1.0 exactly (perfect Spearman correlation)."""
        idx = torch.arange(0, 200)
        from src.core import TERNARY
        vals = TERNARY.valuation(idx)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        radius = 0.9 * (0.5 ** vals.double())
        z = direction.unsqueeze(0) * radius.unsqueeze(1)

        m = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=42)
        assert m["hierarchy"] == pytest.approx(1.0, abs=1e-9)
        assert m["hierarchy_collapsed"] is False

    def test_q_equals_dist_corr_plus_1_5x_hierarchy(self):
        torch.manual_seed(0)
        z = torch.randn(50, 4, dtype=torch.float64) * 0.3
        idx = torch.arange(50)
        m = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=7)
        assert m["Q"] == pytest.approx(m["dist_corr"] + 1.5 * m["hierarchy"], rel=1e-9)

    def test_level_radii_matches_hand_computed_per_level_mean(self):
        idx = torch.tensor([1, 2, 4, 5, 3, 6, 12, 15])  # 4x v=0, 4x v=1
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        r0 = torch.tensor([0.60, 0.601, 0.599, 0.6005], dtype=torch.float64)
        r1 = torch.tensor([0.1, 0.3, 0.05, 0.4], dtype=torch.float64)
        z = direction.unsqueeze(0) * torch.cat([r0, r1]).unsqueeze(1)

        m = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        hr = hyperbolic_radius(z, c=1.0).numpy()
        assert m["level_radii"][0] == pytest.approx(hr[:4].mean(), rel=1e-9)
        assert m["level_radii"][1] == pytest.approx(hr[4:].mean(), rel=1e-9)
        assert m["level_radii"][2] != m["level_radii"][2]  # unpopulated level -> NaN

    def test_worst_level_picks_least_consistent_populated_level(self):
        """level 0 has near-identical radii (consistent); level 1 has
        widely spread radii (inconsistent). worst_level must point at the
        inconsistent one -- 'worst' means least-negative level_hierarchy
        value, i.e. the docstring's own definition."""
        idx = torch.tensor([1, 2, 4, 5, 3, 6, 12, 15])
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        r0 = torch.tensor([0.60, 0.601, 0.599, 0.6005], dtype=torch.float64)  # tight
        r1 = torch.tensor([0.1, 0.3, 0.05, 0.4], dtype=torch.float64)          # spread
        z = direction.unsqueeze(0) * torch.cat([r0, r1]).unsqueeze(1)

        m = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        assert m["worst_level"] == 1
        assert m["level_hierarchy"][0] < m["level_hierarchy"][1], (
            "level 0 (tight) should score more negative (better) than "
            "level 1 (spread)"
        )

    def test_mean_level_hierarchy_averages_only_populated_levels(self):
        idx = torch.tensor([1, 2, 4, 5, 3, 6, 12, 15])
        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
        r0 = torch.tensor([0.60, 0.601, 0.599, 0.6005], dtype=torch.float64)
        r1 = torch.tensor([0.1, 0.3, 0.05, 0.4], dtype=torch.float64)
        z = direction.unsqueeze(0) * torch.cat([r0, r1]).unsqueeze(1)

        m = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        expected = (m["level_hierarchy"][0] + m["level_hierarchy"][1]) / 2
        assert m["mean_level_hierarchy"] == pytest.approx(expected, rel=1e-9)

    def test_small_n_uses_all_pairs_deterministically(self):
        """n<=100 must exhaustively use every pair (np.triu_indices), so the
        result must be identical across different `seed` values -- sampling
        is only supposed to kick in above the n=100 threshold."""
        torch.manual_seed(0)
        z = torch.randn(50, 4, dtype=torch.float64) * 0.3
        idx = torch.arange(50)
        m1 = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        m2 = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=999)
        assert m1["dist_corr"] == m2["dist_corr"]

    def test_large_n_pair_sampling_is_seed_reproducible(self):
        """n>100 samples pairs randomly -- same seed must give the exact
        same dist_corr (reproducible eval runs), different seed generally
        won't."""
        torch.manual_seed(0)
        z = torch.randn(150, 4, dtype=torch.float64) * 0.3
        idx = torch.arange(150)
        m_a = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        m_b = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=1)
        m_c = compute_hierarchy_metrics(z, idx, curvature=1.0, seed=2)
        assert m_a["dist_corr"] == m_b["dist_corr"]
        assert m_a["dist_corr"] != m_c["dist_corr"]

    def test_early_exit_dict_has_same_keys_as_full_path(self):
        """The n<2 early-exit path must return the same key set as the
        normal path (including 'mean_radius'), since src/training/engine.py
        reads hier_metrics_A["mean_radius"] via direct bracket access -- a
        validation split with 0 or 1 samples must degrade gracefully
        (neutral defaults) instead of raising KeyError and crashing training.
        """
        z_small = torch.randn(1, 8, dtype=torch.float64)
        idx_small = torch.tensor([0])
        small = compute_hierarchy_metrics(z_small, idx_small, curvature=1.0)

        z_full = torch.randn(50, 8, dtype=torch.float64) * 0.3
        idx_full = torch.arange(50)
        full = compute_hierarchy_metrics(z_full, idx_full, curvature=1.0)

        assert set(small.keys()) == set(full.keys())
        assert small["mean_radius"] == 0.0
        assert small["hierarchy_collapsed"] is True
        assert small["dist_corr_collapsed"] is True
