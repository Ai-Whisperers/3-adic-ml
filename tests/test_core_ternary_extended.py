# Copyright 2024-2025 AI Whisperers
# Extended tests for core ternary algebra - computation verification and tree structure

"""
Extended Tier 1 Tests: Actual Computation and Tree Structure

These tests verify:
1. Distance formula computes exactly 3^(-v)
2. Property accessors match direct computation from ternary digits
3. Tree structure relationships are internally consistent
4. No assumptions about specific values - only structural invariants
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import TERNARY


class TestDistanceFormulaComputation:
    """Verify distance(i, j) = 3^(-v_3(|i-j|)) exactly."""

    def test_distance_formula_explicit(self):
        """Verify distance = 3^(-valuation) for known differences."""
        # Test pairs where we can compute |i-j| and its valuation
        test_cases = [
            # (i, j, |i-j|, expected_valuation, expected_distance)
            (0, 1, 1, 0, 1.0),          # v_3(1) = 0, d = 3^0 = 1
            (0, 3, 3, 1, 1/3),          # v_3(3) = 1, d = 3^(-1) = 1/3
            (0, 9, 9, 2, 1/9),          # v_3(9) = 2, d = 3^(-2) = 1/9
            (0, 27, 27, 3, 1/27),       # v_3(27) = 3, d = 3^(-3)
            (0, 81, 81, 4, 1/81),       # v_3(81) = 4
            (5, 8, 3, 1, 1/3),          # |5-8| = 3, v_3(3) = 1
            (10, 19, 9, 2, 1/9),        # |10-19| = 9, v_3(9) = 2
            (100, 127, 27, 3, 1/27),    # |100-127| = 27, v_3(27) = 3
        ]

        for i, j, diff, expected_v, expected_d in test_cases:
            actual_d = TERNARY.distance(torch.tensor([i]), torch.tensor([j])).item()

            # Verify via the formula
            v = TERNARY.valuation(torch.tensor([diff])).item()
            computed_d = 3.0 ** (-v)

            assert v == expected_v, (
                f"Valuation mismatch: v_3({diff}) = {v}, expected {expected_v}"
            )
            assert abs(actual_d - expected_d) < 1e-12, (
                f"Distance mismatch: d({i}, {j}) = {actual_d}, expected {expected_d}"
            )
            assert abs(actual_d - computed_d) < 1e-12, (
                f"Distance doesn't match formula: {actual_d} vs 3^(-{v}) = {computed_d}"
            )

    def test_distance_self_is_zero(self):
        """d(i, i) = 0 for all i, not 3^(-MAX_VALUATION)."""
        indices = torch.arange(100)
        distances = TERNARY.distance(indices, indices)

        assert (distances == 0.0).all(), (
            f"Self-distance not zero: max = {distances.max().item()}"
        )

    def test_distance_formula_random_pairs(self):
        """Verify formula holds for random pairs."""
        torch.manual_seed(42)
        n = 500

        i_idx = torch.randint(0, TERNARY.N_OPERATIONS, (n,))
        j_idx = torch.randint(0, TERNARY.N_OPERATIONS, (n,))

        # Compute distance via the function
        distances = TERNARY.distance(i_idx, j_idx)

        # Compute via explicit formula
        diff = torch.abs(i_idx - j_idx)
        valuations = TERNARY.valuation(diff)
        expected_distances = torch.where(
            i_idx == j_idx,
            torch.zeros_like(distances),
            torch.pow(3.0, -valuations.double())
        )

        max_error = (distances - expected_distances).abs().max().item()
        assert max_error < 1e-12, (
            f"Distance formula mismatch: max error = {max_error}"
        )


class TestPropertyAccessorsAgainstDirectComputation:
    """Verify property accessors match direct computation from ternary digits."""

    def _compute_properties_directly(self, n: int) -> dict:
        """Compute all properties directly from the index, no LUT."""
        # Convert to ternary digits
        digits = []
        m = n
        for _ in range(TERNARY.N_DIGITS):
            digits.append((m % 3) - 1)  # {0,1,2} -> {-1,0,1}
            m //= 3

        # Compute properties
        digit_count = sum(1 for d in digits if d != 0)
        digit_sum = sum(digits)

        first_nz = TERNARY.N_DIGITS
        for i, d in enumerate(digits):
            if d != 0:
                first_nz = i
                break

        last_nz = -1
        for i in range(TERNARY.N_DIGITS - 1, -1, -1):
            if digits[i] != 0:
                last_nz = i
                break

        parent = n // 3 if n > 0 else -1

        return {
            'digits': digits,
            'digit_count': digit_count,
            'digit_sum': digit_sum,
            'first_nonzero': first_nz,
            'last_nonzero': last_nz,
            'parent': parent,
        }

    def test_digit_count_matches_direct(self):
        """digit_count accessor matches counting non-zero digits directly."""
        # Test all indices (exhaustive for this small domain)
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        accessor_counts = TERNARY.digit_count(all_indices)

        mismatches = []
        for n in range(TERNARY.N_OPERATIONS):
            direct = self._compute_properties_directly(n)['digit_count']
            accessor = accessor_counts[n].item()
            if direct != accessor:
                mismatches.append((n, direct, accessor))

        assert len(mismatches) == 0, (
            f"digit_count mismatches: {mismatches[:5]}..."
        )

    def test_digit_sum_matches_direct(self):
        """digit_sum accessor matches summing digits directly."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        accessor_sums = TERNARY.digit_sum(all_indices)

        mismatches = []
        for n in range(TERNARY.N_OPERATIONS):
            direct = self._compute_properties_directly(n)['digit_sum']
            accessor = accessor_sums[n].item()
            if direct != accessor:
                mismatches.append((n, direct, accessor))

        assert len(mismatches) == 0, (
            f"digit_sum mismatches: {mismatches[:5]}..."
        )

    def test_first_nonzero_matches_direct(self):
        """first_nonzero accessor matches direct computation."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        accessor_vals = TERNARY.first_nonzero(all_indices)

        mismatches = []
        for n in range(TERNARY.N_OPERATIONS):
            direct = self._compute_properties_directly(n)['first_nonzero']
            accessor = accessor_vals[n].item()
            if direct != accessor:
                mismatches.append((n, direct, accessor))

        assert len(mismatches) == 0, (
            f"first_nonzero mismatches: {mismatches[:5]}..."
        )

    def test_last_nonzero_matches_direct(self):
        """last_nonzero accessor matches direct computation."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        accessor_vals = TERNARY.last_nonzero(all_indices)

        mismatches = []
        for n in range(TERNARY.N_OPERATIONS):
            direct = self._compute_properties_directly(n)['last_nonzero']
            accessor = accessor_vals[n].item()
            if direct != accessor:
                mismatches.append((n, direct, accessor))

        assert len(mismatches) == 0, (
            f"last_nonzero mismatches: {mismatches[:5]}..."
        )

    def test_parent_matches_direct(self):
        """parent accessor matches n // 3 (or -1 for n=0)."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        accessor_parents = TERNARY.parent(all_indices)

        mismatches = []
        for n in range(TERNARY.N_OPERATIONS):
            direct = self._compute_properties_directly(n)['parent']
            accessor = accessor_parents[n].item()
            if direct != accessor:
                mismatches.append((n, direct, accessor))

        assert len(mismatches) == 0, (
            f"parent mismatches: {mismatches[:5]}..."
        )

    def test_properties_dict_consistency(self):
        """properties() dict matches individual accessors."""
        test_indices = torch.tensor([0, 1, 42, 100, 1000, 19682])

        props_dict = TERNARY.properties(test_indices)

        # Compare with individual accessors
        assert torch.equal(props_dict['digit_count'], TERNARY.digit_count(test_indices))
        assert torch.equal(props_dict['digit_sum'], TERNARY.digit_sum(test_indices))
        assert torch.equal(props_dict['first_nonzero'], TERNARY.first_nonzero(test_indices))
        assert torch.equal(props_dict['last_nonzero'], TERNARY.last_nonzero(test_indices))
        assert torch.equal(props_dict['parent'], TERNARY.parent(test_indices))


class TestTreeStructureRelationships:
    """Test tree structure via internal consistency, not assumed values."""

    def test_parent_child_relationship(self):
        """If parent(n) = p, then n must be in {3p, 3p+1, 3p+2}."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        parents = TERNARY.parent(all_indices)

        violations = []
        for n in range(TERNARY.N_OPERATIONS):
            p = parents[n].item()

            if n == 0:
                # Root has parent -1
                if p != -1:
                    violations.append((n, p, "root should have parent -1"))
            else:
                # Non-root: n should be one of {3p, 3p+1, 3p+2}
                expected_children = {3 * p, 3 * p + 1, 3 * p + 2}
                if n not in expected_children:
                    violations.append((n, p, f"n not in children of p: {expected_children}"))

        assert len(violations) == 0, (
            f"Parent-child relationship violated: {violations[:5]}..."
        )

    def test_parent_chain_reaches_root(self):
        """Following parent links from any node reaches 0 (root)."""
        # Sample some indices
        test_indices = [1, 10, 100, 1000, 5000, 10000, 19682]

        for start in test_indices:
            n = start
            visited = {n}
            steps = 0
            max_steps = 20  # Should reach root in at most 9 steps (log_3(19683))

            while n != 0 and steps < max_steps:
                p = TERNARY.parent(torch.tensor([n])).item()
                if p == -1:
                    break  # Reached root indicator
                if p in visited:
                    pytest.fail(f"Cycle detected starting from {start}")
                visited.add(p)
                n = p
                steps += 1

            assert n == 0 or TERNARY.parent(torch.tensor([n])).item() == -1, (
                f"Parent chain from {start} didn't reach root in {max_steps} steps"
            )

    def test_prefix_at_level_zero_is_same_for_all(self):
        """prefix(n, 0) should be the same for all n (root level)."""
        indices = torch.arange(1000)
        prefixes = TERNARY.prefix(indices, level=0)

        unique_prefixes = torch.unique(prefixes)
        assert len(unique_prefixes) == 1, (
            f"Level 0 should have single prefix, got {len(unique_prefixes)}"
        )

    def test_prefix_at_level_nine_is_identity(self):
        """prefix(n, 9) = n for all n (leaf level)."""
        indices = torch.arange(1000)
        prefixes = TERNARY.prefix(indices, level=9)

        assert torch.equal(prefixes, indices), (
            "Level 9 prefix should be identity"
        )

    def test_prefix_hierarchy(self):
        """If two nodes share prefix at level k, they share prefix at all levels < k."""
        torch.manual_seed(42)

        # Sample pairs of indices
        n_pairs = 200
        i_idx = torch.randint(0, TERNARY.N_OPERATIONS, (n_pairs,))
        j_idx = torch.randint(0, TERNARY.N_OPERATIONS, (n_pairs,))

        for level in range(1, 9):
            prefix_at_k = TERNARY.prefix(i_idx, level) == TERNARY.prefix(j_idx, level)

            # If they share prefix at level k, check all lower levels
            for lower_level in range(level):
                prefix_at_lower = TERNARY.prefix(i_idx, lower_level) == TERNARY.prefix(j_idx, lower_level)

                # If prefix matches at level k, it MUST match at lower levels
                violations = prefix_at_k & ~prefix_at_lower
                assert not violations.any(), (
                    f"Prefix hierarchy violated: shared at level {level} but not at {lower_level}"
                )

    def test_prefix_formula(self):
        """Verify prefix(n, level) = n // 3^(9-level)."""
        test_indices = torch.tensor([0, 1, 9, 27, 100, 1000, 19682])

        for level in range(10):
            computed = TERNARY.prefix(test_indices, level)
            divisor = 3 ** (TERNARY.N_DIGITS - level)
            expected = test_indices // divisor

            assert torch.equal(computed, expected), (
                f"Prefix formula mismatch at level {level}"
            )

    def test_sibling_relationship(self):
        """Nodes with same parent are siblings: they differ only in last digit.

        Note: Index 0 is special - it's the root with parent=-1.
        Children of p are {3p, 3p+1, 3p+2}, but we exclude 0 from this check
        since 0 is the root node.
        """
        # For any p > 0, children are 3p, 3p+1, 3p+2
        # For p = 0, children are {0, 1, 2} but 0 is the root (parent=-1)
        test_parents = [1, 10, 100, 1000, 6560]  # Skip 0, start from 1

        for p in test_parents:
            children = [3*p, 3*p + 1, 3*p + 2]
            valid_children = [c for c in children if c < TERNARY.N_OPERATIONS]

            if len(valid_children) < 2:
                continue

            # All valid children should have the same parent
            child_tensor = torch.tensor(valid_children)
            parents = TERNARY.parent(child_tensor)

            assert (parents == p).all(), (
                f"Children of {p} don't all report {p} as parent: got {parents.tolist()}"
            )

    def test_root_special_case(self):
        """Index 0 is the root with parent=-1, but 1 and 2 have parent=0."""
        root_parent = TERNARY.parent(torch.tensor([0])).item()
        assert root_parent == -1, f"Root parent should be -1, got {root_parent}"

        # Children 1 and 2 have parent 0
        children_parents = TERNARY.parent(torch.tensor([1, 2]))
        assert (children_parents == 0).all(), (
            f"Children 1,2 should have parent 0, got {children_parents.tolist()}"
        )


class TestLevelRankConsistency:
    """Test level_rank is consistent with valuation grouping."""

    def test_level_rank_unique_within_valuation(self):
        """Within each valuation level, ranks should be 0, 1, 2, ... (no gaps, no duplicates)."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        valuations = TERNARY.valuation(all_indices)
        ranks = TERNARY.level_rank(all_indices)

        for v in range(TERNARY.MAX_VALUATION + 1):
            mask = valuations == v
            level_ranks = ranks[mask]

            if len(level_ranks) == 0:
                continue

            # Ranks should be 0, 1, 2, ..., n-1
            expected_ranks = torch.arange(len(level_ranks))
            sorted_ranks, _ = torch.sort(level_ranks)

            assert torch.equal(sorted_ranks, expected_ranks), (
                f"Level {v} ranks not contiguous: got {sorted_ranks[:10].tolist()}..."
            )

    def test_level_rank_matches_level_count(self):
        """Max rank at each level should be level_count - 1."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        valuations = TERNARY.valuation(all_indices)
        ranks = TERNARY.level_rank(all_indices)

        for v in range(TERNARY.MAX_VALUATION + 1):
            mask = valuations == v
            level_ranks = ranks[mask]

            expected_count = TERNARY.level_count(v)
            actual_count = len(level_ranks)
            max_rank = level_ranks.max().item() if len(level_ranks) > 0 else -1

            assert actual_count == expected_count, (
                f"Level {v}: count mismatch {actual_count} vs {expected_count}"
            )
            assert max_rank == expected_count - 1, (
                f"Level {v}: max rank {max_rank} != count-1 ({expected_count - 1})"
            )


class TestLevelMask:
    """Test level_mask function."""

    def test_level_mask_matches_valuation(self):
        """level_mask(indices, level) should equal (valuation(indices) == level)."""
        indices = torch.arange(1000)

        for level in range(TERNARY.MAX_VALUATION + 1):
            mask = TERNARY.level_mask(indices, level)
            expected = TERNARY.valuation(indices) == level

            assert torch.equal(mask, expected), (
                f"level_mask mismatch at level {level}"
            )

    def test_level_masks_partition_indices(self):
        """Every index belongs to exactly one level."""
        indices = torch.arange(TERNARY.N_OPERATIONS)

        total_mask = torch.zeros(TERNARY.N_OPERATIONS, dtype=torch.bool)

        for level in range(TERNARY.MAX_VALUATION + 1):
            mask = TERNARY.level_mask(indices, level)

            # No overlap with previous masks
            overlap = total_mask & mask
            assert not overlap.any(), (
                f"Level {level} overlaps with previous levels"
            )

            total_mask = total_mask | mask

        # All indices should be covered
        assert total_mask.all(), (
            f"Not all indices covered by level masks: {(~total_mask).sum()} missing"
        )


class TestValidationFunctions:
    """Test is_valid_index and is_valid_ternary."""

    def test_is_valid_index_range(self):
        """Valid indices are in [0, N_OPERATIONS - 1]."""
        valid = torch.tensor([0, 1, 100, 19682])
        invalid = torch.tensor([-1, -100, 19683, 20000, 100000])

        assert TERNARY.is_valid_index(valid).all(), "Valid indices rejected"
        assert not TERNARY.is_valid_index(invalid).any(), "Invalid indices accepted"

    def test_is_valid_ternary_accepts_valid(self):
        """Valid ternary has all digits in {-1, 0, 1}."""
        valid = torch.tensor([
            [-1, 0, 1, 0, -1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        ], dtype=torch.float64)

        assert TERNARY.is_valid_ternary(valid).all(), "Valid ternary rejected"

    def test_is_valid_ternary_rejects_invalid(self):
        """Invalid ternary has digits outside {-1, 0, 1}."""
        invalid = torch.tensor([
            [2, 0, 0, 0, 0, 0, 0, 0, 0],   # 2 is invalid
            [0, 0, 0, 0, -2, 0, 0, 0, 0],  # -2 is invalid
            [0, 0, 0, 0, 0, 0, 0, 0, 3],   # 3 is invalid
        ], dtype=torch.float64)

        assert not TERNARY.is_valid_ternary(invalid).any(), "Invalid ternary accepted"


class TestValuationHistogram:
    """Test valuation_histogram function."""

    def test_histogram_sums_to_input_count(self):
        """Histogram values should sum to number of input indices."""
        indices = torch.arange(1000)
        hist = TERNARY.valuation_histogram(indices)

        total = sum(hist.values())
        assert total == len(indices), (
            f"Histogram sum {total} != input count {len(indices)}"
        )

    def test_histogram_matches_level_count_for_all(self):
        """Histogram of all indices should match level_count."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        hist = TERNARY.valuation_histogram(all_indices)

        for v in range(TERNARY.MAX_VALUATION + 1):
            assert hist[v] == TERNARY.level_count(v), (
                f"Histogram[{v}] = {hist[v]} != level_count({v}) = {TERNARY.level_count(v)}"
            )


class TestTernaryRepresentationConsistency:
    """Test ternary representation is consistent with other properties."""

    def test_digit_count_matches_ternary_nonzero(self):
        """digit_count should equal number of non-zero values in to_ternary output."""
        indices = torch.arange(1000)

        ternary = TERNARY.to_ternary(indices)
        direct_count = (ternary != 0).sum(dim=-1)
        accessor_count = TERNARY.digit_count(indices)

        assert torch.equal(direct_count, accessor_count), (
            "digit_count doesn't match to_ternary nonzero count"
        )

    def test_digit_sum_matches_ternary_sum(self):
        """digit_sum should equal sum of to_ternary output."""
        indices = torch.arange(1000)

        ternary = TERNARY.to_ternary(indices)
        direct_sum = ternary.sum(dim=-1).long()
        accessor_sum = TERNARY.digit_sum(indices)

        assert torch.equal(direct_sum, accessor_sum), (
            "digit_sum doesn't match to_ternary sum"
        )

    def test_first_nonzero_matches_ternary_scan(self):
        """first_nonzero should match scanning to_ternary from position 0."""
        indices = torch.arange(1000)

        ternary = TERNARY.to_ternary(indices)
        accessor_first = TERNARY.first_nonzero(indices)

        for i, (t, first) in enumerate(zip(ternary, accessor_first)):
            # Find first nonzero manually
            direct_first = TERNARY.N_DIGITS
            for pos in range(TERNARY.N_DIGITS):
                if t[pos] != 0:
                    direct_first = pos
                    break

            assert first.item() == direct_first, (
                f"first_nonzero mismatch at index {indices[i]}: {first.item()} vs {direct_first}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
