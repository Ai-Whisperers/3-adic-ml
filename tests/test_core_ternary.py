# Copyright 2024-2025 AI Whisperers
# Tests for core ternary algebra - mathematical invariants that MUST hold

"""
Tier 1 Critical Tests: Core Ternary Algebra

These tests verify the mathematical foundation of the 3-adic VAE.
Failures here indicate fundamental bugs that cascade through the entire system.
"""

import pytest
import torch
import numpy as np
from itertools import combinations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import TERNARY, valuation, distance, target_radius


class TestValuationFormula:
    """Verify 3-adic valuation v_3(n) = max k where 3^k divides n."""

    def test_valuation_of_zero(self):
        """v_3(0) should be MAX_VALUATION (9) - the "infinitely divisible" case."""
        result = TERNARY.valuation(torch.tensor([0]))
        assert result.item() == TERNARY.MAX_VALUATION, (
            f"v_3(0) should be {TERNARY.MAX_VALUATION}, got {result.item()}"
        )

    def test_valuation_of_one(self):
        """v_3(1) = 0 since 1 is not divisible by 3."""
        result = TERNARY.valuation(torch.tensor([1]))
        assert result.item() == 0, f"v_3(1) should be 0, got {result.item()}"

    def test_valuation_powers_of_three(self):
        """v_3(3^k) = k for k = 0, 1, 2, ..., 9."""
        powers = [3**k for k in range(10)]
        # Clamp to valid index range
        valid_powers = [(k, p) for k, p in enumerate(powers) if p < TERNARY.N_OPERATIONS]

        for expected_valuation, power in valid_powers:
            result = TERNARY.valuation(torch.tensor([power]))
            assert result.item() == expected_valuation, (
                f"v_3({power}) should be {expected_valuation}, got {result.item()}"
            )

    def test_valuation_multiples_of_three(self):
        """v_3(3n) = v_3(n) + 1 for n not divisible by 3."""
        # Test specific cases: 3, 6, 12, 15, 21, 24
        test_cases = [
            (3, 1),   # 3 = 3^1
            (6, 1),   # 6 = 2*3, v_3 = 1
            (9, 2),   # 9 = 3^2
            (12, 1),  # 12 = 4*3, v_3 = 1
            (27, 3),  # 27 = 3^3
            (81, 4),  # 81 = 3^4
            (243, 5), # 243 = 3^5
        ]

        for n, expected in test_cases:
            if n < TERNARY.N_OPERATIONS:
                result = TERNARY.valuation(torch.tensor([n]))
                assert result.item() == expected, (
                    f"v_3({n}) should be {expected}, got {result.item()}"
                )

    def test_valuation_non_multiples_of_three(self):
        """Numbers not divisible by 3 should have valuation 0."""
        non_multiples = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20]

        for n in non_multiples:
            result = TERNARY.valuation(torch.tensor([n]))
            assert result.item() == 0, (
                f"v_3({n}) should be 0 (not divisible by 3), got {result.item()}"
            )

    def test_valuation_batch_consistency(self):
        """Batch valuation should match individual valuations."""
        indices = torch.arange(100)
        batch_result = TERNARY.valuation(indices)

        for i in range(100):
            single_result = TERNARY.valuation(torch.tensor([i]))
            assert batch_result[i] == single_result.item(), (
                f"Batch valuation at {i} differs from single: "
                f"{batch_result[i]} vs {single_result.item()}"
            )


class TestUltrametricInequality:
    """Verify the 3-adic distance satisfies ultrametric inequality.

    The ultrametric inequality is STRONGER than triangle inequality:
        d(a, c) <= max(d(a, b), d(b, c))

    This is the defining property of p-adic metrics.
    """

    def test_ultrametric_sampled_triples(self):
        """Test ultrametric inequality on random triples."""
        torch.manual_seed(42)
        n_samples = 1000

        # Sample random triples
        indices = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples, 3))

        violations = []
        for i in range(n_samples):
            a, b, c = indices[i].tolist()

            d_ab = TERNARY.distance(torch.tensor([a]), torch.tensor([b])).item()
            d_bc = TERNARY.distance(torch.tensor([b]), torch.tensor([c])).item()
            d_ac = TERNARY.distance(torch.tensor([a]), torch.tensor([c])).item()

            max_side = max(d_ab, d_bc)

            # Ultrametric: d(a,c) <= max(d(a,b), d(b,c))
            if d_ac > max_side + 1e-10:  # Small tolerance for floating point
                violations.append((a, b, c, d_ac, max_side))

        assert len(violations) == 0, (
            f"Ultrametric inequality violated in {len(violations)} cases. "
            f"First violation: a={violations[0][0]}, b={violations[0][1]}, c={violations[0][2]}, "
            f"d(a,c)={violations[0][3]:.6f} > max(d(a,b), d(b,c))={violations[0][4]:.6f}"
        )

    def test_ultrametric_specific_cases(self):
        """Test ultrametric on carefully chosen cases."""
        # Case 1: a=0, b=1, c=2 (consecutive integers)
        d_01 = TERNARY.distance(torch.tensor([0]), torch.tensor([1])).item()
        d_12 = TERNARY.distance(torch.tensor([1]), torch.tensor([2])).item()
        d_02 = TERNARY.distance(torch.tensor([0]), torch.tensor([2])).item()

        assert d_02 <= max(d_01, d_12) + 1e-10, (
            f"Ultrametric violated: d(0,2)={d_02} > max(d(0,1)={d_01}, d(1,2)={d_12})"
        )

        # Case 2: Powers of 3
        d_03 = TERNARY.distance(torch.tensor([0]), torch.tensor([3])).item()
        d_39 = TERNARY.distance(torch.tensor([3]), torch.tensor([9])).item()
        d_09 = TERNARY.distance(torch.tensor([0]), torch.tensor([9])).item()

        assert d_09 <= max(d_03, d_39) + 1e-10, (
            f"Ultrametric violated for powers of 3"
        )


class TestDistanceSymmetry:
    """Verify distance is symmetric: d(a, b) = d(b, a)."""

    def test_distance_symmetry_sampled(self):
        """Test symmetry on random pairs."""
        torch.manual_seed(42)
        n_samples = 500

        pairs = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples, 2))

        for i in range(n_samples):
            a, b = pairs[i].tolist()
            d_ab = TERNARY.distance(torch.tensor([a]), torch.tensor([b])).item()
            d_ba = TERNARY.distance(torch.tensor([b]), torch.tensor([a])).item()

            assert abs(d_ab - d_ba) < 1e-10, (
                f"Distance not symmetric: d({a},{b})={d_ab} != d({b},{a})={d_ba}"
            )

    def test_distance_identity(self):
        """d(a, a) = 0 for all a."""
        indices = torch.arange(0, min(1000, TERNARY.N_OPERATIONS))

        for idx in indices:
            d = TERNARY.distance(torch.tensor([idx.item()]), torch.tensor([idx.item()])).item()
            assert d == 0.0, f"d({idx},{idx}) should be 0, got {d}"


class TestRoundTripInvertibility:
    """Verify to_ternary and from_ternary are inverses."""

    def test_full_round_trip(self):
        """from_ternary(to_ternary(i)) == i for all valid indices."""
        # Test all 19,683 indices
        all_indices = torch.arange(TERNARY.N_OPERATIONS)

        # Convert to ternary representation
        ternary_repr = TERNARY.to_ternary(all_indices)

        # Convert back
        recovered = TERNARY.from_ternary(ternary_repr)

        mismatches = (all_indices != recovered).sum().item()
        assert mismatches == 0, (
            f"Round-trip failed for {mismatches} indices out of {TERNARY.N_OPERATIONS}"
        )

    def test_ternary_digit_range(self):
        """All ternary digits should be in {-1, 0, 1}."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        ternary_repr = TERNARY.to_ternary(all_indices)

        valid_digits = torch.tensor([-1, 0, 1])

        for digit_pos in range(TERNARY.N_DIGITS):
            column = ternary_repr[:, digit_pos]
            unique_vals = torch.unique(column)

            for val in unique_vals:
                assert val in valid_digits, (
                    f"Invalid digit {val} at position {digit_pos}. "
                    f"Expected values in {{-1, 0, 1}}"
                )


class TestLevelCounts:
    """Verify level counts partition all operations."""

    def test_level_counts_sum_to_total(self):
        """Sum of counts across all valuation levels = N_OPERATIONS."""
        total = 0
        for v in range(TERNARY.MAX_VALUATION + 1):
            count = TERNARY.level_count(v)
            total += count

        assert total == TERNARY.N_OPERATIONS, (
            f"Level counts sum to {total}, expected {TERNARY.N_OPERATIONS}"
        )

    def test_level_counts_match_actual(self):
        """Verify level_count(v) matches actual count of indices with valuation v."""
        all_valuations = TERNARY.valuation(torch.arange(TERNARY.N_OPERATIONS))

        for v in range(TERNARY.MAX_VALUATION + 1):
            reported_count = TERNARY.level_count(v)
            actual_count = (all_valuations == v).sum().item()

            assert reported_count == actual_count, (
                f"Level {v}: reported count {reported_count} != actual count {actual_count}"
            )

    def test_level_zero_count(self):
        """Level 0 (valuation=0) should have the most elements."""
        level_0_count = TERNARY.level_count(0)

        # Level 0 should have 2 * 3^8 = 13122 elements (non-multiples of 3)
        # Actually: 3^9 - 3^8 = 19683 - 6561 = 13122
        expected = TERNARY.N_OPERATIONS - (TERNARY.N_OPERATIONS // 3)

        assert level_0_count == expected, (
            f"Level 0 count {level_0_count} != expected {expected}"
        )


class TestTargetRadiusMonotonicity:
    """Verify target radii decrease with increasing valuation.

    Note: target_radius takes INDICES, not valuations. It computes
    valuation internally. We test with indices of known valuations.
    """

    def test_strict_monotonic_decrease(self):
        """Indices with higher valuation should have smaller target radius."""
        # Indices with known valuations:
        # v=0: 1, 2, 4, 5, 7, 8 (not divisible by 3)
        # v=1: 3, 6, 12 (divisible by 3, not 9)
        # v=2: 9, 18, 36 (divisible by 9, not 27)
        # v=3: 27, 54 (divisible by 27, not 81)
        # etc.
        # v=9: 0 only
        indices_by_valuation = {
            0: 1,    # v_3(1) = 0
            1: 3,    # v_3(3) = 1
            2: 9,    # v_3(9) = 2
            3: 27,   # v_3(27) = 3
            4: 81,   # v_3(81) = 4
            5: 243,  # v_3(243) = 5
            6: 729,  # v_3(729) = 6
            7: 2187, # v_3(2187) = 7
            8: 6561, # v_3(6561) = 8
            9: 0,    # v_3(0) = 9 (special case)
        }

        radii = {}
        for v, idx in indices_by_valuation.items():
            if idx < TERNARY.N_OPERATIONS:
                r = TERNARY.target_radius(torch.tensor([idx])).item()
                radii[v] = r

        # Higher valuation should have smaller radius
        sorted_vals = sorted(radii.keys())
        for i in range(len(sorted_vals) - 1):
            v1, v2 = sorted_vals[i], sorted_vals[i + 1]
            assert radii[v1] > radii[v2], (
                f"Target radius not decreasing with valuation: "
                f"r(v={v1})={radii[v1]} <= r(v={v2})={radii[v2]}"
            )

    def test_target_radius_bounds(self):
        """Target radii should be in (0, 1) for valid Poincare ball."""
        # Test on indices with various valuations
        test_indices = [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 100, 500, 1000]

        for idx in test_indices:
            if idx < TERNARY.N_OPERATIONS:
                r = TERNARY.target_radius(torch.tensor([idx])).item()
                assert 0 < r < 1, (
                    f"Target radius for index {idx} is {r}, should be in (0, 1)"
                )

    def test_outer_inner_ordering(self):
        """Low valuation (outer) should have largest radius, high valuation (inner) smallest."""
        # Index 1 has valuation 0 (outer boundary)
        # Index 0 has valuation 9 (inner, near origin)
        r_outer = TERNARY.target_radius(torch.tensor([1])).item()  # v=0
        r_inner = TERNARY.target_radius(torch.tensor([0])).item()  # v=9

        assert r_outer > r_inner, (
            f"Outer radius (v=0) {r_outer} should be > inner radius (v=9) {r_inner}"
        )

    def test_target_radius_formula(self):
        """Verify the linear interpolation formula directly."""
        inner, outer = 0.1, 0.9
        max_v = TERNARY.MAX_VALUATION

        # For v=0: should be outer (0.9)
        idx_v0 = torch.tensor([1])  # v_3(1) = 0
        r_v0 = TERNARY.target_radius(idx_v0, inner=inner, outer=outer).item()
        expected_v0 = outer * (1 - 0/max_v) + inner * (0/max_v)
        assert abs(r_v0 - expected_v0) < 1e-10, f"v=0: got {r_v0}, expected {expected_v0}"

        # For v=9: should be inner (0.1)
        idx_v9 = torch.tensor([0])  # v_3(0) = 9
        r_v9 = TERNARY.target_radius(idx_v9, inner=inner, outer=outer).item()
        expected_v9 = outer * (1 - 9/max_v) + inner * (9/max_v)
        assert abs(r_v9 - expected_v9) < 1e-10, f"v=9: got {r_v9}, expected {expected_v9}"


class TestDistanceMatrix:
    """Test properties of the full distance matrix."""

    def test_distance_matrix_symmetry(self):
        """Distance matrix should be symmetric."""
        # Use smaller subset for efficiency
        n = 100
        indices = torch.arange(n)

        dist_matrix = TERNARY.distance_matrix(indices)

        asymmetry = torch.abs(dist_matrix - dist_matrix.T).max().item()
        assert asymmetry < 1e-10, (
            f"Distance matrix asymmetry: max |D - D^T| = {asymmetry}"
        )

    def test_distance_matrix_diagonal_zero(self):
        """Diagonal of distance matrix should be all zeros."""
        n = 100
        indices = torch.arange(n)

        dist_matrix = TERNARY.distance_matrix(indices)
        diagonal = torch.diag(dist_matrix)

        max_diag = diagonal.abs().max().item()
        assert max_diag < 1e-10, (
            f"Distance matrix diagonal not zero: max = {max_diag}"
        )

    def test_distance_values_in_valid_range(self):
        """All distances should be in [0, 1]."""
        n = 100
        indices = torch.arange(n)

        dist_matrix = TERNARY.distance_matrix(indices)

        assert dist_matrix.min() >= 0, f"Negative distance: {dist_matrix.min()}"
        assert dist_matrix.max() <= 1, f"Distance > 1: {dist_matrix.max()}"


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_tensor(self):
        """Empty tensor should return empty result."""
        empty = torch.tensor([], dtype=torch.long)
        result = TERNARY.valuation(empty)

        assert result.numel() == 0, "Valuation of empty tensor should be empty"

    def test_boundary_indices(self):
        """Test first and last valid indices."""
        first = torch.tensor([0])
        last = torch.tensor([TERNARY.N_OPERATIONS - 1])

        # Should not raise
        v_first = TERNARY.valuation(first)
        v_last = TERNARY.valuation(last)

        assert v_first.item() == TERNARY.MAX_VALUATION, "v_3(0) should be max"
        assert 0 <= v_last.item() <= TERNARY.MAX_VALUATION, "Last index valuation out of range"

    def test_out_of_range_clamping(self):
        """Out-of-range indices should be clamped silently."""
        too_large = torch.tensor([TERNARY.N_OPERATIONS + 1000])
        negative = torch.tensor([-1])

        # Should not raise, should clamp
        v_large = TERNARY.valuation(too_large)
        v_neg = TERNARY.valuation(negative)

        assert 0 <= v_large.item() <= TERNARY.MAX_VALUATION
        assert 0 <= v_neg.item() <= TERNARY.MAX_VALUATION

    def test_large_batch(self):
        """Large batch should work efficiently."""
        large_batch = torch.randint(0, TERNARY.N_OPERATIONS, (10000,))

        # Should complete without error
        valuations = TERNARY.valuation(large_batch)

        assert valuations.shape == large_batch.shape
        assert (valuations >= 0).all()
        assert (valuations <= TERNARY.MAX_VALUATION).all()


class TestDeviceConsistency:
    """Test CPU/GPU consistency (if GPU available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_valuation_match(self):
        """Valuations should match between CPU and GPU."""
        indices = torch.arange(1000)

        cpu_result = TERNARY.valuation(indices.cpu())
        gpu_result = TERNARY.valuation(indices.cuda()).cpu()

        assert torch.equal(cpu_result, gpu_result), "CPU and GPU valuations differ"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_distance_match(self):
        """Distances should match between CPU and GPU."""
        a = torch.arange(100)
        b = torch.arange(100, 200)

        cpu_dist = TERNARY.distance(a.cpu(), b.cpu())
        gpu_dist = TERNARY.distance(a.cuda(), b.cuda()).cpu()

        max_diff = (cpu_dist - gpu_dist).abs().max().item()
        assert max_diff < 1e-10, f"CPU/GPU distance mismatch: max diff = {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
