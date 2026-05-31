# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Tests for src/analysis — AnomalyDetector and compute_attribution.

Covers:
  - AnomalyDetector.fit: threshold is positive, normal points below threshold
  - AnomalyDetector.detect: normal points classified as non-anomaly (mostly),
    far-boundary outliers classified as anomaly
  - No Python loops (vectorized implementation)
  - compute_attribution: shape contract and sensitivity values ≥ 0
  - attribute_anomaly: proj_A return-type handled for both factored and
    non-factored models (regression for the tuple-unpack crash)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_embeddings():
    """Small cluster of points near the Poincaré origin."""
    torch.manual_seed(0)
    return torch.randn(40, 8, dtype=torch.float64) * 0.05  # near origin


@pytest.fixture
def outlier_embeddings():
    """Points pushed close to the Poincaré boundary (anomalous)."""
    z = torch.zeros(5, 8, dtype=torch.float64)
    z[:, 0] = 0.92   # very close to boundary; far from origin cluster
    return z


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetectorFit:
    def test_threshold_is_positive(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        det.fit(normal_embeddings, k=3, sigma_factor=2.0)
        assert det.threshold is not None
        assert det.threshold > 0.0

    def test_z_norm_stored_on_device(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector(device=torch.device("cpu"))
        det.fit(normal_embeddings)
        assert det.z_norm is not None
        assert det.z_norm.device.type == "cpu"

    def test_threshold_increases_with_sigma_factor(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det_tight = AnomalyDetector()
        det_wide = AnomalyDetector()
        det_tight.fit(normal_embeddings, k=3, sigma_factor=1.0)
        det_wide.fit(normal_embeddings, k=3, sigma_factor=5.0)
        assert det_wide.threshold > det_tight.threshold

    def test_k_larger_than_n_minus_1_is_clamped(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        # k=1000 >> len(normal_embeddings)-1=39; must not raise
        det.fit(normal_embeddings, k=1000)
        assert det.threshold is not None


class TestAnomalyDetectorDetect:
    def test_detect_raises_if_not_fitted(self):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        with pytest.raises(ValueError, match="fitted"):
            det.detect(torch.randn(3, 8, dtype=torch.float64) * 0.05)

    def test_normal_points_mostly_non_anomaly(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        det.fit(normal_embeddings, k=3, sigma_factor=3.0)
        result = det.detect(normal_embeddings, k=3)
        # Most normal points should not be flagged
        anomaly_rate = result["is_anomaly"].mean()
        assert anomaly_rate < 0.5, f"Too many normal points flagged: {anomaly_rate:.0%}"

    def test_boundary_outliers_flagged(self, normal_embeddings, outlier_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        det.fit(normal_embeddings, k=3, sigma_factor=2.0)
        result = det.detect(outlier_embeddings, k=3)
        assert result["is_anomaly"].all(), (
            "All boundary outliers should be flagged as anomalous"
        )

    def test_output_shapes_match_input(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        det.fit(normal_embeddings)
        queries = torch.randn(7, 8, dtype=torch.float64) * 0.05
        result = det.detect(queries)
        assert result["is_anomaly"].shape == (7,)
        assert result["min_dist"].shape == (7,)

    def test_min_dist_is_non_negative(self, normal_embeddings):
        from src.analysis import AnomalyDetector
        det = AnomalyDetector()
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings)
        assert (result["min_dist"] >= 0.0).all()


class TestAnomalyDetectorNoLoop:
    """The implementation must not use Python loops over points."""

    def test_fit_has_no_for_loop_over_points(self):
        import inspect
        from src.analysis.anomaly_detector import AnomalyDetector
        src = inspect.getsource(AnomalyDetector.fit)
        assert "for" not in src, (
            "AnomalyDetector.fit must use vectorized ops, not a Python loop"
        )

    def test_detect_uses_topk_not_sorted_loop(self):
        import inspect
        from src.analysis.anomaly_detector import AnomalyDetector
        src = inspect.getsource(AnomalyDetector.detect)
        assert "topk" in src, "AnomalyDetector.detect must use .topk() for kNN selection"
        assert "for q in" not in src and "for i in" not in src, (
            "AnomalyDetector.detect must not loop over individual query points"
        )


# ---------------------------------------------------------------------------
# attribute_anomaly — proj_A tuple regression
# ---------------------------------------------------------------------------

class TestAttributeAnomalyProjACompat:
    """Regression: compute_attribution must handle both factored (tuple) and
    non-factored (tensor) return shapes from model.projections.proj_A."""

    def _make_detector_stub(self):
        det = MagicMock()
        det.detect.return_value = {"min_dist": np.array([0.5])}
        return det

    def _make_model_stub(self, factored: bool):
        model = MagicMock()
        latent_dim = 8
        mu = torch.zeros(1, latent_dim, dtype=torch.float64)
        model.get_mu_representations.return_value = mu

        z = torch.zeros(1, latent_dim, dtype=torch.float64)
        if factored:
            model.projections.proj_A.return_value = (z, torch.zeros(1))
        else:
            model.projections.proj_A.return_value = z
        return model

    def test_non_factored_model_does_not_crash(self):
        """Pre-fix: z, _ = proj_A(mu) raised ValueError for non-factored models."""
        from src.analysis.attribute_anomaly import compute_attribution

        seq = "ACG"  # 3 nucleotides → 3 positions to perturb

        # Patch seq_to_ternary_index via import side-effect
        import sys
        # Provide a minimal stub for seq_to_ternary_index
        class _FakeModule:
            @staticmethod
            def seq_to_ternary_index(seq):
                return 0

        real_mod = sys.modules.get("scripts.data.prepare_codon_data")
        try:
            sys.modules["scripts.data.prepare_codon_data"] = _FakeModule()
            import importlib
            import src.analysis.attribute_anomaly as aa_mod
            importlib.reload(aa_mod)

            model = self._make_model_stub(factored=False)
            detector = self._make_detector_stub()
            attribution, orig_dist = aa_mod.compute_attribution(seq, model, detector)
            assert len(attribution) == 3
            assert all(s >= 0.0 for s in attribution)
        finally:
            if real_mod is not None:
                sys.modules["scripts.data.prepare_codon_data"] = real_mod
            else:
                sys.modules.pop("scripts.data.prepare_codon_data", None)

    def test_factored_model_does_not_crash(self):
        """Factored models (tuple return) must also work."""
        from src.analysis.attribute_anomaly import compute_attribution
        import sys

        seq = "ACG"

        class _FakeModule:
            @staticmethod
            def seq_to_ternary_index(seq):
                return 0

        real_mod = sys.modules.get("scripts.data.prepare_codon_data")
        try:
            sys.modules["scripts.data.prepare_codon_data"] = _FakeModule()
            import importlib
            import src.analysis.attribute_anomaly as aa_mod
            importlib.reload(aa_mod)

            model = self._make_model_stub(factored=True)
            detector = self._make_detector_stub()
            attribution, _ = aa_mod.compute_attribution(seq, model, detector)
            assert len(attribution) == 3
        finally:
            if real_mod is not None:
                sys.modules["scripts.data.prepare_codon_data"] = real_mod
            else:
                sys.modules.pop("scripts.data.prepare_codon_data", None)


# ---------------------------------------------------------------------------
# TernarySpace — ternary_add / ternary_mul simplified arithmetic
# ---------------------------------------------------------------------------

class TestTernaryArithmeticSimplified:
    """ternary_add / ternary_mul must give the same results after removing
    the clone+masked-assignment pattern and using arithmetic instead."""

    def _exhaustive_add_table(self):
        """Reference: enumerate all Z_3 digit additions."""
        results = {}
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                # Z_3: map {-1,0,1} → {2,0,1}, add mod 3, map back
                da = a % 3
                db = b % 3
                ds = (da + db) % 3
                results[(a, b)] = ds - 3 if ds == 2 else ds
        return results

    def _exhaustive_mul_table(self):
        """Reference: enumerate all Z_3 digit multiplications."""
        results = {}
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                da = a % 3
                db = b % 3
                dp = (da * db) % 3
                results[(a, b)] = dp - 3 if dp == 2 else dp
        return results

    def test_ternary_add_matches_z3_table(self):
        from src.core import TERNARY
        ref = self._exhaustive_add_table()
        for (a, b), expected in ref.items():
            ia = TERNARY.from_ternary(torch.tensor([[a, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64))
            ib = TERNARY.from_ternary(torch.tensor([[b, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64))
            idx_sum = TERNARY.ternary_add(ia, ib)
            t_sum = TERNARY.to_ternary(idx_sum)
            assert t_sum[0, 0].item() == expected, (
                f"ternary_add({a},{b}): expected digit0={expected}, got {t_sum[0,0].item()}"
            )

    def test_ternary_mul_matches_z3_table(self):
        from src.core import TERNARY
        ref = self._exhaustive_mul_table()
        for (a, b), expected in ref.items():
            ia = TERNARY.from_ternary(torch.tensor([[a, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64))
            ib = TERNARY.from_ternary(torch.tensor([[b, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64))
            idx_prod = TERNARY.ternary_mul(ia, ib)
            t_prod = TERNARY.to_ternary(idx_prod)
            assert t_prod[0, 0].item() == expected, (
                f"ternary_mul({a},{b}): expected digit0={expected}, got {t_prod[0,0].item()}"
            )

    def test_ternary_add_no_clone(self):
        import inspect
        from src.core.ternary import TernarySpace
        src = inspect.getsource(TernarySpace.ternary_add)
        assert ".clone()" not in src, "ternary_add should not use .clone()"

    def test_ternary_mul_no_clone(self):
        import inspect
        from src.core.ternary import TernarySpace
        src = inspect.getsource(TernarySpace.ternary_mul)
        assert ".clone()" not in src, "ternary_mul should not use .clone()"


# ---------------------------------------------------------------------------
# _compute_level_counts and valuation_histogram — torch.bincount
# ---------------------------------------------------------------------------

class TestTernaryBincountOps:
    """Both _compute_level_counts and valuation_histogram must use bincount."""

    def test_level_counts_sum_to_n_operations(self):
        from src.core import TERNARY
        counts = TERNARY._level_counts
        assert counts.sum().item() == TERNARY.N_OPERATIONS

    def test_level_counts_correct_values(self):
        """Known analytical values: |{n : v_3(n)=k}| = 2 * 3^(8-k) for k in 0..8, 1 for k=9."""
        from src.core import TERNARY
        counts = TERNARY._level_counts
        for k in range(9):
            expected = 2 * (3 ** (8 - k))
            assert counts[k].item() == expected, (
                f"Level k={k}: expected {expected}, got {counts[k].item()}"
            )
        assert counts[9].item() == 1

    def test_valuation_histogram_matches_level_counts(self):
        from src.core import TERNARY
        all_idx = torch.arange(TERNARY.N_OPERATIONS)
        hist = TERNARY.valuation_histogram(all_idx)
        for v in range(TERNARY.MAX_VALUATION + 1):
            assert hist[v] == TERNARY._level_counts[v].item(), (
                f"valuation_histogram[{v}] disagrees with _level_counts"
            )

    def test_compute_level_counts_uses_bincount(self):
        import inspect
        from src.core.ternary import TernarySpace
        src = inspect.getsource(TernarySpace._compute_level_counts)
        assert "bincount" in src, "_compute_level_counts must use torch.bincount"
        assert "for v in range" not in src, "_compute_level_counts must not use an element-wise loop"

    def test_valuation_histogram_uses_bincount(self):
        import inspect
        from src.core.ternary import TernarySpace
        src = inspect.getsource(TernarySpace.valuation_histogram)
        assert "bincount" in src, "valuation_histogram must use torch.bincount"
        assert "for val in range" not in src, "valuation_histogram must not use an element-wise loop"
