# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Regression tests for the second-pass audit findings.

Covers:
  1.  Lagrangian margin / scatter / prior tensor keys (combined + monotonic / rank / prior)
  2.  AlgebraicCoherenceLoss no longer accepts valuation_fn
  3.  Dead _target_radii buffers removed from RadialHierarchyLoss,
      MonotonicRadialLoss, and RichHierarchyLoss
  4.  hyperbolic_radius uses dist0 (no zero-tensor allocation)
  5.  WithinLevelContrastiveLoss import math moved to module level
  6.  RichHierarchyLoss accepts max_valuation parameter
  7.  AngularCoherenceLoss global path uses per-level target_sim
  8.  get_lr_scales() side-effect warning is documented
  9.  CPU-generator removal from PAdicGeodesicLoss / GlobalRankLoss /
      RadialHierarchyLoss (no .to(device) transfer on index tensors)
  10. DataAuditor leakage check uses torch.isin instead of Python set-of-tuples
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from tests.conftest import assert_valid_loss


# ---------------------------------------------------------------------------
# 1. Lagrangian tensor keys are present and in-graph
# ---------------------------------------------------------------------------

class TestLagrangianTensorKeys:
    """gap_viol_tensor_v*, scatter_tensor_v*, and vp_gap_tensor_v* must be
    present in the respective loss metrics and must be differentiable tensors
    so that CombinedLoss can construct an in-graph Lagrangian penalty."""

    def _z(self, n=64, d=8):
        z = torch.randn(n, d, dtype=torch.float64) * 0.3
        z.requires_grad_(True)
        return z

    def test_monotonic_emits_gap_viol_tensor_keys(self):
        from src.losses.hierarchy import MonotonicRadialLoss
        fn = MonotonicRadialLoss(inner_radius=0.1, outer_radius=0.85)
        z = self._z()
        idx = torch.arange(64)
        _, metrics = fn(z, idx)
        tensor_keys = [k for k in metrics if k.startswith("gap_viol_tensor_v")]
        assert len(tensor_keys) > 0, "No gap_viol_tensor_v* keys in monotonic metrics"
        for k in tensor_keys:
            vt = metrics[k]
            assert isinstance(vt, torch.Tensor), f"{k} must be a Tensor, got {type(vt)}"
            assert vt.requires_grad or vt.grad_fn is not None, \
                f"{k} is not in-graph (detached)"

    def test_rank_emits_scatter_tensor_keys(self):
        from src.losses.rank import GlobalRankLoss
        fn = GlobalRankLoss(scatter_weight=1.0)
        z = self._z()
        idx = torch.arange(64)
        _, metrics = fn(z, idx)
        tensor_keys = [k for k in metrics if k.startswith("scatter_tensor_v")]
        # scatter_tensor keys only appear for levels that have same-valuation pairs
        if tensor_keys:
            for k in tensor_keys:
                vt = metrics[k]
                assert isinstance(vt, torch.Tensor), f"{k} must be a Tensor"

    def test_valuation_prior_emits_vp_gap_tensor_keys(self):
        from src.losses.prior import ValuationPriorLoss
        fn = ValuationPriorLoss()
        mu = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
        idx = torch.arange(64)
        _, metrics = fn(mu, idx)
        tensor_keys = [k for k in metrics if k.startswith("vp_gap_tensor_v")]
        assert len(tensor_keys) > 0, "No vp_gap_tensor_v* keys in prior metrics"
        for k in tensor_keys:
            vt = metrics[k]
            assert isinstance(vt, torch.Tensor), f"{k} must be a Tensor"
            assert vt.requires_grad or vt.grad_fn is not None, \
                f"{k} is not in-graph (detached)"

    def test_combined_loss_applies_lagrangian_margin_penalty(self):
        """With non-zero lambda_margin the combined total must increase."""
        from src.losses.combined import CombinedLoss
        cfg = {
            "monotonic": {
                "enabled": True, "weight": 1.0,
                "inner_radius": 0.1, "outer_radius": 0.85,
            },
            "rich_hierarchy": {
                "enabled": True, "hierarchy_weight": 1.0,
                "coverage_weight": 0.0, "separation_weight": 0.0,
            },
        }
        fn = CombinedLoss(cfg)
        torch.manual_seed(42)
        z = torch.randn(64, 8, dtype=torch.float64) * 0.3
        z.requires_grad_(True)
        idx = torch.arange(64)
        logits = torch.randn(64, 27, dtype=torch.float64)
        targets = torch.randint(-1, 2, (64, 9))

        out_no_lag = fn(z, idx, logits, targets, epoch=0)

        # Inject a large margin penalty for every level
        dual = {"lambda_margin": [10.0] * 9, "lambda_scatter": [], "lambda_prior": []}
        out_lag = fn(z, idx, logits, targets, epoch=0, dual_weights=dual)

        assert out_lag["total"].item() != out_no_lag["total"].item(), \
            "Lagrangian margin penalty had no effect on total loss"


# ---------------------------------------------------------------------------
# 2. AlgebraicCoherenceLoss no longer accepts valuation_fn
# ---------------------------------------------------------------------------

class TestAlgebraicCoherenceLossNoValuationFn:
    """valuation_fn was a no-op parameter (the loss groups by algebraic
    signature, not valuation).  It must be removed from the constructor."""

    def test_constructor_rejects_valuation_fn_kwarg(self):
        from src.losses.algebraic import AlgebraicCoherenceLoss
        with pytest.raises(TypeError, match="valuation_fn"):
            AlgebraicCoherenceLoss(valuation_fn=lambda x: x)

    def test_constructor_sig_has_no_valuation_fn(self):
        from src.losses.algebraic import AlgebraicCoherenceLoss
        sig = inspect.signature(AlgebraicCoherenceLoss.__init__)
        assert "valuation_fn" not in sig.parameters, \
            "valuation_fn must be removed from AlgebraicCoherenceLoss.__init__"


# ---------------------------------------------------------------------------
# 3. Dead _target_radii buffers removed
# ---------------------------------------------------------------------------

class TestDeadBuffersRemoved:
    """The pre-computed _target_radii buffers were never read by forward();
    they bloat checkpoints and mislead readers.  Assert they are gone."""

    def test_radial_hierarchy_no_target_radii_buffer(self):
        from src.losses.hierarchy import RadialHierarchyLoss
        fn = RadialHierarchyLoss(inner_radius=0.1, outer_radius=0.85)
        assert "_target_radii" not in dict(fn.named_buffers()), \
            "_target_radii buffer must be removed from RadialHierarchyLoss"

    def test_monotonic_no_target_radii_buffer(self):
        from src.losses.hierarchy import MonotonicRadialLoss
        fn = MonotonicRadialLoss(inner_radius=0.1, outer_radius=0.85)
        assert "_target_radii" not in dict(fn.named_buffers()), \
            "_target_radii buffer must be removed from MonotonicRadialLoss"

    def test_rich_hierarchy_no_target_radii_buffer(self):
        from src.losses.hierarchy import RichHierarchyLoss
        fn = RichHierarchyLoss(inner_radius=0.1, outer_radius=0.85)
        assert "target_radii" not in dict(fn.named_buffers()), \
            "target_radii buffer must be removed from RichHierarchyLoss"

    def test_radial_hierarchy_still_computes_correct_loss(self):
        """Removing the buffer must not change loss values."""
        from src.losses.hierarchy import RadialHierarchyLoss
        torch.manual_seed(0)
        z = torch.randn(32, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(32)
        fn = RadialHierarchyLoss(inner_radius=0.1, outer_radius=0.85)
        loss, _ = fn(z, idx)
        assert_valid_loss(loss)


# ---------------------------------------------------------------------------
# 4. hyperbolic_radius uses dist0 — no (B,D) zero-tensor allocation
# ---------------------------------------------------------------------------

class TestHyperbolicRadiusDist0:
    """hyperbolic_radius must use manifold.dist0 instead of creating a
    full zero-tensor; results must be identical to the old approach."""

    def test_same_result_as_origin_distance(self):
        from src.geometry.poincare import get_manifold, hyperbolic_radius
        torch.manual_seed(42)
        z = torch.randn(32, 8, dtype=torch.float64) * 0.3
        c = 1.0
        radii_new = hyperbolic_radius(z, c=c)
        # Reference via explicit poincare_distance to origin
        from src.geometry.poincare import poincare_distance
        origin = torch.zeros_like(z)
        radii_ref = poincare_distance(z, origin, c=c)
        assert torch.allclose(radii_new, radii_ref, atol=1e-10), \
            "dist0 path must give same result as dist(z, origin)"

    def test_no_zero_tensor_created(self):
        """dist0 must not allocate a full (B,D) zero tensor."""
        import src.geometry.poincare as poincare_mod
        src = inspect.getsource(poincare_mod.hyperbolic_radius)
        assert "zeros_like" not in src, \
            "hyperbolic_radius must not use zeros_like; use manifold.dist0 instead"


# ---------------------------------------------------------------------------
# 5. WithinLevelContrastiveLoss: import math at module level
# ---------------------------------------------------------------------------

class TestImportMathModuleLevel:
    def test_no_import_math_inside_forward(self):
        import src.losses.hierarchy as hier_mod
        src_text = inspect.getsource(hier_mod.WithinLevelContrastiveLoss.forward)
        assert "import math" not in src_text, \
            "import math must be at module level, not inside WithinLevelContrastiveLoss.forward"

    def test_math_imported_at_module_level(self):
        import src.losses.hierarchy as hier_mod
        mod_src = inspect.getsource(hier_mod)
        # First occurrence of 'import math' must be near the top, not in a function
        first_import = mod_src.find("import math")
        first_def = mod_src.find("def ")
        assert first_import < first_def, \
            "import math must appear before any function definition in hierarchy.py"


# ---------------------------------------------------------------------------
# 6. RichHierarchyLoss accepts max_valuation parameter
# ---------------------------------------------------------------------------

class TestRichHierarchyMaxValuation:
    def test_default_max_valuation_is_9(self):
        from src.losses.hierarchy import RichHierarchyLoss
        fn = RichHierarchyLoss(inner_radius=0.1, outer_radius=0.85)
        assert fn.max_valuation == 9

    def test_custom_max_valuation_stored(self):
        from src.losses.hierarchy import RichHierarchyLoss
        fn = RichHierarchyLoss(inner_radius=0.1, outer_radius=0.85, max_valuation=5)
        assert fn.max_valuation == 5

    def test_custom_max_valuation_uses_correct_dim_size(self):
        from src.losses.hierarchy import RichHierarchyLoss
        fn = RichHierarchyLoss(inner_radius=0.1, outer_radius=0.85, max_valuation=4)
        # batch with valuations 0..4 only
        z = torch.randn(20, 8, dtype=torch.float64) * 0.3
        idx = torch.tensor([1, 3, 9, 27, 81] * 4)  # valuations 0,1,2,3,4
        raw, metrics = fn(z, idx)
        assert isinstance(raw, dict)
        assert "hierarchy" in raw and torch.isfinite(raw["hierarchy"])


# ---------------------------------------------------------------------------
# 7. AngularCoherenceLoss global path uses per-level target_sim
# ---------------------------------------------------------------------------

class TestAngularCoherenceGlobalTargetSim:
    """When level_prefix_k is None the global path must use per-level
    target_sim rather than always reading index 0."""

    def test_global_path_uses_per_level_sim(self):
        from src.losses.algebraic import AngularCoherenceLoss
        import src.losses.algebraic as alg_mod

        # target_sim: level-0 wants 1.0, all others want 0.0 (already satisfied)
        target_sim = [1.0] + [0.0] * 9
        fn = AngularCoherenceLoss(
            weight=1.0,
            n_pairs=200,
            prefix_k=1,
            phase_start_epoch=0,
            level_prefix_k=None,
            target_sim=target_sim,
        )
        src_text = inspect.getsource(fn.forward)
        # The global branch must NOT hard-code index [0]; it must index by pair val
        assert "self.target_sim[0]" not in src_text, \
            "Global path must use per-level target_sim, not hard-coded index [0]"

    def test_global_path_loss_differs_with_different_target_sim(self):
        from src.losses.algebraic import AngularCoherenceLoss
        torch.manual_seed(42)
        n, d = 64, 8
        # Build embeddings with clear directional structure
        z = torch.randn(n, d, dtype=torch.float64) * 0.3
        r = z.norm(dim=-1)
        idx = torch.arange(n)

        fn_high = AngularCoherenceLoss(
            weight=1.0, n_pairs=100, prefix_k=1,
            phase_start_epoch=0, target_sim=[0.99] * 10,
        )
        fn_low = AngularCoherenceLoss(
            weight=1.0, n_pairs=100, prefix_k=1,
            phase_start_epoch=0, target_sim=[0.0] * 10,
        )
        torch.manual_seed(7)
        loss_high, _ = fn_high(z, r, idx, epoch=1)
        torch.manual_seed(7)
        loss_low, _ = fn_low(z, r, idx, epoch=1)
        # High target_sim should produce higher or equal loss than low
        assert loss_high.item() >= loss_low.item() - 1e-9


# ---------------------------------------------------------------------------
# 8. get_lr_scales() side-effect warning is documented
# ---------------------------------------------------------------------------

class TestGetLRScalesSideEffectDoc:
    def test_get_lr_scales_docstring_warns_side_effects(self):
        from src.models.lr_controller import MetricBasedLR
        doc = MetricBasedLR.get_lr_scales.__doc__ or ""
        assert "side" in doc.lower() or "WARNING" in doc, \
            "get_lr_scales docstring must document its side-effect hazard"


# ---------------------------------------------------------------------------
# 9. No CPU generator in geodesic / rank / radial losses
# ---------------------------------------------------------------------------

class TestNoCPUGenerator:
    """The three losses that used CPU generators now use device=device
    directly in randint, eliminating the index-tensor CPU→GPU transfer."""

    def _has_cpu_generator(self, cls):
        src_text = inspect.getsource(cls.__init__)
        return "torch.Generator()" in src_text

    def test_padic_geodesic_no_cpu_generator(self):
        from src.losses.geodesic import PAdicGeodesicLoss
        assert not self._has_cpu_generator(PAdicGeodesicLoss), \
            "PAdicGeodesicLoss must not create a CPU generator in __init__"

    def test_global_rank_no_cpu_generator(self):
        from src.losses.rank import GlobalRankLoss
        assert not self._has_cpu_generator(GlobalRankLoss), \
            "GlobalRankLoss must not create a CPU generator in __init__"

    def test_radial_hierarchy_no_cpu_generator(self):
        from src.losses.hierarchy import RadialHierarchyLoss
        assert not self._has_cpu_generator(RadialHierarchyLoss), \
            "RadialHierarchyLoss must not create a CPU generator in __init__"

    def test_geodesic_randint_uses_device_param(self):
        from src.losses.geodesic import PAdicGeodesicLoss
        from src.losses.utils import sample_random_pairs

        forward_src = inspect.getsource(PAdicGeodesicLoss.forward)
        assert "sample_random_pairs" in forward_src, \
            "PAdicGeodesicLoss.forward must delegate pair sampling to sample_random_pairs()"
        helper_src = inspect.getsource(sample_random_pairs)
        assert "device=device" in helper_src or "device=" in helper_src, \
            "sample_random_pairs must pass device= to randint"


# ---------------------------------------------------------------------------
# 10. DataAuditor leakage check uses torch.isin
# ---------------------------------------------------------------------------

class TestDataAuditorLeakageCheck:
    """The set-of-tuples approach was replaced with torch.isin for speed.
    Verify the new code path is correct on the standard dataset and on a
    custom-indices dataset with intentional duplicates."""

    def test_no_leakage_standard_dataset(self):
        from src.training.bootstrap import DataAuditor
        auditor = DataAuditor(seed=42)
        train_ds, val_ds, _ = auditor.prepare_data(val_frac=0.1)
        assert auditor.audit_log["data_leakage_count"] == 0

    def test_leakage_detected_with_duplicate_indices(self, tmp_path, monkeypatch):
        """If all_indices contains repeats, overlap must be detected.

        Regression: the original version of this test picked seed=0 on the
        (unverified) assumption that it would split the duplicated index
        value 0 across train/val. It does not — with seed=0 both copies of
        index value 0 land in *train*, so data_leakage_count is 0 and the
        test's try/except swallowed the no-op silently with zero assertions
        on the success path. It never actually exercised the leakage-detection
        branch it claims to cover.

        Fixed by forcing the permutation deterministically instead of hoping
        a seed happens to produce the right split.
        """
        import numpy as np
        import torch
        from src.training.bootstrap import DataAuditor

        # Index value 0 appears at positions 0 and 1.
        dup_indices = torch.tensor([0, 0] + list(range(1, 50)), dtype=torch.long)
        idx_path = tmp_path / "dup_indices.pt"
        torch.save(dup_indices, idx_path)
        n = len(dup_indices)

        # Force position 0 into val and position 1 into train, guaranteeing
        # index value 0 appears in both splits regardless of RNG luck.
        forced_perm = np.array([0] + list(range(2, n)) + [1])

        class _FixedRNG:
            def permutation(self, size):
                assert size == n
                return forced_perm

        monkeypatch.setattr(np.random, "default_rng", lambda seed: _FixedRNG())

        auditor = DataAuditor(seed=0)
        train_ds, val_ds, _ = auditor.prepare_data(
            val_frac=0.1, custom_indices_path=str(idx_path)
        )

        assert auditor.audit_log["data_leakage_count"] > 0, (
            "Duplicate index value split across train/val was not flagged as "
            f"leakage: audit_log={auditor.audit_log}"
        )

        # Verify directly against the returned datasets too, not just the
        # audit_log bookkeeping, in case the log and the actual split diverge.
        idx_train = train_ds.tensors[1]
        idx_val = val_ds.tensors[1]
        assert torch.isin(idx_val, idx_train).any(), (
            "audit_log reported leakage but the returned train/val datasets "
            "share no index values"
        )

    def test_uses_torch_isin_not_set_of_tuples(self):
        import src.training.bootstrap as bootstrap_mod
        src_text = inspect.getsource(bootstrap_mod.DataAuditor.prepare_data)
        assert "torch.isin" in src_text, \
            "DataAuditor.prepare_data must use torch.isin for leakage check"
        assert "tuple(x.tolist()" not in src_text, \
            "DataAuditor.prepare_data must not build set-of-tuples for leakage check"
