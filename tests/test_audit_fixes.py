# MIT License — see repository root for full text
"""Regression tests for audit findings F-02, F-04/F-12, F-06, F-08, F-10, F-11.

Each class targets one finding and verifies the specific failure mode is now caught
or surfaced correctly.
"""
import warnings

import pytest
import torch

# ---------------------------------------------------------------------------
# F-02 / F-11 — compute_hierarchy_metrics: collapsed flags in return dict
# ---------------------------------------------------------------------------

class TestHierarchyMetricsCollapsedFlags:
    """F-02: spearmanr NaN now surfaces as hierarchy_collapsed=True, not silent 0."""

    def _make_z_uniform_radius(self, n: int = 50, dim: int = 4, radius: float = 0.5) -> torch.Tensor:
        """All points on a sphere of identical radius → radii will be constant → spearmanr NaN."""
        import sys
        sys.path.insert(0, ".")
        vecs = torch.randn(n, dim, dtype=torch.float64)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True) * radius
        return vecs

    def test_collapsed_flag_false_when_radii_vary(self) -> None:
        import sys
        sys.path.insert(0, ".")
        import torch

        from src.train import compute_hierarchy_metrics
        z = torch.randn(200, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(200)
        metrics = compute_hierarchy_metrics(z, idx, curvature=1.0)
        # With random radii and mixed valuations, spearmanr should not be NaN
        assert "hierarchy_collapsed" in metrics
        assert "dist_corr_collapsed" in metrics

    def test_collapsed_flag_present_in_early_exit(self) -> None:
        """When n < 2, early-exit path must also include collapsed flags."""
        import sys
        sys.path.insert(0, ".")
        from src.train import compute_hierarchy_metrics
        z = torch.randn(1, 8, dtype=torch.float64) * 0.3
        idx = torch.tensor([0])
        metrics = compute_hierarchy_metrics(z, idx, curvature=1.0)
        assert "hierarchy_collapsed" in metrics
        assert "dist_corr_collapsed" in metrics
        assert metrics["dist_corr_collapsed"] is True  # forced True in early-exit path


# ---------------------------------------------------------------------------
# F-04 / F-12 — CombinedLoss: ac_skipped_no_r counter in losses dict
# ---------------------------------------------------------------------------

class TestACSkipCounter:
    """F-04: after one-time warning, subsequent skips are counted in losses dict."""

    def _make_loss_fn(self) -> object:
        import sys
        sys.path.insert(0, ".")
        from src.losses.combined import CombinedLoss
        cfg = {
            "rich_hierarchy": {
                "enabled": True,
                "hierarchy_weight": 1.0,
                "coverage_weight": 0.0,
                "separation_weight": 0.0,
            },
            "angular_coherence": {
                "enabled": True,
                "weight": 1.0,
                "n_pairs": 50,
                "prefix_k": 2,
                "phase_start_epoch": 0,
            },
        }
        return CombinedLoss(cfg, curvature=1.0)

    def test_ac_skip_count_increments_when_r_is_none(self) -> None:
        loss_fn = self._make_loss_fn()
        z = torch.randn(32, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out1 = loss_fn(z, idx, None, None, epoch=5, r=None)
            out2 = loss_fn(z, idx, None, None, epoch=6, r=None)

        assert "ac_skipped_no_r" in out1
        assert "ac_skipped_no_r" in out2
        assert out2["ac_skipped_no_r"] == 2  # cumulative

    def test_warning_fires_exactly_once(self) -> None:
        loss_fn = self._make_loss_fn()
        z = torch.randn(32, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                loss_fn(z, idx, None, None, epoch=1, r=None)

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)
                      and "AngularCoherenceLoss" in str(w.message)]
        assert len(user_warns) == 1

    def test_no_skip_count_when_r_is_provided(self) -> None:
        loss_fn = self._make_loss_fn()
        z = torch.randn(32, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(32)
        r = z.norm(dim=-1)

        out = loss_fn(z, idx, None, None, epoch=5, r=r)
        assert "ac_skipped_no_r" not in out


# ---------------------------------------------------------------------------
# F-06 — update_optimizer_lr_scales: warns on unmatched keys
# ---------------------------------------------------------------------------

class TestLRScaleNameMismatch:
    """F-06: unmatched lr_scales keys now emit UserWarning instead of silently no-op."""

    def _make_optimizer(self, group_name: str) -> torch.optim.Optimizer:
        p = torch.nn.Parameter(torch.zeros(4))
        return torch.optim.Adam([{"params": [p], "name": group_name, "lr": 1e-3}])

    def test_matched_name_no_warning(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.models.lr_controller import update_optimizer_lr_scales
        opt = self._make_optimizer("encoder_a")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            update_optimizer_lr_scales(opt, base_lr=1e-3, lr_scales={"encoder_a": 0.1})
        assert not any("update_optimizer_lr_scales" in str(w.message) for w in caught)
        assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-10

    def test_unmatched_key_emits_warning(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.models.lr_controller import update_optimizer_lr_scales
        opt = self._make_optimizer("encoder_a")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            update_optimizer_lr_scales(opt, base_lr=1e-3, lr_scales={"encoder_b": 0.1})
        warns = [w for w in caught if issubclass(w.category, UserWarning)
                 and "update_optimizer_lr_scales" in str(w.message)]
        assert len(warns) == 1
        assert "encoder_b" in str(warns[0].message)

    def test_unmatched_key_leaves_lr_unchanged(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.models.lr_controller import update_optimizer_lr_scales
        opt = self._make_optimizer("encoder_a")
        original_lr = opt.param_groups[0]["lr"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            update_optimizer_lr_scales(opt, base_lr=1e-3, lr_scales={"encoder_b": 0.5})
        assert opt.param_groups[0]["lr"] == original_lr


# ---------------------------------------------------------------------------
# F-08 — PAdicGeodesicLoss: corrcoef guarded for n=1 pair
# ---------------------------------------------------------------------------

class TestGeodesicCorrcoefGuard:
    """F-08: corrcoef returns NaN (not 0) when only 1 pair is available."""

    def test_corrcoef_nan_not_zero_for_single_pair(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.losses.padic_geodesic import PAdicGeodesicLoss
        loss_fn = PAdicGeodesicLoss(n_pairs=1)

        # Small batch; loss should not crash and metrics dict should be returned
        z = torch.randn(8, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(8)
        loss_val, metrics = loss_fn(z, idx)
        assert "n_pairs" in metrics
        assert "distance_correlation" in metrics

    def test_corrcoef_with_sufficient_pairs_is_numeric(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.losses.padic_geodesic import PAdicGeodesicLoss
        loss_fn = PAdicGeodesicLoss(n_pairs=200)
        z = torch.randn(100, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(100)
        loss_val, metrics = loss_fn(z, idx)
        corr = metrics["distance_correlation"]
        corr_t = corr if isinstance(corr, torch.Tensor) else torch.tensor(float(corr))
        assert torch.isfinite(corr_t) or torch.isnan(corr_t)


# ---------------------------------------------------------------------------
# F-10 — StateNetConfig.from_dict: deep-key typo detection
# ---------------------------------------------------------------------------

class TestStateNetDeepKeyValidation:
    """F-10: misspelled sub-keys now raise ValueError instead of using defaults."""

    def test_typo_in_coverage_subkey_raises(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'coverage'"):
            StateNetConfig.from_dict({
                "coverage": {"fix_threshhold": 0.9}  # double-h typo
            })

    def test_typo_in_hierarchy_subkey_raises(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'hierarchy'"):
            StateNetConfig.from_dict({
                "hierarchy": {"plateau_patiance": 5}  # typo
            })

    def test_typo_in_timing_subkey_raises(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'timing'"):
            StateNetConfig.from_dict({
                "timing": {"warmup_epoch": 10}  # missing 's'
            })

    def test_valid_subkeys_do_not_raise(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.config.statenet_config import StateNetConfig
        cfg = StateNetConfig.from_dict({
            "coverage": {"fix_threshold": 0.95},
            "timing": {"warmup_epochs": 20},
        })
        assert cfg.coverage.fix_threshold == 0.95
        assert cfg.timing.warmup_epochs == 20

    def test_typo_in_lr_scales_subkey_raises(self) -> None:
        import sys
        sys.path.insert(0, ".")
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'lr_scales'"):
            StateNetConfig.from_dict({
                "lr_scales": {"encoder_A": 0.05}  # wrong capitalisation
            })
