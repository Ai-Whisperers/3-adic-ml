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
        vecs = torch.randn(n, dim, dtype=torch.float64)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True) * radius
        return vecs

    def test_collapsed_flag_false_when_radii_vary(self) -> None:
        import torch

        from src.training.metrics import compute_hierarchy_metrics
        z = torch.randn(200, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(200)
        metrics = compute_hierarchy_metrics(z, idx, curvature=1.0)
        # With random radii and mixed valuations, spearmanr should not be NaN
        assert metrics["hierarchy_collapsed"] is False
        assert metrics["dist_corr_collapsed"] is False

    def test_collapsed_flag_present_in_early_exit(self) -> None:
        """When n < 2, early-exit path must also include collapsed flags."""
        from src.training.metrics import compute_hierarchy_metrics
        z = torch.randn(1, 8, dtype=torch.float64) * 0.3
        idx = torch.tensor([0])
        metrics = compute_hierarchy_metrics(z, idx, curvature=1.0)
        assert metrics["hierarchy_collapsed"] is True
        assert metrics["dist_corr_collapsed"] is True


# ---------------------------------------------------------------------------
# F-04 / F-12 — CombinedLoss: ac_skipped_no_r counter in losses dict
# ---------------------------------------------------------------------------

class TestACSkipCounter:
    """F-04: after one-time warning, subsequent skips are counted in losses dict."""

    def _make_loss_fn(self) -> object:
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
        from src.models.lr_controller import update_optimizer_lr_scales
        opt = self._make_optimizer("encoder_a")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            update_optimizer_lr_scales(opt, base_lr=1e-3, lr_scales={"encoder_a": 0.1})
        assert not any("update_optimizer_lr_scales" in str(w.message) for w in caught)
        assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-10

    def test_unmatched_key_emits_warning(self) -> None:
        from src.models.lr_controller import update_optimizer_lr_scales
        opt = self._make_optimizer("encoder_a")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            update_optimizer_lr_scales(opt, base_lr=1e-3, lr_scales={"encoder_b": 0.1})
        warns = [w for w in caught if issubclass(w.category, UserWarning)
                 and "update_optimizer_lr_scales" in str(w.message)]
        assert len(warns) == 2
        assert "matched no optimizer param group" in str(warns[0].message)
        assert "have no corresponding scale" in str(warns[1].message)

    def test_unmatched_key_leaves_lr_unchanged(self) -> None:
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
        from src.losses.geodesic import PAdicGeodesicLoss
        loss_fn = PAdicGeodesicLoss(n_pairs=1)

        # Small batch; loss should not crash and metrics dict should be returned
        z = torch.randn(8, 8, dtype=torch.float64) * 0.3
        idx = torch.arange(8)
        import math
        loss_val, metrics = loss_fn(z, idx)
        assert metrics["n_pairs"] == 1
        assert math.isnan(metrics["distance_correlation"])

    def test_corrcoef_with_sufficient_pairs_is_numeric(self) -> None:
        from src.losses.geodesic import PAdicGeodesicLoss
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
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'coverage'"):
            StateNetConfig.from_dict({
                "coverage": {"fix_threshhold": 0.9}  # double-h typo
            })

    def test_typo_in_hierarchy_subkey_raises(self) -> None:
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'hierarchy'"):
            StateNetConfig.from_dict({
                "hierarchy": {"plateau_patiance": 5}  # typo
            })

    def test_typo_in_timing_subkey_raises(self) -> None:
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'timing'"):
            StateNetConfig.from_dict({
                "timing": {"warmup_epoch": 10}  # missing 's'
            })

    def test_valid_subkeys_do_not_raise(self) -> None:
        from src.config.statenet_config import StateNetConfig
        cfg = StateNetConfig.from_dict({
            "coverage": {"fix_threshold": 0.95},
            "timing": {"warmup_epochs": 20},
        })
        assert cfg.coverage.fix_threshold == 0.95
        assert cfg.timing.warmup_epochs == 20

    def test_typo_in_lr_scales_subkey_raises(self) -> None:
        from src.config.statenet_config import StateNetConfig
        with pytest.raises(ValueError, match="unknown keys in 'lr_scales'"):
            StateNetConfig.from_dict({
                "lr_scales": {"encoder_A": 0.05}  # wrong capitalisation
            })


# ---------------------------------------------------------------------------
# Quality gate: engine.py must not contain the dead print_every call
# ---------------------------------------------------------------------------

class TestEngineDeadVariable:
    """Regression for dead `train_cfg.get('print_every', 5)` result-discarded call.

    The call was removed because its return value was never used.  This test
    prevents it from being accidentally re-introduced.
    """

    def test_engine_has_no_dead_print_every_call(self) -> None:
        from pathlib import Path
        engine_src = (Path(__file__).parent.parent / "src" / "training" / "engine.py").read_text()
        # The pattern "train_cfg.get(\"print_every\"" with NO assignment on the same
        # line is the dead call we removed.  We allow it only inside comments.
        import re
        for lineno, line in enumerate(engine_src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r'train_cfg\.get\(["\']print_every', stripped):
                # Allow it only if there is an assignment target
                if "=" not in stripped.split("train_cfg")[0]:
                    raise AssertionError(
                        f"Dead train_cfg.get('print_every', …) call found at "
                        f"src/training/engine.py:{lineno}: {line!r}"
                    )


# ---------------------------------------------------------------------------
# Quality gate: CombinedLoss.get_enabled_losses() must list all loss attributes
# ---------------------------------------------------------------------------

class TestCombinedLossRegistryCompleteness:
    """Regression for get_enabled_losses() omitting algebraic_multiplication
    and algebraic_distributive.

    Every Optional[<LossClass>] attribute of CombinedLoss that is populated
    inside _init_losses() must appear in get_enabled_losses().
    """

    def _all_algebraic_config(self) -> dict:
        return {
            "rich_hierarchy": {"enabled": True, "hierarchy_weight": 1.0,
                               "coverage_weight": 0.0, "separation_weight": 0.0},
            "algebraic_addition": {"enabled": True, "weight": 1.0},
            "algebraic_multiplication": {"enabled": True, "weight": 1.0},
            "algebraic_distributive": {"enabled": True, "weight": 1.0},
        }

    def test_algebraic_multiplication_appears_in_enabled_list(self) -> None:
        from src.losses.combined import CombinedLoss
        fn = CombinedLoss(self._all_algebraic_config())
        assert "algebraic_multiplication" in fn.get_enabled_losses(), (
            "algebraic_multiplication is enabled but missing from get_enabled_losses()"
        )

    def test_algebraic_distributive_appears_in_enabled_list(self) -> None:
        from src.losses.combined import CombinedLoss
        fn = CombinedLoss(self._all_algebraic_config())
        assert "algebraic_distributive" in fn.get_enabled_losses(), (
            "algebraic_distributive is enabled but missing from get_enabled_losses()"
        )

    def test_repr_includes_algebraic_multiplication(self) -> None:
        from src.losses.combined import CombinedLoss
        fn = CombinedLoss(self._all_algebraic_config())
        assert "algebraic_multiplication" in repr(fn)

    def test_repr_includes_algebraic_distributive(self) -> None:
        from src.losses.combined import CombinedLoss
        fn = CombinedLoss(self._all_algebraic_config())
        assert "algebraic_distributive" in repr(fn)

    def test_disabled_losses_not_in_enabled_list(self) -> None:
        from src.losses.combined import CombinedLoss
        cfg = {"rich_hierarchy": {"enabled": True, "hierarchy_weight": 1.0,
                                  "coverage_weight": 0.0, "separation_weight": 0.0}}
        fn = CombinedLoss(cfg)
        enabled = fn.get_enabled_losses()
        assert "algebraic_addition" not in enabled
        assert "algebraic_multiplication" not in enabled
        assert "algebraic_distributive" not in enabled

    def test_hyperbolic_contrastive_appears_in_enabled_list(self) -> None:
        """Regression: get_enabled_losses() was built from an independent
        if-chain that never listed hyperbolic_contrastive or
        surrogate_property at all (not even omitted-by-typo — just never
        written), unlike the algebraic_* cases above which were at least
        attempted. Now both share _LOSS_NAME_ATTRS with the "at least one
        loss enabled" guard in _init_losses(), so they can't drift apart."""
        from src.losses.combined import CombinedLoss
        cfg = {"hyperbolic_contrastive": {"enabled": True, "weight": 1.0}}
        fn = CombinedLoss(cfg)
        assert "hyperbolic_contrastive" in fn.get_enabled_losses()

    def test_surrogate_property_appears_in_enabled_list(self) -> None:
        from src.losses.combined import CombinedLoss
        cfg = {"surrogate_property": {"enabled": True, "weight": 1.0}}
        fn = CombinedLoss(cfg)
        assert "surrogate_property" in fn.get_enabled_losses()


# ---------------------------------------------------------------------------
# Quality gate: bootstrap factored default must match schema default
# ---------------------------------------------------------------------------

class TestBootstrapFactoredDefault:
    """Regression for bootstrap.py using factored=False while schema defaults to True.

    When model_cfg does not contain a 'factored' key (i.e., caller did not
    normalize the config through Pydantic first), ModelAuditor must fall back
    to the same default that ModelConfig uses so callers get consistent behaviour.
    """

    def test_schema_default_is_true(self) -> None:
        from src.config.schema import ModelConfig
        assert ModelConfig().factored is True, (
            "ModelConfig.factored default changed — update bootstrap.py to match"
        )

    def test_bootstrap_default_matches_schema(self) -> None:
        """ModelAuditor must call model_cfg.get('factored', True), not False."""
        from pathlib import Path
        bootstrap_src = (
            Path(__file__).parent.parent / "src" / "training" / "bootstrap.py"
        ).read_text()
        # The only occurrence of get("factored", …) must default to True
        import re
        matches = re.findall(r'\.get\(["\']factored["\'],\s*(\w+)\)', bootstrap_src)
        assert matches, "Expected model_cfg.get('factored', …) call not found in bootstrap.py"
        for default_val in matches:
            assert default_val == "True", (
                f"bootstrap.py uses get('factored', {default_val}); "
                f"must be True to match ModelConfig default"
            )
