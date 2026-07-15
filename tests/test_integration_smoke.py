# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Integration smoke test — full training loop, 3 epochs, CPU only.

Exercises the entire path from data loading through train_model() to
checkpoint creation without touching any mocked internals.  Runtime
target: < 15 s on a modern CPU core (tiny model, batch_size=512).

Marked with ``@pytest.mark.integration`` so CI can gate on them
separately if needed.  Run locally with:
    pytest tests/test_integration_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Minimal config — bypasses normalize_config / Pydantic so the test stays
# self-contained and doesn't depend on schema evolution.
# ---------------------------------------------------------------------------

_SMOKE_CONFIG: Dict[str, Any] = {
    "device": {"use_amp": False},
    "data": {"valuation_type": "index"},
    "model": {
        "name": "TernaryVAEV6Controllable",
        "latent_dim": 8,
        "hidden_dim": 16,
        "max_radius": 0.95,
        "curvature": 1.0,
        "factored": False,
        "radial_dims": 4,
        "projection_layers": 1,
        "projection_dropout": 0.0,
        "init_identity": True,
        "tangent_scale": 0.1,
        "encoder_type": "improved",
        "decoder_type": "improved",
        "learnable_curvature": False,
        "positional_encoding": False,
        "pos_weight_base": 3.0,
        "detach_radial": False,
    },
    "training": {
        "epochs": 3,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "eval_every": 1,
        "save_every": 100,      # won't fire in 3 epochs
        "val_frac": 0.1,
        "num_workers": 0,       # avoid multiprocessing in tests
        "seed": 42,
        "scheduler": {"type": "cosine"},
    },
    "loss": {
        "rich_hierarchy": {
            "enabled": True,
            "hierarchy_weight": 1.0,
            "coverage_weight": 1.0,
            "separation_weight": 0.5,
        },
        "hyperbolic_kl": {
            "enabled": True,
            "beta": 0.1,
            "weight": 0.01,
            "free_bits": 0.0,
            "variance_only": False,
            "warmup_epochs": 2,   # exercises the new KL warmup path
        },
    },
    "statenet": {"enabled": False},
    "option_c": {"enabled": False},
    "riemannian": {"enabled": False},
}


# ---------------------------------------------------------------------------
# Module-scoped fixture: run training once, share results across all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory):
    """Run 3 training epochs and return (TrainingResults, log_dir)."""
    from src.models.vae import TernaryVAEV6Controllable
    from src.training.bootstrap import DataAuditor, set_determinism
    from src.training.engine import train_model
    from src.training.reporting import ReportingManager
    from src.training.setup import (
        setup_dataloaders,
        setup_losses,
        setup_optimizer,
        setup_scheduler,
    )

    set_determinism(seed=42)
    device = torch.device("cpu")
    cfg = _SMOKE_CONFIG

    # Data
    auditor = DataAuditor(seed=42)
    train_ds, val_ds, _ = auditor.prepare_data(
        val_frac=cfg["training"]["val_frac"], device=device
    )
    train_loader, val_loader = setup_dataloaders(train_ds, val_ds, cfg, seed=42)

    # Model — both encoders trainable so all param groups are exercised
    model = TernaryVAEV6Controllable(
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        max_radius=cfg["model"]["max_radius"],
        curvature=cfg["model"]["curvature"],
        factored=cfg["model"]["factored"],
        radial_dims=cfg["model"]["radial_dims"],
        n_projection_layers=cfg["model"]["projection_layers"],
        projection_dropout=cfg["model"]["projection_dropout"],
        init_identity=cfg["model"]["init_identity"],
        tangent_scale_init=cfg["model"]["tangent_scale"],
        encoder_type=cfg["model"]["encoder_type"],
        decoder_type=cfg["model"]["decoder_type"],
        learnable_curvature=cfg["model"]["learnable_curvature"],
        positional_encoding=cfg["model"]["positional_encoding"],
        encoder_a_trainable=True,   # ensure all param groups exist
        encoder_b_trainable=True,
        projections_trainable=True,
    ).to(device)

    # Losses + optimizer + scheduler
    loss_fn, loss_fn_b = setup_losses(cfg, device)
    loss_params = list(loss_fn.parameters())
    optimizer = setup_optimizer(model, cfg, loss_params)
    scheduler = setup_scheduler(optimizer, cfg)

    # Reporting — stub TensorBoard logger so no file I/O overhead
    log_dir = tmp_path_factory.mktemp("smoke_run")
    reporting = ReportingManager(log_dir, cfg, tb_logger=MagicMock())

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        loss_fn_b=loss_fn_b,
        device=device,
        reporting=reporting,
        config=cfg,
        lr_controller=None,
        dual_state=None,
        grokking_detector=None,
        vis_pipeline=None,
        use_amp=False,
    )

    # Mirror what main() does after train_model returns
    reporting.save_results(results)

    return results, log_dir, model, optimizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSmokeTrainingLoop:
    """Training loop produces finite, valid outputs after 3 epochs."""

    def test_results_keys_present(self, smoke_results):
        results, *_ = smoke_results
        for key in ("epochs_trained", "best_Q", "best_hierarchy", "best_coverage"):
            assert key in results, f"Missing key '{key}' in TrainingResults"

    def test_epochs_trained(self, smoke_results):
        results, *_ = smoke_results
        assert results["epochs_trained"] == _SMOKE_CONFIG["training"]["epochs"]

    def test_loss_is_finite(self, smoke_results):
        """Total loss must be finite (no NaN/Inf explosions)."""
        # train_model would have raised RuntimeError on NaN; reaching here is enough.
        # Additional check: best_Q is a valid float.
        results, *_ = smoke_results
        assert isinstance(results["best_Q"], float)
        assert not (results["best_Q"] != results["best_Q"]), "best_Q is NaN"

    def test_coverage_is_valid_float(self, smoke_results):
        """best_coverage must be a float in [0, 1] (even if 0 in a 3-epoch toy run)."""
        results, *_ = smoke_results
        cov = results["best_coverage"]
        assert isinstance(cov, float), f"best_coverage is not float: {type(cov)}"
        assert 0.0 <= cov <= 1.0, f"best_coverage={cov} out of [0, 1]"

    def test_hierarchy_computed(self, smoke_results):
        """Hierarchy correlation is returned as a valid float (even if near 0)."""
        results, *_ = smoke_results
        h = results["best_hierarchy"]
        assert isinstance(h, float)
        assert -1.0 <= h <= 1.0, f"hierarchy={h} out of [-1, 1]"

    def test_final_checkpoint_saved(self, smoke_results):
        _, log_dir, *_ = smoke_results
        final_ckpt = log_dir / "checkpoints" / "final.pt"
        assert final_ckpt.exists(), f"final.pt checkpoint not written to {final_ckpt}"

    def test_final_checkpoint_loadable(self, smoke_results):
        """Checkpoint must be loadable and contain expected keys."""
        _, log_dir, *_ = smoke_results
        ckpt = torch.load(log_dir / "checkpoints" / "final.pt", weights_only=False)
        for key in ("epoch", "model_state_dict", "optimizer_state_dict"):
            assert key in ckpt, f"Checkpoint missing key '{key}'"
        assert ckpt["epoch"] == _SMOKE_CONFIG["training"]["epochs"]

    def test_results_json_written(self, smoke_results):
        """results.json must exist in the log directory."""
        import json
        _, log_dir, *_ = smoke_results
        results_path = log_dir / "results.json"
        assert results_path.exists(), "results.json not written"
        data = json.loads(results_path.read_text())
        assert "best_Q" in data

    def test_model_parameters_changed(self, smoke_results):
        """At least some parameters must have non-zero gradients (VAE-B not dead)."""
        _, _, model, optimizer = smoke_results
        # Check that the optimizer has accumulated state (Adam exp_avg) for params
        has_state = sum(1 for g in optimizer.param_groups for p in g["params"]
                        if p in optimizer.state)
        assert has_state > 0, "Optimizer has no state — no backward pass occurred"

    def test_vae_b_gradients_flow(self, smoke_results):
        """VAE-B must have been trained (regression: VAE-B was dead before V6.2 fix)."""
        _, _, model, optimizer = smoke_results
        # Every named param group should have optimizer state if it participated
        b_params = list(model.head_B.parameters()) + list(model.decoder_B.parameters())
        b_with_state = sum(1 for p in b_params if p in optimizer.state)
        assert b_with_state > 0, (
            "VAE-B parameters have no optimizer state — VAE-B did not participate in training"
        )

    def test_best_q_checkpoint_saved_when_positive(self, smoke_results):
        """best_Q.pt must be present if Q ever exceeded the initial sentinel -1."""
        results, log_dir, *_ = smoke_results
        if results["best_Q"] > -1.0:
            best_ckpt = log_dir / "checkpoints" / "best_Q.pt"
            assert best_ckpt.exists(), (
                f"best_Q={results['best_Q']:.4f} was recorded but best_Q.pt not written"
            )


@pytest.mark.integration
class TestSmokeNumericalHealth:
    """Numerical-stability invariants must hold throughout training."""

    def test_no_nan_in_model_weights(self, smoke_results):
        _, _, model, _ = smoke_results
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in model parameter '{name}'"

    def test_radii_inside_poincare_ball(self, smoke_results):
        """After training, all embeddings must remain strictly inside the ball."""
        _, _, model, _ = smoke_results
        from src.core import TERNARY
        from src.geometry import hyperbolic_radius

        model.eval()
        device = next(model.parameters()).device
        # Sample a small subset to keep the test fast
        idx = torch.arange(200)
        ops = TERNARY.to_ternary(idx).to(device)
        with torch.no_grad():
            out = model(ops)
        z_A = out["z_A_hyp"]
        norms = torch.norm(z_A, dim=-1)
        assert (norms < 1.0).all(), (
            f"Some embeddings escaped the Poincaré ball: max_norm={norms.max():.6f}"
        )
