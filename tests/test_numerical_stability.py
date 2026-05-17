# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Numerical Integrity and Tensor Safeguard Tests.

Verifies the 'Safety Wrapper' (NaN/Inf sanitizer) and curvature clamping.
"""

import pytest
import torch
import torch.nn as nn
from src.training.engine import train_epoch
from src.models.hyperbolic_projection import HyperbolicProjection
from unittest.mock import MagicMock


class TestNumericalStability:
    def test_nan_sanitizer_catches_vae_a_instability(self):
        """Verify that train_epoch raises RuntimeError when VAE-A loss is NaN."""
        # Mock components
        model = MagicMock(spec=nn.Module)
        model.train.return_value = None
        # Mock model.projections.get_curvature()
        model.projections = MagicMock()
        model.projections.get_curvature.return_value = 1.0
        
        # Mock loss function to return NaN in losses_A
        loss_fn = MagicMock()
        loss_fn.return_value = ({"total": torch.tensor(float('nan')), "recon": torch.tensor(0.1)}, {})
        
        loss_fn_b = MagicMock()
        loss_fn_b.return_value = ({"total": torch.tensor(0.5)}, {})
        
        # Setup minimal inputs for train_epoch
        loader = [(torch.randn(1, 9), torch.tensor([0]))]
        optimizer = MagicMock()
        device = torch.device("cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        
        with pytest.raises(RuntimeError, match="Numerical instability: VAE-A total loss is nan"):
            train_epoch(
                epoch=1, model=model, loader=loader, optimizer=optimizer,
                loss_fn=loss_fn, loss_fn_b=loss_fn_b, device=device,
                scaler=scaler, max_grad_norm=1.0, use_amp=False,
                dual_state=None, current_dual_weights=None,
                hw_monitor=None, reporting=MagicMock(), global_step_start=0
            )

    def test_inf_sanitizer_catches_vae_b_instability(self):
        """Verify that train_epoch raises RuntimeError when VAE-B loss is Inf."""
        model = MagicMock(spec=nn.Module)
        model.train.return_value = None
        model.projections = MagicMock()
        model.projections.get_curvature.return_value = 1.0
        
        loss_fn = MagicMock()
        loss_fn.return_value = ({"total": torch.tensor(0.5)}, {})
        
        loss_fn_b = MagicMock()
        loss_fn_b.return_value = ({"total": torch.tensor(float('inf'))}, {})
        
        loader = [(torch.randn(1, 9), torch.tensor([0]))]
        optimizer = MagicMock()
        device = torch.device("cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        
        with pytest.raises(RuntimeError, match="Numerical instability: VAE-B total loss is inf"):
            train_epoch(
                epoch=1, model=model, loader=loader, optimizer=optimizer,
                loss_fn=loss_fn, loss_fn_b=loss_fn_b, device=device,
                scaler=scaler, max_grad_norm=1.0, use_amp=False,
                dual_state=None, current_dual_weights=None,
                hw_monitor=None, reporting=MagicMock(), global_step_start=0
            )


class TestCurvatureSafeguards:
    def test_curvature_clamping_min(self):
        """Verify that learnable curvature is brought to a safe range if too low."""
        proj = HyperbolicProjection(learnable_curvature=True, curvature=1.0)
        proj.to(torch.float64)
        
        param = None
        for name, p in proj.manifold.named_parameters():
            if "c" in name or "k" in name:
                param = p
                break
        assert param is not None

        # Manually set curvature EXTREMELY low
        with torch.no_grad():
            param.data.fill_(-100.0) # For softplus, this would be near zero c
        
        # Trigger clamping
        proj._clamp_curvature()
        
        # Should be at least 0.1 (actual range depends on softplus, but -3.0 -> softplus(-3) ≈ 0.048)
        # Wait, if I set min=-3.0 in code, softplus(-3) is ~0.05.
        # Let's just check it's >= 0.04
        assert proj.manifold.c.item() >= 0.04

    def test_curvature_clamping_max(self):
        """Verify that learnable curvature is restricted if too high."""
        proj = HyperbolicProjection(learnable_curvature=True, curvature=1.0)
        proj.to(torch.float64)
        
        param = None
        for name, p in proj.manifold.named_parameters():
            if "c" in name or "k" in name:
                param = p
                break
        assert param is not None

        # Manually set curvature EXTREMELY high
        with torch.no_grad():
            param.data.fill_(100.0)
            
        # Trigger clamping
        proj._clamp_curvature()
        
        # Should be restricted (max=5.0 in code for parameter -> softplus(5) ≈ 5.0)
        assert proj.manifold.c.item() == pytest.approx(5.0, abs=1e-2)

    def test_non_learnable_curvature_not_clamped(self):
        """Verify that fixed curvature is NOT modified by clamping."""
        proj = HyperbolicProjection(learnable_curvature=False, curvature=0.05)
        
        # Forward pass
        proj(torch.randn(1, 16))
        
        # Should stay at 0.05 even though it's < 0.1, because it's not learnable
        assert proj.manifold.c.item() == pytest.approx(0.05)
