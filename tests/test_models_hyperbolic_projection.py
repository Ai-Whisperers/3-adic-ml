# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Regression tests for src/models/hyperbolic_projection.py.

This module is the site of the two most severe historical bugs in the
project (see CLAUDE.md):

  1. V6.2 "max_radius saturation" — before a learnable tangent_scale
     existed, encoder outputs (~norm 4) fed directly into expmap0 saturate
     the Poincaré ball: expmap0's output norm asymptotes to 1 very quickly
     for large tangent-vector magnitudes, so after the max_radius clamp
     *every* embedding landed at exactly 0.95 regardless of its true
     hierarchy level. Fixed by adding a learnable tangent_scale (init=0.1)
     that keeps expmap0's input in its non-saturating regime.

  2. V24 "tangent_scale directional collapse" — the raw `tangent_scale`
     nn.Parameter could be driven to ~0 by gradient descent. At scale=0,
     `tangent_net(0 * z_theta)` is the same constant for every sample
     regardless of z_theta, so in the factored path all directions
     collapse to one vector (pairwise cosine similarity -> 1.0,
     empirically observed epochs 25-75 on VAE-B). This is a
     self-reinforcing fixed point: once collapsed, there is no
     per-sample gradient signal left to recover. Fixed by storing
     log_tangent_scale and using exp() so the effective scale can never
     reach exactly 0, plus a hard clamp on the log-space value itself.

Nuance uncovered while writing these tests: `init_identity=True` makes the
collapse mechanism inert *at initialization* only. With the last tangent_net
layer zeroed, tangent_net(x) == 0 for any x, so direction =
normalize(scale * z_theta) == normalize(z_theta) regardless of scale --
scale cancels out of the normalization entirely. The actual V24 collapse
happened after tangent_net's weights drifted away from zero during
training (epochs 25-75), reintroducing the bias-dominated collapse
mechanism. The collapse-regression tests below therefore use
init_identity=False (trained-like weights) to actually exercise the
failure mode; init_identity=True would pass vacuously.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.hyperbolic_projection import (
    DualHyperbolicProjection,
    HyperbolicProjection,
)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------

class TestTangentScaleValidation:
    def test_zero_tangent_scale_init_rejected(self):
        """Regression: validation used to be `< 0`, which let 0 through and
        started training already at the degenerate fixed point."""
        with pytest.raises(ValueError, match="tangent_scale_init must be > 0"):
            HyperbolicProjection(tangent_scale_init=0.0)

    def test_negative_tangent_scale_init_rejected(self):
        with pytest.raises(ValueError, match="tangent_scale_init must be > 0"):
            HyperbolicProjection(tangent_scale_init=-0.1)

    def test_positive_tangent_scale_init_accepted(self):
        proj = HyperbolicProjection(latent_dim=16, tangent_scale_init=0.1)
        assert proj.tangent_scale.item() == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# Bug 1 (V6.2): max_radius saturation
# ---------------------------------------------------------------------------

def _encoder_like_batch(seed: int = 0) -> torch.Tensor:
    """A batch spanning small and large tangent-space magnitudes, similar to
    real encoder outputs before the fix (~norm 4)."""
    g = torch.Generator().manual_seed(seed)
    small = torch.randn(32, 16, generator=g, dtype=torch.float64) * 0.5
    large = torch.randn(32, 16, generator=g, dtype=torch.float64) * 4.0
    return torch.cat([small, large], dim=0)


class TestMaxRadiusSaturationRegression:
    """Without a small enough tangent_scale, expmap0 saturates every sample
    to exactly max_radius, destroying all radial hierarchy signal."""

    def test_default_scale_does_not_saturate(self):
        torch.manual_seed(0)
        proj = HyperbolicProjection(
            latent_dim=16, hidden_dim=32, tangent_scale_init=0.1,
            max_radius=0.95, init_identity=True,
        )
        z = _encoder_like_batch()
        with torch.no_grad():
            z_hyp = proj(z)
        norms = z_hyp.norm(dim=-1)

        assert norms.std().item() > 0.05, (
            f"Radii barely vary (std={norms.std().item():.6f}) — looks saturated: "
            f"min={norms.min():.4f} max={norms.max():.4f}"
        )
        frac_saturated = (norms > proj.max_radius - 1e-4).float().mean().item()
        assert frac_saturated < 0.5, (
            f"{frac_saturated:.0%} of embeddings pinned at max_radius={proj.max_radius}"
        )

    def test_unscaled_tangent_saturates_almost_everything(self):
        """Differential proof that the default init is load-bearing, not
        incidental: tangent_scale_init=1.0 (effectively unscaled, matching
        pre-V6.2 behavior) collapses most embeddings onto the ball boundary
        for the exact same encoder-realistic input magnitudes that the
        default 0.1 handles fine."""
        torch.manual_seed(0)
        proj = HyperbolicProjection(
            latent_dim=16, hidden_dim=32, tangent_scale_init=1.0,
            max_radius=0.95, init_identity=True,
        )
        z = _encoder_like_batch()
        with torch.no_grad():
            z_hyp = proj(z)
        norms = z_hyp.norm(dim=-1)
        frac_saturated = (norms > proj.max_radius - 1e-4).float().mean().item()
        assert frac_saturated > 0.7, (
            f"Expected near-total saturation with unscaled tangent input, "
            f"got only {frac_saturated:.0%} (std={norms.std().item():.6f})"
        )

    def test_radial_ordering_preserved_along_fixed_direction(self):
        """Larger tangent-space magnitude along a fixed direction must map
        to a strictly larger Poincaré radius — the mechanism every radial
        hierarchy loss in this codebase relies on."""
        torch.manual_seed(0)
        proj = HyperbolicProjection(
            latent_dim=16, hidden_dim=32, tangent_scale_init=0.1,
            max_radius=0.95, init_identity=True,
        )
        direction = torch.randn(1, 16, dtype=torch.float64)
        direction = direction / direction.norm()

        radii = []
        for input_norm in (0.5, 1.0, 2.0, 4.0, 8.0):
            with torch.no_grad():
                z_hyp = proj(direction * input_norm)
            radii.append(z_hyp.norm().item())

        assert radii == sorted(radii), (
            f"Radii not monotonically increasing with input magnitude: {radii}"
        )
        assert radii[0] < radii[-1] - 1e-6


# ---------------------------------------------------------------------------
# Bug 2 (V24): tangent_scale directional collapse
# ---------------------------------------------------------------------------

class TestTangentScaleDirectionalCollapseRegression:
    """tangent_scale must never reach exactly 0, and gradient descent must
    always retain an escape route from a near-collapsed state."""

    def test_tangent_scale_always_strictly_positive(self):
        proj = HyperbolicProjection(latent_dim=16, tangent_scale_init=0.1)
        with torch.no_grad():
            proj.log_tangent_scale.data.fill_(-1e6)  # simulate runaway collapse
        assert proj.tangent_scale.item() > 0.0

    def test_tangent_scale_clamped_to_documented_floor(self):
        proj = HyperbolicProjection(latent_dim=16, tangent_scale_init=0.1)
        with torch.no_grad():
            proj.log_tangent_scale.data.fill_(-1e6)
        scale = proj.tangent_scale.item()
        assert scale == pytest.approx(math.exp(-10.0), rel=1e-6)

    def test_tangent_scale_clamped_to_documented_ceiling(self):
        proj = HyperbolicProjection(latent_dim=16, tangent_scale_init=0.1)
        with torch.no_grad():
            proj.log_tangent_scale.data.fill_(1e6)
        scale = proj.tangent_scale.item()
        assert scale == pytest.approx(math.exp(3.0), rel=1e-6)

    def test_clamp_mutates_underlying_parameter_on_access(self):
        """The property clamps log_tangent_scale.data in place on every
        read — this is what actually prevents the raw parameter from
        drifting into a pathological range over training, since forward()
        reads self.tangent_scale on every step."""
        proj = HyperbolicProjection(latent_dim=16, tangent_scale_init=0.1)
        with torch.no_grad():
            proj.log_tangent_scale.data.fill_(-1e6)
        _ = proj.tangent_scale  # trigger the clamp side-effect
        assert proj.log_tangent_scale.item() == pytest.approx(-10.0)

    def test_gradient_never_vanishes_at_the_collapsed_floor(self):
        """The core V24 fix guarantee: even parked at the lowest allowed
        scale, gradient must still flow to log_tangent_scale so an
        optimizer *can* climb back out. CLAUDE.md documents that the
        pre-fix raw parameterization had no such guarantee: 'once
        collapsed, cannot recover'.

        Uses init_identity=False — see module docstring for why
        init_identity=True cannot exhibit this failure mode at all.
        """
        torch.manual_seed(0)
        proj = HyperbolicProjection(
            latent_dim=16, hidden_dim=32, factored=True, radial_dims=4,
            tangent_scale_init=0.1, init_identity=False,
        )
        with torch.no_grad():
            proj.log_tangent_scale.data.fill_(-10.0)  # parked at the clamp floor

        z = torch.randn(64, 16, dtype=torch.float64)
        z_hyp, r = proj(z)
        direction = z_hyp / z_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        # Pairwise cosine similarity across the batch — exactly the quantity
        # documented at ~1.000000 in the real V24 collapse.
        cos_sim = direction @ direction.T
        assert cos_sim.mean().item() > 0.99, (
            "Sanity check failed: batch is not actually near-collapsed at "
            "the floor, so this test isn't exercising the failure mode"
        )

        loss = cos_sim.mean()  # angular-coherence-style pressure to diversify
        loss.backward()

        assert proj.log_tangent_scale.grad is not None
        assert proj.log_tangent_scale.grad.abs().item() > 1e-12, (
            "Gradient vanished at the collapsed floor — an optimizer would "
            "be permanently stuck there, reproducing the V24 bug"
        )

    def test_dual_projection_tangent_scales_are_independent(self):
        """VAE-A and VAE-B must have independent tangent_scale parameters.
        V24's bug affected only VAE-B ('VAE-A escaped by chance of
        initialization'), which is only possible if the two projections
        don't share this piece of state."""
        torch.manual_seed(0)
        dual = DualHyperbolicProjection(
            latent_dim=16, hidden_dim=32, factored=True, radial_dims=4,
        )
        assert dual.proj_A.log_tangent_scale is not dual.proj_B.log_tangent_scale

        with torch.no_grad():
            dual.proj_B.log_tangent_scale.data.fill_(-10.0)

        scale_A = dual.proj_A.tangent_scale.item()
        scale_B = dual.proj_B.tangent_scale.item()
        assert abs(scale_A - scale_B) > 0.01, (
            f"proj_B's collapsed tangent_scale leaked into proj_A: "
            f"A={scale_A}, B={scale_B}"
        )
