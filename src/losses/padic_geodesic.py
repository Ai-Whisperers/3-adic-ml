# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Unified P-Adic Geodesic Loss for V5.11.

THE KEY V5.11 INNOVATION: Unify hierarchy and correlation in one loss.

Instead of competing losses:
  - ranking_loss: compares triplet orderings (relative)
  - radial_loss: pushes to target radii (absolute)

Use single geodesic loss:
  - For each pair (i,j): |d_poincare(z_i, z_j) - target(v_3(|i-j|))|²
  - Target maps valuation to hyperbolic distance
  - High valuation → small geodesic distance (automatically near origin)

In proper hyperbolic geometry, hierarchy IS correlation. The Poincaré metric
naturally couples them - two points near origin have smaller geodesic distance
than two at boundary.

Single responsibility: Unified p-adic geodesic alignment.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import TERNARY
from ..geometry import hyperbolic_radius, poincare_distance
from .base import HierarchyLossBase, MetricsDict, RichHierarchyLossBase


def _exponential_target_radii(
    max_valuation: int, inner_radius: float, outer_radius: float, scale: float = 3.0
) -> torch.Tensor:
    """Precompute target radii using exponential decay matching p-adic structure.

    Formula: r(v) = inner + (outer - inner) * (exp(-v/s) - exp(-M/s)) / (1 - exp(-M/s))
    where M = max_valuation, s = scale.

    This creates exponentially-spaced radial bands that match the ultrametric
    structure of 3-adic distances: large gaps at low valuation (boundary),
    smaller gaps at high valuation (origin). Exact endpoints: r(0)=outer, r(M)=inner.

    Args:
        max_valuation: Maximum valuation level (typically 9 for 3^9).
        inner_radius: Target radius for highest valuation (near origin).
        outer_radius: Target radius for lowest valuation (near boundary).
        scale: Exponential decay rate. Higher = more gradual decay.
            Default 3.0 matches PAdicGeodesicLoss.valuation_scale.

    Returns:
        Tensor of shape (max_valuation + 1,) with target radius per level.
    """
    import math

    levels = torch.arange(max_valuation + 1, dtype=torch.float64)
    exp_levels = torch.exp(-levels / scale)
    exp_max = math.exp(-max_valuation / scale)
    denom = 1.0 - exp_max
    return inner_radius + (outer_radius - inner_radius) * (exp_levels - exp_max) / denom


def _euclidean_to_hyperbolic_radius(r_euclid: torch.Tensor) -> torch.Tensor:
    """Convert Poincaré ball Euclidean radius to hyperbolic geodesic distance.

    The Poincaré ball model uses: r_hyp = 2 * arctanh(||x||)
    where ||x|| is the Euclidean norm of the point.

    This converts config targets (in Euclidean [0,1) coords) to hyperbolic
    distance units (what hyperbolic_radius() returns).

    Args:
        r_euclid: Euclidean radius in [0, 1) Poincaré coordinates
    Returns:
        Hyperbolic distance from origin
    """
    r_euclid = r_euclid.clamp(min=0.0, max=0.999)
    return 2.0 * torch.atanh(r_euclid)


class PAdicGeodesicLoss(HierarchyLossBase):
    """Unified P-Adic Geodesic Loss.

    THE KEY V5.11 INSIGHT: In hyperbolic geometry, distance ordering and
    radial hierarchy are coupled via the metric itself.

    For pairs (i, j) with 3-adic valuation v_3(|i-j|):
    - High valuation → small target geodesic distance
    - Small geodesic distance in hyperbolic space → both points near origin

    This single loss enforces BOTH:
    - Hierarchy: High-valuation pairs close → both near origin
    - Correlation: Distance ordering matches valuation ordering

    No competing objectives - geometry handles both automatically.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_target_distance: float = 3.0,
        valuation_scale: float = 3.0,
        n_pairs: int = 2000,
        use_smooth_l1: bool = True,
        seed: int = 42,
    ):
        """Initialize PAdicGeodesicLoss.

        Args:
            curvature: Poincaré ball curvature
            max_target_distance: Maximum target geodesic distance (for v=0)
            valuation_scale: Scale factor for valuation→distance mapping
            n_pairs: Number of pairs to sample per batch
            use_smooth_l1: Use SmoothL1 (Huber) loss instead of MSE
            seed: Random seed for reproducible pair sampling
        """
        super().__init__()
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        self._validate_positive(max_target_distance, "max_target_distance", self.__class__.__name__)
        self._validate_positive(valuation_scale, "valuation_scale", self.__class__.__name__)
        if n_pairs < 1:
            raise ValueError(f"PAdicGeodesicLoss: n_pairs must be >= 1, got {n_pairs}")
        self.curvature = curvature
        self.max_target = max_target_distance
        self.valuation_scale = valuation_scale
        self.n_pairs = n_pairs
        self.use_smooth_l1 = use_smooth_l1
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
        """Map 3-adic valuation to target hyperbolic distance.

        v_3 = 0 (not divisible by 3)   → large distance (far apart)
        v_3 = 9 (divisible by 3^9)     → tiny distance (nearly same point)

        Formula: d_target = max_dist * exp(-valuation / scale)

        This exponential mapping matches the ultrametric structure of 3-adics.
        """
        return self.max_target * torch.exp(-valuation / self.valuation_scale)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Compute unified geodesic loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices for each sample (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z_hyp.size(0)
        device = z_hyp.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {"n_pairs": 0}

        # Sample random pairs (reproducible via generator)
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        # Generate on CPU with seeded generator, then move to device
        i_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(
            device
        )
        j_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(
            device
        )

        # Avoid self-pairs
        same_mask = i_idx == j_idx
        j_idx[same_mask] = (j_idx[same_mask] + 1) % batch_size

        # Compute actual Poincaré distance
        d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], self.curvature)

        # Compute target distance from 3-adic valuation
        diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())
        valuation = TERNARY.valuation(diff).double()
        d_target = self.target_distance(valuation)

        # Loss: align actual geodesic distance with target
        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(d_actual, d_target)
        else:
            loss = F.mse_loss(d_actual, d_target)

        # Compute metrics
        with torch.no_grad():
            # Correlation between actual and target distances
            corr = torch.corrcoef(torch.stack([d_actual, d_target]))[0, 1]
            if torch.isnan(corr):
                corr = torch.tensor(0.0, device=device, dtype=torch.float64)

            # Mean distances by valuation level
            mean_d_low_v = (
                d_actual[valuation < 2].mean()
                if (valuation < 2).any()
                else torch.tensor(0.0, device=device, dtype=torch.float64)
            )
            mean_d_high_v = (
                d_actual[valuation >= 4].mean()
                if (valuation >= 4).any()
                else torch.tensor(0.0, device=device, dtype=torch.float64)
            )

        metrics = {
            "n_pairs": n_pairs,
            "mean_d_actual": d_actual.mean().item(),
            "mean_d_target": d_target.mean().item(),
            "distance_correlation": corr.item(),
            "mean_d_low_valuation": mean_d_low_v.item(),
            "mean_d_high_valuation": mean_d_high_v.item(),
        }

        return loss, metrics


class RadialHierarchyLoss(HierarchyLossBase):
    """Radial Hierarchy Loss - direct radius enforcement.

    While PAdicGeodesicLoss handles hierarchy implicitly via geodesics,
    this loss provides explicit radial control for faster convergence.

    Target: v_3(n) high → radius small (near origin)
            v_3(n) low  → radius large (near boundary)

    V5.11.1 FIX: Added margin loss to enforce separation between valuation levels.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        valuation_weighting: bool = True,
        margin_weight: float = 1.0,
        use_margin_loss: bool = True,
        curvature: float = 1.0,
        seed: int = 42,
        valuation_weight_exponent: float = 0.25,
        margin_step_factor: float = 0.5,
    ):
        """Initialize RadialHierarchyLoss.

        Args:
            inner_radius: Target radius for highest valuation (near origin)
            outer_radius: Target radius for lowest valuation (near boundary)
            max_valuation: Maximum valuation (log_3(19683) = 9)
            valuation_weighting: Weight high-valuation points more
            margin_weight: Weight for margin-based separation loss
            use_margin_loss: Enable pairwise margin loss for radial separation
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
            seed: Random seed for reproducible pair sampling
            valuation_weight_exponent: Exponent for valuation-based sample weighting.
                Higher values weight rare high-valuation samples more heavily.
                Default 0.25 provides stable gradients. Range: [0.1, 0.5]
            margin_step_factor: Factor applied to radius_step for margin computation.
                Controls tightness of hierarchical separation. Default 0.5 means
                margins are half the per-level radius step. Range: [0.3, 1.0]
        """
        super().__init__()
        self._validate_radii(inner_radius, outer_radius, self.__class__.__name__)
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        self._validate_weight(margin_weight, "margin_weight", self.__class__.__name__)
        if max_valuation < 1:
            raise ValueError(f"RadialHierarchyLoss: max_valuation must be >= 1, got {max_valuation}")
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.valuation_weighting = valuation_weighting
        self.margin_weight = margin_weight
        self.use_margin_loss = use_margin_loss
        self.curvature = curvature
        self.valuation_weight_exponent = valuation_weight_exponent
        self.margin_step_factor = margin_step_factor
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Precompute target radii in hyperbolic distance units
        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(
                max_valuation, inner_radius, outer_radius, scale=3.0
            )
        )
        self.register_buffer('_target_radii', target_radii)
    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Compute radial hierarchy loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = z_hyp.device
        batch_size = z_hyp.size(0)

        # Compute 3-adic valuation for each index
        valuations = TERNARY.valuation(batch_indices).double()

        # V5.12.2: Compute actual radius using hyperbolic distance, not Euclidean norm
        # This ensures consistent geometry throughout the system
        actual_radius = hyperbolic_radius(z_hyp, c=self.curvature)

        # Compute target radius using exponential p-adic mapping (via precomputed LUT)
        v_clamped = valuations.long().clamp(0, self.max_valuation).cpu()
        target_radius = self._target_radii[v_clamped].to(device)

        # Weighted loss (high-valuation points are rarer, more important)
        if self.valuation_weighting:
            # V5.12.3: Clamped exponential weighting for gradient stability
            # v=0: weight=1, v=4: weight~3, v=9: weight=10 (clamped)
            # Softer than v5.11.2 to prevent gradient instability on rare samples
            raw_weights = 1.0 + torch.exp(valuations * self.valuation_weight_exponent)
            weights = torch.clamp(raw_weights, min=1.0, max=10.0)
        else:
            weights = torch.ones_like(valuations)

        # Primary loss: push each point to its target radius
        primary_loss = (
            F.mse_loss(actual_radius, target_radius, reduction="none") * weights
        ).mean()

        # Margin loss: enforce separation between different valuation levels
        margin_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if self.use_margin_loss and batch_size >= 2:
            # Sample pairs (reproducible via generator)
            n_pairs = min(1000, batch_size * (batch_size - 1) // 2)
            i_idx = torch.randint(
                0, batch_size, (n_pairs,), generator=self.generator
            ).to(device)
            j_idx = torch.randint(
                0, batch_size, (n_pairs,), generator=self.generator
            ).to(device)

            # Avoid same index
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % batch_size

            v_i = valuations[i_idx]
            v_j = valuations[j_idx]
            r_i = actual_radius[i_idx]
            r_j = actual_radius[j_idx]

            # For pairs where v_i > v_j (i has higher valuation),
            # we want r_i < r_j (i should be closer to origin)
            higher_v_mask = v_i > v_j

            if higher_v_mask.any():
                # Expected margin: difference in exponential target radii
                # (p-adic structure → larger margins at low valuation, smaller at high)
                vi_clamped = v_i[higher_v_mask].long().clamp(0, self.max_valuation).cpu()
                vj_clamped = v_j[higher_v_mask].long().clamp(0, self.max_valuation).cpu()
                target_r_j = self._target_radii[vj_clamped].to(device)
                target_r_i = self._target_radii[vi_clamped].to(device)
                expected_margin = (target_r_j - target_r_i) * self.margin_step_factor

                # Actual difference: r_j - r_i (should be positive)
                actual_diff = r_j[higher_v_mask] - r_i[higher_v_mask]

                # Margin violation: max(0, expected_margin - actual_diff)
                violations = F.relu(expected_margin - actual_diff)
                margin_loss = violations.mean()

        total_loss = primary_loss + self.margin_weight * margin_loss

        # Metrics
        with torch.no_grad():
            radial_corr = torch.corrcoef(torch.stack([valuations, -actual_radius]))[
                0, 1
            ]
            if torch.isnan(radial_corr):
                radial_corr = torch.tensor(0.0, device=device, dtype=torch.float64)

            # Compute actual range
            radius_range = actual_radius.max() - actual_radius.min()

        metrics = {
            "mean_radius": actual_radius.mean().item(),
            "mean_target_radius": target_radius.mean().item(),
            "radial_hierarchy_corr": radial_corr.item(),
            "radius_min": actual_radius.min().item(),
            "radius_max": actual_radius.max().item(),
            "radius_range": radius_range.item(),
            "primary_loss": primary_loss.item(),
            "margin_loss": (
                margin_loss.item()
                if isinstance(margin_loss, torch.Tensor)
                else margin_loss
            ),
        }

        return total_loss, metrics


class GlobalRankLoss(HierarchyLossBase):
    """Global Rank Loss - enforces monotonic radius ordering by valuation.

    Unlike RadialHierarchyLoss which uses point-wise MSE and sampled pair margins,
    this loss directly optimizes the global ranking of radii.

    Key insight: We want radius to be monotonically decreasing with valuation.
    This is a differentiable surrogate for Spearman correlation.

    The loss penalizes ALL pairs where the ordering is violated:
    - If v_i > v_j (i has higher valuation), then r_i should be < r_j
    - Violation: r_i >= r_j when v_i > v_j

    Uses soft ranking via sigmoid for differentiability.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        n_pairs: int = 2000,
        use_all_pairs: bool = False,
        curvature: float = 1.0,
        seed: int = 42,
        scatter_weight: float = 0.0,
    ):
        """Initialize GlobalRankLoss.

        Args:
            temperature: Softness of the ranking (lower = sharper)
            n_pairs: Number of pairs to sample (if not using all pairs)
            use_all_pairs: If True, use all O(n²) pairs (expensive but exact)
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
            seed: Random seed for reproducible pair sampling
            scatter_weight: Weight for within-level scatter penalty. For same-valuation
                pairs (v_i == v_j), penalises (r_i - r_j)^2. Direct differentiable
                proxy for dist_corr improvement. Default 0.0 (disabled); 0.3 recommended.
        """
        super().__init__()
        self._validate_positive(temperature, "temperature", self.__class__.__name__)
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        if n_pairs < 1:
            raise ValueError(f"GlobalRankLoss: n_pairs must be >= 1, got {n_pairs}")
        self._validate_weight(scatter_weight, "scatter_weight", self.__class__.__name__)
        self.temperature = temperature
        self.n_pairs = n_pairs
        self.use_all_pairs = use_all_pairs
        self.curvature = curvature
        self.scatter_weight = scatter_weight
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Compute global rank loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = z_hyp.device
        batch_size = z_hyp.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_violations": 0
            }

        # Get valuations and radii
        valuations = TERNARY.valuation(batch_indices).double()
        # V5.12.2: Use hyperbolic distance instead of Euclidean norm
        radii = hyperbolic_radius(z_hyp, c=self.curvature)

        if self.use_all_pairs:
            # All pairs (expensive: O(n²))
            i_idx = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(-1, batch_size)
                .reshape(-1)
            )
            j_idx = (
                torch.arange(batch_size, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
            )
        else:
            # Sample pairs (reproducible via generator)
            n_pairs = min(self.n_pairs, batch_size * (batch_size - 1))
            i_idx = torch.randint(
                0, batch_size, (n_pairs,), generator=self.generator
            ).to(device)
            j_idx = torch.randint(
                0, batch_size, (n_pairs,), generator=self.generator
            ).to(device)

            # Avoid self-pairs
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % batch_size

        # Get valuations and radii for pairs
        v_i = valuations[i_idx]
        v_j = valuations[j_idx]
        r_i = radii[i_idx]
        r_j = radii[j_idx]

        # Valuation difference: positive means i has higher valuation
        v_diff = v_i - v_j

        # Only consider pairs where valuations differ
        diff_mask = v_diff != 0
        if not diff_mask.any():
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_violations": 0,
                "n_pairs": 0,
            }

        v_diff = v_diff[diff_mask]
        r_i = r_i[diff_mask]
        r_j = r_j[diff_mask]

        # For pairs where v_i > v_j (v_diff > 0), we want r_i < r_j
        # i.e., r_j - r_i > 0
        # For pairs where v_i < v_j (v_diff < 0), we want r_i > r_j
        # i.e., r_i - r_j > 0, or -(r_j - r_i) > 0

        # Normalize: if v_diff > 0, expect r_diff = r_j - r_i > 0
        #            if v_diff < 0, expect r_diff = r_j - r_i < 0
        # So: sign(v_diff) * (r_j - r_i) should be > 0

        r_diff = r_j - r_i
        expected_sign = torch.sign(v_diff)

        # Signed radius difference: should be positive if ordering is correct
        signed_r_diff = expected_sign * r_diff

        # Soft violation: sigmoid(-signed_r_diff / temperature)
        # When signed_r_diff > 0 (correct), sigmoid → 0
        # When signed_r_diff < 0 (violation), sigmoid → 1
        violations = torch.sigmoid(-signed_r_diff / self.temperature)

        # Weight by valuation difference magnitude (larger gaps more important)
        weights = torch.abs(v_diff)
        weighted_violations = violations * weights

        loss = weighted_violations.mean()

        # Within-level scatter penalty: for same-valuation pairs, push radii together.
        # This is the direct differentiable proxy for dist_corr: dist_corr measures
        # Spearman(|r_i - r_j|, |v_i - v_j|), so same-v pairs (|v_i - v_j|=0) should
        # have |r_i - r_j|=0. The diff_mask above excludes these pairs from the ranking
        # loss — we handle them here separately.
        scatter_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if self.scatter_weight > 0.0:
            # Use original (unfiltered) pair tensors: v_i_full, r_i_full before diff_mask
            v_i_full = valuations[i_idx]
            v_j_full = valuations[j_idx]
            r_i_full = radii[i_idx]
            r_j_full = radii[j_idx]
            same_v_mask = v_i_full == v_j_full
            if same_v_mask.any():
                scatter_loss = ((r_i_full[same_v_mask] - r_j_full[same_v_mask]) ** 2).mean()
                loss = loss + self.scatter_weight * scatter_loss

        # Metrics
        with torch.no_grad():
            hard_violations = (signed_r_diff < 0).double().sum().item()
            n_pairs_used = len(v_diff)
            violation_rate = hard_violations / n_pairs_used if n_pairs_used > 0 else 0

        metrics = {
            "n_pairs": n_pairs_used,
            "n_violations": int(hard_violations),
            "violation_rate": violation_rate,
            "mean_signed_diff": signed_r_diff.mean().item(),
            "scatter_loss": scatter_loss.item(),
        }

        return loss, metrics


class MonotonicRadialLoss(HierarchyLossBase):
    """Monotonic Radial Loss - enforces strict per-level radius ordering.

    V5.11.4 IMPROVEMENT: Instead of sampling random pairs, this loss:
    1. Groups points by valuation level (0-9)
    2. Computes mean radius per level
    3. Enforces: mean_r[v] > mean_r[v+1] + margin for all consecutive levels

    This creates explicit "radial bands" where each valuation level occupies
    a distinct range of radii with guaranteed separation.

    Key insight: The weakness of pairwise sampling is that it may miss
    enforcing structure between specific adjacent levels. This loss
    directly targets the level-wise means, ensuring monotonicity.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        min_margin: float = 0.02,
        margin_scale: float = 1.0,
        use_soft_margin: bool = True,
        temperature: float = 0.05,
        curvature: float = 1.0,
        target_loss_weight: float = 0.5,
    ):
        """Initialize MonotonicRadialLoss.

        Args:
            inner_radius: Target radius for highest valuation (v=9)
            outer_radius: Target radius for lowest valuation (v=0)
            max_valuation: Maximum valuation level
            min_margin: Minimum margin between adjacent levels
            margin_scale: Scale factor for adaptive margins
            use_soft_margin: Use soft hinge loss (differentiable)
            temperature: Softness for soft margin (lower = sharper)
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
            target_loss_weight: Weight for target radius MSE relative to margin loss.
                Controls balance between ordinal constraints and absolute positioning.
                Default 0.5 gives equal importance. Range: [0.0, 1.0]
        """
        super().__init__()
        self._validate_radii(inner_radius, outer_radius, self.__class__.__name__)
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        self._validate_positive(temperature, "temperature", self.__class__.__name__)
        self._validate_weight(target_loss_weight, "target_loss_weight", self.__class__.__name__)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.curvature = curvature
        self.min_margin = min_margin
        self.margin_scale = margin_scale
        self.use_soft_margin = use_soft_margin
        self.temperature = temperature
        self.target_loss_weight = target_loss_weight

        # Precompute target radii in hyperbolic distance units
        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(
                max_valuation, inner_radius, outer_radius, scale=3.0
            )
        )
        self.register_buffer('_target_radii', target_radii)
    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        """Compute monotonic radial loss.

        Args:
            z_hyp: Points in Poincaré ball (batch, latent_dim)
            batch_indices: Operation indices (batch,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = z_hyp.device
        batch_size = z_hyp.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_levels": 0
            }

        # Get valuations and radii
        valuations = TERNARY.valuation(batch_indices)
        # V5.12.2: Use hyperbolic distance instead of Euclidean norm
        radii = hyperbolic_radius(z_hyp, c=self.curvature)

        # Compute mean radius per valuation level
        level_means = []
        level_counts = []
        levels_present = []

        for v in range(self.max_valuation + 1):
            mask = valuations == v
            if mask.any():
                level_means.append(radii[mask].mean())
                level_counts.append(mask.sum().item())
                levels_present.append(v)

        if len(levels_present) < 2:
            # Need at least 2 levels to enforce ordering
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_levels": len(levels_present),
                "margin_violations": 0,
            }

        level_means = torch.stack(level_means)
        n_levels = len(levels_present)

        # Compute target margins between adjacent present levels
        # Using exponential target radii for p-adic structure
        target_lut = self._target_radii
        margins = []
        for i in range(n_levels - 1):
            v_curr = levels_present[i]
            v_next = levels_present[i + 1]
            # Margin from exponential target difference (scaled, with floor)
            exp_margin = (
                target_lut[v_curr] - target_lut[v_next]
            ).item() * self.margin_scale
            margin = max(self.min_margin, exp_margin)
            margins.append(margin)

        margins = torch.tensor(margins, device=device, dtype=torch.float64)

        # Monotonicity constraint: r[v] > r[v+1] + margin
        # Equivalently: r[v] - r[v+1] > margin
        # Violation: margin - (r[v] - r[v+1]) > 0
        radius_diffs = level_means[:-1] - level_means[1:]  # r[v] - r[v+1]
        violations = margins - radius_diffs  # positive when violated

        if self.use_soft_margin:
            # Soft hinge: softplus(beta=1/temperature) gives near-ReLU sharpness
            # beta=1/0.05=20 → sharp gradient at violation boundary, smooth elsewhere
            loss = F.softplus(violations, beta=1.0 / self.temperature).mean()
        else:
            # Hard hinge
            loss = F.relu(violations).mean()

        # Also add a gentle pull toward exponential target radii per level
        target_radii = target_lut[torch.tensor(levels_present, dtype=torch.long)].to(device)

        target_loss = F.mse_loss(level_means, target_radii)

        # Combined loss: margin enforcement + target guidance
        total_loss = loss + self.target_loss_weight * target_loss

        # Metrics
        with torch.no_grad():
            hard_violations = (violations > 0).sum().item()
            mean_violation = (
                violations[violations > 0].mean().item() if hard_violations > 0 else 0.0
            )

            # Build level radius map for logging
            level_radius_map = {
                f"r_v{levels_present[i]}": level_means[i].item()
                for i in range(n_levels)
            }

        metrics = {
            "n_levels": n_levels,
            "margin_violations": hard_violations,
            "mean_violation_magnitude": mean_violation,
            "monotonic_loss": loss.item(),
            "target_loss": target_loss.item(),
            "min_radius_diff": (
                radius_diffs.min().item() if len(radius_diffs) > 0 else 0
            ),
            "mean_radius_diff": (
                radius_diffs.mean().item() if len(radius_diffs) > 0 else 0
            ),
            **level_radius_map,
        }

        return total_loss, metrics


class RichHierarchyLoss(RichHierarchyLossBase):
    """Rich Hierarchy Loss - Unified Training Objective.

    Combines:
    1. Hierarchy Loss: Enforces target radii per valuation level
    2. Coverage Loss: Standard CrossEntropy on reconstruction
    3. Separation Loss: Enforces margins between valuation levels

    Refactored to use pure Hyperbolic distance for radii.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        curvature: float = 1.0,
        separation_margin: float = 0.01,
        variance_weight: float = 0.1,
    ) -> None:
        """Initialize RichHierarchyLoss.

        Args:
            inner_radius: Target radius for highest valuation (v=9), must be in (0, 1)
            outer_radius: Target radius for lowest valuation (v=0), must be in (0, 1)
            curvature: Hyperbolic curvature, must be > 0
            separation_margin: Minimum margin between adjacent levels, must be >= 0
            variance_weight: Weight for within-level radius variance penalty.
                Higher values tighten clusters per valuation level, directly
                improving dist_corr. Default 0.1 (original); 0.5 recommended.
        """
        super().__init__()
        self._validate_radii(inner_radius, outer_radius, self.__class__.__name__)
        self._validate_positive(curvature, "curvature", self.__class__.__name__)
        self._validate_weight(separation_margin, "separation_margin", self.__class__.__name__)
        self._validate_weight(variance_weight, "variance_weight", self.__class__.__name__)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.curvature = curvature
        self.separation_margin = separation_margin
        self.variance_weight = variance_weight
        # Precompute target radii in hyperbolic distance units
        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(
                9, inner_radius, outer_radius, scale=3.0
            )
        )
        self.register_buffer("target_radii", target_radii)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], MetricsDict]:
        """Compute rich hierarchy loss.

        Args:
            z_hyp: Points on Poincaré ball (B, latent_dim), float64
            batch_indices: Ternary operation indices (B,)
            **kwargs: Must contain 'logits' (B, 9, 3) or (B, 27) and 'targets' (B, 9)

        Returns:
            loss: Weighted scalar loss (hierarchy + coverage + separation)
            metrics: Dict with 'hierarchy', 'coverage', 'separation' float values
        """
        logits = kwargs["logits"]
        targets = kwargs["targets"]
        indices_batch = batch_indices
        device = z_hyp.device

        # Use hyperbolic radius (distance from origin), not Euclidean norm
        radii = hyperbolic_radius(z_hyp, c=self.curvature)
        # Use precomputed hyperbolic target radii from __init__
        target_radii = self.target_radii.to(device)

        valuations = TERNARY.valuation(indices_batch).long().to(device)

        # 1. Hierarchy loss (MSE on mean radius + within-level variance)
        # Mean-only MSE leaves per-level scatter unpunished (CV up to 34% at v=3),
        # directly limiting Spearman Q. Adding variance_loss tightens each level.
        hierarchy_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        variance_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        present_levels = torch.unique(valuations)

        for v in present_levels:
            mask = valuations == v
            level_radii = radii[mask]
            if level_radii.numel() > 0:
                mean_r = level_radii.mean()
                target_r = target_radii[int(v.item())]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
                if level_radii.numel() > 1:
                    variance_loss = variance_loss + level_radii.var()

        n_levels = len(present_levels)
        if n_levels > 0:
            hierarchy_loss = hierarchy_loss / n_levels
            variance_loss = variance_loss / n_levels
        # Fold variance penalty into hierarchy_loss.
        # variance_weight controls within-level tightening; higher values drive dist_corr.
        # At plateau, mean-MSE gradient ≈ 0 so variance_loss becomes the primary radial signal.
        hierarchy_loss = hierarchy_loss + self.variance_weight * variance_loss

        # 2. Coverage loss (Reconstruction)
        # logits may be None when coverage_weight=0.0 (e.g. VAE-B loss_fn_b).
        # Guard here avoids running F.cross_entropy for a term that will be
        # multiplied by zero in CombinedLoss — eliminating wasted forward compute.
        if logits is None:
            coverage_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        else:
            # logits shape: (B, 9, 3) or (B, 27) depending on decoder
            # targets are in {-1, 0, 1}, shift to {0, 1, 2} for cross_entropy
            targets_shifted = (targets + 1).long()
            if logits.shape[-1] == 3:  # (B, 9, 3)
                coverage_loss = F.cross_entropy(
                    logits.view(-1, 3),
                    targets_shifted.view(-1),
                )
            elif logits.shape[-1] == 27:  # (B, 27) flattened
                coverage_loss = F.cross_entropy(
                    logits.view(-1, 9, 3).permute(0, 2, 1),  # (B, 3, 9)
                    targets_shifted,  # (B, 9) - no clamp, data must be valid
                )
            else:
                coverage_loss = torch.tensor(0.0, device=device, dtype=torch.float64)

        # 3. Separation loss (Margin between levels)
        separation_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        mean_radii = []
        # Sort levels to ensure we compare adjacent levels correctly
        # v=0 (Outer) -> v=9 (Inner)
        # We want r[v] > r[v+1] (Outer > Inner)
        sorted_levels = sorted(present_levels.tolist())

        for v in sorted_levels:
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii.append(radii[mask].mean())

        # Enforce r[v] > r[v+1] + margin
        # violation = max(0, r[v+1] - r[v] + margin)
        # Original logic: violation = torch.relu(mean_radii[i + 1] - mean_radii[i] + 0.01)
        # Here mean_radii list corresponds to sorted levels (0, 1, 2...)
        # So mean_radii[i] is level v, mean_radii[i+1] is level v+k
        # We expect mean_radii[i] (outer) > mean_radii[i+1] (inner)
        # So mean_radii[i+1] - mean_radii[i] should be negative.
        # If it's positive, inner is larger than outer -> violation.
        min_margin = torch.tensor(
            self.separation_margin, device=device, dtype=torch.float64
        )
        for i in range(len(mean_radii) - 1):
            v_curr = sorted_levels[i]
            v_next = sorted_levels[i + 1]
            level_margin = torch.maximum(
                min_margin, target_radii[v_curr] - target_radii[v_next]
            )
            violation = F.relu(mean_radii[i + 1] - mean_radii[i] + level_margin)
            separation_loss = separation_loss + violation

        # Return (scalar_loss=zero placeholder, components_dict).
        # CombinedLoss reads individual components and applies config weights.
        # The zero scalar is intentional: CombinedLoss owns the weighting.
        metrics: Dict[str, float] = {
            "hierarchy": hierarchy_loss.item(),
            "coverage": coverage_loss.item(),
            "separation": separation_loss.item(),
            "variance": variance_loss.item(),
        }
        # Pass raw tensors under _tensors key for CombinedLoss gradient flow
        _raw = {
            "hierarchy": hierarchy_loss,
            "coverage": coverage_loss,
            "separation": separation_loss,
        }
        return _raw, metrics


class ValuationPriorLoss(nn.Module):
    """Valuation-conditioned radial prior operating in tangent space.

    Replaces the N(0,I) mean term of KL with a target that pulls ||μ||
    toward target_tangent[v₃(i)] per sample, based on the 3-adic valuation
    of the corresponding operation index.

    target_tangent[v] = arctanh(r_euclid[v]) / sqrt(c)

    This is the inverse of geoopt's expmap0 formula:
        ||expmap0(v)||_euclid = tanh(sqrt(c) * ||v||)
    So the tangent norm needed to reach a given Euclidean radius is:
        ||v|| = arctanh(r_euclid) / sqrt(c)

    Why tangent space (not post-expmap):
    - μ is in tangent space BEFORE reparameterization and expmap0
    - Gradient path is clean: no expmap Jacobian, no max_radius clamping
    - This directly corrects the μ→0 pull from standard KL

    Use in conjunction with HyperbolicKLDivergence(variance_only=True) to
    avoid double-counting: KL handles σ, this handles μ.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        inner_radius: float = 0.08,
        outer_radius: float = 0.85,
        scale: float = 3.0,
        max_valuation: int = 9,
    ):
        super().__init__()
        self.curvature_init = curvature
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.scale = scale
        self.max_valuation = max_valuation
        # Precompute Euclidean target radii (fixed by geometry, not by curvature)
        target_r_euclid = _exponential_target_radii(
            max_valuation, inner_radius, outer_radius, scale
        )
        self.register_buffer('target_r_euclid', target_r_euclid)  # (max_v+1,)

    def forward(
        self,
        mu: torch.Tensor,
        batch_indices: torch.Tensor,
        curvature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute valuation-conditioned mean prior loss.

        Args:
            mu: Tangent space posterior means, shape (B, d), float64
            batch_indices: Ternary operation indices, shape (B,)
            curvature: Current curvature (use model's if learnable). Falls
                       back to curvature_init if None.

        Returns:
            loss: Scalar MSE((||μ||, target_tangent[v]))
            metrics: Logging dict with mean norms per valuation level
        """
        import math

        if curvature is None:
            curvature = self.curvature_init

        mu = mu.to(torch.float64)
        device = mu.device

        # Compute target tangent norms from current curvature
        # target_tangent[v] = arctanh(r_euclid[v]) / sqrt(c)
        sqrt_c = math.sqrt(max(curvature, 1e-6))
        target_r = self.target_r_euclid.to(device)  # (max_v+1,)
        target_tangent_norms = torch.atanh(target_r.clamp(max=0.9999)) / sqrt_c  # (max_v+1,)

        # Get valuation per sample using precomputed LUT
        valuations = TERNARY.valuation(batch_indices).long().clamp(0, self.max_valuation)
        target_norms = target_tangent_norms[valuations.cpu()].to(device)  # (B,)

        # ||μ|| per sample
        mu_norms = torch.norm(mu, dim=-1)  # (B,)

        loss = F.mse_loss(mu_norms, target_norms)

        # Metrics: mean norm per valuation level (diagnostic)
        with torch.no_grad():
            mean_mu_norm = mu_norms.mean().item()
            mean_target = target_norms.mean().item()
            # Per-level breakdown for the first few levels
            metrics: Dict[str, float] = {
                'vp_mean_mu_norm': mean_mu_norm,
                'vp_mean_target': mean_target,
                'vp_gap': abs(mean_mu_norm - mean_target),
            }
            for v in range(min(5, self.max_valuation + 1)):
                mask = valuations == v
                if mask.any():
                    metrics[f'vp_mu_norm_v{v}'] = mu_norms[mask].mean().item()

        return loss, metrics


__all__ = [
    "PAdicGeodesicLoss",
    "RadialHierarchyLoss",
    "GlobalRankLoss",
    "MonotonicRadialLoss",
    "RichHierarchyLoss",
    "ValuationPriorLoss",
]
