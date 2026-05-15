# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Radial and hierarchical consistency losses for p-adic VAE."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import TERNARY
from ..geometry import hyperbolic_radius, poincare_distance
from ..utils.scatter_utils import level_has_data, level_scatter_mean
from .base import HierarchyLossBase, MetricsDict, RichHierarchyLossBase
from .utils import _euclidean_to_hyperbolic_radius, _exponential_target_radii


class RadialHierarchyLoss(HierarchyLossBase):
    """Radial Hierarchy Loss - direct radius enforcement."""

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        valuation_weighting: bool = True,
        margin_weight: float = 1.0,
        use_margin_loss: bool = True,
        curvature: float = 1.0,
        valuation_fn=None,
        seed: int = 42,
        valuation_weight_exponent: float = 0.25,
        margin_step_factor: float = 0.5,
    ):
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
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
        self.valuation_weight_exponent = valuation_weight_exponent
        self.margin_step_factor = margin_step_factor
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(max_valuation, inner_radius, outer_radius, scale=3.0)
        )
        self.register_buffer('_target_radii', target_radii)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        device = z_hyp.device
        batch_size = z_hyp.size(0)
        cur_c = kwargs.get("curvature", self.curvature)
        valuations = self._valuation_fn(batch_indices).double()
        actual_radius = hyperbolic_radius(z_hyp, c=cur_c)

        target_radii_all = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(self.max_valuation, self.inner_radius, self.outer_radius, scale=3.0).to(device),
            c=cur_c
        )
        v_clamped = valuations.long().clamp(0, self.max_valuation)
        target_radius = target_radii_all[v_clamped]

        if self.valuation_weighting:
            raw_weights = 1.0 + torch.exp(valuations * self.valuation_weight_exponent)
            weights = torch.clamp(raw_weights, min=1.0, max=10.0)
        else:
            weights = torch.ones_like(valuations)

        primary_loss = (F.mse_loss(actual_radius, target_radius, reduction="none") * weights).mean()

        margin_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if self.use_margin_loss and batch_size >= 2:
            n_pairs = min(1000, batch_size * (batch_size - 1) // 2)
            i_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(device)
            j_idx = torch.randint(0, batch_size, (n_pairs,), generator=self.generator).to(device)
            same = i_idx == j_idx
            j_idx[same] = (j_idx[same] + 1) % batch_size

            v_i, v_j = valuations[i_idx], valuations[j_idx]
            r_i, r_j = actual_radius[i_idx], actual_radius[j_idx]
            higher_v_mask = v_i > v_j

            if higher_v_mask.any():
                vi_clamped = v_i[higher_v_mask].long().clamp(0, self.max_valuation).cpu()
                vj_clamped = v_j[higher_v_mask].long().clamp(0, self.max_valuation).cpu()
                target_r_j = self._target_radii[vj_clamped].to(device)
                target_r_i = self._target_radii[vi_clamped].to(device)
                expected_margin = (target_r_j - target_r_i) * self.margin_step_factor
                actual_diff = r_j[higher_v_mask] - r_i[higher_v_mask]
                violations = F.relu(expected_margin - actual_diff)
                margin_loss = violations.mean()

        total_loss = primary_loss + self.margin_weight * margin_loss

        with torch.no_grad():
            if valuations.numel() >= 2:
                radial_corr = torch.corrcoef(torch.stack([valuations, -actual_radius]))[0, 1]
                if torch.isnan(radial_corr):
                    radial_corr = torch.tensor(0.0, device=device)
            else:
                radial_corr = torch.tensor(0.0, device=device)
            radius_range = actual_radius.max() - actual_radius.min()

        metrics = {
            "mean_radius": actual_radius.mean().item(),
            "mean_target_radius": target_radius.mean().item(),
            "radial_hierarchy_corr": radial_corr.item(),
            "radius_min": actual_radius.min().item(),
            "radius_max": actual_radius.max().item(),
            "radius_range": radius_range.item(),
            "primary_loss": primary_loss.item(),
            "margin_loss": margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
        }
        return total_loss, metrics


class MonotonicRadialLoss(HierarchyLossBase):
    """Monotonic Radial Loss - per-level radius ordering constraint."""

    def __init__(
        self,
        inner_radius: float = 0.08,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        min_margin: float = 0.02,
        margin_scale: float = 1.0,
        use_soft_margin: bool = True,
        temperature: float = 0.05,
        curvature: float = 1.0,
        target_loss_weight: float = 0.5,
        valuation_fn=None,
    ):
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
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation

        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(max_valuation, inner_radius, outer_radius, scale=3.0)
        )
        self.register_buffer('_target_radii', target_radii)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        device = z_hyp.device
        batch_size = z_hyp.size(0)
        cur_c = kwargs.get("curvature", self.curvature)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {"n_levels": 0}

        valuations = self._valuation_fn(batch_indices)
        radii = hyperbolic_radius(z_hyp, c=cur_c)

        target_radii_all = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(self.max_valuation, self.inner_radius, self.outer_radius, scale=3.0).to(device),
            c=cur_c
        )

        dim_size = self.max_valuation + 1
        vals_long = valuations.long()
        present_mask = level_has_data(vals_long, dim_size=dim_size)
        levels_present = present_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        if len(levels_present) < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64), {
                "n_levels": len(levels_present),
                "margin_violations": 0,
            }

        level_means_all = level_scatter_mean(radii, vals_long, dim_size=dim_size)
        level_means = level_means_all[present_mask]
        n_levels = len(levels_present)

        margins = []
        for i in range(n_levels - 1):
            v_curr = levels_present[i]
            v_next = levels_present[i + 1]
            exp_margin = (target_radii_all[v_curr] - target_radii_all[v_next]).item() * self.margin_scale
            margin = max(self.min_margin, exp_margin)
            margins.append(margin)

        margins_t = torch.tensor(margins, device=device, dtype=torch.float64)
        radius_diffs = level_means[:-1] - level_means[1:]
        violations = margins_t - radius_diffs

        if self.use_soft_margin:
            loss = F.softplus(violations, beta=1.0 / self.temperature).mean()
        else:
            loss = F.relu(violations).mean()

        target_guidance = target_radii_all[torch.tensor(levels_present, dtype=torch.long, device=device)]
        target_loss = F.mse_loss(level_means, target_guidance)
        total_loss = loss + self.target_loss_weight * target_loss

        per_gap_tensors: Dict[str, torch.Tensor] = {}
        for i in range(n_levels - 1):
            per_gap_tensors[f"gap_viol_tensor_v{levels_present[i]}"] = F.relu(violations[i])

        metrics = {
            "n_levels": n_levels,
            "margin_violations": (violations > 0).sum().item(),
            "monotonic_loss": loss.item(),
            "target_loss": target_loss.item(),
        }
        for i in range(n_levels):
            metrics[f"r_v{levels_present[i]}"] = level_means[i].item()
        for i in range(n_levels - 1):
            metrics[f"gap_viol_v{levels_present[i]}"] = float(max(0.0, violations[i].item()))
        metrics.update(per_gap_tensors)

        return total_loss, metrics


class RichHierarchyLoss(RichHierarchyLossBase):
    """Unified Hierarchy Loss combining multiple objectives."""

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        curvature: float = 1.0,
        separation_margin: float = 0.01,
        variance_weight: float = 0.1,
        valuation_fn=None,
    ) -> None:
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
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
        target_radii = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(9, inner_radius, outer_radius, scale=3.0)
        )
        self.register_buffer("target_radii", target_radii)

    def forward(
        self,
        z_hyp: torch.Tensor,
        batch_indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], MetricsDict]:
        logits, targets = kwargs.get("logits"), kwargs.get("targets")
        device, cur_c = z_hyp.device, kwargs.get("curvature", self.curvature)
        radii = hyperbolic_radius(z_hyp, c=cur_c)
        target_radii_adj = _euclidean_to_hyperbolic_radius(
            _exponential_target_radii(9, self.inner_radius, self.outer_radius, scale=3.0).to(device),
            c=cur_c
        )
        valuations = self._valuation_fn(batch_indices).long().to(device)

        dim_size = 10
        present_mask = level_has_data(valuations, dim_size=dim_size)
        means_all = level_scatter_mean(radii, valuations, dim_size=dim_size)

        deviations = radii - means_all[valuations]
        variance_all = torch.zeros(dim_size, dtype=radii.dtype, device=device)
        counts_all = torch.zeros(dim_size, dtype=radii.dtype, device=device)
        variance_all.scatter_add_(0, valuations, deviations ** 2)
        counts_all.scatter_add_(0, valuations, torch.ones_like(radii))
        variance_all = variance_all / counts_all.clamp(min=1.0)

        with torch.no_grad():
            stds_all = variance_all.clamp(min=0.0).sqrt()

        hierarchy_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        variance_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if present_mask.any():
            hierarchy_loss = F.mse_loss(means_all[present_mask], target_radii_adj[present_mask])
            variance_loss = variance_all[present_mask].mean()

        total_hier_loss = hierarchy_loss + self.variance_weight * variance_loss

        coverage_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        if logits is not None and targets is not None:
            targets_shifted = (targets + 1).long().clamp(0, 2)
            if logits.shape[-1] == 3:
                coverage_loss = F.cross_entropy(logits.view(-1, 3), targets_shifted.view(-1))
            elif logits.shape[-1] == 27:
                coverage_loss = F.cross_entropy(logits.view(-1, 9, 3).permute(0, 2, 1), targets_shifted)

        separation_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        present_lvls = present_mask.nonzero(as_tuple=True)[0].tolist()
        sorted_lvls = sorted(present_lvls)
        min_m = torch.tensor(self.separation_margin, device=device, dtype=torch.float64)
        for i in range(len(sorted_lvls) - 1):
            v, v_next = sorted_lvls[i], sorted_lvls[i+1]
            margin = torch.maximum(min_m, target_radii_adj[v] - target_radii_adj[v_next])
            separation_loss = separation_loss + F.relu(means_all[v_next] - means_all[v] + margin)

        metrics = {"hierarchy": hierarchy_loss.item(), "coverage": coverage_loss.item(),
                   "separation": separation_loss.item(), "variance": variance_loss.item()}
        with torch.no_grad():
            for v in range(dim_size):
                if present_mask[v]:
                    metrics[f"r_mean_v{v}"] = means_all[v].item()
                    metrics[f"r_std_v{v}"] = stds_all[v].item()

        return {"hierarchy": total_hier_loss, "coverage": coverage_loss, "separation": separation_loss}, metrics


class WithinLevelContrastiveLoss(nn.Module):
    """Pull same-valuation points geodesically together."""

    def __init__(
        self,
        curvature: float = 1.0,
        max_pairs_per_level: int = 500,
        weight: float = 1.0,
        valuation_fn=None,
    ):
        super().__init__()
        self.curvature = curvature
        self.max_pairs_per_level = max_pairs_per_level
        self.weight = weight
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        device, cur_c = z_hyp.device, kwargs.get("curvature", self.curvature)
        vals = self._valuation_fn(indices)
        total_loss, n_active = torch.tensor(0.0, device=device, dtype=z_hyp.dtype), 0
        metrics: MetricsDict = {}

        for v in range(10):
            mask = (vals == v).nonzero(as_tuple=True)[0]
            if mask.numel() < 2: continue
            z_v = z_hyp[mask]
            if z_v.size(0) * (z_v.size(0)-1) // 2 > self.max_pairs_per_level:
                import math
                n_take = min(z_v.size(0), int(math.ceil((2 * self.max_pairs_per_level) ** 0.5)) + 1)
                z_v = z_v[torch.randperm(z_v.size(0), device=device)[:n_take]]

            idx_i, idx_j = torch.triu_indices(z_v.size(0), z_v.size(0), offset=1, device=device)
            if idx_i.numel() == 0: continue
            d_sq = poincare_distance(z_v[idx_i], z_v[idx_j], c=cur_c) ** 2
            level_loss = d_sq.mean()
            total_loss = total_loss + level_loss
            n_active += 1
            metrics[f'wlc_d2_v{v}'] = level_loss.item()
            metrics[f'wlc_n_pairs_v{v}'] = idx_i.numel()

        loss = self.weight * (total_loss / max(1, n_active))
        metrics.update({'wlc_loss': loss.item(), 'wlc_n_levels': n_active})
        return loss, metrics
