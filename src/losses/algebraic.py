# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Algebraic coherence and addition losses for p-adic VAE."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import TERNARY
from .base import MetricsDict
from .utils import make_zero_loss


class AngularCoherenceLoss(nn.Module):
    """Pull same-digit-prefix operations together angularly within each valuation level."""

    def __init__(
        self,
        weight: float = 0.3,
        n_pairs: int = 1000,
        prefix_k: int = 2,
        phase_start_epoch: int = 50,
        level_prefix_k: Optional[List[int]] = None,
        target_sim: Union[float, List[float]] = 1.0,
        valuation_fn=None,
    ):
        super().__init__()
        self.weight = weight
        self.n_pairs = n_pairs
        self.prefix_k = prefix_k
        self.phase_start_epoch = phase_start_epoch
        self.level_prefix_k = level_prefix_k
        self._valuation_fn = valuation_fn if valuation_fn is not None else TERNARY.valuation
        if isinstance(target_sim, (int, float)):
            self.target_sim: List[float] = [float(target_sim)] * 10
        else:
            self.target_sim = list(target_sim)

    def forward(
        self,
        z_hyp: torch.Tensor,
        r: torch.Tensor,
        indices: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        metrics: MetricsDict = {}
        zero = make_zero_loss(z_hyp.device, z_hyp.dtype)

        if epoch < self.phase_start_epoch:
            metrics["angular_coherence_loss"] = 0.0
            metrics["angular_coherence_pairs"] = 0
            return zero, metrics

        eps = torch.tensor(1e-10, device=z_hyp.device, dtype=z_hyp.dtype)
        dir_vecs = z_hyp / r.unsqueeze(-1).clamp(min=eps)

        vals = self._valuation_fn(indices)
        B = dir_vecs.shape[0]
        if B < 4:
            metrics["angular_coherence_loss"] = 0.0
            metrics["angular_coherence_pairs"] = 0
            return zero, metrics

        if self.level_prefix_k is not None:
            total_loss = zero
            total_pairs = 0
            n_active_levels = 0

            for v in range(10):
                k = self.level_prefix_k[v]
                if k == 0:
                    continue

                t_sim = self.target_sim[v]
                mask_v = (vals == v)
                if mask_v.sum() < 4:
                    continue

                idx_v = mask_v.nonzero(as_tuple=True)[0]
                nv = idx_v.shape[0]
                perm = torch.randperm(nv, device=z_hyp.device)
                num_v_pairs = self.n_pairs // max(1, sum(kk > 0 for kk in self.level_prefix_k))
                half = min(num_v_pairs, nv // 2)
                if half < 2:
                    continue

                i_idx = idx_v[perm[:half]]
                j_idx = idx_v[perm[half:half * 2]]

                prefix_i = TERNARY.digit_prefix_class(indices[i_idx], k)
                prefix_j = TERNARY.digit_prefix_class(indices[j_idx], k)
                same_cls = prefix_i == prefix_j
                n_same = same_cls.sum().item()

                if n_same < 2:
                    continue

                di = dir_vecs[i_idx[same_cls]]
                dj = dir_vecs[j_idx[same_cls]]
                cos_sim = (di * dj).sum(dim=-1)
                t = torch.tensor(t_sim, device=z_hyp.device, dtype=z_hyp.dtype)
                level_loss = F.relu(t - cos_sim).mean()
                total_loss = total_loss + level_loss
                total_pairs += int(n_same)
                n_active_levels += 1
                metrics[f"ac_loss_v{v}"] = level_loss.item()

            if n_active_levels == 0:
                metrics["angular_coherence_loss"] = 0.0
                metrics["angular_coherence_pairs"] = 0
                return zero, metrics

            loss = self.weight * (total_loss / n_active_levels)
            metrics["angular_coherence_loss"] = loss.item()
            metrics["angular_coherence_pairs"] = total_pairs
            return loss, metrics

        prefix = TERNARY.digit_prefix_class(indices, self.prefix_k)
        key = vals * (3 ** self.prefix_k) + prefix
        perm = torch.randperm(B, device=z_hyp.device)
        half = min(self.n_pairs, B // 2)
        i_idx = perm[:half]
        j_idx = perm[half:half * 2]

        same_cls = key[i_idx] == key[j_idx]
        n_same = same_cls.sum().item()

        if n_same < 4:
            metrics["angular_coherence_loss"] = 0.0
            metrics["angular_coherence_pairs"] = int(n_same)
            return zero, metrics

        di = dir_vecs[i_idx[same_cls]]
        dj = dir_vecs[j_idx[same_cls]]
        cos_sim = (di * dj).sum(dim=-1)
        pair_vals = vals[i_idx[same_cls]].clamp(0, len(self.target_sim) - 1)
        t_values = torch.tensor(self.target_sim, device=z_hyp.device, dtype=z_hyp.dtype)
        t_per_pair = t_values[pair_vals]
        loss = self.weight * F.relu(t_per_pair - cos_sim).mean()

        metrics["angular_coherence_loss"] = loss.item()
        metrics["angular_coherence_pairs"] = int(n_same)
        return loss, metrics


class AlgebraicCoherenceLoss(nn.Module):
    """Attract same-algebraic-class operations in embedding direction space."""

    def __init__(
        self,
        weight: float = 1.0,
        n_pairs: int = 2000,
        target_sim: float = 0.70,
        phase_start_epoch: int = 20,
        min_class_size: int = 3,
    ):
        super().__init__()
        self.weight = weight
        self.n_pairs = n_pairs
        self.target_sim = target_sim
        self.phase_start_epoch = phase_start_epoch
        self.min_class_size = min_class_size

    def forward(
        self,
        z_hyp: torch.Tensor,
        r: torch.Tensor,
        indices: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        metrics: MetricsDict = {}
        zero = make_zero_loss(z_hyp.device, z_hyp.dtype)

        if epoch < self.phase_start_epoch:
            metrics["alg_coherence_loss"] = 0.0
            metrics["alg_coherence_pairs"] = 0
            return zero, metrics

        B = z_hyp.shape[0]
        if B < 4:
            metrics["alg_coherence_loss"] = 0.0
            metrics["alg_coherence_pairs"] = 0
            return zero, metrics

        eps = torch.tensor(1e-10, device=z_hyp.device, dtype=z_hyp.dtype)
        dir_vecs = z_hyp / r.unsqueeze(-1).clamp(min=eps)

        sigs = TERNARY.algebraic_signature(indices)

        t = torch.tensor(self.target_sim, device=z_hyp.device, dtype=z_hyp.dtype)
        total_loss = zero
        total_pairs = 0
        n_active = 0

        for sig_val in range(1, 8):
            mask = (sigs == sig_val)
            n_in_class = int(mask.sum().item())
            if n_in_class < self.min_class_size:
                continue

            class_idx = mask.nonzero(as_tuple=True)[0]
            nc = class_idx.shape[0]
            half = min(nc // 2, self.n_pairs)
            if half < 2:
                continue

            perm = torch.randperm(nc, device=z_hyp.device)
            di = dir_vecs[class_idx[perm[:half]]]
            dj = dir_vecs[class_idx[perm[half:half * 2]]]

            cos_sim = (di * dj).sum(dim=-1)
            cls_loss = F.relu(t - cos_sim).mean()

            total_loss = total_loss + cls_loss
            total_pairs += half
            n_active += 1
            metrics[f"alg_loss_sig{sig_val}"] = cls_loss.item()

        if n_active == 0:
            metrics["alg_coherence_loss"] = 0.0
            metrics["alg_coherence_pairs"] = 0
            return zero, metrics

        loss = self.weight * (total_loss / n_active)
        metrics["alg_coherence_loss"] = loss.item()
        metrics["alg_coherence_pairs"] = total_pairs
        return loss, metrics


class _AlgebraicBinaryLoss(nn.Module):
    """Base for binary algebraic homomorphism losses (addition, multiplication).

    Subclasses define the ternary operation and the target combination rule.
    Template: sample pairs → apply op → compare mu_result vs mu_target via smooth_l1.
    """

    _loss_key: str
    _sim_key: str

    def __init__(
        self,
        weight: float = 1.0,
        n_pairs: int = 512,
        phase_start_epoch: int = 0,
    ):
        super().__init__()
        self.weight = weight
        self.n_pairs = n_pairs
        self.phase_start_epoch = phase_start_epoch
        self.generator = torch.Generator()
        self.generator.manual_seed(42)

    def _ternary_op(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _mu_target(
        self, mu_A: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        mu_A: torch.Tensor,
        indices: torch.Tensor,
        model: nn.Module,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        metrics: MetricsDict = {}
        zero = make_zero_loss(mu_A.device, mu_A.dtype)

        if epoch < self.phase_start_epoch or self.weight <= 0:
            metrics[self._loss_key] = 0.0
            return zero, metrics

        B = mu_A.shape[0]
        n_triplets = min(self.n_pairs, B // 2)
        if B < 2 or n_triplets < 1:
            metrics[self._loss_key] = 0.0
            return zero, metrics

        perm = torch.randperm(B, generator=self.generator).to(mu_A.device)
        idx_a_local = perm[:n_triplets]
        idx_b_local = perm[n_triplets:2 * n_triplets]

        idx_result = self._ternary_op(indices[idx_a_local], indices[idx_b_local])
        mu_result = model.get_mu_representations(idx_result, mu_A.device)
        mu_target = self._mu_target(mu_A, idx_a_local, idx_b_local)

        if mu_result.shape != mu_target.shape:
            raise RuntimeError(
                f"{type(self).__name__}: shape mismatch — "
                f"mu_result {mu_result.shape} vs mu_target {mu_target.shape}. "
                f"model.get_mu_representations must return (n_pairs, latent_dim)."
            )

        loss = self.weight * F.smooth_l1_loss(mu_result, mu_target)
        metrics[self._loss_key] = loss.item()

        with torch.no_grad():
            metrics[self._sim_key] = F.cosine_similarity(mu_result, mu_target).mean().item()

        return loss, metrics


class AlgebraicAdditionLoss(_AlgebraicBinaryLoss):
    r"""Enforce additive homomorphism: z(a ⊕ b) ≈ z(a) + z(b)."""

    _loss_key = "alg_addition_loss"
    _sim_key = "alg_addition_sim"

    def _ternary_op(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        return TERNARY.ternary_add(idx_a, idx_b)

    def _mu_target(
        self, mu_A: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
    ) -> torch.Tensor:
        return mu_A[idx_a] + mu_A[idx_b]


class AlgebraicMultiplicationLoss(_AlgebraicBinaryLoss):
    r"""Enforce multiplicative homomorphism: z(a ⊗ b) ≈ z(a) ⊙ z(b) (element-wise)."""

    _loss_key = "alg_multiplication_loss"
    _sim_key = "alg_multiplication_sim"

    def _ternary_op(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        return TERNARY.ternary_mul(idx_a, idx_b)

    def _mu_target(
        self, mu_A: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
    ) -> torch.Tensor:
        return mu_A[idx_a] * mu_A[idx_b]


class AlgebraicDistributiveLoss(nn.Module):
    r"""Enforce the distributive law: z(a ⊗ (b ⊕ c)) ≈ z(a) ⊙ (z(b) + z(c))."""

    def __init__(
        self,
        weight: float = 1.0,
        n_triplets: int = 512,
        phase_start_epoch: int = 0,
    ):
        super().__init__()
        self.weight = weight
        self.n_triplets = n_triplets
        self.phase_start_epoch = phase_start_epoch
        self.generator = torch.Generator()
        self.generator.manual_seed(42)

    def forward(
        self,
        mu_A: torch.Tensor,
        indices: torch.Tensor,
        model: nn.Module,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, MetricsDict]:
        metrics: MetricsDict = {}
        zero = make_zero_loss(mu_A.device, mu_A.dtype)

        if epoch < self.phase_start_epoch or self.weight <= 0:
            metrics["alg_distributive_loss"] = 0.0
            return zero, metrics

        B = mu_A.shape[0]
        n_samples = min(self.n_triplets, B // 3)
        if B < 3 or n_samples < 1:
            metrics["alg_distributive_loss"] = 0.0
            return zero, metrics

        perm = torch.randperm(B, generator=self.generator).to(mu_A.device)
        idx_a = perm[:n_samples]
        idx_b = perm[n_samples:2 * n_samples]
        idx_c = perm[2 * n_samples:3 * n_samples]

        idx_sum_bc = TERNARY.ternary_add(indices[idx_b], indices[idx_c])
        idx_dist_gt = TERNARY.ternary_mul(indices[idx_a], idx_sum_bc)
        mu_res = model.get_mu_representations(idx_dist_gt, mu_A.device)
        mu_target = mu_A[idx_a] * (mu_A[idx_b] + mu_A[idx_c])

        if mu_res.shape != mu_target.shape:
            raise RuntimeError(
                f"AlgebraicDistributiveLoss: shape mismatch — "
                f"mu_res {mu_res.shape} vs mu_target {mu_target.shape}. "
                f"model.get_mu_representations must return (n_samples, latent_dim)."
            )

        loss = self.weight * F.smooth_l1_loss(mu_res, mu_target)
        metrics["alg_distributive_loss"] = loss.item()

        with torch.no_grad():
            metrics["alg_distributive_sim"] = F.cosine_similarity(mu_res, mu_target).mean().item()

        return loss, metrics
