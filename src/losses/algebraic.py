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
        zero = torch.tensor(0.0, device=z_hyp.device, dtype=z_hyp.dtype)

        if epoch < self.phase_start_epoch:
            metrics["angular_coherence_loss"] = 0.0
            metrics["angular_coherence_pairs"] = 0
            return zero, metrics

        eps = torch.tensor(1e-10, device=z_hyp.device, dtype=z_hyp.dtype)
        dir_vecs = z_hyp / r.unsqueeze(-1).clamp(min=eps)

        vals = self._valuation_fn(indices)
        B = len(dir_vecs)
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
                nv = len(idx_v)
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
                level_loss = torch.nn.functional.relu(t - cos_sim).mean()
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
        # Use the per-valuation target_sim for each same-class pair.
        # Same-class pairs share the same (vals, prefix_k) key, so they share
        # the same valuation level — use that level's target similarity.
        pair_vals = vals[i_idx[same_cls]].clamp(0, len(self.target_sim) - 1)
        t_values = torch.tensor(self.target_sim, device=z_hyp.device, dtype=z_hyp.dtype)
        t_per_pair = t_values[pair_vals]
        loss = self.weight * torch.nn.functional.relu(t_per_pair - cos_sim).mean()

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
        zero = torch.tensor(0.0, device=z_hyp.device, dtype=z_hyp.dtype)

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
            nc = len(class_idx)
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


class AlgebraicAdditionLoss(nn.Module):
    r"""Enforce additive consistency in tangent space (Mu space)."""

    def __init__(
        self,
        weight: float = 1.0,
        n_pairs: int = 512,
        phase_start_epoch: int = 0,
        valuation_fn=None,
    ):
        super().__init__()
        self.weight = weight
        self.n_pairs = n_pairs
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
        zero = torch.tensor(0.0, device=mu_A.device, dtype=mu_A.dtype)

        if epoch < self.phase_start_epoch or self.weight <= 0:
            metrics["alg_addition_loss"] = 0.0
            return zero, metrics

        B = mu_A.shape[0]
        if B < 2:
            metrics["alg_addition_loss"] = 0.0
            return zero, metrics

        n_triplets = min(self.n_pairs, B // 2)
        if n_triplets < 1:
            metrics["alg_addition_loss"] = 0.0
            return zero, metrics

        perm = torch.randperm(B, generator=self.generator).to(mu_A.device)
        idx_a_local = perm[:n_triplets]
        idx_b_local = perm[n_triplets : 2 * n_triplets]

        idx_sum = TERNARY.ternary_add(indices[idx_a_local], indices[idx_b_local])
        mu_sum = model.get_mu_representations(idx_sum, mu_A.device)

        mu_target = mu_A[idx_a_local] + mu_A[idx_b_local]
        if mu_sum.shape != mu_target.shape:
            raise RuntimeError(
                f"AlgebraicAdditionLoss: shape mismatch — "
                f"mu_sum {mu_sum.shape} vs mu_target {mu_target.shape}. "
                f"model.get_mu_representations must return shape "
                f"(n_pairs, latent_dim) matching the input batch."
            )
        loss_val = F.smooth_l1_loss(mu_sum, mu_target)

        loss = self.weight * loss_val
        metrics["alg_addition_loss"] = loss.item()

        with torch.no_grad():
            cos_sim = F.cosine_similarity(mu_sum, mu_target).mean()
            metrics["alg_addition_sim"] = cos_sim.item()

        return loss, metrics


class AlgebraicMultiplicationLoss(nn.Module):
    r"""Enforce multiplicative consistency in tangent space (Mu space).
    
    Objective: z(a ⊗ b) ≈ z(a) ⊙ z(b) where ⊙ is element-wise product.
    """

    def __init__(
        self,
        weight: float = 1.0,
        n_pairs: int = 512,
        phase_start_epoch: int = 0,
        valuation_fn=None,
    ):
        super().__init__()
        self.weight = weight
        self.n_pairs = n_pairs
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
        zero = torch.tensor(0.0, device=mu_A.device, dtype=mu_A.dtype)

        if epoch < self.phase_start_epoch or self.weight <= 0:
            metrics["alg_multiplication_loss"] = 0.0
            return zero, metrics

        B = mu_A.shape[0]
        if B < 2:
            metrics["alg_multiplication_loss"] = 0.0
            return zero, metrics

        n_triplets = min(self.n_pairs, B // 2)
        if n_triplets < 1:
            metrics["alg_multiplication_loss"] = 0.0
            return zero, metrics

        perm = torch.randperm(B, generator=self.generator).to(mu_A.device)
        idx_a_local = perm[:n_triplets]
        idx_b_local = perm[n_triplets : 2 * n_triplets]

        idx_prod = TERNARY.ternary_mul(indices[idx_a_local], indices[idx_b_local])
        mu_prod = model.get_mu_representations(idx_prod, mu_A.device)

        # Multiplicative homomorphism: mu(a*b) ≈ mu(a) * mu(b) (element-wise)
        mu_target = mu_A[idx_a_local] * mu_A[idx_b_local]
        if mu_prod.shape != mu_target.shape:
            raise RuntimeError(
                f"AlgebraicMultiplicationLoss: shape mismatch — "
                f"mu_prod {mu_prod.shape} vs mu_target {mu_target.shape}. "
                f"model.get_mu_representations must return shape "
                f"(n_pairs, latent_dim) matching the input batch."
            )
        loss_val = F.smooth_l1_loss(mu_prod, mu_target)

        loss = self.weight * loss_val
        metrics["alg_multiplication_loss"] = loss.item()

        with torch.no_grad():
            cos_sim = F.cosine_similarity(mu_prod, mu_target).mean()
            metrics["alg_multiplication_sim"] = cos_sim.item()

        return loss, metrics


class AlgebraicDistributiveLoss(nn.Module):
    r"""Enforce the distributive law in tangent space (Mu space).
    
    Objective: z(a ⊗ (b ⊕ c)) ≈ z(a) ⊙ (z(b) + z(c))
    """

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
        zero = torch.tensor(0.0, device=mu_A.device, dtype=mu_A.dtype)

        if epoch < self.phase_start_epoch or self.weight <= 0:
            metrics["alg_distributive_loss"] = 0.0
            return zero, metrics

        B = mu_A.shape[0]
        if B < 3:
            metrics["alg_distributive_loss"] = 0.0
            return zero, metrics

        n_samples = min(self.n_triplets, B // 3)
        if n_samples < 1:
            metrics["alg_distributive_loss"] = 0.0
            return zero, metrics

        perm = torch.randperm(B, generator=self.generator).to(mu_A.device)
        idx_a = perm[:n_samples]
        idx_b = perm[n_samples : 2 * n_samples]
        idx_c = perm[2 * n_samples : 3 * n_samples]

        # Ground truth: a * (b + c)
        idx_sum_bc = TERNARY.ternary_add(indices[idx_b], indices[idx_c])
        idx_dist_gt = TERNARY.ternary_mul(indices[idx_a], idx_sum_bc)
        
        # Representations
        mu_res = model.get_mu_representations(idx_dist_gt, mu_A.device)

        # Distributive target: mu(a) * (mu(b) + mu(c))
        mu_target = mu_A[idx_a] * (mu_A[idx_b] + mu_A[idx_c])

        if mu_res.shape != mu_target.shape:
            raise RuntimeError(
                f"AlgebraicDistributiveLoss: shape mismatch — "
                f"mu_res {mu_res.shape} vs mu_target {mu_target.shape}. "
                f"model.get_mu_representations must return shape "
                f"(n_samples, latent_dim) matching the input batch."
            )
        loss_val = F.smooth_l1_loss(mu_res, mu_target)
        loss = self.weight * loss_val
        metrics["alg_distributive_loss"] = loss.item()

        with torch.no_grad():
            cos_sim = F.cosine_similarity(mu_res, mu_target).mean()
            metrics["alg_distributive_sim"] = cos_sim.item()

        return loss, metrics
