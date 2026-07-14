# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Configuration and component setup for p-adic VAE training."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..config import StateNetConfig
from ..config.statenet_config import LRScales
from ..core import TERNARY
from ..core import get_valuation_fn
from ..geometry import get_riemannian_optimizer
from ..losses import CombinedLoss
from ..losses.lagrangian import LagrangianDualState
from ..models import MetricBasedLR


def setup_dataloaders(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    config: Dict[str, Any],
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """Setup train and validation DataLoaders with stratified sampling."""
    train_cfg = config.get("training") or {}
    batch_size = train_cfg.get("batch_size", 512)
    num_workers = train_cfg.get("num_workers", 4)

    valuation_type = (config.get("data") or {}).get("valuation_type", "index")
    valuation_fn = get_valuation_fn(valuation_type)

    def worker_init_fn(worker_id: int) -> None:
        import numpy as np
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_indices = train_ds.tensors[1]
    train_valuations = valuation_fn(train_indices)
    level_counts = torch.bincount(train_valuations, minlength=TERNARY.MAX_VALUATION + 1)

    level_weights = torch.zeros(TERNARY.MAX_VALUATION + 1, dtype=torch.float64)
    nonzero = level_counts > 0
    level_weights[nonzero] = 1.0 / level_counts[nonzero].to(torch.float64).sqrt()
    sample_weights = level_weights[train_valuations.long()]
    sample_weights_list = sample_weights.tolist()

    stratified_generator = torch.Generator()
    stratified_generator.manual_seed(seed)
    stratified_sampler = WeightedRandomSampler(
        weights=sample_weights_list,
        num_samples=len(train_ds),
        replacement=True,
        generator=stratified_generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=stratified_sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    return train_loader, val_loader


def setup_losses(
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[CombinedLoss, CombinedLoss]:
    """Setup CombinedLoss for VAE-A and VAE-B."""
    loss_cfg = config.get("loss") or {}
    model_cfg = config.get("model") or {}
    curvature = model_cfg.get("curvature", 1.0)
    valuation_type = (config.get("data") or {}).get("valuation_type", "index")

    latent_dim = model_cfg.get("latent_dim", 64)

    loss_fn = CombinedLoss(
        loss_cfg, curvature=curvature, device=device, valuation_type=valuation_type, latent_dim=latent_dim
    )

    # VAE-B loss: hierarchy-only (deep copy to avoid shared nested dict refs)
    loss_cfg_b = copy.deepcopy(loss_cfg)
    if "rich_hierarchy" in loss_cfg_b and isinstance(loss_cfg_b["rich_hierarchy"], dict):
        loss_cfg_b["rich_hierarchy"]["coverage_weight"] = 0.0

    loss_fn_b = CombinedLoss(
        loss_cfg_b, curvature=curvature, device=device, valuation_type=valuation_type, latent_dim=latent_dim
    )

    return loss_fn, loss_fn_b


def setup_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    loss_params: List[nn.Parameter],
) -> torch.optim.Optimizer:
    """Setup Riemannian or standard optimizer."""
    train_cfg = config.get("training") or {}
    riemannian_cfg = config.get("riemannian") or {}
    base_lr = train_cfg.get("lr", 8e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    param_groups = model.get_param_groups(base_lr)
    if loss_params:
        param_groups.append(
            {"params": loss_params, "lr": base_lr, "name": "loss_weights"}
        )

    if riemannian_cfg.get("enabled", True):
        stabilize = riemannian_cfg.get("stabilize", 10)
        optimizer = get_riemannian_optimizer(
            param_groups,
            lr=base_lr,
            stabilize=stabilize,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )

    return optimizer


def _cosine_warmup_restarts_factor(epoch: int, T_0: int, T_mult: int) -> float:
    """Return the cosine-annealing-with-warm-restarts multiplier in [0, 1].

    Analytically reimplements CosineAnnealingWarmRestarts so the factor can
    be composed with a phase scale inside a single LambdaLR.  Using
    ChainedScheduler([CosineAnnealingWarmRestarts, LambdaLR]) was wrong:
    LambdaLR uses optimizer base_lr as its reference, overwriting cosine's
    output and making the cosine schedule a no-op.
    """
    if T_0 <= 0:
        return 1.0
    if T_mult == 1:
        T_cur = epoch % T_0
        T_i = T_0
    else:
        T_i = T_0
        t_start = 0
        while t_start + T_i <= epoch:
            t_start += T_i
            T_i = max(1, int(T_i * T_mult))
        T_cur = epoch - t_start
    return 0.5 * (1.0 + math.cos(math.pi * T_cur / T_i))


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Any:
    """Setup LR scheduler."""
    train_cfg = config.get("training") or {}
    epochs = train_cfg.get("epochs", 100)
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "cosine")

    if scheduler_type == "cosine_warmup_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get("T_0", 25),
            T_mult=scheduler_cfg.get("T_mult", 2),
        )
    elif scheduler_type == "multi_phase_cosine":
        phases = scheduler_cfg.get("phases", [])
        if phases:
            first_phase = phases[0]
            T_0 = int(first_phase.get("T_0", scheduler_cfg.get("T_0", 25)))
            T_mult = int(first_phase.get("T_mult", scheduler_cfg.get("T_mult", 2)))

            phase_schedule = []
            for p in phases:
                epoch_range = p.get("epoch_range", [0, epochs])
                scale = p.get("base_lr_scale", 1.0)
                phase_schedule.append((epoch_range[0], epoch_range[1], scale))
            phase_schedule.sort(key=lambda x: x[0])

            # Single LambdaLR that multiplies base_lr by (cosine_factor × phase_scale).
            # This fixes the old ChainedScheduler approach where LambdaLR overwrote
            # the cosine output, making the cosine schedule a no-op.
            def multi_phase_lr_lambda(epoch: int) -> float:
                phase_scale = phase_schedule[-1][2]
                for start, end, scale in phase_schedule:
                    if start <= epoch < end:
                        phase_scale = scale
                        break
                cosine_factor = _cosine_warmup_restarts_factor(epoch, T_0, T_mult)
                return phase_scale * cosine_factor

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=multi_phase_lr_lambda
            )

    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)


def setup_controller(
    config: Dict[str, Any],
) -> Optional[MetricBasedLR]:
    """Setup StateNet training controller."""
    statenet_cfg = config.get("statenet") or {}
    if not statenet_cfg.get("enabled", True):
        return None

    sn_config = StateNetConfig.from_dict(statenet_cfg)
    option_c_cfg = config.get("option_c") or {}

    if option_c_cfg.get("enabled", True):
        sn_config.lr_scales = LRScales(
            encoder_a=option_c_cfg.get("encoder_a_lr_scale", sn_config.lr_scales.encoder_a),
            encoder_b=option_c_cfg.get("encoder_b_lr_scale", sn_config.lr_scales.encoder_b),
            projections=option_c_cfg.get("projections_lr_scale", sn_config.lr_scales.projections),
            decoders=sn_config.lr_scales.decoders,
        )

    return MetricBasedLR(sn_config)


def setup_lagrangian(
    config: Dict[str, Any],
) -> Optional[LagrangianDualState]:
    """Setup Lagrangian dual adaptive weighting."""
    lagrangian_cfg = config.get('loss', {}).get('lagrangian', {})
    if lagrangian_cfg.get('enabled', False):
        return LagrangianDualState.from_config(lagrangian_cfg)
    return None
