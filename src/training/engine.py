# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Core training engine for p-adic VAE."""

from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tqdm for progress bars (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

from ..core.contracts import TrainingResults, VAEOutput
from ..core import get_valuation_fn
from ..models import (
    TrainingMetrics,
    get_optimizer_grad_stats,
    update_optimizer_lr_scales,
)
from .metrics import (
    compute_accuracy,
    compute_coverage,
    compute_hierarchy_metrics,
    plot_algebraic_consistency,
)
from .reporting import ReportingManager
from .setup import DEFAULT_LR


_TRACKED_LOSS_KEYS = {
    "hyperbolic_contrastive": "contrastive",
    "surrogate_property": "surrogate",
    "algebraic_coherence": "alg_coherence",
}


def _assert_finite_losses(losses: Dict[str, Any], label: str, epoch: int) -> None:
    """Raise on NaN/Inf in any tensor-valued loss entry."""
    for name, val in losses.items():
        if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
            raise RuntimeError(
                f"Numerical instability: {label} {name} loss is {val} at epoch {epoch}"
            )


def _apply_controller_update(
    lr_controller: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    model: nn.Module,
    reporting: ReportingManager,
    train_cfg: Dict[str, Any],
    epoch: int,
    val_metrics: Dict[str, Any],
    hier_metrics_A: Dict[str, Any],
    hier_metrics_B: Dict[str, Any],
) -> None:
    """Update the StateNet LR controller and apply its output.

    Mutates `optimizer` param-group LRs/requires_grad (via
    update_optimizer_lr_scales) and the model's internal trainable flags in
    place, and logs controller metrics to `reporting`.
    """
    grad_stats = get_optimizer_grad_stats(optimizer, "projections")
    controller_grad_norm = grad_stats.get("grad_norm_estimate", 0.0)

    metrics = TrainingMetrics(
        epoch=epoch,
        coverage=val_metrics["avg_val_acc"],
        hierarchy_a=hier_metrics_A["hierarchy"],
        hierarchy_b=hier_metrics_B["hierarchy"],
        dist_corr_a=hier_metrics_A["dist_corr"],
        q_value=hier_metrics_A["Q"],
        grad_norm_projections=controller_grad_norm,
        hierarchy_a_collapsed=bool(hier_metrics_A.get("hierarchy_collapsed", False)),
        hierarchy_b_collapsed=bool(hier_metrics_B.get("hierarchy_collapsed", False)),
    )
    controller_state = lr_controller.update(metrics)

    # Apply new scales — also syncs requires_grad for each param group
    cosine_factor = scheduler.get_last_lr()[0] / scheduler.base_lrs[0]
    current_base_lr = train_cfg.get("lr", DEFAULT_LR) * cosine_factor
    new_scales = controller_state["lr_scales"]
    update_optimizer_lr_scales(optimizer, current_base_lr, new_scales)

    # Keep model's internal _trainable state consistent with LR scales.
    # requires_grad is already synced by update_optimizer_lr_scales above;
    # these calls only update the model's _trainable flag (used by
    # is_trainable / get_trainable_params for introspection).
    model.set_encoder_a_trainable(new_scales.get('encoder_a', 0.0) > 0)
    model.set_encoder_b_trainable(new_scales.get('encoder_b', 0.0) > 0)
    model.set_projections_trainable(new_scales.get('projections', 0.0) > 0)

    reporting.log_metrics({
        "encoder_a_lr_scale": new_scales.get("encoder_a", 0),
        "encoder_b_lr_scale": new_scales.get("encoder_b", 0),
        "projections_lr_scale": new_scales.get("projections", 0),
        "best_Q": controller_state.get("best_q", 0),
    }, epoch, prefix="LRController")


def _build_epoch_log_dict(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    hier_metrics_A: Dict[str, Any],
    hier_metrics_B: Dict[str, Any],
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """Assemble the flat metrics dict logged to TensorBoard once per eval epoch."""
    log_dict: Dict[str, float] = {
        "Accuracy/train": train_metrics["avg_train_acc"],
        "Accuracy/val": val_metrics["avg_val_acc"],
        "Loss/train": train_metrics["avg_train_loss"],
        "Coverage": val_metrics["avg_val_coverage"],
        "Hierarchy/corr_VAE_A": hier_metrics_A["hierarchy"],
        "Hierarchy/corr_VAE_B": hier_metrics_B["hierarchy"],
        "Hierarchy/Q_VAE_A": hier_metrics_A["Q"],
        "Hierarchy/Q_VAE_B": hier_metrics_B["Q"],
        "Hierarchy/dist_corr": hier_metrics_A["dist_corr"],
        "Radius/mean_VAE_A": hier_metrics_A["mean_radius"],
        "Radius/mean_VAE_B": hier_metrics_B["mean_radius"],
    }

    # Tracked per-loss metrics (contrastive, surrogate, algebraic coherence),
    # derived from _TRACKED_LOSS_KEYS so every loss tracked in train_epoch is
    # logged here without hand-listing each name.
    for short_name in _TRACKED_LOSS_KEYS.values():
        for vae in ("A", "B"):
            key = f"avg_train_{short_name}_{vae}"
            if key in train_metrics:
                log_dict[f"Loss/train_{short_name}_{vae}"] = train_metrics[key]

    log_dict["Loss/biological_losses_active"] = (
        1.0 if getattr(loss_fn, "biological_losses_active", True) else 0.0
    )

    for v, r_val in hier_metrics_A.get("level_radii", {}).items():
        if not np.isnan(r_val):
            log_dict[f"Radius/level_v{v}"] = r_val

    for name, val in val_metrics.get("alg_metrics", {}).items():
        log_dict[f"Algebraic/{name}"] = val

    return log_dict


def train_model(
    config: Dict[str, Any],
    device: torch.device,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    loss_fn_b: nn.Module,
    reporting: ReportingManager,
    lr_controller: Optional[Any] = None,
    grokking_detector: Optional[Any] = None,
    dual_state: Optional[Any] = None,
    vis_pipeline: Optional[Any] = None,
    use_amp: bool = False,
    start_epoch: int = 0,
    best_stats: Optional[Dict[str, float]] = None,
) -> TrainingResults:
    """Main training orchestration loop."""
    train_cfg = config.get("training") or {}
    loss_cfg = config.get("loss") or {}
    epochs = train_cfg.get("epochs", 100)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    eval_every = train_cfg.get("eval_every", 5)
    save_every = train_cfg.get("save_every", 25)

    # Resolve valuation function
    valuation_type = (config.get("data") or {}).get("valuation_type", "index")
    valuation_fn = get_valuation_fn(valuation_type)

    # Best metrics tracking
    if best_stats is None:
        best_stats = {"best_Q": -1.0, "best_hierarchy": 0.0, "best_coverage": 0.0}

    best_Q = best_stats["best_Q"]
    best_hierarchy = best_stats["best_hierarchy"]
    best_coverage = best_stats["best_coverage"]

    ckpt_dir = reporting.log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Common args shared by every checkpoint save site below; only epoch/extra vary.
    build_checkpoint = partial(
        reporting.build_checkpoint_payload,
        model=model, optimizer=optimizer, scheduler=scheduler,
        lr_controller=lr_controller, dual_state=dual_state,
        loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
        grokking_detector=grokking_detector,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    current_dual_weights = None
    if dual_state:
        current_dual_weights = dual_state.get_dual_weights()

    global_step = start_epoch * len(train_loader)

    epoch_iter = (
        tqdm(range(start_epoch, epochs), desc="Training", unit="epoch")
        if TQDM_AVAILABLE
        else range(start_epoch, epochs)
    )

    for epoch in epoch_iter:
        # 1. Training Phase
        train_metrics = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, loss_fn_b,
            device, scaler, max_grad_norm, use_amp,
            dual_state, current_dual_weights, reporting, global_step
        )
        global_step += len(train_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        reporting.log_metrics({"LR/scheduled": current_lr}, epoch)

        # Lagrangian dual ascent update — runs every epoch so dual variables
        # track constraint violations without waiting for eval windows.
        if dual_state and train_metrics.get("dual_violation_count", 0) > 0:
            # dual_violation_acc is already the per-epoch mean (train_epoch divides
            # by n_batches once) — do not divide again here.
            dual_state.update(train_metrics["dual_violation_acc"])
            current_dual_weights = dual_state.get_dual_weights()

        # 2. Validation & Reporting Phase
        if epoch % eval_every == 0 or epoch == epochs - 1:
            val_metrics = validate_epoch(
                epoch, model, val_loader, device, valuation_fn, config, loss_fn=loss_fn
            )

            # Check dynamic curriculum threshold
            dc_cfg = loss_cfg.get("dynamic_curriculum") or {}
            if dc_cfg.get("enabled", False):
                threshold = dc_cfg.get("coverage_threshold", 0.90)
                if val_metrics["avg_val_acc"] >= threshold:
                    if not getattr(loss_fn, "biological_losses_active", True):
                        print(f"\n  [CURRICULUM] Validation accuracy {val_metrics['avg_val_acc']:.4f} >= threshold {threshold:.4f}. Activating biological/radial/hierarchy losses!")
                    loss_fn.biological_losses_active = True
                    loss_fn_b.biological_losses_active = True

            # Extract hierarchy metrics (using VAE-A as primary)
            hier_metrics_A = val_metrics["hier_metrics_A"]
            hier_metrics_B = val_metrics["hier_metrics_B"]

            # Controller update
            if lr_controller:
                _apply_controller_update(
                    lr_controller, optimizer, scheduler, model, reporting,
                    train_cfg, epoch, val_metrics, hier_metrics_A, hier_metrics_B,
                )

            # Grokking detection
            if grokking_detector:
                grok_state = grokking_detector.update(
                    epoch, train_metrics["avg_train_loss"],
                    train_metrics["avg_train_acc"], val_metrics["avg_val_acc"]
                )
                if grok_state["event"]:
                    print(f"  [GROKKING] Event detected at epoch {epoch}!")

            # Log everything to TensorBoard
            log_dict = _build_epoch_log_dict(train_metrics, val_metrics, hier_metrics_A, hier_metrics_B, loss_fn)
            reporting.log_metrics(log_dict, epoch)

            # Algebraic Consistency Figure
            if reporting.tb_logger and "mu_A_cat" in val_metrics and val_metrics["mu_A_cat"] is not None:
                fig = plot_algebraic_consistency(
                    model, val_metrics["idx_cat"], val_metrics["mu_A_cat"], device
                )
                if fig is not None:
                    reporting.tb_logger.add_figure("Algebraic/AdditionConsistency", fig, global_step=epoch)

            # Visualization
            if vis_pipeline and epoch > 0:
                vis_pipeline.run(epoch, val_metrics["z_A_cat"].cpu(),
                                 valuation_fn(val_metrics["idx_cat"].cpu()),
                                 val_metrics["idx_cat"].cpu())

            # Checkpoint: best Q
            if hier_metrics_A["Q"] > best_Q:
                best_Q = hier_metrics_A["Q"]
                torch.save(
                    build_checkpoint(
                        epoch,
                        extra={
                            "Q": best_Q,
                            "hierarchy_A": hier_metrics_A["hierarchy"],
                            "hierarchy_B": hier_metrics_B["hierarchy"],
                            "coverage": val_metrics["avg_val_coverage"],
                        },
                    ),
                    ckpt_dir / "best_Q.pt",
                )

            # Update best stats
            best_coverage = max(best_coverage, val_metrics["avg_val_coverage"])
            best_hierarchy = max(best_hierarchy, hier_metrics_A["hierarchy"])

        # Periodic checkpoint
        if epoch % save_every == 0 and epoch > 0:
            torch.save(build_checkpoint(epoch), ckpt_dir / f"epoch_{epoch}.pt")

    # Final results
    results: TrainingResults = {
        "epochs_trained": epochs,
        "best_Q": best_Q,
        "best_hierarchy": best_hierarchy,
        "best_coverage": best_coverage,
        "grokking_events": [e.__dict__ for e in grokking_detector.events] if grokking_detector else [],
    }

    # Save final checkpoint
    torch.save(build_checkpoint(epochs, extra=results), ckpt_dir / "final.pt")

    return results


def train_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loss_fn_b: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
    use_amp: bool,
    dual_state: Optional[Any],
    current_dual_weights: Optional[Any],
    reporting: ReportingManager,
    global_step_start: int,
) -> Dict[str, Any]:
    """Train for one epoch."""
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    loss_accums: Dict[str, float] = {}
    # Violation floats for Lagrangian dual ascent; accumulated across batches.
    viol_accums: Dict[str, float] = {}

    if dual_state:
        dual_state.step_epoch(epoch)

    batch_iter = tqdm(loader, desc=f"Ep {epoch:03d}", leave=False, unit="batch") if TQDM_AVAILABLE else loader

    # Parameter identities are fixed for the duration of the epoch; build once
    # instead of re-concatenating three lists on every batch.
    all_params = (
        list(model.parameters())
        + list(loss_fn.parameters())
        + list(loss_fn_b.parameters())
    )

    for batch_ops, batch_idx in batch_iter:
        batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out: VAEOutput = model(batch_ops, decode_b=False)
            curr_c = model.projections.get_curvature()

            losses_A = loss_fn(
                out["z_A_hyp"], batch_idx, out.get("logits_A"), batch_ops,
                epoch=epoch, mu=out.get("mu_A"), logvar=out.get("logvar_A"),
                curvature=curr_c, dual_weights=current_dual_weights, r=out.get("r_A"),
                model=model
            )
            losses_B = loss_fn_b(
                out["z_B_hyp"], batch_idx, None, batch_ops,
                epoch=epoch, mu=out.get("mu_B"), logvar=out.get("logvar_B"),
                curvature=curr_c, dual_weights=current_dual_weights, r=out.get("r_B"),
                model=model
            )

            _assert_finite_losses(losses_A, "VAE-A", epoch)
            _assert_finite_losses(losses_B, "VAE-B", epoch)

            loss = losses_A["total"] + losses_B["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Includes learnable loss weight params so they're bounded if enabled.
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        acc_sum += compute_accuracy(out["logits_A"], batch_ops)

        n_batches += 1

        step_metrics: Dict[str, float] = {"Batch/Loss": loss.item(), "Batch/GradNorm": grad_norm.item()}
        for loss_key, short_name in _TRACKED_LOSS_KEYS.items():
            for vae, losses in (("A", losses_A), ("B", losses_B)):
                if loss_key in losses:
                    val = losses[loss_key].item()
                    acc_key = f"{loss_key}_{vae}"
                    loss_accums[acc_key] = loss_accums.get(acc_key, 0.0) + val
                    step_metrics[f"Batch/Loss_{short_name}_{vae}"] = val

        # Accumulate per-level float violations for Lagrangian dual ascent.
        # Keys: gap_viol_v{v}, scatter_v{v}, vp_gap_v{v} (floats, not tensors).
        for src_losses in (losses_A, losses_B):
            for metric_key in ("monotonic_metrics", "rank_metrics", "valuation_prior_metrics"):
                m = src_losses.get(metric_key)
                if m is None:
                    continue
                for k, v in m.items():
                    if isinstance(v, float) and (
                        k.startswith("gap_viol_v") or
                        k.startswith("scatter_v") or
                        k.startswith("vp_gap_v")
                    ):
                        viol_accums[k] = viol_accums.get(k, 0.0) + v

        reporting.log_metrics(step_metrics, global_step_start + n_batches)

    epoch_metrics: Dict[str, Any] = {
        "avg_train_loss": loss_sum / n_batches,
        "avg_train_acc": acc_sum / n_batches,
    }
    for loss_key, short_name in _TRACKED_LOSS_KEYS.items():
        for vae in ("A", "B"):
            acc_key = f"{loss_key}_{vae}"
            if acc_key in loss_accums:
                epoch_metrics[f"avg_train_{short_name}_{vae}"] = loss_accums[acc_key] / n_batches
    if viol_accums:
        epoch_metrics["dual_violation_acc"] = {k: v / n_batches for k, v in viol_accums.items()}
        epoch_metrics["dual_violation_count"] = n_batches
    return epoch_metrics


def validate_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    valuation_fn: Any,
    config: Dict[str, Any],
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Validate for one epoch."""
    model.eval()
    acc_sum = 0.0
    cov_sum = 0.0
    n_batches = 0
    z_A_all, z_B_all, idx_all = [], [], []
    mu_A_all = []

    with torch.no_grad():
        for batch_ops, batch_idx in loader:
            batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
            out: VAEOutput = model(batch_ops)
            logits_A = out.get("logits_A")
            if logits_A is not None:
                acc_sum += compute_accuracy(logits_A, batch_ops)
                cov_sum += compute_coverage(logits_A, batch_ops)
            n_batches += 1
            z_A_all.append(out["z_A_hyp"].detach())
            z_B_all.append(out["z_B_hyp"].detach())
            idx_all.append(batch_idx.detach())
            if "mu_A" in out:
                mu_A_all.append(out["mu_A"].detach())

    z_A_cat, z_B_cat, idx_cat = torch.cat(z_A_all), torch.cat(z_B_all), torch.cat(idx_all)
    mu_A_cat = torch.cat(mu_A_all) if mu_A_all else None
    curr_c = model.projections.get_curvature()

    hier_A = compute_hierarchy_metrics(z_A_cat, idx_cat, curr_c, seed=42+epoch, valuation_fn=valuation_fn)
    hier_B = compute_hierarchy_metrics(z_B_cat, idx_cat, curr_c, seed=1042+epoch, valuation_fn=valuation_fn)

    # Compute algebraic metrics if loss_fn is provided. Sample a subset to
    # avoid OOM/slow computation on the full validation set.
    alg_metrics: Dict[str, Any] = {}
    if loss_fn is not None and mu_A_cat is not None:
        n_alg = min(mu_A_cat.size(0), 1024)
        perm = torch.randperm(mu_A_cat.size(0), device=device)[:n_alg]
        mu_sub = mu_A_cat[perm]
        idx_sub = idx_cat[perm]

        for attr_name in (
            "algebraic_addition_loss",
            "algebraic_multiplication_loss",
            "algebraic_distributive_loss",
        ):
            sub_loss = getattr(loss_fn, attr_name, None)
            if sub_loss:
                _, m = sub_loss(mu_sub, idx_sub, model, epoch)
                alg_metrics.update(m)

        coherence_loss = getattr(loss_fn, "algebraic_coherence_loss", None)
        if coherence_loss:
            z_sub = z_A_cat[perm]
            r_sub = z_sub.norm(dim=-1)
            _, m = coherence_loss(z_sub, r_sub, idx_sub, epoch, model=model)
            alg_metrics.update(m)

    return {
        "avg_val_acc": acc_sum / n_batches,
        "avg_val_coverage": cov_sum / n_batches,
        "hier_metrics_A": hier_A,
        "hier_metrics_B": hier_B,
        "z_A_cat": z_A_cat,
        "z_B_cat": z_B_cat,
        "mu_A_cat": mu_A_cat,
        "idx_cat": idx_cat,
        "alg_metrics": alg_metrics,
    }
