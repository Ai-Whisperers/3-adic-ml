# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Core training engine for p-adic VAE."""

import time
from typing import Any, Dict, Optional

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
from ..core.ternary import get_valuation_fn
from ..models import (
    TrainingMetrics,
    get_optimizer_grad_stats,
    update_optimizer_lr_scales,
)
from ..utils import HardwareMonitor
from .metrics import (
    compute_accuracy,
    compute_coverage,
    compute_hierarchy_metrics,
)
from .reporting import ReportingManager, _build_checkpoint_payload


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
    hw_monitor: Optional[HardwareMonitor] = None,
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
    train_cfg.get("print_every", 5)

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
        time.time()

        # 1. Training Phase
        train_metrics = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, loss_fn_b,
            device, scaler, max_grad_norm, use_amp,
            dual_state, current_dual_weights, hw_monitor, reporting, global_step
        )
        global_step += len(train_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        reporting.log_metrics({"LR/scheduled": current_lr}, epoch)

        # 2. Validation & Reporting Phase
        if epoch % eval_every == 0 or epoch == epochs - 1:
            val_metrics = validate_epoch(
                epoch, model, val_loader, device, valuation_fn, config
            )

            # Extract hierarchy metrics (using VAE-A as primary)
            hier_metrics_A = val_metrics["hier_metrics_A"]
            hier_metrics_B = val_metrics["hier_metrics_B"]

            # Controller update
            if lr_controller:
                grad_stats = get_optimizer_grad_stats(optimizer, "projections")
                controller_grad_norm = grad_stats.get("grad_norm_estimate", 0.0)

                # Sync logic from original monolith...
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

                # Apply new scales
                cosine_factor = scheduler.get_last_lr()[0] / scheduler.base_lrs[0]
                current_base_lr = train_cfg.get("lr", 1e-3) * cosine_factor
                new_scales = controller_state["lr_scales"]
                update_optimizer_lr_scales(optimizer, current_base_lr, new_scales)

                # Sync model requires_grad
                model.set_encoder_a_trainable(new_scales.get('encoder_a', 0.0) > 0)
                model.set_encoder_b_trainable(new_scales.get('encoder_b', 0.0) > 0)
                model.set_projections_trainable(new_scales.get('projections', 0.0) > 0)

                # Log controller metrics
                reporting.log_metrics({
                    "encoder_a_lr_scale": new_scales.get("encoder_a", 0),
                    "encoder_b_lr_scale": new_scales.get("encoder_b", 0),
                    "projections_lr_scale": new_scales.get("projections", 0),
                    "best_Q": controller_state.get("best_q", 0),
                }, epoch, prefix="LRController")

            # Lagrangian update
            if dual_state and val_metrics.get("dual_violation_count", 0) > 0:
                avg_violations = {
                    k: v / val_metrics["dual_violation_count"]
                    for k, v in val_metrics["dual_violation_acc"].items()
                }
                dual_state.update(avg_violations)
                current_dual_weights = dual_state.get_dual_weights()

            # Grokking detection
            if grokking_detector:
                grok_state = grokking_detector.update(
                    epoch, train_metrics["avg_train_loss"],
                    train_metrics["avg_train_acc"], val_metrics["avg_val_acc"]
                )
                if grok_state["event"]:
                    print(f"  [GROKKING] Event detected at epoch {epoch}!")

            # Log everything to TensorBoard
            log_dict = {
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
            reporting.log_metrics(log_dict, epoch)

            # Visualization
            if vis_pipeline and epoch > 0:
                vis_pipeline.run(epoch, val_metrics["z_A_cat"].cpu(),
                                 valuation_fn(val_metrics["idx_cat"].cpu()),
                                 val_metrics["idx_cat"].cpu())

            # Checkpoint: best Q
            if hier_metrics_A["Q"] > best_Q:
                best_Q = hier_metrics_A["Q"]
                torch.save(
                    _build_checkpoint_payload(
                        epoch, model, optimizer, scheduler,
                        lr_controller=lr_controller, dual_state=dual_state,
                        loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
                        grokking_detector=grokking_detector,
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
            torch.save(
                _build_checkpoint_payload(
                    epoch, model, optimizer, scheduler,
                    lr_controller=lr_controller, dual_state=dual_state,
                    loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
                    grokking_detector=grokking_detector,
                ),
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    # Final results
    results: TrainingResults = {
        "epochs_trained": epochs,
        "best_Q": best_Q,
        "best_hierarchy": best_hierarchy,
        "best_coverage": best_coverage,
        "grokking_events": [e.__dict__ for e in grokking_detector.events] if grokking_detector else [],
    }

    # Save final checkpoint
    torch.save(
        _build_checkpoint_payload(
            epochs, model, optimizer, scheduler,
            lr_controller=lr_controller, dual_state=dual_state,
            loss_cfg=loss_cfg, loss_fn=loss_fn, loss_fn_b=loss_fn_b,
            grokking_detector=grokking_detector,
            extra=results,
        ),
        ckpt_dir / "final.pt",
    )

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
    hw_monitor: Optional[HardwareMonitor],
    reporting: ReportingManager,
    global_step_start: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0

    if dual_state:
        dual_state.step_epoch(epoch)

    batch_iter = tqdm(loader, desc=f"Ep {epoch:03d}", leave=False, unit="batch") if TQDM_AVAILABLE else loader

    for batch_ops, batch_idx in batch_iter:
        batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out: VAEOutput = model(batch_ops, decode_b=False)
            curr_c = model.projections.get_curvature()

            losses_A, metrics_A = loss_fn(
                out["z_A_hyp"], batch_idx, out.get("logits_A"), batch_ops,
                epoch=epoch, mu=out.get("mu_A"), logvar=out.get("logvar_A"),
                curvature=curr_c, dual_weights=current_dual_weights, r=out.get("r_A"),
                model=model
            )
            losses_B, metrics_B = loss_fn_b(
                out["z_B_hyp"], batch_idx, None, batch_ops,
                epoch=epoch, mu=out.get("mu_B"), logvar=out.get("logvar_B"),
                curvature=curr_c, dual_weights=current_dual_weights, r=out.get("r_B"),
                model=model
            )

            # --- Numerical Integrity: NaN/Inf Sanitizer ---
            for name, loss_val in losses_A.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(f"Numerical instability: VAE-A {name} loss is {loss_val} at epoch {epoch}")
            for name, loss_val in losses_B.items():
                if not torch.isfinite(loss_val):
                    raise RuntimeError(f"Numerical instability: VAE-B {name} loss is {loss_val} at epoch {epoch}")

            loss = losses_A["total"] + losses_B["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        logits_A = out.get("logits_A")
        if logits_A is not None:
            acc_sum += compute_accuracy(logits_A, batch_ops)
        n_batches += 1

        step = global_step_start + n_batches
        reporting.log_metrics({"Batch/Loss": loss.item(), "Batch/GradNorm": grad_norm.item()}, step)

    return {
        "avg_train_loss": loss_sum / n_batches,
        "avg_train_acc": acc_sum / n_batches,
    }


def validate_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    valuation_fn: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate for one epoch."""
    model.eval()
    acc_sum = 0.0
    cov_sum = 0.0
    n_batches = 0
    z_A_all, z_B_all, idx_all = [], [], []

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

    z_A_cat, z_B_cat, idx_cat = torch.cat(z_A_all), torch.cat(z_B_all), torch.cat(idx_all)
    curr_c = model.projections.get_curvature()

    hier_A = compute_hierarchy_metrics(z_A_cat, idx_cat, curr_c, seed=42+epoch, valuation_fn=valuation_fn)
    hier_B = compute_hierarchy_metrics(z_B_cat, idx_cat, curr_c, seed=1042+epoch, valuation_fn=valuation_fn)

    return {
        "avg_val_acc": acc_sum / n_batches,
        "avg_val_coverage": cov_sum / n_batches,
        "hier_metrics_A": hier_A,
        "hier_metrics_B": hier_B,
        "z_A_cat": z_A_cat,
        "idx_cat": idx_cat,
    }
