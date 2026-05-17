# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Reporting, logging, and checkpointing for p-adic VAE training."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from ..utils import TensorBoardLogger


def _build_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    lr_controller: Any = None,
    dual_state: Any = None,
    loss_cfg: Optional[Dict[str, Any]] = None,
    loss_fn: Optional[nn.Module] = None,
    loss_fn_b: Optional[nn.Module] = None,
    grokking_detector: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a checkpoint payload dict with all resumable state.

    Centralises the checkpoint construction that was previously duplicated
    at best-Q, periodic, and final saving sites.

    Args:
        epoch: Current epoch number.
        model: The VAE model.
        optimizer: Optimizer with state dict.
        scheduler: LR scheduler with state dict.
        lr_controller: Optional MetricBasedLR controller.
        dual_state: Optional LagrangianDualState.
        loss_cfg: Optional loss configuration dict.
        loss_fn: Optional CombinedLoss instance (for VAE-A).
        loss_fn_b: Optional CombinedLoss instance (for VAE-B).
        grokking_detector: Optional GrokkingDetector instance.
        extra: Optional dict of extra metrics/metadata.

    Returns:
        Checkpoint payload dictionary.
    """
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    if lr_controller:
        payload["lr_controller_state"] = lr_controller.state_dict()
    if dual_state:
        payload["lagrangian_state"] = dual_state.state_dict()
    if loss_cfg:
        payload["loss_config"] = loss_cfg
    if loss_fn and hasattr(loss_fn, "state_dict"):
        payload["loss_fn_state_dict"] = loss_fn.state_dict()
    if loss_fn_b and hasattr(loss_fn_b, "state_dict"):
        payload["loss_fn_b_state_dict"] = loss_fn_b.state_dict()
    if grokking_detector:
        payload["grokking_state"] = grokking_detector.state_dict()
        # Also include events summary for easy parsing without full load
        payload["grokking_events"] = [
            {"epoch": e.epoch, "lift": e.val_lift} for e in grokking_detector.events
        ]
    return payload


class ReportingManager:
    """Manages TensorBoard, console logging, and result persistence."""

    def __init__(
        self,
        log_dir: Path,
        config: Dict[str, Any],
        tb_logger: Optional[TensorBoardLogger] = None,
    ):
        self.log_dir = log_dir
        self.config = config
        self.tb_logger = tb_logger
        self.results_path = log_dir / "results.json"

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log metrics to TensorBoard and console."""
        if self.tb_logger and self.tb_logger.is_available:
            for name, val in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.tb_logger.log_scalar(tag, val, step)

    def save_results(self, results: Mapping[str, Any]) -> None:
        """Persist results to JSON."""
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

    def print_summary(self, results: Mapping[str, Any]) -> None:
        """Print training summary to console."""
        print(f"\n{'=' * 60}")
        print("  TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Best Q:         {results.get('best_Q', 0):.4f}")
        print(f"  Best Hierarchy: {results.get('best_hierarchy', 0):.4f}")
        print(f"  Best Coverage:  {results.get('best_coverage', 0):.2%}")
        print(f"  Grokking Events: {len(results.get('grokking_events', []))}")
        print(f"  Results: {self.results_path}")
        if self.tb_logger:
            print(f"  TensorBoard: tensorboard --logdir {self.log_dir}")
        print(f"{'=' * 60}\n")
