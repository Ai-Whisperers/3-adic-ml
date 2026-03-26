# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""TensorBoard logging for training visualization.

This module handles all TensorBoard-related logging:
- Batch-level metrics logging
- Epoch-level metrics logging
- Hyperbolic geometry metrics
- Weight histograms

Single responsibility: TensorBoard visualization only.
Latent space visualization is handled by VisualizationPipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

# TensorBoard integration (optional)
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as SummaryWriterType


class TensorBoardLogger:
    """Handles all TensorBoard logging operations.

    Provides methods for logging metrics, histograms, and embeddings
    to TensorBoard for visualization.

    Attributes:
        writer: TensorBoard SummaryWriter instance
        log_callback: Optional callback for log messages
    """

    def __init__(
        self,
        tensorboard_dir: Optional[str],
        experiment_name: str,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize TensorBoard logger.

        Args:
            tensorboard_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment run
            log_callback: Optional callback for log messages
        """
        self.writer: Optional[SummaryWriterType] = None
        self.log_callback = log_callback or (lambda msg: None)

        if TENSORBOARD_AVAILABLE and tensorboard_dir is not None:
            log_path = Path(tensorboard_dir) / f"ternary_vae_{experiment_name}"
            self.writer = SummaryWriter(str(log_path))
            self.log_callback(f"TensorBoard logging to: {log_path}")
        elif tensorboard_dir is not None and not TENSORBOARD_AVAILABLE:
            self.log_callback(
                "Warning: TensorBoard requested but not installed "
                "(pip install tensorboard)"
            )

    @property
    def is_available(self) -> bool:
        """Check if TensorBoard logging is available."""
        return self.writer is not None

    def log_batch(
        self,
        global_step: int,
        loss: float,
        ce_A: float = 0.0,
        ce_B: float = 0.0,
        kl_A: float = 0.0,
        kl_B: float = 0.0,
    ) -> None:
        """Log batch-level metrics.

        Args:
            global_step: Global batch step
            loss: Current batch loss
            ce_A: VAE-A cross-entropy
            ce_B: VAE-B cross-entropy
            kl_A: VAE-A KL divergence
            kl_B: VAE-B KL divergence
        """
        if self.writer is None:
            return

        self.writer.add_scalar("Batch/Loss", loss, global_step)
        self.writer.add_scalar("Batch/CE_A", ce_A, global_step)
        self.writer.add_scalar("Batch/CE_B", ce_B, global_step)
        self.writer.add_scalar("Batch/KL_A", kl_A, global_step)
        self.writer.add_scalar("Batch/KL_B", kl_B, global_step)

    def log_histograms(self, epoch: int, model: torch.nn.Module) -> None:
        """Log model weight histograms.

        Args:
            epoch: Current epoch
            model: Model to log weights from
        """
        if self.writer is None:
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Weights/{name}", param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    def flush(self) -> None:
        """Flush pending TensorBoard events."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.log_callback("TensorBoard writer closed")


__all__ = ["TensorBoardLogger", "TENSORBOARD_AVAILABLE"]
