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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union

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

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Tag for the metric (e.g. 'Loss/train')
            value: Scalar value to log
            step: Global step or epoch
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_hparams(
        self,
        hparam_dict: Mapping[str, Union[str, float, int, bool]],
        metric_dict: Mapping[str, float],
        run_name: Optional[str] = None,
    ) -> None:
        """Log hyperparameters and associated best metrics.

        Args:
            hparam_dict: Hyperparameter configuration
            metric_dict: Best results metrics
            run_name: Optional custom run name for the hparams entry
        """
        if self.writer is not None:
            self.writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    def log_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        """Log a text string.

        Args:
            tag: Tag for the text
            text_string: Text to log (supports markdown)
            global_step: Optional global step
        """
        if self.writer is not None:
            self.writer.add_text(tag, text_string, global_step)

    def add_custom_scalars(self, layout: dict[str, dict[str, list[Any]]]) -> None:
        """Add custom scalars layout for dashboard organization.

        Args:
            layout: Layout dictionary defining multi-line charts
        """
        if self.writer is not None:
            self.writer.add_custom_scalars(layout)

    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: Optional[list[Any]] = None,
        label_img: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        tag: str = "default",
        metadata_header: Optional[list[str]] = None,
    ) -> None:
        """Add high-dimensional embeddings for TensorBoard Projector.

        Args:
            mat: Embedding matrix (N, D)
            metadata: List of labels per sample
            label_img: Images associated with samples
            global_step: Current step/epoch
            tag: Name for this embedding
            metadata_header: Header for the metadata columns
        """
        if self.writer is not None:
            self.writer.add_embedding(
                mat,
                metadata=metadata,
                label_img=label_img,
                global_step=global_step,
                tag=tag,
                metadata_header=metadata_header,
            )

    def add_figure(
        self,
        tag: str,
        figure: Any,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> None:
        """Add a Matplotlib figure to TensorBoard.

        Args:
            tag: Name for the figure
            figure: Matplotlib figure or list of figures
            global_step: Current step/epoch
            close: If True, close the figure after logging
            walltime: Custom timestamp
        """
        if self.writer is not None:
            self.writer.add_figure(
                tag, figure, global_step=global_step, close=close, walltime=walltime
            )

    def flush(self) -> None:
        """Flush pending TensorBoard events."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer (idempotent)."""
        if self.writer is not None:
            self.writer.close()
            self.log_callback("TensorBoard writer closed")
            self.writer = None


__all__ = ["TensorBoardLogger", "TENSORBOARD_AVAILABLE"]
