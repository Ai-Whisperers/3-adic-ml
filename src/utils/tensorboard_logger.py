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
    SummaryWriter = None  # type: ignore[assignment, misc]

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
        log_dir: Optional[Union[str, Path]],
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory where TensorBoard logs will be saved
            log_callback: Optional callback for log messages
        """
        self.writer: Optional[SummaryWriterType] = None
        self.log_callback = log_callback or (lambda msg: None)

        if TENSORBOARD_AVAILABLE and log_dir is not None:
            self.writer = SummaryWriter(str(log_dir))
            self.log_callback(f"TensorBoard logging to: {log_dir}")
        elif log_dir is not None and not TENSORBOARD_AVAILABLE:
            self.log_callback(
                "[WARN] TensorBoard not available. Metrics will not be visualized."
            )

    @property
    def is_available(self) -> bool:
        """Check if TensorBoard is available and initialized."""
        return self.writer is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name/tag
            value: Scalar value to log
            step: Current training step or epoch
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(
        self, metrics: Mapping[str, float], step: int, prefix: str = ""
    ) -> None:
        """Log multiple scalar metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step or epoch
            prefix: Optional prefix for metric tags
        """
        if self.writer:
            for name, val in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, val, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of values.

        Args:
            tag: Histogram name/tag
            values: Tensor of values
            step: Current training step or epoch
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log histograms of model weights.

        Args:
            model: PyTorch model
            step: Current training step or epoch
        """
        if self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"Weights/{name}", param, step)

    def log_hparams(
        self, hparams: Mapping[str, Any], metrics: Mapping[str, float]
    ) -> None:
        """Log hyperparameters and final metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        if self.writer:
            # SummaryWriter.add_hparams expects specific types; ensure consistency
            clean_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool, torch.Tensor)):
                    clean_hparams[k] = v
                else:
                    clean_hparams[k] = str(v)

            self.writer.add_hparams(clean_hparams, dict(metrics))

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
            self.writer = None

    def flush(self) -> None:
        """Flush the TensorBoard writer."""
        if self.writer:
            self.writer.flush()

    def add_embedding(
        self,
        mat: torch.Tensor,
        metadata: Optional[Any] = None,
        label_img: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        tag: str = "default",
        metadata_header: Optional[Any] = None,
    ) -> None:
        """Add embedding to TensorBoard."""
        if self.writer:
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
    ) -> None:
        """Add a matplotlib figure to TensorBoard."""
        if self.writer:
            self.writer.add_figure(tag, figure, global_step=global_step, close=close)
