# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""TensorBoard logging for training visualization.

This module handles all TensorBoard-related logging:
- Batch-level metrics logging
- Epoch-level metrics logging
- Hyperbolic geometry metrics
- Weight histograms
- Latent space embedding visualization

Single responsibility: TensorBoard visualization only.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from src.core import TERNARY
from src.geometry import poincare_distance

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

    def log_manifold_embedding(
        self,
        model: torch.nn.Module,
        epoch: int,
        device: str,
        n_samples: int = 5000,
        include_all: bool = False,
        seed: int = 42,
    ) -> None:
        """Log latent embeddings for 3D visualization.

        Uses TensorBoard's embedding projector for interactive
        visualization of the latent space with 3-adic metadata.

        Args:
            model: The VAE model to encode samples
            epoch: Current epoch for step tracking
            device: Device to run inference on
            n_samples: Number of samples to embed
            include_all: If True, embed all 19,683 operations
            seed: Random seed for reproducible sampling
        """
        if self.writer is None:
            return

        model.eval()

        # Generate operations (uses cached LUT from TERNARY singleton)
        all_operations = TERNARY.all_ternary()
        total_ops = len(all_operations)

        # Sample or use all (reproducible via seeded RNG)
        if include_all or n_samples >= total_ops:
            indices = list(range(total_ops))
        else:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(total_ops), n_samples))

        x = all_operations[indices].to(device)

        with torch.no_grad():
            # V6 model: forward(x) returns dict with z_A_hyp, z_B_hyp (Poincare manifold)
            outputs = model(x)
            z_A_hyp = outputs["z_A_hyp"]
            z_B_hyp = outputs["z_B_hyp"]
            # Tangent space representations for Euclidean embedding view
            mu_A = outputs["mu_A"]
            mu_B = outputs["mu_B"]

            # Compute hyperbolic radii (z_A_hyp/z_B_hyp are already on the Poincare manifold)
            origin_A = torch.zeros_like(z_A_hyp)
            origin_B = torch.zeros_like(z_B_hyp)
            hyp_radii_A = poincare_distance(z_A_hyp, origin_A, c=1.0)
            hyp_radii_B = poincare_distance(z_B_hyp, origin_B, c=1.0)

        # Compute 3-adic metadata
        metadata = []
        metadata_header = [
            "index",
            "prefix_1",
            "prefix_2",
            "prefix_3",
            "tree_depth",
            "radius_A",
            "radius_B",
        ]

        for idx, op_idx in enumerate(indices):
            prefix_1 = op_idx % 3
            prefix_2 = op_idx % 9
            prefix_3 = op_idx % 27
            depth = self._compute_3adic_depth(op_idx)
            r_A = hyp_radii_A[idx].item()
            r_B = hyp_radii_B[idx].item()

            metadata.append([
                str(op_idx),
                str(prefix_1),
                str(prefix_2),
                str(prefix_3),
                str(depth),
                f"{r_A:.3f}",
                f"{r_B:.3f}",
            ])

        # Log embeddings (mu = tangent space means, hyp = Poincare manifold points)
        self.writer.add_embedding(
            mu_A.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_A_TangentSpace",
        )
        self.writer.add_embedding(
            z_A_hyp.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_A_Poincare",
        )
        self.writer.add_embedding(
            mu_B.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_B_TangentSpace",
        )
        self.writer.add_embedding(
            z_B_hyp.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_B_Poincare",
        )

        self.writer.flush()
        self.log_callback(
            f"Logged {len(indices)} embeddings to TensorBoard (epoch {epoch})"
        )

    def _compute_3adic_depth(self, n: int) -> int:
        """Compute 3-adic valuation (tree depth) of integer n.

        Returns the largest k such that 3^k divides n.
        For n=0, returns 9 (maximum depth for 3^9 space).

        Args:
            n: Integer index

        Returns:
            3-adic valuation (depth in tree)
        """
        # Delegate to canonical implementation in TERNARY singleton
        return TERNARY.valuation(torch.tensor([n])).item()

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
