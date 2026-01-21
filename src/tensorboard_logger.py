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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from src.data.generation import generate_all_ternary_operations
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

    def log_hyperbolic_batch(
        self,
        global_step: int,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
    ) -> None:
        """Log v5.10 hyperbolic metrics at batch level.

        Args:
            global_step: Global batch step
            ranking_loss: Hyperbolic ranking loss
            radial_loss: Radial hierarchy loss
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
        """
        if self.writer is None:
            return

        self.writer.add_scalar("Batch/HypRankingLoss", ranking_loss, global_step)
        self.writer.add_scalar("Batch/RadialLoss", radial_loss, global_step)
        self.writer.add_scalar("Batch/HypKL_A", hyp_kl_A, global_step)
        self.writer.add_scalar("Batch/HypKL_B", hyp_kl_B, global_step)
        self.writer.add_scalar("Batch/CentroidLoss", centroid_loss, global_step)

    def log_hyperbolic_epoch(
        self,
        epoch: int,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        homeostatic_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log v5.10 hyperbolic metrics at epoch level.

        Args:
            epoch: Current epoch
            corr_A_hyp: VAE-A hyperbolic correlation
            corr_B_hyp: VAE-B hyperbolic correlation
            corr_A_euc: VAE-A Euclidean correlation
            corr_B_euc: VAE-B Euclidean correlation
            mean_radius_A: VAE-A mean latent radius
            mean_radius_B: VAE-B mean latent radius
            ranking_weight: Current ranking loss weight
            ranking_loss: Hyperbolic ranking loss
            radial_loss: Radial hierarchy loss
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
            homeostatic_metrics: Dict of homeostatic adaptation metrics
        """
        if self.writer is None:
            return

        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        self.writer.add_scalars(
            "Hyperbolic/Correlation_Hyp",
            {
                "VAE_A": corr_A_hyp,
                "VAE_B": corr_B_hyp,
                "Mean": corr_mean_hyp,
            },
            epoch,
        )

        self.writer.add_scalars(
            "Hyperbolic/Correlation_Euc",
            {
                "VAE_A": corr_A_euc,
                "VAE_B": corr_B_euc,
                "Mean": corr_mean_euc,
            },
            epoch,
        )

        self.writer.add_scalars(
            "Hyperbolic/MeanRadius",
            {"VAE_A": mean_radius_A, "VAE_B": mean_radius_B},
            epoch,
        )

        self.writer.add_scalar("Hyperbolic/RankingWeight", ranking_weight, epoch)
        self.writer.add_scalar("Hyperbolic/RankingLoss", ranking_loss, epoch)
        self.writer.add_scalar("Hyperbolic/RadialLoss", radial_loss, epoch)

        # v5.10 specific
        self.writer.add_scalars(
            "v5.10/HyperbolicKL",
            {"VAE_A": hyp_kl_A, "VAE_B": hyp_kl_B},
            epoch,
        )
        self.writer.add_scalar("v5.10/CentroidLoss", centroid_loss, epoch)

        # Homeostatic metrics
        if homeostatic_metrics:
            if "prior_sigma_A" in homeostatic_metrics:
                self.writer.add_scalars(
                    "v5.10/HomeostaticSigma",
                    {
                        "VAE_A": homeostatic_metrics.get("prior_sigma_A", 1.0),
                        "VAE_B": homeostatic_metrics.get("prior_sigma_B", 1.0),
                    },
                    epoch,
                )
            if "prior_curvature_A" in homeostatic_metrics:
                self.writer.add_scalars(
                    "v5.10/HomeostaticCurvature",
                    {
                        "VAE_A": homeostatic_metrics.get("prior_curvature_A", 2.0),
                        "VAE_B": homeostatic_metrics.get("prior_curvature_B", 2.0),
                    },
                    epoch,
                )

    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, Any],
        val_losses: Dict[str, Any],
        unique_A: int,
        unique_B: int,
        cov_A: float,
        cov_B: float,
    ) -> None:
        """Log epoch-level metrics.

        Args:
            epoch: Current epoch
            train_losses: Training losses dict
            val_losses: Validation losses dict
            unique_A: VAE-A unique operations
            unique_B: VAE-B unique operations
            cov_A: VAE-A coverage percentage
            cov_B: VAE-B coverage percentage
        """
        if self.writer is None:
            return

        # Primary losses
        self.writer.add_scalars(
            "Loss/Total",
            {"train": train_losses["loss"], "val": val_losses["loss"]},
            epoch,
        )

        # VAE-A metrics
        self.writer.add_scalar("VAE_A/CrossEntropy", train_losses["ce_A"], epoch)
        self.writer.add_scalar("VAE_A/KL_Divergence", train_losses["kl_A"], epoch)
        self.writer.add_scalar("VAE_A/Entropy", train_losses["H_A"], epoch)
        self.writer.add_scalar("VAE_A/Coverage_Count", unique_A, epoch)
        self.writer.add_scalar("VAE_A/Coverage_Pct", cov_A, epoch)

        # VAE-B metrics
        self.writer.add_scalar("VAE_B/CrossEntropy", train_losses["ce_B"], epoch)
        self.writer.add_scalar("VAE_B/KL_Divergence", train_losses["kl_B"], epoch)
        self.writer.add_scalar("VAE_B/Entropy", train_losses["H_B"], epoch)
        self.writer.add_scalar("VAE_B/Coverage_Count", unique_B, epoch)
        self.writer.add_scalar("VAE_B/Coverage_Pct", cov_B, epoch)

        # Comparative metrics
        self.writer.add_scalars(
            "Compare/Entropy",
            {"VAE_A": train_losses["H_A"], "VAE_B": train_losses["H_B"]},
            epoch,
        )
        self.writer.add_scalars(
            "Compare/Coverage",
            {"VAE_A": cov_A, "VAE_B": cov_B},
            epoch,
        )

        # Training dynamics
        self.writer.add_scalar("Dynamics/Phase", train_losses["phase"], epoch)
        self.writer.add_scalar("Dynamics/Rho", train_losses["rho"], epoch)
        self.writer.add_scalar("Dynamics/GradRatio", train_losses["grad_ratio"], epoch)
        self.writer.add_scalar(
            "Dynamics/EMA_Momentum", train_losses["ema_momentum"], epoch
        )

        # Lambda weights
        self.writer.add_scalars(
            "Lambdas",
            {
                "lambda1": train_losses["lambda1"],
                "lambda2": train_losses["lambda2"],
                "lambda3": train_losses["lambda3"],
            },
            epoch,
        )

        # Temperature scheduling
        self.writer.add_scalars(
            "Temperature",
            {"VAE_A": train_losses["temp_A"], "VAE_B": train_losses["temp_B"]},
            epoch,
        )

        # Beta scheduling
        self.writer.add_scalars(
            "Beta",
            {"VAE_A": train_losses["beta_A"], "VAE_B": train_losses["beta_B"]},
            epoch,
        )

        # Learning rate
        self.writer.add_scalar("LR/Scheduled", train_losses["lr_scheduled"], epoch)
        if "lr_corrected" in train_losses:
            self.writer.add_scalar("LR/Corrected", train_losses["lr_corrected"], epoch)
            self.writer.add_scalar("LR/Delta", train_losses.get("delta_lr", 0), epoch)

        # StateNet corrections
        if "delta_lambda1" in train_losses:
            self.writer.add_scalars(
                "StateNet/Deltas",
                {
                    "delta_lr": train_losses.get("delta_lr", 0),
                    "delta_lambda1": train_losses.get("delta_lambda1", 0),
                    "delta_lambda2": train_losses.get("delta_lambda2", 0),
                    "delta_lambda3": train_losses.get("delta_lambda3", 0),
                },
                epoch,
            )

        # p-Adic losses
        self._log_padic_losses(epoch, train_losses)

        # Single flush per epoch
        self.writer.flush()

    def _log_padic_losses(self, epoch: int, train_losses: Dict[str, Any]) -> None:
        """Log p-adic loss components.

        Args:
            epoch: Current epoch
            train_losses: Training losses dict
        """
        if self.writer is None:
            return

        has_padic = (
            train_losses.get("padic_metric_A", 0) > 0
            or train_losses.get("padic_ranking_A", 0) > 0
            or train_losses.get("padic_norm_A", 0) > 0
        )

        if not has_padic:
            return

        if train_losses.get("padic_metric_A", 0) > 0:
            self.writer.add_scalars(
                "PAdicLoss/Metric",
                {
                    "VAE_A": train_losses.get("padic_metric_A", 0),
                    "VAE_B": train_losses.get("padic_metric_B", 0),
                },
                epoch,
            )
        if train_losses.get("padic_ranking_A", 0) > 0:
            self.writer.add_scalars(
                "PAdicLoss/Ranking",
                {
                    "VAE_A": train_losses.get("padic_ranking_A", 0),
                    "VAE_B": train_losses.get("padic_ranking_B", 0),
                },
                epoch,
            )
        if train_losses.get("padic_norm_A", 0) > 0:
            self.writer.add_scalars(
                "PAdicLoss/Norm",
                {
                    "VAE_A": train_losses.get("padic_norm_A", 0),
                    "VAE_B": train_losses.get("padic_norm_B", 0),
                },
                epoch,
            )

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
        """
        if self.writer is None:
            return

        model.eval()

        # Generate operations
        all_operations = generate_all_ternary_operations()
        total_ops = len(all_operations)

        # Sample or use all
        if include_all or n_samples >= total_ops:
            indices = list(range(total_ops))
        else:
            indices = sorted(random.sample(range(total_ops), n_samples))

        operations = all_operations[np.array(indices)]
        x = torch.from_numpy(operations).float().to(device)

        with torch.no_grad():
            outputs = model(x, 1.0, 1.0, 0.5, 0.5)
            z_A = outputs["z_A"]
            z_B = outputs["z_B"]

            # Project to Poincare ball for visualization
            z_A_euc_norm = torch.norm(z_A, dim=1, keepdim=True)
            z_A_poincare = z_A / (1 + z_A_euc_norm) * 0.95

            z_B_euc_norm = torch.norm(z_B, dim=1, keepdim=True)
            z_B_poincare = z_B / (1 + z_B_euc_norm) * 0.95

            # V5.12.2: Compute hyperbolic radii for metadata
            origin_A = torch.zeros_like(z_A_poincare)
            origin_B = torch.zeros_like(z_B_poincare)
            hyp_radii_A = poincare_distance(z_A_poincare, origin_A, c=1.0)
            hyp_radii_B = poincare_distance(z_B_poincare, origin_B, c=1.0)

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

        # Log embeddings
        self.writer.add_embedding(
            z_A.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_A_Euclidean",
        )
        self.writer.add_embedding(
            z_A_poincare.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_A_Poincare",
        )
        self.writer.add_embedding(
            z_B.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag="Embedding/VAE_B_Euclidean",
        )
        self.writer.add_embedding(
            z_B_poincare.cpu(),
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
        if n == 0:
            return 9

        depth = 0
        while n % 3 == 0:
            depth += 1
            n //= 3
        return depth

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
