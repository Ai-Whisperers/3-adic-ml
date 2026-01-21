# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Coverage evaluation for VAE models.

This module handles operation coverage evaluation:
- Sampling from VAE models
- Counting unique operations generated
- Computing coverage percentages

Single responsibility: Model coverage evaluation only.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from src.config.constants import N_TERNARY_OPERATIONS


def evaluate_coverage(
    model: torch.nn.Module,
    num_samples: int,
    device: str,
    vae: str = "A",
    batch_size: int = 1000,
) -> Tuple[int, float]:
    """Evaluate operation coverage for a VAE.

    Vectorized implementation using torch.unique.
    Reduces 500+ GPU syncs to 1 sync per batch.

    Args:
        model: Model to evaluate
        num_samples: Number of samples to generate
        device: Device to run on
        vae: Which VAE to evaluate ('A' or 'B')
        batch_size: Samples per batch

    Returns:
        Tuple of (unique_count, coverage_percentage)
    """
    model.eval()

    with torch.no_grad():
        num_batches = max(1, num_samples // batch_size)

        # Collect all samples first (on GPU)
        all_samples_list: List[torch.Tensor] = []
        for _ in range(num_batches):
            samples = model.sample(batch_size, device, vae)
            samples_rounded = torch.round(samples).long()
            all_samples_list.append(samples_rounded)

        # Concatenate and find unique (vectorized, single GPU->CPU transfer)
        all_samples = torch.cat(all_samples_list, dim=0)
        unique_samples = torch.unique(all_samples, dim=0)
        unique_count = unique_samples.size(0)

    coverage_pct = (unique_count / N_TERNARY_OPERATIONS) * 100
    return unique_count, coverage_pct


class CoverageEvaluator:
    """Evaluator for model operation coverage.

    Provides cached and efficient coverage evaluation
    with configurable sample sizes.

    Attributes:
        num_samples: Default number of samples for evaluation
        batch_size: Samples per batch
    """

    def __init__(
        self,
        num_samples: int = 100000,
        batch_size: int = 1000,
    ):
        """Initialize coverage evaluator.

        Args:
            num_samples: Default number of samples for evaluation
            batch_size: Samples per batch
        """
        self.num_samples = num_samples
        self.batch_size = batch_size

    def evaluate(
        self,
        model: torch.nn.Module,
        device: str,
        vae: str = "A",
        num_samples: int | None = None,
    ) -> Tuple[int, float]:
        """Evaluate coverage for a VAE.

        Args:
            model: Model to evaluate
            device: Device to run on
            vae: Which VAE to evaluate ('A' or 'B')
            num_samples: Override default sample count

        Returns:
            Tuple of (unique_count, coverage_percentage)
        """
        samples = num_samples or self.num_samples
        return evaluate_coverage(
            model=model,
            num_samples=samples,
            device=device,
            vae=vae,
            batch_size=self.batch_size,
        )

    def evaluate_both(
        self,
        model: torch.nn.Module,
        device: str,
        num_samples: int | None = None,
    ) -> Tuple[int, float, int, float]:
        """Evaluate coverage for both VAEs.

        Args:
            model: Model to evaluate
            device: Device to run on
            num_samples: Override default sample count

        Returns:
            Tuple of (unique_A, cov_A, unique_B, cov_B)
        """
        unique_A, cov_A = self.evaluate(
            model, device, "A", num_samples
        )
        unique_B, cov_B = self.evaluate(
            model, device, "B", num_samples
        )
        return unique_A, cov_A, unique_B, cov_B


__all__ = ["evaluate_coverage", "CoverageEvaluator"]
