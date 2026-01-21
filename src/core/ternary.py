# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Core ternary algebra module - Single Source of Truth.

This module owns ALL ternary-related computation for the entire codebase.
No other module should implement valuation, distance, or index conversion.

Architecture:
    TernarySpace is a singleton that precomputes and caches all ternary
    algebra operations. All other modules import from here.

Usage:
    from src.core.ternary import TERNARY

    # Valuation (3-adic)
    v = TERNARY.valuation(indices)  # O(1) lookup

    # Distance (3-adic metric)
    d = TERNARY.distance(i, j)  # O(1) lookup

    # Conversion
    ternary = TERNARY.to_ternary(indices)  # O(1) lookup
    indices = TERNARY.from_ternary(ternary)  # O(n) vectorized

Why this matters:
    Before: 4+ implementations of valuation with 9-iteration loops each
    After: 1 implementation, precomputed LUT, O(1) lookups

    Before: Scattered constants (19683, 3^9, etc.)
    After: Single source of truth (TERNARY.N_OPERATIONS)

    Before: Each module re-implements ternary <-> index conversion
    After: Canonical implementation in one place
"""

from typing import Optional

import torch


class TernarySpace:
    """Singleton managing all ternary algebra operations.

    This class precomputes and caches:
    - 3-adic valuations for all 19,683 indices
    - Ternary representations for all indices
    - Base-3 weights for fast conversion

    All operations are O(1) lookups after initialization.
    """

    # Class constants - canonical values
    N_DIGITS = 9  # 9 trits per operation
    N_OPERATIONS = 19683  # 3^9 = 19,683 total operations
    MAX_VALUATION = 9  # Maximum 3-adic valuation
    TERNARY_VALUES = (-1, 0, 1)  # Valid trit values

    def __init__(self):
        """Initialize precomputed lookup tables."""
        self._device = "cpu"

        # Precompute valuation LUT: index -> v_3(index)
        # Memory: 19,683 * 8 bytes = ~157 KB
        self._valuation_lut = self._build_valuation_lut()

        # Precompute ternary LUT: index -> (d0, d1, ..., d8)
        # Memory: 19,683 * 9 * 4 bytes = ~708 KB
        self._ternary_lut = self._build_ternary_lut()

        # Base-3 weights for index computation: [1, 3, 9, 27, ...]
        self._base3_weights = torch.tensor([3**i for i in range(self.N_DIGITS)], dtype=torch.long)

        # Device-cached versions (populated on first use)
        self._device_cache = {}

    def _build_valuation_lut(self) -> torch.Tensor:
        """Build 3-adic valuation lookup table."""
        valuations = []
        for n in range(self.N_OPERATIONS):
            if n == 0:
                valuations.append(self.MAX_VALUATION)
            else:
                v = 0
                m = n
                while m % 3 == 0:
                    v += 1
                    m //= 3
                valuations.append(v)
        return torch.tensor(valuations, dtype=torch.long)

    def _build_ternary_lut(self) -> torch.Tensor:
        """Build index -> ternary representation lookup table."""
        ternary = []
        for n in range(self.N_OPERATIONS):
            digits = []
            m = n
            for _ in range(self.N_DIGITS):
                digits.append((m % 3) - 1)  # Convert 0,1,2 to -1,0,1
                m //= 3
            ternary.append(digits)
        return torch.tensor(ternary, dtype=torch.float32)

    def _get_cached_lut(self, name: str, lut: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Get device-cached version of a LUT."""
        device_str = str(device)
        cache_key = f"{name}_{device_str}"

        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = lut.to(device)

        return self._device_cache[cache_key]

    # =========================================================================
    # Core Operations - All O(1) lookups
    # =========================================================================

    def valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute 3-adic valuation for indices.

        v_3(n) = max k such that 3^k divides n
        v_3(0) = MAX_VALUATION (infinity in theory)

        Args:
            indices: Tensor of indices in [0, N_OPERATIONS-1], any shape

        Returns:
            Tensor of valuations (long), same shape as input
        """
        device = indices.device
        lut = self._get_cached_lut("valuation", self._valuation_lut, device)

        # Clamp to valid range
        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        return lut[indices]

    def valuation_of_difference(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        """Compute v_3(|i - j|) for pairs of indices.

        This is the key operation for 3-adic distance computation.

        Args:
            idx_i, idx_j: Tensors of indices, same shape

        Returns:
            Tensor of valuations, same shape as input
        """
        diff = torch.abs(idx_i.long() - idx_j.long())
        diff = torch.clamp(diff, 0, self.N_OPERATIONS - 1)
        return self.valuation(diff)

    def distance(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        """Compute 3-adic distance between pairs of indices.

        d_3(i, j) = 3^(-v_3(|i - j|))
        d_3(i, i) = 0

        Args:
            idx_i, idx_j: Tensors of indices, same shape

        Returns:
            Tensor of distances in (0, 1], same shape as input
        """
        # Handle identical indices (zero distance)
        zero_mask = idx_i == idx_j

        # Compute valuation of difference
        v = self.valuation_of_difference(idx_i, idx_j)

        # Convert to distance: d = 3^(-v)
        distances = torch.pow(3.0, -v.float())

        # Set distance to 0 for identical indices
        distances[zero_mask] = 0.0

        return distances

    def to_ternary(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to ternary representation.

        Args:
            indices: Tensor of indices in [0, N_OPERATIONS-1], shape (N,)

        Returns:
            Tensor of ternary representations, shape (N, 9)
            Values in {-1, 0, 1}
        """
        device = indices.device
        lut = self._get_cached_lut("ternary", self._ternary_lut, device)

        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        return lut[indices]

    def from_ternary(self, ternary: torch.Tensor) -> torch.Tensor:
        """Convert ternary representation to indices.

        Args:
            ternary: Tensor of shape (..., 9) with values in {-1, 0, 1}

        Returns:
            Tensor of indices, shape (...)
        """
        device = ternary.device
        weights = self._get_cached_lut("weights", self._base3_weights, device)

        # Convert {-1, 0, 1} to {0, 1, 2}
        digits = (ternary + 1).long()

        # Compute index as base-3 number
        return (digits * weights).sum(dim=-1)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def is_valid_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Check if indices are valid operation indices."""
        return (indices >= 0) & (indices < self.N_OPERATIONS)

    def is_valid_ternary(self, ternary: torch.Tensor) -> torch.Tensor:
        """Check if ternary representation is valid."""
        return ((ternary == -1) | (ternary == 0) | (ternary == 1)).all(dim=-1)

    def sample_indices(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample random operation indices.

        Args:
            n: Number of indices to sample
            device: Device to create tensor on

        Returns:
            Tensor of random indices, shape (n,)
        """
        return torch.randint(0, self.N_OPERATIONS, (n,), device=device)

    def all_indices(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get tensor of all valid indices [0, 1, ..., N_OPERATIONS-1]."""
        return torch.arange(self.N_OPERATIONS, device=device)

    # =========================================================================
    # Batch Operations for GPU Efficiency
    # =========================================================================

    def all_ternary(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get all 19,683 ternary representations at once.

        Useful for GPU-resident dataset - load once, index thereafter.

        Args:
            device: Device to place tensor on

        Returns:
            Tensor of shape (19683, 9) with all ternary representations
        """
        if device is None:
            return self._ternary_lut.clone()
        return self._get_cached_lut("ternary", self._ternary_lut, device)

    def prefix(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """Compute tree prefix for given level (vectorized).

        In the 3-adic tree, nodes at level k share the same prefix.
        prefix(n, k) = n // 3^(9-k)

        This is used by HyperbolicCentroidLoss for tree structure.

        Args:
            indices: Operation indices, any shape
            level: Tree level (0 = root, 9 = leaves)

        Returns:
            Prefix indices, same shape as input
        """
        level = max(0, min(level, self.N_DIGITS))
        divisor = 3 ** (self.N_DIGITS - level)
        return indices.long() // divisor

    def level_mask(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """Get mask for indices at specific tree level.

        Args:
            indices: Operation indices
            level: Tree level

        Returns:
            Boolean mask where True = index is at this level
        """
        v = self.valuation(indices)
        return v == level

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def valuation_histogram(self, indices: torch.Tensor) -> dict:
        """Compute histogram of valuations in a set of indices.

        Args:
            indices: Tensor of indices

        Returns:
            Dict mapping valuation -> count
        """
        v = self.valuation(indices)
        hist = {}
        for val in range(self.MAX_VALUATION + 1):
            hist[val] = (v == val).sum().item()
        return hist

    def expected_valuation(self) -> float:
        """Compute expected valuation over uniform distribution.

        E[v_3(n)] for n ~ Uniform(1, N_OPERATIONS-1)
        """
        # Exclude 0 which has infinite valuation
        v = self._valuation_lut[1:].float()
        return v.mean().item()


# =============================================================================
# Singleton Instance
# =============================================================================

# Global singleton - the ONE source of truth for ternary algebra
TERNARY = TernarySpace()


# =============================================================================
# Module-level convenience functions (delegate to singleton)
# =============================================================================


def valuation(indices: torch.Tensor) -> torch.Tensor:
    """Compute 3-adic valuation. See TernarySpace.valuation."""
    return TERNARY.valuation(indices)


def distance(idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
    """Compute 3-adic distance. See TernarySpace.distance."""
    return TERNARY.distance(idx_i, idx_j)


def to_ternary(indices: torch.Tensor) -> torch.Tensor:
    """Convert to ternary. See TernarySpace.to_ternary."""
    return TERNARY.to_ternary(indices)


def from_ternary(ternary: torch.Tensor) -> torch.Tensor:
    """Convert from ternary. See TernarySpace.from_ternary."""
    return TERNARY.from_ternary(ternary)


__all__ = [
    "TernarySpace",
    "TERNARY",
    "valuation",
    "distance",
    "to_ternary",
    "from_ternary",
]
