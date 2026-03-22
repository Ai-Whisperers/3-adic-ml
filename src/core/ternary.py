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

    # Structured properties (Option B)
    props = TERNARY.properties(indices)  # Returns dict of all properties
    dc = TERNARY.digit_count(indices)    # Number of non-zero digits
    parent = TERNARY.parent(indices)     # Parent in 3-adic tree

Why this matters:
    Before: 4+ implementations of valuation with 9-iteration loops each
    After: 1 implementation, precomputed LUT, O(1) lookups

    Before: Scattered constants (19683, 3^9, etc.)
    After: Single source of truth (TERNARY.N_OPERATIONS)

    Before: Each module re-implements ternary <-> index conversion
    After: Canonical implementation in one place

Future (Option C):
    Tree navigation with precomputed children, siblings, and subtree info.
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

    Precision:
        All floating-point operations use float64 for consistency with
        geoopt's hyperbolic geometry (boundary stability near radius=1).

    Thread Safety:
        The device cache is not thread-safe. In multi-threaded scenarios,
        concurrent first-access to a new device may cause redundant copies.
        This is benign (no corruption) but slightly wasteful.

    Memory Usage (per device):
        - Valuation LUT: 19,683 × 8 bytes = 157 KB (torch.long)
        - Ternary LUT: 19,683 × 9 × 8 bytes = 1.4 MB (float64)
        - Base-3 weights: 9 × 8 bytes = 72 bytes
        - Total: ~1.6 MB per device
    """

    # Class constants - canonical values
    N_DIGITS = 9  # 9 trits per operation
    N_OPERATIONS = 19683  # 3^9 = 19,683 total operations
    MAX_VALUATION = 9  # Maximum 3-adic valuation
    TERNARY_VALUES = (-1, 0, 1)  # Valid trit values

    # Property indices for structured LUT (Option B)
    # Each property is a column in the properties tensor
    PROP_VALUATION = 0       # 3-adic valuation v_3(n)
    PROP_DIGIT_COUNT = 1     # Number of non-zero digits
    PROP_DIGIT_SUM = 2       # Sum of digits (shifted: actual + 9 to be non-negative)
    PROP_FIRST_NONZERO = 3   # Position of first non-zero digit (9 if all zero)
    PROP_LAST_NONZERO = 4    # Position of last non-zero digit (-1 if all zero)
    PROP_PARENT = 5          # Parent index in 3-adic tree (n // 3)
    PROP_LEVEL_RANK = 6      # Rank within same-valuation cohort
    N_PROPERTIES = 7         # Total number of properties

    # Property names for introspection (tuple: immutable, avoids RUF012 mutable default)
    PROPERTY_NAMES = (
        'valuation', 'digit_count', 'digit_sum', 'first_nonzero',
        'last_nonzero', 'parent', 'level_rank'
    )

    def __init__(self):
        """Initialize precomputed lookup tables."""
        # Precompute valuation LUT: index -> v_3(index)
        # Memory: 19,683 * 8 bytes = ~157 KB
        self._valuation_lut = self._build_valuation_lut()

        # Precompute ternary LUT: index -> (d0, d1, ..., d8)
        # Memory: 19,683 * 9 * 8 bytes = ~1.4 MB (float64 for geoopt compatibility)
        self._ternary_lut = self._build_ternary_lut()

        # Base-3 weights for index computation: [1, 3, 9, 27, ...]
        self._base3_weights = torch.tensor([3**i for i in range(self.N_DIGITS)], dtype=torch.long)

        # Precompute structured properties LUT (Option B)
        # Memory: 19,683 * 7 * 8 bytes = ~1.1 MB
        self._properties_lut = self._build_properties_lut()

        # Precompute level population counts (how many indices at each valuation)
        self._level_counts = self._compute_level_counts()

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
        return torch.tensor(ternary, dtype=torch.float64)

    def _build_properties_lut(self) -> torch.Tensor:
        """Build structured properties lookup table (Option B).

        Each row contains N_PROPERTIES values for one index.
        Properties are indexed by PROP_* constants.

        Returns:
            Tensor of shape (N_OPERATIONS, N_PROPERTIES), dtype=long
        """
        props = torch.zeros((self.N_OPERATIONS, self.N_PROPERTIES), dtype=torch.long)

        # Track rank within each valuation level
        level_counters = [0] * (self.MAX_VALUATION + 1)

        for n in range(self.N_OPERATIONS):
            # Get ternary representation
            digits = []
            m = n
            for _ in range(self.N_DIGITS):
                digits.append((m % 3) - 1)
                m //= 3

            # Valuation (already computed, but recompute for consistency)
            v = self._valuation_lut[n].item()
            props[n, self.PROP_VALUATION] = v

            # Digit count (number of non-zero digits)
            digit_count = sum(1 for d in digits if d != 0)
            props[n, self.PROP_DIGIT_COUNT] = digit_count

            # Digit sum (shifted by 9 to ensure non-negative: range -9 to +9 -> 0 to 18)
            digit_sum = sum(digits) + self.N_DIGITS
            props[n, self.PROP_DIGIT_SUM] = digit_sum

            # First non-zero digit position (from LSB, 9 if all zero)
            first_nz = self.N_DIGITS
            for i, d in enumerate(digits):
                if d != 0:
                    first_nz = i
                    break
            props[n, self.PROP_FIRST_NONZERO] = first_nz

            # Last non-zero digit position (from LSB, -1 if all zero)
            last_nz = -1
            for i in range(self.N_DIGITS - 1, -1, -1):
                if digits[i] != 0:
                    last_nz = i
                    break
            props[n, self.PROP_LAST_NONZERO] = last_nz

            # Parent in 3-adic tree (n // 3, or -1 for n=0)
            parent = n // 3 if n > 0 else -1
            props[n, self.PROP_PARENT] = parent

            # Level rank (position within same-valuation cohort)
            props[n, self.PROP_LEVEL_RANK] = level_counters[int(v)]
            level_counters[int(v)] += 1

        return props

    def _compute_level_counts(self) -> torch.Tensor:
        """Compute population count for each valuation level.

        Returns:
            Tensor of shape (MAX_VALUATION + 1,) with count at each level
        """
        counts = torch.zeros(self.MAX_VALUATION + 1, dtype=torch.long)
        for v in range(self.MAX_VALUATION + 1):
            counts[v] = (self._valuation_lut == v).sum()
        return counts

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

    def valuation(self, indices: torch.Tensor, strict: bool = False) -> torch.Tensor:
        """Compute 3-adic valuation for indices.

        v_3(n) = max k such that 3^k divides n
        v_3(0) = MAX_VALUATION (infinity in theory)

        Args:
            indices: Tensor of indices in [0, N_OPERATIONS-1], any shape
            strict: If True, raise ValueError for out-of-range indices.
                    If False (default), clamp to valid range silently.

        Returns:
            Tensor of valuations (long), same shape as input

        Raises:
            ValueError: If strict=True and any index is outside [0, N_OPERATIONS-1]

        Note:
            By default, indices outside [0, N_OPERATIONS-1] are clamped to valid range.
            Use strict=True during debugging to catch data loading bugs early.
        """
        device = indices.device
        lut = self._get_cached_lut("valuation", self._valuation_lut, device)

        if strict:
            invalid = (indices < 0) | (indices >= self.N_OPERATIONS)
            if invalid.any():
                invalid_vals = indices[invalid]
                raise ValueError(
                    f"Invalid indices (outside [0, {self.N_OPERATIONS - 1}]): "
                    f"{invalid_vals[:5].tolist()}{'...' if len(invalid_vals) > 5 else ''}"
                )

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
        distances = torch.pow(3.0, -v.double())

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

    def sample_indices(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample random operation indices.

        Args:
            n: Number of indices to sample
            device: Device to create tensor on
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            Tensor of random indices, shape (n,)
        """
        return torch.randint(0, self.N_OPERATIONS, (n,), device=device, generator=generator)

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
            Tensor of shape (19683, 9) with all ternary representations.
            Always returns a clone to prevent accidental mutation of cached data.
        """
        if device is None:
            return self._ternary_lut.clone()
        # Return clone of cached tensor to prevent mutation
        return self._get_cached_lut("ternary", self._ternary_lut, device).clone()

    def all_valuations(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get valuations for all 19,683 indices at once.

        Useful for batch analysis and loss computation.

        Args:
            device: Device to place tensor on

        Returns:
            Tensor of shape (19683,) with valuations in [0, 9].
            Always returns a clone to prevent accidental mutation.
        """
        if device is None:
            return self._valuation_lut.clone()
        return self._get_cached_lut("valuation", self._valuation_lut, device).clone()

    def distance_matrix(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute pairwise 3-adic distance matrix.

        d_3(i, j) = 3^(-v_3(|i - j|)) for all pairs.

        Args:
            indices: Tensor of shape (N,) with operation indices

        Returns:
            Distance matrix of shape (N, N) where D[i,j] = d_3(indices[i], indices[j])
        """
        n = indices.shape[0]
        # Expand for pairwise: (N, 1) vs (1, N) -> (N, N)
        i_idx = indices.unsqueeze(1).expand(n, n)
        j_idx = indices.unsqueeze(0).expand(n, n)
        return self.distance(i_idx, j_idx)

    def target_radius(
        self,
        indices: torch.Tensor,
        inner: float = 0.1,
        outer: float = 0.9,
    ) -> torch.Tensor:
        """Compute target Poincaré radius based on 3-adic valuation.

        Maps valuation to radius: high valuation → small radius (near origin),
        low valuation → large radius (near boundary).

        This is the canonical mapping for p-adic hierarchy in hyperbolic space.

        Args:
            indices: Operation indices, any shape
            inner: Target radius for v=9 (highest valuation, near origin)
            outer: Target radius for v=0 (lowest valuation, near boundary)

        Returns:
            Target radii, same shape as indices
        """
        v = self.valuation(indices).double()
        # Linear interpolation: v=0 → outer, v=MAX_VALUATION → inner
        t = v / self.MAX_VALUATION
        return outer * (1.0 - t) + inner * t

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
        v = self._valuation_lut[1:].double()
        return v.mean().item()

    # =========================================================================
    # Structured Property Accessors (Option B)
    # =========================================================================

    def _get_property(self, indices: torch.Tensor, prop_idx: int) -> torch.Tensor:
        """Internal helper to get a single property column."""
        device = indices.device
        lut = self._get_cached_lut("properties", self._properties_lut, device)
        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        return lut[indices, prop_idx]

    def digit_count(self, indices: torch.Tensor) -> torch.Tensor:
        """Get number of non-zero digits for each index.

        Args:
            indices: Operation indices, any shape

        Returns:
            Digit counts in [0, 9], same shape as indices
        """
        return self._get_property(indices, self.PROP_DIGIT_COUNT)

    def digit_sum(self, indices: torch.Tensor) -> torch.Tensor:
        """Get sum of digits for each index.

        The raw sum of {-1, 0, 1} digits, range [-9, +9].
        (Stored shifted by +9 internally for non-negative values.)

        Args:
            indices: Operation indices, any shape

        Returns:
            Digit sums in [-9, +9], same shape as indices
        """
        raw = self._get_property(indices, self.PROP_DIGIT_SUM)
        return raw - self.N_DIGITS  # Unshift: stored + 9, so subtract 9

    def first_nonzero(self, indices: torch.Tensor) -> torch.Tensor:
        """Get position of first non-zero digit (from LSB).

        Args:
            indices: Operation indices, any shape

        Returns:
            Positions in [0, 8], or 9 if all digits are zero
        """
        return self._get_property(indices, self.PROP_FIRST_NONZERO)

    def last_nonzero(self, indices: torch.Tensor) -> torch.Tensor:
        """Get position of last non-zero digit (from LSB).

        Args:
            indices: Operation indices, any shape

        Returns:
            Positions in [0, 8], or -1 if all digits are zero
        """
        return self._get_property(indices, self.PROP_LAST_NONZERO)

    def parent(self, indices: torch.Tensor) -> torch.Tensor:
        """Get parent index in 3-adic tree (n // 3).

        The 3-adic tree has root at 0, with each node n having
        children {3n, 3n+1, 3n+2} (when in range).

        Args:
            indices: Operation indices, any shape

        Returns:
            Parent indices, or -1 for index 0 (root)
        """
        return self._get_property(indices, self.PROP_PARENT)

    def level_rank(self, indices: torch.Tensor) -> torch.Tensor:
        """Get rank within same-valuation cohort.

        Indices with the same valuation form a cohort. This returns
        the position (0-indexed) within that cohort.

        Args:
            indices: Operation indices, any shape

        Returns:
            Ranks within valuation cohort, same shape as indices
        """
        return self._get_property(indices, self.PROP_LEVEL_RANK)

    def level_count(self, level: int) -> int:
        """Get number of indices at a specific valuation level.

        Args:
            level: Valuation level in [0, MAX_VALUATION]

        Returns:
            Count of indices with this valuation
        """
        if level < 0 or level > self.MAX_VALUATION:
            return 0
        return int(self._level_counts[level].item())

    def properties(self, indices: torch.Tensor) -> dict:
        """Get all properties for indices as a dictionary.

        Convenient for analysis and debugging.

        Args:
            indices: Operation indices, any shape

        Returns:
            Dict with keys: valuation, digit_count, digit_sum,
            first_nonzero, last_nonzero, parent, level_rank
        """
        device = indices.device
        lut = self._get_cached_lut("properties", self._properties_lut, device)
        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        props = lut[indices]  # Shape: (*indices.shape, N_PROPERTIES)

        return {
            'valuation': props[..., self.PROP_VALUATION],
            'digit_count': props[..., self.PROP_DIGIT_COUNT],
            'digit_sum': props[..., self.PROP_DIGIT_SUM] - self.N_DIGITS,
            'first_nonzero': props[..., self.PROP_FIRST_NONZERO],
            'last_nonzero': props[..., self.PROP_LAST_NONZERO],
            'parent': props[..., self.PROP_PARENT],
            'level_rank': props[..., self.PROP_LEVEL_RANK],
        }

    def all_properties(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get properties tensor for all indices.

        Useful for batch analysis.

        Args:
            device: Device to place tensor on

        Returns:
            Tensor of shape (N_OPERATIONS, N_PROPERTIES)
        """
        if device is None:
            return self._properties_lut.clone()
        return self._get_cached_lut("properties", self._properties_lut, device).clone()


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


def distance_matrix(indices: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrix. See TernarySpace.distance_matrix."""
    return TERNARY.distance_matrix(indices)


def target_radius(indices: torch.Tensor, inner: float = 0.1, outer: float = 0.9) -> torch.Tensor:
    """Compute target radii. See TernarySpace.target_radius."""
    return TERNARY.target_radius(indices, inner, outer)


# Property accessors (Option B)
def digit_count(indices: torch.Tensor) -> torch.Tensor:
    """Get number of non-zero digits. See TernarySpace.digit_count."""
    return TERNARY.digit_count(indices)


def digit_sum(indices: torch.Tensor) -> torch.Tensor:
    """Get sum of digits. See TernarySpace.digit_sum."""
    return TERNARY.digit_sum(indices)


def first_nonzero(indices: torch.Tensor) -> torch.Tensor:
    """Get position of first non-zero digit. See TernarySpace.first_nonzero."""
    return TERNARY.first_nonzero(indices)


def last_nonzero(indices: torch.Tensor) -> torch.Tensor:
    """Get position of last non-zero digit. See TernarySpace.last_nonzero."""
    return TERNARY.last_nonzero(indices)


def parent(indices: torch.Tensor) -> torch.Tensor:
    """Get parent index in 3-adic tree. See TernarySpace.parent."""
    return TERNARY.parent(indices)


def level_rank(indices: torch.Tensor) -> torch.Tensor:
    """Get rank within valuation cohort. See TernarySpace.level_rank."""
    return TERNARY.level_rank(indices)


__all__ = [
    # Core types
    "TernarySpace",
    "TERNARY",
    # Basic operations
    "valuation",
    "distance",
    "distance_matrix",
    "to_ternary",
    "from_ternary",
    "target_radius",
    # Property accessors (Option B)
    "digit_count",
    "digit_sum",
    "first_nonzero",
    "last_nonzero",
    "parent",
    "level_rank",
]
