# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

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

    # Algebraic property indices — binary-operation interpretation of ternary digits.
    # Each operation is f: {-1,0,1}² → {-1,0,1} with digit[k] = f(a,b),
    # a=(k//3)-1, b=(k%3)-1 (row-major 3×3 table).
    # These are precomputed from the ternary LUT; O(1) lookup after init.
    PROP_ALG_COMMUTATIVE  = 0   # f(a,b) = f(b,a) for all a,b
    PROP_ALG_IDEMPOTENT   = 1   # f(a,a) = a for all a ∈ {-1,0,1}
    PROP_ALG_HAS_IDENTITY = 2   # ∃ e ∈ {-1,0,1}: f(e,a)=f(a,e)=a for all a
    PROP_ALG_HAS_ABSORBING = 3  # ∃ z ∈ {-1,0,1}: f(z,a)=f(a,z)=z for all a
    N_ALG_PROPERTIES = 4

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

        # Precompute algebraic properties LUT (binary-operation interpretation)
        # Memory: 19,683 × 4 × 1 byte = ~79 KB (bool)
        self._algebraic_lut = self._build_algebraic_lut()

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

    def _build_algebraic_lut(self) -> torch.Tensor:
        """Build algebraic property LUT for binary-operation interpretation.

        Interprets each operation as f: {-1,0,1}² → {-1,0,1} via its 9-digit table:
            digit[k] = f(a, b),  a = (k // 3) - 1,  b = (k % 3) - 1

        Anti-symmetric pairs (commutativity check): (1,3), (2,6), (5,7)
        Diagonal positions (idempotency check):     0, 4, 8  → values -1, 0, 1

        Identity element e (row+col conditions):
          e=-1: d[0]==-1, d[1]==0, d[2]==1 AND d[3]==0, d[6]==1
          e= 0: d[3]==-1, d[4]==0, d[5]==1 AND d[1]==-1, d[7]==1
          e= 1: d[6]==-1, d[7]==0, d[8]==1 AND d[2]==-1, d[5]==0

        Absorbing element z:
          z=-1: d[0,1,2]==-1 AND d[3,6]==-1
          z= 0: d[3,4,5]==0  AND d[1,7]==0
          z= 1: d[6,7,8]==1  AND d[2,5]==1

        Returns:
            Bool tensor of shape (N_OPERATIONS, N_ALG_PROPERTIES)
        """
        import numpy as np
        ops = self._ternary_lut.numpy()  # (19683, 9) float64, values in {-1,0,1}
        N = self.N_OPERATIONS
        result = torch.zeros((N, self.N_ALG_PROPERTIES), dtype=torch.bool)

        # Commutative: 3 anti-symmetric pairs must match
        comm = (
            (ops[:, 1] == ops[:, 3]) &
            (ops[:, 2] == ops[:, 6]) &
            (ops[:, 5] == ops[:, 7])
        )
        result[:, self.PROP_ALG_COMMUTATIVE] = torch.from_numpy(comm)

        # Idempotent: all 3 diagonal entries equal their expected value
        idmpt = (
            (ops[:, 0] == -1.0) &
            (ops[:, 4] ==  0.0) &
            (ops[:, 8] ==  1.0)
        )
        result[:, self.PROP_ALG_IDEMPOTENT] = torch.from_numpy(idmpt)

        # Has identity element (any of e=-1, e=0, e=1)
        id_e_neg1 = (
            (ops[:, 0] == -1.0) & (ops[:, 1] ==  0.0) & (ops[:, 2] ==  1.0) &
            (ops[:, 3] ==  0.0) & (ops[:, 6] ==  1.0)
        )
        id_e_zero = (
            (ops[:, 3] == -1.0) & (ops[:, 4] ==  0.0) & (ops[:, 5] ==  1.0) &
            (ops[:, 1] == -1.0) & (ops[:, 7] ==  1.0)
        )
        id_e_pos1 = (
            (ops[:, 6] == -1.0) & (ops[:, 7] ==  0.0) & (ops[:, 8] ==  1.0) &
            (ops[:, 2] == -1.0) & (ops[:, 5] ==  0.0)
        )
        has_id = id_e_neg1 | id_e_zero | id_e_pos1
        result[:, self.PROP_ALG_HAS_IDENTITY] = torch.from_numpy(has_id)

        # Has absorbing element (any of z=-1, z=0, z=1)
        abs_z_neg1 = (
            (ops[:, 0] == -1.0) & (ops[:, 1] == -1.0) & (ops[:, 2] == -1.0) &
            (ops[:, 3] == -1.0) & (ops[:, 6] == -1.0)
        )
        abs_z_zero = (
            (ops[:, 3] ==  0.0) & (ops[:, 4] ==  0.0) & (ops[:, 5] ==  0.0) &
            (ops[:, 1] ==  0.0) & (ops[:, 7] ==  0.0)
        )
        abs_z_pos1 = (
            (ops[:, 6] ==  1.0) & (ops[:, 7] ==  1.0) & (ops[:, 8] ==  1.0) &
            (ops[:, 2] ==  1.0) & (ops[:, 5] ==  1.0)
        )
        has_abs = abs_z_neg1 | abs_z_zero | abs_z_pos1
        result[:, self.PROP_ALG_HAS_ABSORBING] = torch.from_numpy(has_abs)

        return result

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

    def ternary_add(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        """Perform 3-adic modular addition of two indices.
        
        Operation-wise addition: (a_i + b_i + 1) % 3 - 1
        Maps {-1, 0, 1} digits such that:
           1 + 1 = -1
          -1 - 1 = 1
           0 + x = x
        
        Args:
            idx_a: Tensor of indices, shape (N,)
            idx_b: Tensor of indices, shape (N,)
            
        Returns:
            Tensor of indices of the sums, shape (N,)
        """
        t_a = self.to_ternary(idx_a) # (N, 9)
        t_b = self.to_ternary(idx_b) # (N, 9)
        
        # Shift to {0, 1, 2} for standard modulo
        d_a = t_a + 1
        d_b = t_b + 1
        
        # Modular addition in {0, 1, 2}
        d_sum = (d_a + d_b) % 3
        
        # Shift back to {-1, 0, 1}
        t_sum = d_sum - 1
        
        return self.from_ternary(t_sum)

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

    def zero_count_valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Content-based hierarchy: number of zero digits in ternary representation.

        Maps sparse operations (many zeros) to high "valuation" levels, which
        the loss functions place near the Poincaré origin — the same convention
        as index-derived v_3(n).

        Distribution is near-binomial (peak ~27% at 3 zeros vs. 66% at v=0 for
        index-derived valuation), eliminating the Spearman tied-rank ceiling.

        Args:
            indices: Operation indices, any shape

        Returns:
            Zero counts in [0, 9] (= N_DIGITS − digit_count), same shape as indices
        """
        return self.N_DIGITS - self._get_property(indices, self.PROP_DIGIT_COUNT)

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

    # =========================================================================
    # Algebraic Pattern Classifiers (for direction geometry analysis)
    # =========================================================================

    def digit_prefix_class(self, indices: torch.Tensor, k: int = 2) -> torch.Tensor:
        """Classify operations by their first k digits interpreted as base-3.

        Returns class label in [0, 3^k). Within a valuation level, operations
        sharing a prefix class form natural sub-clusters in direction space.

        Args:
            indices: Operation indices, any shape
            k: Number of leading digits to use (default 2 → 9 classes)

        Returns:
            Class labels in [0, 3^k), same shape as indices
        """
        ops = self.to_ternary(indices)                          # (..., 9) float64
        ops_shifted = (ops[..., :k] + 1).long()                # (..., k) in {0,1,2}
        device = indices.device
        weights = torch.tensor(
            [3 ** i for i in range(k - 1, -1, -1)],
            dtype=torch.long, device=device,
        )
        return (ops_shifted * weights).sum(dim=-1)              # (...,) in [0, 3^k)

    def nonzero_pattern(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode which digit positions are non-zero as a 9-bit integer.

        Operations sharing nonzero_pattern have identical zero-structure,
        regardless of the sign of non-zero digits.

        Returns:
            9-bit codes in [0, 512), same shape as indices
        """
        ops = self.to_ternary(indices)                          # (..., 9)
        nonzero = (ops != 0).long()                            # (..., 9) binary
        device = indices.device
        weights = torch.tensor(
            [2 ** i for i in range(self.N_DIGITS)],
            dtype=torch.long, device=device,
        )
        return (nonzero * weights).sum(dim=-1)                  # (...,) in [0, 512)

    def valuation_prefix_class(self, indices: torch.Tensor) -> torch.Tensor:
        """Within-level sub-class: sign of first non-zero digit × value of next digit.

        For v=k operations, the first k digits are 0. The k-th digit has sign ±1
        and the (k+1)-th digit takes values {-1, 0, +1}. Together they give 6
        sub-classes per level that capture secondary p-adic tree branching.

        Returns:
            Sub-class labels in [0, 6), same shape as indices
        """
        ops = self.to_ternary(indices)                          # (..., 9) float64
        fz = self.first_nonzero(indices).clamp(0, 8)           # (...,)

        # Sign class: whether first non-zero digit is positive (0 or 1)
        sign_digit = torch.gather(
            ops.view(-1, self.N_DIGITS),
            1, fz.view(-1, 1),
        ).squeeze(1).view(indices.shape)
        sign_cls = (sign_digit > 0).long()                     # (...,) in {0,1}

        # Next digit value class: -1→0, 0→1, +1→2
        next_pos = (fz + 1).clamp(0, 8)
        next_digit = torch.gather(
            ops.view(-1, self.N_DIGITS),
            1, next_pos.view(-1, 1),
        ).squeeze(1).view(indices.shape)
        next_cls = (next_digit + 1).long()                     # (...,) in {0,1,2}

        return sign_cls * 3 + next_cls                         # (...,) in [0, 6)

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

    # =========================================================================
    # Algebraic Property Accessors — binary-operation analysis (v10)
    # =========================================================================

    def _get_alg_property(self, indices: torch.Tensor, prop_idx: int) -> torch.Tensor:
        """Internal helper to get a single algebraic property column."""
        device = indices.device
        lut = self._get_cached_lut("algebraic", self._algebraic_lut, device)
        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        return lut[indices, prop_idx]

    def is_commutative(self, indices: torch.Tensor) -> torch.Tensor:
        """Return bool tensor: True where f(a,b) = f(b,a) for all a,b.

        Args:
            indices: Operation indices, any shape

        Returns:
            Bool tensor, same shape as indices
        """
        return self._get_alg_property(indices, self.PROP_ALG_COMMUTATIVE)

    def is_idempotent(self, indices: torch.Tensor) -> torch.Tensor:
        """Return bool tensor: True where f(a,a) = a for all a ∈ {-1,0,1}.

        Args:
            indices: Operation indices, any shape

        Returns:
            Bool tensor, same shape as indices
        """
        return self._get_alg_property(indices, self.PROP_ALG_IDEMPOTENT)

    def has_identity_element(self, indices: torch.Tensor) -> torch.Tensor:
        """Return bool tensor: True where ∃ e s.t. f(e,a) = f(a,e) = a.

        Args:
            indices: Operation indices, any shape

        Returns:
            Bool tensor, same shape as indices
        """
        return self._get_alg_property(indices, self.PROP_ALG_HAS_IDENTITY)

    def has_absorbing_element(self, indices: torch.Tensor) -> torch.Tensor:
        """Return bool tensor: True where ∃ z s.t. f(z,a) = f(a,z) = z.

        Args:
            indices: Operation indices, any shape

        Returns:
            Bool tensor, same shape as indices
        """
        return self._get_alg_property(indices, self.PROP_ALG_HAS_ABSORBING)

    def algebraic_signature(self, indices: torch.Tensor) -> torch.Tensor:
        """3-bit algebraic signature packed as integer in [0, 7].

        Bit layout (MSB → LSB):
            bit 2: is_commutative
            bit 1: has_identity_element
            bit 0: has_absorbing_element

        Class 0 (000) is the bulk (~95% of ops, none of the special properties).
        Classes 1–7 are algebraically significant sub-populations.

        Args:
            indices: Operation indices, any shape

        Returns:
            Integer tensor in [0, 7], same shape as indices
        """
        device = indices.device
        lut = self._get_cached_lut("algebraic", self._algebraic_lut, device)
        indices = torch.clamp(indices.long(), 0, self.N_OPERATIONS - 1)
        props = lut[indices]  # (..., 4) bool
        return (
            props[..., self.PROP_ALG_COMMUTATIVE].long()  * 4 +
            props[..., self.PROP_ALG_HAS_IDENTITY].long() * 2 +
            props[..., self.PROP_ALG_HAS_ABSORBING].long()
        )


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


def digit_prefix_class(indices: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Classify by first k digits. See TernarySpace.digit_prefix_class."""
    return TERNARY.digit_prefix_class(indices, k)


def nonzero_pattern(indices: torch.Tensor) -> torch.Tensor:
    """9-bit nonzero structure encoding. See TernarySpace.nonzero_pattern."""
    return TERNARY.nonzero_pattern(indices)


def valuation_prefix_class(indices: torch.Tensor) -> torch.Tensor:
    """Within-level sub-class. See TernarySpace.valuation_prefix_class."""
    return TERNARY.valuation_prefix_class(indices)


def ternary_add(idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """Perform 3-adic modular addition. See TernarySpace.ternary_add."""
    return TERNARY.ternary_add(idx_a, idx_b)


def is_commutative(indices: torch.Tensor) -> torch.Tensor:
    """Check commutativity. See TernarySpace.is_commutative."""
    return TERNARY.is_commutative(indices)


def is_idempotent(indices: torch.Tensor) -> torch.Tensor:
    """Check idempotency. See TernarySpace.is_idempotent."""
    return TERNARY.is_idempotent(indices)


def has_identity_element(indices: torch.Tensor) -> torch.Tensor:
    """Check identity element existence. See TernarySpace.has_identity_element."""
    return TERNARY.has_identity_element(indices)


def has_absorbing_element(indices: torch.Tensor) -> torch.Tensor:
    """Check absorbing element existence. See TernarySpace.has_absorbing_element."""
    return TERNARY.has_absorbing_element(indices)


def algebraic_signature(indices: torch.Tensor) -> torch.Tensor:
    """3-bit algebraic signature in [0,7]. See TernarySpace.algebraic_signature."""
    return TERNARY.algebraic_signature(indices)


def get_valuation_fn(valuation_type: str):
    """Return the valuation callable for the given type.

    Args:
        valuation_type: "index" for 3-adic v_3(n), or "digit_count" for
            zero_count_valuation (number of zero digits — content-based hierarchy).

    Returns:
        Callable(indices: Tensor) -> Tensor of long valuations in [0, MAX_VALUATION]

    Raises:
        ValueError: If valuation_type is not recognized.
    """
    if valuation_type == "index":
        return TERNARY.valuation
    if valuation_type == "digit_count":
        return TERNARY.zero_count_valuation
    raise ValueError(
        f"Unknown valuation_type={valuation_type!r}. Valid options: 'index', 'digit_count'."
    )


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
    "ternary_add",
    "target_radius",
    # Property accessors (Option B)
    "digit_count",
    "digit_sum",
    "first_nonzero",
    "last_nonzero",
    "parent",
    "level_rank",
    # Algebraic pattern classifiers
    "digit_prefix_class",
    "nonzero_pattern",
    "valuation_prefix_class",
    # Algebraic property accessors (v10)
    "is_commutative",
    "is_idempotent",
    "has_identity_element",
    "has_absorbing_element",
    "algebraic_signature",
    # Valuation dispatch
    "get_valuation_fn",
]
