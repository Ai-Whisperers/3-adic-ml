# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary operation data generation.

This module generates all possible ternary operations for the 3^9 space.

Single responsibility: Data generation only.
"""

from typing import List

import numpy as np


def generate_all_ternary_operations() -> np.ndarray:
    """Generate all 19,683 possible ternary operations.

    Each operation is a 9-element vector with values in {-1, 0, 1},
    representing all possible outputs for a ternary logic function
    with inputs from {-1, 0, 1}^2.

    Returns:
        Array of shape (19683, 9) with all ternary operations
    """
    operations = []
    for i in range(3**9):
        op = []
        num = i
        for _ in range(9):
            op.append(num % 3 - 1)  # Convert 0,1,2 to -1,0,1
            num //= 3
        operations.append(op)
    return np.array(operations, dtype=np.float32)


def count_ternary_operations() -> int:
    """Return the total number of ternary operations (3^9).

    Returns:
        19683
    """
    return 3**9


def generate_ternary_operation_by_index(index: int) -> List[int]:
    """Generate a specific ternary operation by its index.

    Args:
        index: Index in range [0, 19683)

    Returns:
        List of 9 values in {-1, 0, 1}

    Raises:
        ValueError: If index is out of range
    """
    if not 0 <= index < 3**9:
        raise ValueError(f"Index must be in range [0, {3**9}), got {index}")

    op = []
    num = index
    for _ in range(9):
        op.append(num % 3 - 1)
        num //= 3
    return op
