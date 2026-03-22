from .ternary import (
    # Core types
    TERNARY,
    TernarySpace,
    # Property accessors (Option B)
    digit_count,
    digit_sum,
    distance,
    distance_matrix,
    first_nonzero,
    from_ternary,
    last_nonzero,
    level_rank,
    parent,
    target_radius,
    to_ternary,
    # Basic operations
    valuation,
)

__all__ = [
    # Core types
    "TERNARY",
    "TernarySpace",
    # Basic operations
    "valuation",
    "distance",
    "distance_matrix",
    "to_ternary",
    "from_ternary",
    "target_radius",
    # Property accessors
    "digit_count",
    "digit_sum",
    "first_nonzero",
    "last_nonzero",
    "parent",
    "level_rank",
]
