"""Scatter utilities with optional torch_scatter acceleration.

All functions are differentiable (autograd-compatible) in both modes.
The fallback uses torch.zeros + scatter_add_ which supports autograd.

Usage:
    from src.utils.scatter_utils import level_scatter_mean, level_scatter_std, level_has_data
"""
import torch

try:
    from torch_scatter import scatter_mean as _ts_scatter_mean
    from torch_scatter import scatter_std as _ts_scatter_std
    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False


def level_scatter_mean(
    src: torch.Tensor,
    index: torch.LongTensor,
    dim_size: int = 10,
) -> torch.Tensor:
    """Per-level mean. Differentiable. Returns (dim_size,) tensor; 0.0 where count=0."""
    if _HAS_TORCH_SCATTER:
        return _ts_scatter_mean(src, index, dim=0, dim_size=dim_size)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    sums = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(src))
    sums.scatter_add_(0, index, src)
    return sums / counts.clamp(min=1.0)


def level_scatter_std(  # noqa: used for metrics only — NOT for differentiable loss terms (NaN grad at std=0)
    src: torch.Tensor,
    index: torch.LongTensor,
    dim_size: int = 10,
) -> torch.Tensor:
    """Per-level std (population). Differentiable. Returns (dim_size,); 0.0 where count=0."""
    if _HAS_TORCH_SCATTER:
        return _ts_scatter_std(src, index, dim=0, dim_size=dim_size)
    # Unbiased (N-1) variance to match torch_scatter.scatter_std
    means = level_scatter_mean(src, index, dim_size)
    deviations = src - means[index]
    sum_sq = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    sum_sq.scatter_add_(0, index, deviations ** 2)
    counts.scatter_add_(0, index, torch.ones_like(src))
    variance = sum_sq / (counts - 1.0).clamp(min=1.0)
    return variance.clamp(min=0.0).sqrt()


def level_has_data(index: torch.LongTensor, dim_size: int = 10) -> torch.BoolTensor:
    """(dim_size,) bool: which levels have at least 1 sample."""
    counts = torch.zeros(dim_size, dtype=torch.long, device=index.device)
    counts.scatter_add_(0, index, torch.ones_like(index))
    return counts > 0
