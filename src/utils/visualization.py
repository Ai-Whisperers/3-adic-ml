# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Phase 2 Visualization Pipeline.

Computes and logs geometric structure of the learned hyperbolic latent space
using the *hyperbolic* (Poincaré) distance matrix as input — not raw Euclidean
coordinates. This is the correct metric for 3-adic ultrametric structure.

Pipeline per eval call:
  1. Stratified subsample (up to max_per_level points per valuation level)
  2. Compute pairwise hyperbolic distance matrix D_hyp (shared by all algorithms)
  3. UMAP(metric='precomputed') → 3D → TensorBoard add_embedding
  4. PaCMAP → 2D → TensorBoard add_figure + HTML
  5. TriMAP(use_dist_matrix=True) → 2D → TensorBoard add_figure + HTML
  6. Poincaré ball 3D scatter (logmap0 tangent space PCA) → add_figure + HTML
  7. [Every persist_every epochs] Persistent homology → Betti numbers + diagram

All heavy work runs under torch.no_grad() on CPU numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as SummaryWriterType

# Optional imports — all installed, but guard defensively
try:
    import umap as umap_lib
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    import pacmap as pacmap_lib
    _HAS_PACMAP = True
except ImportError:
    _HAS_PACMAP = False

try:
    import trimap as trimap_lib
    _HAS_TRIMAP = True
except ImportError:
    _HAS_TRIMAP = False

try:
    from ripser import ripser as _ripser
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from src.geometry import log_map_zero, poincare_distance_matrix
from src.utils.poincare_renderer import save_poincare_disk
from src.utils.svg_renderer import save_poincare_disk_svg

# Colourmap: 10 valuation levels → distinct colours
_LEVEL_CMAP = "plasma"
_LEVEL_COLORS_MPL = None  # Lazily initialised


@dataclass(frozen=True)
class VisualizationRuntimeConfig:
    """Validated runtime config for VisualizationPipeline.

    Keeps constructor validation local to the pipeline so callers get a small,
    direct error surface instead of failures later inside optional backends.
    """

    max_per_level: int = 500
    persist_every: int = 50
    html_dir: Path = Path("visualizations")
    save_html: bool = True
    umap_neighbors: int = 15
    curvature: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
    ) -> VisualizationRuntimeConfig:
        cfg = dict(config or {})
        runtime = cls(
            max_per_level=int(cfg.get("max_per_level", 500)),
            persist_every=int(cfg.get("persist_every", 50)),
            html_dir=Path(cfg.get("html_dir", "visualizations")),
            save_html=bool(cfg.get("save_html", True)),
            umap_neighbors=int(cfg.get("umap_neighbors", 15)),
            curvature=float(cfg.get("curvature", 1.0)),
        )
        if runtime.max_per_level < 1:
            raise ValueError("VisualizationPipeline: max_per_level must be >= 1")
        if runtime.persist_every < 1:
            raise ValueError("VisualizationPipeline: persist_every must be >= 1")
        if runtime.umap_neighbors < 2:
            raise ValueError("VisualizationPipeline: umap_neighbors must be >= 2")
        if runtime.curvature <= 0:
            raise ValueError("VisualizationPipeline: curvature must be > 0")
        return runtime


def _level_colors(n_levels: int = 10) -> list[str]:
    """Return hex colour strings for each valuation level."""
    global _LEVEL_COLORS_MPL
    if _LEVEL_COLORS_MPL is None and _HAS_MPL:
        # matplotlib 3.7+ uses matplotlib.colormaps; older uses plt.cm.get_cmap
        try:
            cmap = matplotlib.colormaps[_LEVEL_CMAP].resampled(n_levels)
        except AttributeError:
            cmap = plt.cm.get_cmap(_LEVEL_CMAP, n_levels)  # type: ignore[attr-defined]
        _LEVEL_COLORS_MPL = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b, _ in [cmap(i) for i in range(n_levels)]
        ]

    if _LEVEL_COLORS_MPL:
        return _LEVEL_COLORS_MPL

    # Fallback: High-quality hardcoded plasma-like hex colors for 10 levels
    # Generated from matplotlib.cm.plasma for 10 steps
    return [
        "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786",
        "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"
    ]


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def _stratified_subsample(
    z_hyp: torch.Tensor,
    valuations: torch.Tensor,
    indices: torch.Tensor,
    max_per_level: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_np, val_np, idx_np) arrays, subsampled stratified by valuation level.

    For N=19,683 full dataset: keeps ≤500 per level → ≤4,501 points total.
    The v=9 singleton (just index 0) is always included.

    Args:
        z_hyp: (N, D) Poincaré ball embeddings, any dtype
        valuations: (N,) integer valuation labels 0..9
        indices: (N,) original operation indices
        max_per_level: Maximum points per valuation level
        seed: RNG seed for reproducibility

    Returns:
        z_np: (M, D) float32 numpy array
        val_np: (M,) int32 numpy array
        idx_np: (M,) int32 numpy array
    """
    rng = np.random.default_rng(seed)
    z_np = z_hyp.detach().cpu().float().numpy()
    val_np = valuations.detach().cpu().numpy().astype(np.int32)
    idx_all_np = indices.detach().cpu().numpy().astype(np.int32)

    selected = []
    for v in range(10):
        idx = np.where(val_np == v)[0]
        if len(idx) == 0:
            continue
        n_take = min(len(idx), max_per_level)
        chosen = rng.choice(idx, size=n_take, replace=False)
        selected.append(chosen)

    all_idx = np.concatenate(selected)
    all_idx.sort()
    return z_np[all_idx], val_np[all_idx], idx_all_np[all_idx]


# ---------------------------------------------------------------------------
# Hyperbolic distance matrix
# ---------------------------------------------------------------------------

def compute_hyperbolic_distance_matrix(
    z_np: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    """Compute pairwise Poincaré distance matrix from numpy array.

    Uses the codebase's geoopt-backed poincare_distance_matrix for
    numerical stability near the ball boundary.

    Args:
        z_np: (N, D) float32 array of Poincaré ball points
        c: Curvature

    Returns:
        D: (N, N) float32 symmetric distance matrix
    """
    z_t = torch.from_numpy(z_np).double()
    with torch.no_grad():
        D_t = poincare_distance_matrix(z_t, c=c)
    D = D_t.cpu().float().numpy()
    # Symmetrize and zero diagonal for numerical cleanliness
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    # Clamp any tiny negatives from float32 rounding
    np.clip(D, 0.0, None, out=D)
    return D


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _run_umap(D: np.ndarray, n_components: int = 3, n_neighbors: int = 15) -> Optional[np.ndarray]:
    """UMAP on precomputed hyperbolic distance matrix.

    Uses metric='precomputed' so UMAP respects Poincaré distances, not
    Euclidean embedding coordinates.

    Critical: UMAP's fast_knn_indices() does row-wise argsort; with diagonal=0
    point i is always its own nearest neighbour (D[i,i]=0 is minimum). We set
    diagonal to np.inf on a copy so UMAP picks genuine neighbours — the original
    D is unchanged and still correct for ripser/TriMAP.

    Returns (N, n_components) float32 or None if unavailable/fails.
    """
    if not _HAS_UMAP:
        return None
    try:
        # Copy and push diagonal to a large finite value so self-loops rank last
        # in UMAP's row-wise argsort (np.inf fails float32 validation in UMAP).
        D_umap = D.copy()
        large_val = float(np.nanmax(D_umap[D_umap < np.inf])) * 10.0 + 1.0
        np.fill_diagonal(D_umap, large_val)
        reducer = umap_lib.UMAP(
            n_components=n_components,
            metric="precomputed",
            n_neighbors=min(n_neighbors, D_umap.shape[0] - 1),
            min_dist=0.1,
            random_state=42,
            low_memory=False,
            verbose=False,
        )
        return reducer.fit_transform(D_umap).astype(np.float32)
    except Exception:
        return None


def _run_pacmap(
    z_np: np.ndarray,
    D: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
) -> Optional[np.ndarray]:
    """PaCMAP with hyperbolic-aware kNN injection.

    PaCMAP has no 'metric=precomputed' mode — it always takes feature vectors.
    However, it accepts `pair_neighbors` (Nx k, 2) pre-built from any distance
    matrix, which we derive from D (the Poincaré distance matrix). This injects
    the hyperbolic kNN structure into PaCMAP's near-pair optimisation while
    PaCMAP handles mid-near and far-pair sampling on its own.

    The feature vectors z_np are still required for PaCMAP's internal
    optimisation (it learns an embedding starting from them). This gives a
    hybrid view: topology from hyperbolic kNN, embedding quality from PaCMAP.

    Args:
        z_np: (N, D) raw Poincaré embedding coordinates (feature vectors)
        D: (N, N) pairwise Poincaré distance matrix (used only for kNN)
        n_components: Output dimensionality
        n_neighbors: k for kNN extraction from D

    Returns (N, n_components) float32 or None if unavailable/fails.
    """
    if not _HAS_PACMAP:
        return None
    try:
        k = min(n_neighbors, z_np.shape[0] - 2)
        # Extract kNN from hyperbolic distance matrix (exclude self: push diagonal
        # to large finite so self ranks last in argsort — np.inf causes issues)
        D_no_self = D.copy()
        large_val = float(np.nanmax(D_no_self)) * 10.0 + 1.0
        np.fill_diagonal(D_no_self, large_val)
        knn_idx = np.argsort(D_no_self, axis=1)[:, :k]  # (N, k)
        n = z_np.shape[0]
        rows = np.repeat(np.arange(n), k)
        pair_neighbors = np.column_stack([rows, knn_idx.ravel()]).astype(np.int32)

        emb = pacmap_lib.PaCMAP(
            n_components=n_components,
            n_neighbors=k,
            MN_ratio=0.5,
            FP_ratio=2.0,
            pair_neighbors=pair_neighbors,
            random_state=42,
            verbose=False,
        )
        return emb.fit_transform(z_np).astype(np.float32)
    except Exception:
        return None


def _run_trimap(D: np.ndarray, n_components: int = 2) -> Optional[np.ndarray]:
    """TriMAP on precomputed distance matrix — preserves ordering of distances.

    use_dist_matrix=True tells TriMAP to treat input as a full NxN distance
    matrix rather than feature vectors. This is the correct mode for
    hyperbolic distances, since TriMAP's triplet constraints then reflect
    actual Poincaré ordering (v=0 farther than v=1 etc.).

    Returns (N, n_components) or None if unavailable/fails.
    """
    if not _HAS_TRIMAP:
        return None
    try:
        n = D.shape[0]
        # TriMAP uses n_extra = min(n_inliers + 50, n) for argpartition.
        # kth must be strictly < n; clamp n_inliers so n_extra <= n - 1.
        n_inliers = min(12, max(1, n - 51))
        emb = trimap_lib.TRIMAP(
            n_dims=n_components,
            n_inliers=n_inliers,
            use_dist_matrix=True,
            verbose=False,
        )
        return emb.fit_transform(D).astype(np.float32)
    except Exception:
        return None


def _poincare_tangent_pca(
    z_np: np.ndarray,
    n_components: int = 3,
    c: float = 1.0,
) -> Optional[np.ndarray]:
    """Project z_hyp → tangent space via logmap0, then PCA to n_components.

    logmap0 maps Poincaré ball points to the tangent space at the origin,
    which IS Euclidean ℝ^D. PCA in tangent space preserves the radial
    origin-distance relationship (radius is invariant to PCA direction).
    This is the geometrically correct way to get a low-dim Poincaré view
    compared to naive PCA on raw embedding coordinates.

    Returns (N, n_components) float32 or None.
    """
    try:
        z_t = torch.from_numpy(z_np).double()
        with torch.no_grad():
            v_t = log_map_zero(z_t, c=c)  # (N, D) tangent vectors
        v = v_t.float().numpy()

        # sklearn PCA on tangent space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(v).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------

def compute_persistence(
    D: np.ndarray,
    max_dimension: int = 1,
) -> dict:
    """Vietoris-Rips persistent homology on hyperbolic distance matrix.

    The 3-adic ultrametric has a known signature:
    - H0: should show a dendrogram pattern — 3^k components that merge
      at threshold corresponding to v_3=k. Clean hierarchy → the H0
      persistence diagram reflects the tree structure.
    - H1: trees have zero 1-cycles. Large H1 indicates non-ultrametric
      loops in the learned geometry (a failure mode).

    Uses ripser with distance_matrix=True for direct metric input.

    Args:
        D: (N, N) symmetric float32 pairwise distance matrix
        max_dimension: Maximum homology dimension (1 = H0 + H1)

    Returns:
        dict with keys:
            dgms: list of (n_pts, 2) birth/death arrays, one per dim
            betti_0: number of H0 features with infinite persistence
            betti_1: number of H1 features with infinite persistence
            persistence_entropy: Shannon entropy of H0 finite lifetimes
    """
    if not _HAS_RIPSER:
        return {}
    try:
        result = _ripser(D, maxdim=max_dimension, distance_matrix=True)
        dgms = result["dgms"]

        # H0 Betti number = number of connected components
        dgm0 = dgms[0]
        betti_0 = int(np.sum(dgm0[:, 1] == np.inf))

        # H1 Betti number
        betti_1 = 0
        if len(dgms) > 1:
            dgm1 = dgms[1]
            betti_1 = int(np.sum(dgm1[:, 1] == np.inf)) if len(dgm1) > 0 else 0

        # Persistence entropy of H0 (finite lifetimes only)
        finite_mask = dgm0[:, 1] < np.inf
        finite_lifetimes = dgm0[finite_mask, 1] - dgm0[finite_mask, 0]
        finite_lifetimes = finite_lifetimes[finite_lifetimes > 0]
        persistence_entropy = 0.0
        if len(finite_lifetimes) > 0:
            total = finite_lifetimes.sum()
            p = finite_lifetimes / (total + 1e-10)
            persistence_entropy = float(-(p * np.log(p + 1e-10)).sum())

        return {
            "dgms": dgms,
            "betti_0": betti_0,
            "betti_1": betti_1,
            "persistence_entropy": persistence_entropy,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Matplotlib figure helpers
# ---------------------------------------------------------------------------

def _scatter_2d_by_level(
    coords: np.ndarray,
    val_np: np.ndarray,
    title: str,
    n_levels: int = 10,
) -> plt.Figure:
    """2D scatter coloured by valuation level for TensorBoard add_figure."""
    colors = _level_colors(n_levels)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    for v in range(n_levels):
        mask = val_np == v
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[v], s=6, alpha=0.6, label=f"v={v}", linewidths=0,
        )
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _scatter_3d_by_level_mpl(
    coords: np.ndarray,
    val_np: np.ndarray,
    title: str,
    n_levels: int = 10,
) -> plt.Figure:
    """3D scatter coloured by valuation level for TensorBoard add_figure."""
    colors = _level_colors(n_levels)
    fig = plt.figure(figsize=(8, 7), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for v in range(n_levels):
        mask = val_np == v
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=colors[v], s=5, alpha=0.5, label=f"v={v}",
        )
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def _persistence_figure(dgms: list) -> plt.Figure:
    """Persistence diagram (birth vs death scatter) for H0 and H1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    dim_names = ["H0 (components)", "H1 (loops)"]
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        if i >= len(dgms) or len(dgms[i]) == 0:
            ax.set_title(f"{name}: no features")
            continue
        dgm = dgms[i]
        finite = dgm[dgm[:, 1] < np.inf]
        inf_pts = dgm[dgm[:, 1] == np.inf]
        max_val = finite[:, 1].max() if len(finite) > 0 else 1.0
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], c="steelblue", s=10, alpha=0.7)
        if len(inf_pts) > 0:
            ax.scatter(
                inf_pts[:, 0], [max_val * 1.1] * len(inf_pts),
                c="red", s=20, marker="^", label=f"∞ ({len(inf_pts)})",
            )
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"{name} — {len(dgm)} features")
        ax.legend(fontsize=8)
    fig.suptitle("Persistent Homology (Poincaré distances)", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotly HTML helpers
# ---------------------------------------------------------------------------

def _plotly_scatter_3d(
    coords: np.ndarray,
    val_np: np.ndarray,
    title: str,
) -> Optional[go.Figure]:
    """Interactive 3D Plotly scatter coloured by valuation level."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    colors = _level_colors(10)
    for v in range(10):
        mask = val_np == v
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0].tolist(),
            y=coords[mask, 1].tolist(),
            z=coords[mask, 2].tolist(),
            mode="markers",
            marker=dict(size=3, color=colors[v], opacity=0.7),
            name=f"v={v}",
        ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )
    return fig


def _plotly_scatter_2d(
    coords: np.ndarray,
    val_np: np.ndarray,
    title: str,
) -> Optional[go.Figure]:
    """Interactive 2D Plotly scatter."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    colors = _level_colors(10)
    for v in range(10):
        mask = val_np == v
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=coords[mask, 0].tolist(),
            y=coords[mask, 1].tolist(),
            mode="markers",
            marker=dict(size=4, color=colors[v], opacity=0.7),
            name=f"v={v}",
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, b=20, t=40),
    )
    return fig


def _plotly_persistence(dgms: list) -> Optional[go.Figure]:
    """Interactive Plotly persistence diagram."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    dim_colors = ["steelblue", "darkorange"]
    for i, color in enumerate(dim_colors):
        if i >= len(dgms) or len(dgms[i]) == 0:
            continue
        dgm = dgms[i]
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) > 0:
            fig.add_trace(go.Scatter(
                x=finite[:, 0].tolist(), y=finite[:, 1].tolist(),
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.8),
                name=f"H{i} finite ({len(finite)})",
            ))
        inf_pts = dgm[dgm[:, 1] == np.inf]
        if len(inf_pts) > 0:
            max_death = finite[:, 1].max() if len(finite) > 0 else 1.0
            fig.add_trace(go.Scatter(
                x=inf_pts[:, 0].tolist(),
                y=[max_death * 1.15] * len(inf_pts),
                mode="markers",
                marker=dict(size=8, color=color, symbol="triangle-up"),
                name=f"H{i} infinite ({len(inf_pts)})",
            ))
    fig.update_layout(
        title="Persistent Homology — Poincaré Distance Matrix",
        xaxis_title="Birth",
        yaxis_title="Death",
    )
    return fig


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class VisualizationPipeline:
    """Phase 2 visualization pipeline for hyperbolic latent geometry.

    Computes all projections from a shared Poincaré distance matrix per eval.
    Logs to TensorBoard and saves interactive HTML to disk.

    Usage in train.py:
        vis = VisualizationPipeline(config.get('visualization', {}), writer)
        # Inside eval block (every eval_every epochs):
        vis.run(epoch, z_A_hyp, valuations)

    Config keys (all optional with defaults):
        max_per_level: int = 500     — points per valuation level in subsample
        persist_every: int = 50      — epochs between persistence computation
        html_dir: str = 'visualizations'
        save_html: bool = True
        umap_neighbors: int = 15
        curvature: float = 1.0
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        writer: Optional[SummaryWriterType],
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        runtime_cfg = VisualizationRuntimeConfig.from_mapping(config)
        self.max_per_level: int = runtime_cfg.max_per_level
        self.persist_every: int = runtime_cfg.persist_every
        self.html_dir: Path = runtime_cfg.html_dir
        self.save_html: bool = runtime_cfg.save_html
        self.umap_neighbors: int = runtime_cfg.umap_neighbors
        self.curvature: float = runtime_cfg.curvature
        self.writer = writer
        self._log = log_callback or (lambda msg: None)

        # Report availability
        missing = []
        if not _HAS_UMAP:
            missing.append("umap-learn")
        if not _HAS_PACMAP:
            missing.append("pacmap")
        if not _HAS_TRIMAP:
            missing.append("trimap")
        if not _HAS_RIPSER:
            missing.append("ripser")
        if not _HAS_PLOTLY:
            missing.append("plotly")
        if missing:
            self._log(f"[VisualizationPipeline] Missing optional packages: {missing}. "
                      "Some outputs will be skipped.")

    def run(
        self,
        epoch: int,
        z_hyp: torch.Tensor,
        valuations: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Run the full visualization pipeline for one eval point.

        Args:
            epoch: Current training epoch (used for TB step and HTML dir names)
            z_hyp: (N, D) Poincaré ball embeddings — can be on any device
            valuations: (N,) integer valuation labels 0..9
            indices: (N,) original operation indices
        """
        if self.writer is None and not self.save_html:
            return

        self._validate_inputs(epoch, z_hyp, valuations)

        # 1. Stratified subsample — runs entirely on CPU numpy
        z_np, val_np, idx_np = _stratified_subsample(
            z_hyp, valuations, indices,
            max_per_level=self.max_per_level,
        )
        N = len(z_np)
        self._log(f"[VisualizationPipeline] epoch={epoch} subsample={N}")

        # 2. Shared pairwise Poincaré distance matrix (O(N²), computed once)
        D = compute_hyperbolic_distance_matrix(z_np, c=self.curvature)

        html_epoch_dir = self.html_dir / f"epoch_{epoch:05d}"
        if self.save_html and _HAS_PLOTLY:
            os.makedirs(html_epoch_dir, exist_ok=True)

        # 3. UMAP 3D (precomputed metric = Poincaré)
        self._run_umap_step(epoch, D, z_np, val_np, html_epoch_dir)

        # 4. PaCMAP 2D (Euclidean on raw coords — complementary view)
        self._run_pacmap_step(epoch, D, z_np, val_np, html_epoch_dir)

        # 5. TriMAP 2D (precomputed distance matrix)
        self._run_trimap_step(epoch, D, val_np, html_epoch_dir)

        # 6. Poincaré ball 3D (logmap0 → PCA in tangent space)
        self._run_poincare3d_step(epoch, z_np, val_np, html_epoch_dir)

        # 7. Native Poincaré Disk (r-theta projection)
        self._run_native_poincare_step(epoch, z_np, val_np, idx_np, html_epoch_dir)

        # 8. SVG Native Tree (Zero-dependency vector graphics)
        self._run_svg_step(epoch, z_np, val_np, idx_np, html_epoch_dir)

        # 9. Persistent homology (every persist_every epochs)
        if epoch % self.persist_every == 0 or epoch == 1:
            self._run_persistence_step(epoch, D, html_epoch_dir)

        # Update 'latest' symlink
        if self.save_html and _HAS_PLOTLY:
            latest = self.html_dir / "latest"
            if latest.is_symlink():
                latest.unlink()
            try:
                latest.symlink_to(html_epoch_dir.name)
            except OSError:
                pass

        if self.writer is not None:
            self.writer.flush()

    @staticmethod
    def _validate_inputs(
        epoch: int,
        z_hyp: torch.Tensor,
        valuations: torch.Tensor,
    ) -> None:
        """Validate runtime inputs before calling optional visualization backends."""
        if epoch < 1:
            raise ValueError(f"VisualizationPipeline: epoch must be >= 1, got {epoch}")
        if z_hyp.ndim != 2:
            raise ValueError(
                f"VisualizationPipeline: z_hyp must have shape (N, D), got ndim={z_hyp.ndim}"
            )
        if valuations.ndim != 1:
            raise ValueError(
                "VisualizationPipeline: valuations must have shape (N,), "
                f"got ndim={valuations.ndim}"
            )
        if z_hyp.shape[0] == 0:
            raise ValueError("VisualizationPipeline: z_hyp must be non-empty")
        if z_hyp.shape[0] != valuations.shape[0]:
            raise ValueError(
                "VisualizationPipeline: z_hyp and valuations must have the same first "
                f"dimension, got {z_hyp.shape[0]} and {valuations.shape[0]}"
            )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_umap_step(
        self,
        epoch: int,
        D: np.ndarray,
        z_np: np.ndarray,
        val_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        coords = _run_umap(D, n_components=3, n_neighbors=self.umap_neighbors)
        if coords is None:
            return

        # TensorBoard add_embedding (replaces the Euclidean one from tensorboard_logger)
        if self.writer is not None:
            # Build metadata matching TensorBoard's expected format
            metadata = [[str(v)] for v in val_np.tolist()]
            self.writer.add_embedding(
                torch.from_numpy(coords),
                metadata=metadata,
                metadata_header=["valuation"],
                global_step=epoch,
                tag="Embedding/UMAP3D_HyperbolicMetric_A",
            )

        # TensorBoard 3D scatter figure
        if self.writer is not None and _HAS_MPL:
            fig = _scatter_3d_by_level_mpl(coords, val_np, f"UMAP 3D — Poincaré metric (epoch {epoch})")
            self.writer.add_figure("Topology/UMAP3D", fig, global_step=epoch)
            plt.close(fig)

        # Plotly HTML
        if self.save_html and _HAS_PLOTLY:
            plotly_fig = _plotly_scatter_3d(coords, val_np, f"UMAP 3D — Poincaré metric (epoch {epoch})")
            if plotly_fig is not None:
                plotly_fig.write_html(str(html_dir / "umap_3d.html"))

    def _run_pacmap_step(
        self,
        epoch: int,
        D: np.ndarray,
        z_np: np.ndarray,
        val_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        coords = _run_pacmap(z_np, D, n_components=2, n_neighbors=self.umap_neighbors)
        if coords is None:
            return

        if self.writer is not None and _HAS_MPL:
            fig = _scatter_2d_by_level(coords, val_np, f"PaCMAP 2D — mid-range structure (epoch {epoch})")
            self.writer.add_figure("Topology/PaCMAP2D", fig, global_step=epoch)
            plt.close(fig)

        if self.save_html and _HAS_PLOTLY:
            plotly_fig = _plotly_scatter_2d(coords, val_np, f"PaCMAP 2D (epoch {epoch})")
            if plotly_fig is not None:
                plotly_fig.write_html(str(html_dir / "pacmap_2d.html"))

    def _run_trimap_step(
        self,
        epoch: int,
        D: np.ndarray,
        val_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        coords = _run_trimap(D, n_components=2)
        if coords is None:
            return

        if self.writer is not None and _HAS_MPL:
            fig = _scatter_2d_by_level(coords, val_np, f"TriMAP 2D — distance ordering (epoch {epoch})")
            self.writer.add_figure("Topology/TriMAP2D", fig, global_step=epoch)
            plt.close(fig)

        if self.save_html and _HAS_PLOTLY:
            plotly_fig = _plotly_scatter_2d(coords, val_np, f"TriMAP 2D (epoch {epoch})")
            if plotly_fig is not None:
                plotly_fig.write_html(str(html_dir / "trimap_2d.html"))

    def _run_poincare3d_step(
        self,
        epoch: int,
        z_np: np.ndarray,
        val_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        coords = _poincare_tangent_pca(z_np, n_components=3, c=self.curvature)
        if coords is None:
            return

        if self.writer is not None and _HAS_MPL:
            fig = _scatter_3d_by_level_mpl(
                coords, val_np,
                f"Poincaré ball (tangent PCA, epoch {epoch})"
            )
            self.writer.add_figure("Topology/Poincare3D", fig, global_step=epoch)
            plt.close(fig)

        if self.save_html and _HAS_PLOTLY:
            plotly_fig = _plotly_scatter_3d(
                coords, val_np,
                f"Poincaré ball — logmap0 PCA (epoch {epoch})"
            )
            if plotly_fig is not None:
                plotly_fig.write_html(str(html_dir / "poincare_3d.html"))

    def _run_persistence_step(
        self,
        epoch: int,
        D: np.ndarray,
        html_dir: Path,
    ) -> None:
        result = compute_persistence(D, max_dimension=1)
        if not result:
            return

        dgms = result["dgms"]
        betti_0 = result["betti_0"]
        betti_1 = result["betti_1"]
        entropy = result["persistence_entropy"]

        # TensorBoard scalars
        if self.writer is not None:
            self.writer.add_scalar("Topology/betti_0", betti_0, epoch)
            self.writer.add_scalar("Topology/betti_1", betti_1, epoch)
            self.writer.add_scalar("Topology/persistence_entropy", entropy, epoch)
            # ratio: H1/H0 — should be near 0 for a tree-like structure
            self.writer.add_scalar(
                "Topology/betti_ratio", betti_1 / (betti_0 + 1e-6), epoch
            )

        self._log(
            f"[VisualizationPipeline] Topology epoch={epoch}: "
            f"β0={betti_0} β1={betti_1} entropy={entropy:.3f}"
        )

        # TensorBoard persistence diagram figure
        if self.writer is not None and _HAS_MPL:
            fig = _persistence_figure(dgms)
            self.writer.add_figure("Topology/PersistenceDiagram", fig, global_step=epoch)
            plt.close(fig)

        # Plotly HTML
        if self.save_html and _HAS_PLOTLY:
            plotly_fig = _plotly_persistence(dgms)
            if plotly_fig is not None:
                plotly_fig.write_html(str(html_dir / "persistence.html"))

    def _run_native_poincare_step(
        self,
        epoch: int,
        z_np: np.ndarray,
        val_np: np.ndarray,
        indices_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        """Render native 2D Poincaré disk (r-theta projection)."""
        colors = _level_colors(10)

        # 0. Define landmark algebraic walks for visualization
        # We pick indices that are likely to be in the sample
        walks = []
        # Walk 1: Powers of 3 (radial path)
        p3 = [1, 3, 9, 27, 81, 243, 729, 2187, 6561]
        walks.append(np.array(p3))
        # Walk 2: Successive addition around a landmark
        s1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        walks.append(np.array(s1))

        # 1. Always try to save image via Matplotlib
        if _HAS_MPL:
            img_path = html_dir / "poincare_disk_native.png"
            save_poincare_disk(
                z_np, val_np,
                output_path=str(img_path),
                indices=indices_np,
                title=f"Native Poincaré Disk (epoch {epoch})",
                c=self.curvature,
                colors=colors,
                show_tree=True,
                walks=walks
            )
            # Add to TensorBoard if writer is present
            if self.writer is not None:
                from src.utils.poincare_renderer import render_poincare_disk_mpl
                fig = render_poincare_disk_mpl(
                    z_np, val_np,
                    indices=indices_np,
                    title=f"Native Poincaré Disk (epoch {epoch})",
                    c=self.curvature,
                    colors=colors,
                    show_tree=True,
                    walks=walks
                )
                self.writer.add_figure("Topology/PoincareDiskNative", fig, global_step=epoch)
                plt.close(fig)

        # 2. Try to save interactive HTML if Plotly available
        if self.save_html and _HAS_PLOTLY:
            html_path = html_dir / "poincare_disk_native.html"
            save_poincare_disk(
                z_np, val_np,
                output_path=str(html_path),
                indices=indices_np,
                title=f"Native Poincaré Disk (epoch {epoch})",
                c=self.curvature,
                colors=colors,
                show_tree=True,
                walks=walks
            )

    def _run_svg_step(
        self,
        epoch: int,
        z_np: np.ndarray,
        val_np: np.ndarray,
        indices_np: np.ndarray,
        html_dir: Path,
    ) -> None:
        """Render native SVG Poincaré disk (zero-dependency)."""
        # Always generate SVG as it has no dependencies — ensure dir exists
        os.makedirs(html_dir, exist_ok=True)
        svg_path = html_dir / "poincare_disk_native.svg"

        # 0. Define landmark algebraic walks
        walks = []
        walks.append(np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561]))
        walks.append(np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))

        # 1. Project to 2D disk (r-theta) exactly like the renderer
        r_euclidean = np.linalg.norm(z_np, axis=1)
        z_torch = torch.from_numpy(z_np).double()
        with torch.no_grad():
            v_tangent = log_map_zero(z_torch, c=self.curvature).float().numpy()

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        v_2d = pca.fit_transform(v_tangent)
        v_norms = np.clip(np.linalg.norm(v_2d, axis=1, keepdims=True), 1e-10, None)
        v_dir = v_2d / v_norms
        z_2d = r_euclidean[:, np.newaxis] * v_dir

        colors = _level_colors(10)
        save_poincare_disk_svg(
            z_2d, val_np,
            output_path=str(svg_path),
            indices=indices_np,
            title=f"3-Adic Latent Tree (Epoch {epoch})",
            colors=colors,
            show_tree=True,
            walks=walks
        )


__all__ = [
    "VisualizationPipeline",
    "VisualizationRuntimeConfig",
    "compute_hyperbolic_distance_matrix",
    "compute_persistence",
]
