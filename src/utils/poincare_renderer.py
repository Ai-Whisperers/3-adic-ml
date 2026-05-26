# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Native Poincaré Disk Renderer.

Renders high-dimensional hyperbolic embeddings directly into a 2D Poincaré disk
by preserving the radial distance (from the origin) and using PCA or specific
angular dimensions for the direction.

Features:
- True radial preservation (meaningful hierarchy visualization)
- Interactive Plotly-based HTML output
- Algebraic transformation overlays (Phase 2.2)
"""

from typing import Any, List, Optional

import numpy as np
import torch
import plotly.graph_objects as go

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from src.geometry import log_map_zero

def render_poincare_disk_mpl(
    z_hyp: np.ndarray,
    valuations: np.ndarray,
    indices: Optional[np.ndarray] = None,
    title: str = "Native Poincaré Disk",
    c: float = 1.0,
    colors: Optional[list] = None,
    show_tree: bool = False,
    walks: Optional[List[np.ndarray]] = None,
) -> Any:
    """Render embeddings in a 2D Poincaré disk using Matplotlib."""
    if not _HAS_MPL:
        return None

    r_euclidean = np.linalg.norm(z_hyp, axis=1)

    z_torch = torch.from_numpy(z_hyp).double()
    with torch.no_grad():
        v_tangent = log_map_zero(z_torch, c=c).float().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    v_2d = pca.fit_transform(v_tangent)

    v_norms = np.linalg.norm(v_2d, axis=1, keepdims=True)
    v_norms = np.clip(v_norms, 1e-10, None)
    v_dir = v_2d / v_norms
    z_2d = r_euclidean[:, np.newaxis] * v_dir

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Draw boundary
    boundary = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(boundary)

    if colors is None:
        colors = [plt.cm.plasma(i/10) for i in range(10)]

    # 1. Draw Tree Edges (Cayley Graph)
    if show_tree and indices is not None:
        from src.core import TERNARY
        from src.utils.geodesic_utils import get_geodesic_arc
        idx_map = {idx: i for i, idx in enumerate(indices)}

        parents = TERNARY.parent(torch.from_numpy(indices)).numpy()
        for i, p_idx in enumerate(parents):
            if p_idx in idx_map:
                p_i = idx_map[p_idx]
                arc = get_geodesic_arc(z_2d[i], z_2d[p_i], n_points=10)
                ax.plot(
                    arc[:, 0], arc[:, 1],
                    color='gray', alpha=0.1, linewidth=0.5, zorder=1
                )

    # 2. Draw Algebraic Walks (Flow lines)
    if walks is not None:
        for walk_indices in walks:
            idx_map_local: dict[Any, int] = {idx: i for i, idx in enumerate(indices)} if indices is not None else {}
            if idx_map_local:
                pts = []
                for idx in walk_indices:
                    if idx in idx_map_local:
                        pts.append(z_2d[idx_map_local[idx]])
                if len(pts) > 1:
                    pts_np = np.array(pts)
                    ax.plot(pts_np[:, 0], pts_np[:, 1], color='cyan', alpha=0.6, linewidth=1.5, zorder=3, marker='>')

    # 3. Draw Prefix Territory Shading
    if indices is not None:
        from src.core import TERNARY
        prefix_classes = TERNARY.digit_prefix_class(torch.from_numpy(indices), k=3).numpy()
        unique_prefixes = np.unique(prefix_classes)
        for p in unique_prefixes:
            mask = prefix_classes == p
            if mask.sum() >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(z_2d[mask])
                    polygon = z_2d[mask][hull.vertices]
                    poly_patch = plt.Polygon(polygon, facecolor=plt.cm.tab20(p % 20), alpha=0.05, edgecolor='none', zorder=0)
                    ax.add_patch(poly_patch)
                except Exception:
                    pass

    # 4. Draw Scatter Points
    for v in range(10):
        mask = valuations == v
        if not mask.any():
            continue
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=[colors[v]], s=10, alpha=0.7, label=f"v={v}",
            edgecolors='none', zorder=2
        )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    ax.legend(loc='upper right', markerscale=2, fontsize='small')

    return fig

def render_poincare_disk(
    z_hyp: np.ndarray,
    valuations: np.ndarray,
    indices: Optional[np.ndarray] = None,
    title: str = "Native Poincaré Disk",
    c: float = 1.0,
    colors: Optional[list] = None,
    show_tree: bool = False,
    walks: Optional[List[np.ndarray]] = None,
    decision_threshold: Optional[float] = None,
    query_z: Optional[np.ndarray] = None,
) -> Any:
    """Render embeddings in a 2D Poincaré disk with interactive Plotly."""
    _N, _D = z_hyp.shape
    r_euclidean = np.linalg.norm(z_hyp, axis=1)
    z_torch = torch.from_numpy(z_hyp).double()
    with torch.no_grad():
        v_tangent = log_map_zero(z_torch, c=c).float().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    v_2d = pca.fit_transform(v_tangent)

    v_norms = np.linalg.norm(v_2d, axis=1, keepdims=True)
    v_norms = np.clip(v_norms, 1e-10, None)
    v_dir = v_2d / v_norms
    z_2d = r_euclidean[:, np.newaxis] * v_dir

    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Boundary
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode='lines',
        line={"color": 'rgba(255,255,255,0.6)', "width": 2}, showlegend=False
    ))

    # Decision Boundary
    if decision_threshold is not None:
        r_thresh = np.tanh(decision_threshold / 2)
        fig.add_trace(go.Scatter(
            x=(r_thresh * np.cos(theta)), y=(r_thresh * np.sin(theta)),
            mode='lines', line={"color": 'rgba(248, 113, 113, 0.7)', "width": 2, "dash": 'dot'},
            name='Boundary'
        ))

    # Geodesic
    if query_z is not None:
        # Simplified: linear path to origin is a geodesic in Poincaré projection
        fig.add_trace(go.Scatter(
            x=[0, query_z[0, 0]], y=[0, query_z[0, 1]],
            mode='lines', line={"color": 'rgba(255,255,255,0.4)', "width": 1},
            name='Geodesic'
        ))

    if colors is None:
        colors = [f"hsl({i*36}, 70%, 50%)" for i in range(10)]

    for v in range(10):
        mask = valuations == v
        if not mask.any(): continue
        fig.add_trace(go.Scatter(
            x=z_2d[mask, 0], y=z_2d[mask, 1],
            mode='markers',
            marker={"size": 6, "color": colors[v], "opacity": 0.8, "line": {"width": 0.5, "color": 'white'}},
            name=f"v={v}", hoverinfo='name'
        ))

    fig.update_layout(
        title=title,
        xaxis={"range": [-1.1, 1.1], "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"range": [-1.1, 1.1], "showgrid": False, "zeroline": False},
        width=800, height=800, plot_bgcolor='white', legend={"itemsizing": 'constant'},
        margin={"l": 20, "r": 20, "t": 60, "b": 20}
    )
    return fig

def save_poincare_disk(z_hyp, valuations, output_path, indices=None, title="Native Poincaré Disk", c=1.0, colors=None, show_tree=False, walks=None):
    fig = render_poincare_disk(z_hyp, valuations, indices, title, c, colors, show_tree, walks)
    fig.write_html(output_path)
