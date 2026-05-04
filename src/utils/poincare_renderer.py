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

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

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
        idx_map = {idx: i for i, idx in enumerate(indices)}
        
        parents = TERNARY.parent(torch.from_numpy(indices)).numpy()
        for i, p_idx in enumerate(parents):
            if p_idx in idx_map:
                p_i = idx_map[p_idx]
                # Draw a simple line for now (geodesics are circular arcs, but
                # in PCA-tangent space, straight lines are a decent first approx)
                ax.plot(
                    [z_2d[i, 0], z_2d[p_i, 0]],
                    [z_2d[i, 1], z_2d[p_i, 1]],
                    color='gray', alpha=0.1, linewidth=0.5, zorder=1
                )

    # 2. Draw Scatter Points
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
) -> Any:
    """Render embeddings in a 2D Poincaré disk."""
    if not _HAS_PLOTLY:
        return None
    
    # This Plotly implementation currently ignores show_tree for simplicity
    N, D = z_hyp.shape
    
    # 1. Calculate Poincaré Radius (r)
    # Norm in Poincaré ball is NOT Euclidean distance to origin, 
    # but we can use the Euclidean norm ||z|| for the 2D plot 
    # because it maps monotonically to hyperbolic distance.
    r_euclidean = np.linalg.norm(z_hyp, axis=1)
    
    # 2. Project Directions to 2D
    # We use logmap0 to get tangent vectors, then PCA to 2D for the direction.
    # This preserves the "relative orientation" of vectors at the origin.
    z_torch = torch.from_numpy(z_hyp).double()
    with torch.no_grad():
        v_tangent = log_map_zero(z_torch, c=c).float().numpy()
    
    # Use PCA to get the two most significant direction components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    v_2d = pca.fit_transform(v_tangent)
    
    # Normalize v_2d to unit length to get 'pure' direction (theta)
    v_norms = np.linalg.norm(v_2d, axis=1, keepdims=True)
    v_norms = np.clip(v_norms, 1e-10, None)
    v_dir = v_2d / v_norms
    
    # 3. Reconstruct 2D Poincaré coordinates
    # (x, y) = r_euclidean * v_dir
    z_2d = r_euclidean[:, np.newaxis] * v_dir
    
    # 4. Create Plotly Figure
    fig = go.Figure()
    
    # Draw the boundary circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
        mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dash'),
        name='Boundary',
        showlegend=False
    ))

    if colors is None:
        # Default fallback (HSL)
        colors = [f"hsl({i*36}, 70%, 50%)" for i in range(10)]

    for v in range(10):
        mask = valuations == v
        if not mask.any():
            continue
        
        fig.add_trace(go.Scatter(
            x=z_2d[mask, 0].tolist(),
            y=z_2d[mask, 1].tolist(),
            mode='markers',
            marker=dict(
                size=6,
                color=colors[v],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            name=f"v={v}",
            hoverinfo='name'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False),
        width=800,
        height=800,
        plot_bgcolor='white',
        legend=dict(itemsizing='constant'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def save_poincare_disk(
    z_hyp: np.ndarray,
    valuations: np.ndarray,
    output_path: str,
    indices: Optional[np.ndarray] = None,
    title: str = "Native Poincaré Disk",
    c: float = 1.0,
    colors: Optional[list] = None,
    show_tree: bool = False,
):
    """Convenience helper to render and save. Supports .html and .png/.pdf."""
    if output_path.endswith('.html'):
        if not _HAS_PLOTLY:
            return
        fig = render_poincare_disk(z_hyp, valuations, indices, title, c, colors, show_tree)
        fig.write_html(output_path)
    else:
        if not _HAS_MPL:
            return
        fig = render_poincare_disk_mpl(z_hyp, valuations, indices, title, c, colors, show_tree)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
