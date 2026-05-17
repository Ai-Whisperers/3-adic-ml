# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Pure Python SVG Renderer for Poincaré Disk.

Generates beautiful, standalone vector graphics with true hyperbolic geodesics.
Zero dependencies beyond numpy.
"""

from typing import List, Optional

import numpy as np

from src.utils.geodesic_utils import get_geodesic_arc


def render_poincare_disk_svg(
    z_2d: np.ndarray,
    valuations: np.ndarray,
    indices: Optional[np.ndarray] = None,
    title: str = "3-Adic Latent Manifold",
    colors: Optional[List[str]] = None,
    show_tree: bool = True,
    walks: Optional[List[np.ndarray]] = None,
    width: int = 800,
    height: int = 800,
) -> str:
    """Render Poincaré disk as an SVG string.

    Args:
        z_2d: (N, 2) coordinates in unit disk
        valuations: (N,) integer valuations
        indices: (N,) original indices for tree structure
        title: SVG Title
        colors: Hex color strings for levels 0-9
        show_tree: Whether to draw hyperbolic arcs
        walks: List of index-arrays for flow lines
    """
    if colors is None:
        colors = [
            "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786",
            "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"
        ]

    # Padding and scale
    margin = 50
    scale = (min(width, height) - 2 * margin) / 2
    cx, cy = width / 2, height / 2

    def to_svg(coords):
        # Flip Y for SVG coordinate system
        x = cx + coords[0] * scale
        y = cy - coords[1] * scale
        return x, y

    # SVG Header
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#0b0e14; font-family: sans-serif;">',
        '<rect width="100%" height="100%" fill="#0b0e14"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{scale}" fill="none" stroke="#2a2e37" stroke-dasharray="4,2"/>'
    ]

    # 1. Tree Edges (Geodesic Arcs)
    idx_map = {idx: i for i, idx in enumerate(indices)} if indices is not None else None
    if show_tree and indices is not None and idx_map:
        import torch

        from src.core import TERNARY
        parents = TERNARY.parent(torch.from_numpy(indices)).numpy()

        for i, p_idx in enumerate(parents):
            if p_idx in idx_map:
                p_i = idx_map[p_idx]
                # True Hyperbolic Geodesic
                arc = get_geodesic_arc(z_2d[i], z_2d[p_i], n_points=12)
                path_data = []
                for k, pt in enumerate(arc):
                    sx, sy = to_svg(pt)
                    cmd = "M" if k == 0 else "L"
                    path_data.append(f"{cmd}{sx:.2f},{sy:.2f}")

                path_str = " ".join(path_data)
                svg.append(f'<path d="{path_str}" fill="none" stroke="#4a4e57" stroke-width="0.5" stroke-opacity="0.15"/>')

    # 2. Algebraic Walks (Flow lines)
    if walks is not None and idx_map:
        for walk_indices in walks:
            pts = []
            for idx in walk_indices:
                if idx in idx_map:
                    pts.append(z_2d[idx_map[idx]])

            if len(pts) > 1:
                path_data = []
                for k, pt in enumerate(pts):
                    sx, sy = to_svg(pt)
                    cmd = "M" if k == 0 else "L"
                    path_data.append(f"{cmd}{sx:.2f},{sy:.2f}")

                path_str = " ".join(path_data)
                svg.append(f'<path d="{path_str}" fill="none" stroke="#00f2ff" stroke-width="1.5" stroke-opacity="0.7" marker-end="url(#arrowhead)"/>')

    # 3. Prefix Territory Shading
    if indices is not None:
        import torch

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
                    points_str = " ".join([f"{to_svg(pt)[0]:.2f},{to_svg(pt)[1]:.2f}" for pt in polygon])
                    color = colors[p % 10]
                    svg.append(f'<polygon points="{points_str}" fill="{color}" fill-opacity="0.05" stroke="none"/>')
                except Exception:
                    pass

    # Markers for flow lines
    svg.append('<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#00f2ff" /></marker></defs>')

    # 4. Points
    for v in range(10):
        mask = valuations == v
        if not mask.any():
            continue

        color = colors[v]
        v_indices = np.where(mask)[0]
        for i in v_indices:
            sx, sy = to_svg(z_2d[i])
            # Add simple tooltip and glow
            svg.append(
                f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="2.5" fill="{color}" fill-opacity="0.8">'
                f'<title>Index: {indices[i] if indices is not None else "?"} | Val: {v}</title>'
                f'</circle>'
            )

    # Title
    svg.append(f'<text x="20" y="30" fill="#8b949e" font-size="14" font-weight="bold">{title}</text>')

    # Legend
    for v in range(10):
        svg.append(f'<circle cx="20" cy="{50 + v*15}" r="4" fill="{colors[v]}"/>')
        svg.append(f'<text x="35" y="{54 + v*15}" fill="#8b949e" font-size="10">v={v}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def save_poincare_disk_svg(
    z_2d: np.ndarray,
    valuations: np.ndarray,
    output_path: str,
    indices: Optional[np.ndarray] = None,
    title: str = "3-Adic Latent Manifold",
    colors: Optional[List[str]] = None,
    show_tree: bool = True,
    walks: Optional[List[np.ndarray]] = None,
):
    """Save Poincaré disk SVG to file."""
    content = render_poincare_disk_svg(z_2d, valuations, indices, title, colors, show_tree, walks)
    with open(output_path, "w") as f:
        f.write(content)
