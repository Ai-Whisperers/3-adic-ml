# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Hyperbolic Geodesic Utilities.

Calculates circular arc paths for geodesics in the Poincaré disk, and shared
2D-projection / tree / territory geometry reused by the mpl, plotly, and SVG
Poincaré disk renderers.
"""

from typing import List, Tuple

import numpy as np
import torch

from src.core import TERNARY
from src.geometry import log_map_zero


def get_geodesic_arc(u: np.ndarray, v: np.ndarray, n_points: int = 30) -> np.ndarray:
    """Calculate points on the hyperbolic geodesic between u and v in the Poincaré disk.

    If u and v are near-collinear with the origin, returns a straight line.
    Otherwise, calculates the circular arc orthogonal to the boundary.

    Args:
        u, v: (2,) arrays in the Poincaré disk (norm < 1)
        n_points: Number of points to sample along the arc

    Returns:
        (n_points, 2) array of coordinates
    """
    # 1. Check for collinearity with origin (straight line case)
    # Cross product in 2D
    det = u[0] * v[1] - u[1] * v[0]
    if np.abs(det) < 1e-6:
        # Collinear or one point is origin
        t = np.linspace(0, 1, n_points)
        return u[np.newaxis, :] * (1 - t)[:, np.newaxis] + v[np.newaxis, :] * t[:, np.newaxis]

    # 2. Find the center and radius of the circle orthogonal to the boundary
    # and passing through u and v.
    # From: "Drawing Hyperbolic Tilings"
    # The circle's center O(x0, y0) must satisfy:
    # (u_x - x0)^2 + (u_y - y0)^2 = R^2
    # (v_x - x0)^2 + (v_y - y0)^2 = R^2
    # x0^2 + y0^2 = R^2 + 1  (orthogonality to unit circle)

    # Let u2 = u_x^2 + u_y^2, v2 = v_x^2 + v_y^2
    u2 = np.sum(u**2)
    v2 = np.sum(v**2)

    # Linear system for (x0, y0):
    # 2*x0*(v_x - u_x) + 2*y0*(v_y - u_y) = v2 - u2
    # 2*x0*u_x + 2*y0*u_y = u2 + 1
    # ... derived from: u2 - 2(u.O) + x0^2 + y0^2 = R^2 => u2 - 2(u.O) + 1 = 0

    A = np.array([
        [2*u[0], 2*u[1]],
        [2*v[0], 2*v[1]]
    ])
    b = np.array([u2 + 1, v2 + 1])

    try:
        center = np.linalg.solve(A, b)
        radius = np.sqrt(np.sum(center**2) - 1)
    except np.linalg.LinAlgError:
        # Fallback to straight line if system is singular
        t = np.linspace(0, 1, n_points)
        return u[np.newaxis, :] * (1 - t)[:, np.newaxis] + v[np.newaxis, :] * t[:, np.newaxis]

    # 3. Parametrize the arc
    # Calculate angles of u and v relative to the center
    angle_u = np.arctan2(u[1] - center[1], u[0] - center[0])
    angle_v = np.arctan2(v[1] - center[1], v[0] - center[0])

    # Ensure we take the shorter path
    diff = angle_v - angle_u
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    angles = np.linspace(angle_u, angle_u + diff, n_points)

    arc = np.zeros((n_points, 2))
    arc[:, 0] = center[0] + radius * np.cos(angles)
    arc[:, 1] = center[1] + radius * np.sin(angles)

    return arc


def project_to_poincare_disk_2d(z_hyp: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project (N, D) Poincaré-ball points to 2D disk coordinates.

    Preserves the original Euclidean radius (meaningful hierarchy distance);
    direction comes from a 2-component PCA of the log-mapped tangent vectors.
    """
    from sklearn.decomposition import PCA

    r_euclidean = np.linalg.norm(z_hyp, axis=1)
    z_torch = torch.from_numpy(z_hyp).double()
    with torch.no_grad():
        v_tangent = log_map_zero(z_torch, c=c).float().numpy()

    pca = PCA(n_components=2)
    v_2d = pca.fit_transform(v_tangent)
    v_norms = np.clip(np.linalg.norm(v_2d, axis=1, keepdims=True), 1e-10, None)
    v_dir = v_2d / v_norms
    return r_euclidean[:, np.newaxis] * v_dir


def compute_tree_edge_arcs(
    z_2d: np.ndarray, indices: np.ndarray, n_points: int = 10
) -> List[np.ndarray]:
    """Return geodesic arcs (n_points, 2) connecting each index to its 3-adic parent.

    Only edges whose parent index is itself present in `indices` are included.
    """
    idx_map = {idx: i for i, idx in enumerate(indices)}
    parents = TERNARY.parent(torch.from_numpy(indices)).numpy()
    arcs = []
    for i, p_idx in enumerate(parents):
        if p_idx in idx_map:
            arcs.append(get_geodesic_arc(z_2d[i], z_2d[idx_map[p_idx]], n_points=n_points))
    return arcs


def compute_prefix_hulls(
    z_2d: np.ndarray, indices: np.ndarray, k: int = 3, min_points: int = 3
) -> List[Tuple[int, np.ndarray]]:
    """Return (prefix_class, hull_polygon) pairs for digit-prefix classes with enough points.

    Classes with fewer than `min_points` members, or a degenerate (non-hull-able)
    point set, are silently skipped.
    """
    prefix_classes = TERNARY.digit_prefix_class(torch.from_numpy(indices), k=k).numpy()
    hulls = []
    for p in np.unique(prefix_classes):
        mask = prefix_classes == p
        if mask.sum() >= min_points:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(z_2d[mask])
                hulls.append((int(p), z_2d[mask][hull.vertices]))
            except Exception:
                pass
    return hulls
