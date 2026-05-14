# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Hyperbolic Geodesic Utilities.

Calculates circular arc paths for geodesics in the Poincaré disk.
"""

import numpy as np


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
