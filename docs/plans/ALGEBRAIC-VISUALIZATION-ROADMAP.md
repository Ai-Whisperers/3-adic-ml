# Roadmap: Algebraic Consistency & Native Hyperbolic Visualization

**Date**: May 3, 2026  
**Status**: Draft / Strategic Direction  
**Target**: V10.0 "Algebraic Geometry"

---

## Executive Summary
This roadmap outlines the dual-track improvement of the 3-Adic ML project. We aim to prove that the latent space is not just a hierarchical map, but a functional algebraic manifold, and to make that manifold interactively visible through native hyperbolic rendering.

---

## Phase 1: Algebraic Consistency Probing (The Science)
**Goal**: Quantify how well the latent space preserves 3-adic field operations.

### 1.1 Additive Vector Logic
*   **Task**: Measure "Latent Addition Error".
*   **Metric**: $E_{add} = || \text{Decoder}(z(a) \oplus z(b)) - (a + b \pmod{3^9}) ||$.
*   **Hypothesis**: Addition in the 3-adic field maps to a specific isometric translation in the Poincaré ball.

### 1.2 Symmetry & Negation Probing
*   **Task**: Analyze $z(n)$ vs $z(-n)$.
*   **Implementation**: Check for reflection symmetry in $z_\theta$ (direction space).
*   **Validation**: Train a linear probe to predict ternary "parity" or "sign" from latent vectors.

### 1.3 Valuation Shift Operators
*   **Task**: Map the $n \to 3n$ transformation (valuation increment).
*   **Geometry**: This should correspond to a radial step $\Delta r$ toward the origin.
*   **Outcome**: Identification of a "Scale Operator" in latent space.

---

## Phase 2: Native Poincaré Renderer (The UX)
**Goal**: Move beyond Euclidean "flattening" (UMAP) to true hyperbolic navigation.

### 2.1 The Poincaré Disk Engine
*   **Technology**: D3.js or Three.js (GLSL shaders).
*   **Core Feature**: Render points directly in the disk using $(r, \theta)$ from the factored projection.
*   **Interaction**: Implement "Hyperbolic Panning" (Möbius transformations/Isometries) where clicking a point "centers" the disk on that point, expanding its local hierarchy.

### 2.2 Algebraic Overlays
*   **Task**: Visualize "Algebraic Walks".
*   **Feature**: User selects operation $a$; UI shows "shadows" or arrows pointing to $a+1, a+2, 3a$, etc.
*   **Outcome**: Visual confirmation of the algebraic consistency found in Phase 1.

### 2.3 Tree-Topology Integration
*   **Task**: Dynamic adjacency lines.
*   **Feature**: Draw lines between points that share a common $k$-digit prefix, making the "Cayley Graph" visible within the continuous ball.

---

## Phase 3: Integration & Foundation (The ROI)
**Goal**: Converge findings into the V10.0 release.

*   **V10.0 Model**: Architecture refined to enforce the algebraic properties found (e.g., via "algebraic loss" terms).
*   **Public Dashboard**: A hosted interactive page showcasing the live 3-adic logic geometry.
*   **Paper/Preprint**: "Emergent Algebraic Structure in Hyperbolic P-Adic VAEs".

---

## Implementation Reference
*   **Precision**: Maintain `float64` for all coordinate calculations.
*   **Geometry Backend**: `src/geometry/poincare.py` (geoopt).
*   **Data Source**: `src/core/ternary.py` (TernarySpace lookup tables).
*   **Verification**: All probes must be integrated into `scripts/diagnostics/`.
