# ADR-001: Architectural Foundation (V6.0 Pivot)

**Status**: Accepted  
**Date**: 2026-05-17  
**Deciders**: AI Whisperers Core Team

## Context

Early versions of the p-adic VAE struggled with a "stability-coverage" trade-off. Optimizing for hierarchy (ultrametric structure) often collapsed the latent space, while optimizing for coverage (reconstruction) led to unorganized, Euclidean-like distributions that failed to capture the 3-adic tree structure.

## Decision

We moved to a **Dual-VAE + True Hyperbolic Projection** architecture (V6.0).

### 1. Dual-VAE (Coverage/Hierarchy Split)
We maintain two independent encoder heads (A and B):
*   **VAE-A**: Primary head for reconstruction.
*   **VAE-B**: Specialized head for learning hierarchy.
*   **Rationale**: By splitting the heads, we can use the StateNet controller to dynamically freeze/unfreeze the reconstruction head (A) while allowing the hierarchy head (B) to refine the global topology without constant pressure to satisfy reconstruction logits.

### 2. True Hyperbolic Projections (expmap0)
We transitioned from "pseudo-hyperbolic" constraints to true manifold projections:
*   **Tangent residual network** maps raw encoder outputs to the tangent space at the origin.
*   **expmap0** projects tangent vectors to the Poincaré ball.
*   **logmap0** projects manifold points back to the tangent space for the decoder.
*   **Rationale**: Using true hyperbolic distances in the loss functions ensures that the latent space correctly approximates an ultrametric (hierarchical) tree, as hyperbolic space is the continuous limit of such trees.

### 3. StateNet Controller (Q-Gating)
We implemented a dynamic training controller:
*   **Metric**: $Q = dist\_corr + 1.5 \times |hierarchy|$
*   **Action**: Automatically manages learning rate scales for components.
*   **Rationale**: Prevents training stalls and "grokking" regressions by ensuring components are only trainable when prerequisites (like reconstruction coverage) are stable.

## Consequences

*   **Complexity**: Higher parameter count and more complex training loops.
*   **Numerical Stability**: Requires float64 and strict curvature clamping.
*   **Reliability**: Significant improvement in $Q$-scores and hierarchical consistency compared to V5.x.
