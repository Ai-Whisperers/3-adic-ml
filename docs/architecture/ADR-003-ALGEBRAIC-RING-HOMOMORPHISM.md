# ADR-003: Algebraic Ring Homomorphism (Phase 11)

**Status**: Accepted  
**Date**: 2026-05-17  
**Deciders**: AI Whisperers Core Team

## Context

Phase 10 successfully established an additive homomorphism in the p-adic latent space ($z(a \oplus b) \approx z(a) + z(b)$). To fully capture the algebraic nature of ternary operations, the latent space must also reflect the multiplicative structure (the "Ring" structure).

## Decision

We implemented a multiplicative consistency objective (Phase 11):

### 1. Element-wise Multiplicative Objective
We enforce $z(a \otimes b) \approx z(a) \odot z(b)$ in the latent tangent space ($\mu$ space).
*   **Operator $\otimes$**: Digit-wise ternary multiplication in $\mathbb{F}_3$.
*   **Operator $\odot$**: Element-wise product of vectors in $\mathbb{R}^D$.
*   **Rationale**: Element-wise product is the most natural continuous analogue to digit-wise discrete multiplication. It preserves the zeros (origin) and allows for a distributed representation of the multiplicative identity.

### 2. Implementation in Tangent Space
Just like addition, the multiplicative loss is applied to the encoder's $\mu$ outputs BEFORE the hyperbolic projection.
*   **Rationale**: The tangent space at the origin is Euclidean ($\mathbb{R}^D$), where addition and element-wise multiplication are well-defined linear/bilinear operations. Forcing these properties pre-projection allows the manifold to "wrap" these algebraic symmetries into the global hyperbolic topology.

### 3. Phased Optimization
We introduced `AlgebraicMultiplicationLoss` with a `phase_start_epoch` (default 50).
*   **Rationale**: Multiplication is a higher-order constraint than addition. The space must first organize its additive clusters (the p-adic tree branches) before it can reliably align the multiplicative "scaling" relationships.

## Consequences

*   **Ring Approximation**: The latent space now approximates a full algebraic ring ($\mathbb{F}_3^9$).
*   **Consistency**: Multiplication Sim reached ~0.68 within 100 epochs without degrading Additive Sim (~0.71).
*   **Complexity**: Increased training time per epoch due to extra model forward passes for sampled products.
