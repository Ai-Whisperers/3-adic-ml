# ADR-004: Latent Symbolic Reasoning (Symbolic Calculator)

**Status**: Accepted  
**Date**: 2026-05-18  
**Deciders**: AI Whisperers Core Team

## Context

We have established that the V11 model embeds a continuous ring homomorphism into its hyperbolic latent space. To utilize this beyond simple reconstruction, we need a way to perform and verify symbolic reasoning tasks (like addition and multiplication of operations) directly within the latent space, without relying on full-domain decoding for every step.

## Decision

We implemented the `AlgebraicLatentCalculator` toolset.

### 1. Functional Scope
The calculator provides a CLI for zero-shot algebraic operations:
- Converts symbolic ternary operations $	o$ indices $	o$ latent space ($\mu$).
- Performs arithmetic operations (Addition, Multiplication) directly on latent tensors.
- Decodes the result to evaluate if the homomorphic approximation holds.

### 2. Verification Protocol
The tool includes diagnostic capabilities to measure:
- Mu-space MSE and Cosine Similarity (evaluating the algebraic homomorphism).
- Decoded digit accuracy (evaluating the practical resolution of the operation).

## Consequences

- **Latent Prototyping**: Allows for rapid experimentation with symbolic chains ($a \oplus b \otimes c$) without re-training or deploying full models.
- **LSB Bottleneck Identification**: Directly revealed the "LSB Resolution Gap," where reconstruction accuracy for the least significant digit is suboptimal.
- **Foundation for Phase 13**: Provides the tools necessary to develop residual correction networks or refinement layers in the future.
