# ADR-002: Modular Training Architecture (V6.2)

**Status**: Accepted  
**Date**: 2026-05-17  
**Deciders**: AI Whisperers Core Team

## Context

As the project moved toward Phase 10 (Algebraic Consistency), the monolithic `src/train.py` (exceeding 2,000 lines) became a significant bottleneck. It suffered from:
1.  **Low Testability**: Core training logic was intertwined with CLI boilerplate and hardware setup.
2.  **Configuration Drift**: Changes in loss functions often required manual updates in multiple distant parts of the file.
3.  **Type-Safety Brittleness**: Mypy struggled to trace types through the monolithic loops, leading to silent metric-parsing bugs.

## Decision

We refactored the training pipeline into a modular package-based structure:

### 1. Separation of Concerns
*   `src/training/bootstrap.py`: Handles hardware initialization, data auditing, and deterministic setup.
*   `src/training/setup.py`: A factory module for optimizers, schedulers, and loss composition.
*   `src/training/engine.py`: Pure training and validation logic, decoupled from CLI and file system concerns.
*   `src/training/reporting.py`: Centralized management of checkpointing, TensorBoard logging, and final results persistence.

### 2. Standardized Package Layout
We added `src/__init__.py` and eliminated `sys.path` manipulation. The project is now a proper Python package that supports standard absolute imports and `pip install -e .` installation.

### 3. Numerical Integrity Gates
We integrated a "Safety Wrapper" within `src/training/engine.py` to proactively catch `NaN` or `Inf` loss values before the backward pass, identifying exactly which mathematical component failed.

## Consequences

*   **Maintainability**: Significantly easier to add new V10 objectives by updating the `setup.py` factory.
*   **Quality**: Allowed for 100% `mypy` type-safety across the core training infrastructure.
*   **Portability**: The project can now be imported as a library by external scripts without path-hacking.
*   **Observability**: Improved logging detail, specifically for numerical instability events.
