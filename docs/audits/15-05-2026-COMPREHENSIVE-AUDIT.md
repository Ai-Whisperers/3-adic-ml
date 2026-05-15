# Audit Report: 3-Adic ML (2026-05-15)

## 1. Executive Summary
The project has undergone a significant architectural refactor (V6.2 Modularization) since the last major audit in March 2026. This refactor successfully resolved several critical bugs related to configuration drift and runtime validation failures. The core mathematical and geometric foundations (p-adic logic, hyperbolic projections) remain robust and well-tested.

However, the refactor has left behind a substantial amount of technical debt in the form of static analysis warnings and stale documentation.

**Overall Health: 🟢 STABLE (Functional) / 🟡 UNTIDY (Technical Debt)**

## 2. Key Findings

### 2.1 Configuration & Validation (Improved)
- **Resolved**: The critical `NoneType` crash on `anchor_checkpoint: null` is fixed.
- **Resolved**: Missing imports in `train.py` (e.g., `get_model_state_dict`) have been corrected in the modularized `src/training/bootstrap.py`.
- **Resolved**: Defaults for StateNet coverage thresholds are now synchronized between `schema.py` and `statenet_config.py` (using the new `0.35 / 0.45` standard).
- **Improved**: `train.py --validate-only` now successfully validates all primary presets (`v6`, `v7_large`, `5.12.4`).

### 2.2 Static Analysis & Type Safety (High Debt)
- **Mypy**: 32 errors remain, primarily in `src/training/` and `src/losses/`. Key issues include incompatible return types in the training engine and improper handling of `TypedDict` vs `Dict[str, Any]`.
- **Ruff**: 177 warnings found. Major categories:
    - `E402`: Module-level imports not at top (caused by `sys.path` hacks for project root).
    - `F401`: Unused imports (cleanup needed after modularization).
    - `W293`: Trailing whitespace and blank line noise.
    - `C408`: Unnecessary `dict()` calls.

### 2.3 Documentation Drift (Stale)
- **src/README.md**: Highly stale. Still refers to the old monolith `train.py`, quotes outdated defaults (0.995), and lists files that have been deleted or moved.
- **STATUS.md**: Needs update to reflect the modularized V6.2 state.

### 2.4 Dependency Management (Inconsistent)
- `pyproject.toml` is missing `yamllint`, which is present in `requirements.txt`.
- Several visualization dependencies (`umap-learn`, `pacmap`, etc.) are correctly relegated to optional dependencies but are inconsistently listed between files.

## 3. Recommended Remediation

### Priority 1: High-Impact Code Hygiene (Immediate)
1. Fix mypy errors in `src/training/engine.py` and `reporting.py` to ensure type-safe checkpointing.
2. Run `ruff check --fix` to resolve 100+ trivial linting issues.
3. Remove unused imports in `src/training/` package.

### Priority 2: Metadata & Documentation Sync (Near-term)
1. Add `yamllint` to `pyproject.toml`.
2. Rewrite `src/README.md` to reflect the Modularized (V6.2) structure and current StateNet defaults.
3. Update `STATUS.md` with current audit results.

### Priority 3: Process Hardening (Long-term)
1. Add a CI gate for `mypy` on at least the `src/training/` and `src/models/` directories.
2. Investigate removing `sys.path` hacks in favor of proper package installation for scripts to resolve `E402` linting errors.

## 4. Verification
- All 399 tests pass.
- `train.py --validate-only` passes for all core presets.
- Coverage: 64% (Core modules like `ternary.py` and `poincare.py` have high coverage; `train.py` wrapper has low coverage as logic moved to `src/training/`).
