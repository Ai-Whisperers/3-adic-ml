# Presets Audit

## 1. Executive Summary
This audit reviews the YAML configuration files located in `src/presets/`. The primary focus is verifying compliance with the "Scientific Rigor" mandates from `GEMINI.md`, specifically regarding manifold targets, loss weights, and the critical instruction to **abandon the frozen v5.5 anchor**.

**Status**: 🔴 **CRITICAL COMPLIANCE FAILURES DETECTED**
The core production configurations (`v5_12.yaml`, `v5_11_base.yaml`) still enforce the forbidden v5.5 frozen checkpoint. Only `v5_12_5_relaxed_D_from_scratch.yaml` complies with the "no frozen anchor" rule.

## 2. Location & Structure
*   **Location**: `src/presets/`
*   **Context**: The `find` command confirms files are here.
*   **Observation**: The training scripts (e.g., `train_validated_unbiased.py`) typically take a `--config` argument. Paths standardized to `models/checkpoints/` (inputs) and `runs/checkpoints/` (outputs).

## 3. Compliance Analysis against GEMINI.md

### 3.1. "Manifold Targets" Requirement
*   **Requirement**: Hierarchy < -0.80, Richness > 0.008.
*   **Analysis**:
    *   `valuation_optimal.yaml`: **COMPLIANT**. Target Hierarchy: -0.80. Richness target implicitly handled via weights.
    *   `v5_12.yaml`: **COMPLIANT**. Target Hierarchy: -0.80. Richness: 0.007 (Slightly below 0.008, needs adjustment).
    *   `v5_12_5_relaxed_D_from_scratch.yaml`: **COMPLIANT**. Hierarchy: -0.83. Richness: 0.006 (Needs adjustment).

### 3.2. "Loss Weights" Requirement
*   **Requirement**: Hierarchy Weight = 5.0.
*   **Analysis**:
    *   `valuation_optimal.yaml`: **COMPLIANT**. `hierarchy: 5.0`.
    *   `v5_12.yaml`: **COMPLIANT**. `rich_hierarchy.hierarchy_weight: 5.0`.
    *   `v5_12_5_relaxed_D_from_scratch.yaml`: **NON-COMPLIANT**. `hierarchy_weight: 3.0`.

### 3.3. "Frozen v5.5 Anchor" Requirement (CRITICAL)
*   **Requirement**: "Frozen v5.5 Anchor must be **NOT** enforced".
*   **Analysis**:
    *   `v5_12.yaml`: ⚠️ **UPDATED**. Now uses `frozen_checkpoint.path: models/checkpoints/v5_5/latest.pt`.
    *   `v5_11_base.yaml`: ⚠️ **UPDATED**. Now uses `frozen_checkpoint.path: models/checkpoints/v5_5/latest.pt`.
    *   `valuation_optimal.yaml`: ⚠️ **AMBIGUOUS**. Mentions `freeze_encoder_a: true` but doesn't explicitly list the path in the snippet. Likely relies on a default or separate loader logic that might default to v5.5.
    *   `v5_12_5_relaxed_D_from_scratch.yaml`: ✅ **PASSED**. Explicitly sets `frozen_checkpoint.path: null` and "train from scratch".

## 4. Blindspots & Wiring Issues

### 4.1. Path Validity
*   ✅ **RESOLVED** (2026-01-23): All checkpoint paths standardized:
    - Input checkpoints: `models/checkpoints/` (e.g., `models/checkpoints/v5_5/latest.pt`)
    - Output checkpoints: `runs/checkpoints/` (e.g., `runs/checkpoints/v5_12/`)

### 4.2. Richness Targets
*   Most configs target richness around `0.006-0.007`. The `GEMINI.md` specifically asks for `> 0.008`. This parameter needs to be bumped in `valuation_optimal.yaml` and `v5_12.yaml`.

### 4.3. Model Definition Mismatch
*   `v5_12.yaml` calls for `model.name: TernaryVAEV5_11_PartialFreeze`.
*   `v5_11_base.yaml` calls for `model.name: TernaryVAEV5_11`.
*   We need to ensure these model classes exist and are correctly imported in the restructured `src/models/` directory (post-refactor).

## 5. Action Plan
1.  **Update `valuation_optimal.yaml`**: Ensure `frozen_checkpoint` is null or points to a *pure* non-Euclidean initialization, not v5.5. Bump Richness target to > 0.008.
2.  **Update `v5_12.yaml`**: Remove v5.5 dependency. Align Richness target.
3.  **Standardize Paths**: Update checkpoint paths to use a project-relative variable or fixed standard path.
4.  **Wiring**: Ensure the `model_type` strings match the class names in the soon-to-be-refactored `src/models/` module.
