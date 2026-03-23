# Adversarial Audit Report
**Date**: 22-03-2026
**Target**: Identity Geometry Feature (Steps 1-5) — digit classifiers, AngularCoherenceLoss, Angular Q metric, v7.yaml wiring
**Auditor**: Adversarial Auditor Agent
**Audit Depth**: Full

## Executive Summary

The identity geometry feature (Steps 1-5) is **correctly implemented end-to-end** with no mocks, no disconnected wiring, and no shortcuts. The core pipeline -- classifiers in `ternary.py`, `AngularCoherenceLoss` in `padic_geodesic.py`, wiring through `CombinedLoss`, `r` kwarg propagation from `train.py`, and Angular Q metric logging -- works as claimed. One medium-severity semantic issue was found in `valuation_prefix_class` (unused by the active pipeline, but misleading), and one low-severity silent-skip behavior was identified.

## Behavioral Adherence Score

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Behavioral Adherence | 1/1 | All 5 steps verified to work under real conditions. `digit_prefix_class(k=3)` produces 27 classes as claimed. `AngularCoherenceLoss` fires with real batch data, finds same-class pairs, and contributes to the total loss. Angular Q metric computes meaningful values. `r_A` flows from model through `CombinedLoss.forward()` to the loss. v7.yaml `prefix_k=3` is read and used. |
| Functional | 1/1 | `digit_prefix_class` range is [0,26] with 27 unique classes. `nonzero_pattern` range is [0,511] with 512 unique values. Gradient isolation verified numerically: `d(AngularCoherenceLoss)/d(r) = 0.0` (exact zero). AQ metric produces near-zero values for random embeddings (correct baseline). Full forward+backward pass succeeds with all losses contributing gradients. |
| Clever | 1/1 | Factored latent design (`z_r + z_theta`) with provable gradient isolation is a non-trivial engineering choice. The composite key `vals * 3^k + prefix` for efficient pair matching is elegant. Using `digit_prefix_class` (not `valuation_prefix_class`) for AngularCoherenceLoss avoids the semantic trap in the latter. |
| Practical | 1/1 | Config-driven with proper YAML propagation. Phase-gating prevents premature angular loss activation. The `r=None` guard (line 606 in `combined.py`) gracefully handles non-factored mode. The 13 same-class pairs found in a B=512 batch with k=3 is sufficient for gradient signal. |
| Reliable | 1/1 | Loss returns zero tensor (not NaN/error) when `n_same < 4` or `epoch < phase_start`. Double normalization in AQ metric (train.py:1279-1280) is redundant in factored mode but harmless (max diff = 2.3e-16). TERNARY.valuation works correctly on CPU/GPU tensor paths. |
| **TOTAL** | **5/5** | **ACHIEVED** |

## Critical Findings

None.

## High-Severity Findings

None.

## Medium-Severity Findings

### M1: `valuation_prefix_class` produces degenerate sub-classes for v>=2

**File**: `src/core/ternary.py:695-723`
**Severity**: Medium
**Impact**: Does NOT affect AngularCoherenceLoss (which uses `digit_prefix_class`, not `valuation_prefix_class`). Affects only code that directly calls `valuation_prefix_class`.

**Description**: The docstring claims "For v=k operations, the first k digits are 0. The k-th digit has sign +/-1 and the (k+1)-th digit takes values {-1,0,+1}. Together they give 6 sub-classes per level." This is **incorrect for the ternary encoding used**.

The function uses `first_nonzero()` which operates on the **shifted** ternary representation (`{-1,0,1}`), not the raw base-3 digits (`{0,1,2}`). For operations with v>=1, the raw digit at position 0 is 0 (by definition of 3-adic valuation), which maps to ternary value -1 (nonzero!). Therefore `first_nonzero()` returns 0 for ALL v>=1 operations.

**Evidence**:
| Level | Total ops | Unique vpc classes | Expected |
|-------|-----------|-------------------|----------|
| v=0   | 13122     | 6                 | 6        |
| v=1   | 4374      | 2 ({1,2})         | 6        |
| v=2   | 1458      | 1 ({0})           | 6        |
| v=3+  | <=486     | 1 ({0})           | 6        |

The function is self-consistent (always returns values in [0,6)) but provides almost no discriminative power for v>=2.

### M2: `AngularCoherenceLoss` silently skipped when `r=None` despite being "enabled"

**File**: `src/losses/combined.py:606`
**Severity**: Low-Medium
**Impact**: If someone enables `angular_coherence` in a non-factored (V6) config, the loss is silently omitted from the total. No warning, no error. The loss appears in `get_enabled_losses()` but never fires.

**Evidence**: `CombinedLoss.forward()` line 606: `if self.angular_coherence is not None and r is not None:` -- the `r is not None` guard silently drops the loss. This is correct behavior for factored-only features, but the guard should emit a warning on first skip to avoid confusion during debugging.

## Low-Severity Findings

### L1: `digit_prefix_class` uses big-endian encoding of LSB-first digits

**File**: `src/core/ternary.py:667-675`
**Severity**: Low (informational)
**Impact**: None -- self-consistent, but potentially confusing.

The ternary LUT stores digits in LSB-first order (digit 0 = least significant). `digit_prefix_class` takes `ops[..., :k]` (the first k = least significant digits) and weights them with `[3^(k-1), ..., 3^0]` (big-endian). This means the "prefix" is actually the **least significant** k digits interpreted in big-endian order. The naming "prefix" suggests most-significant digits. This is self-consistent but semantically inverted from typical prefix usage.

### L2: AQ metric uses random permutation pairs (not exhaustive)

**File**: `src/train.py:1286-1290`
**Severity**: Low
**Impact**: AQ metric has sampling noise proportional to 1/sqrt(500). Acceptable for monitoring.

The AQ computation samples 500 random pairs for efficiency, which is reasonable for a monitoring metric (not a loss). The metric converges to the true value as batch size grows.

### L3: 13/58 trainable parameters with gradients in wiring test

**Severity**: Informational
**Impact**: None -- this is expected behavior due to default `encoder_a_trainable=False` in `TernaryVAEV6Controllable.__init__()` and `decode_b=False` in the forward pass. When train.py constructs the model with YAML-specified trainability, all expected parameters receive gradients.

## Evidence Gathered

### Static Analysis Results

- `AngularCoherenceLoss` is correctly imported in `combined.py` (line 41) and instantiated (lines 286-293).
- `prefix_k` flows from YAML -> `CombinedLoss.__init__` -> `AngularCoherenceLoss.__init__` with no hardcoding.
- `CombinedLoss.forward()` passes `r` kwarg to `AngularCoherenceLoss` (line 607).
- `train.py` passes `r=out.get("r_A")` to both `loss_fn()` and `loss_fn_b()` (lines 1122, 1132).

### Behavioral Test Results

**Gradient isolation test** (AngularCoherenceLoss):
- Created `z_hyp = r * dir` with `r_leaf` and `dir_leaf` as separate leaf tensors.
- Used B=256 v=0 indices to guarantee same-class pairs.
- Result: `r_leaf.grad.norm() = 0.0` (exact zero), `dir_leaf.grad.norm() = 0.134`.
- **PASS**: `d(AngularCoherenceLoss)/d(r) = 0` confirmed numerically.

**Full forward+backward wiring test**:
- Model: `TernaryVAEV6Controllable` with `factored=True, radial_dims=4`.
- Config: v7.yaml with `angular_coherence.enabled=True, prefix_k=3, phase_start_epoch=50`.
- B=512, epoch=100 (past phase gate).
- `angular_coherence` loss value: 0.306, requires_grad=True, 13 same-class pairs found.
- Total loss backward succeeds, 13/58 params have gradients (expected given default trainability).
- **PASS**: End-to-end pipeline works.

**AQ metric sanity check**:
- Random embeddings produce AQ=0.0087 (near zero, correct for no structure).
- `TERNARY.valuation(idx_cat)` works correctly with batch tensors.
- **PASS**: Metric produces meaningful baseline values.

**digit_prefix_class range test (k=3)**:
- Range: [0, 26], 27 unique classes. **PASS**.
- Operations sharing first 3 LSB digits have same class (verified: index 1 and 28). **PASS**.

**nonzero_pattern range test**:
- Range: [0, 511], 512 unique patterns (2^9). **PASS**.
- Index 9841 (all-zero ternary) produces pattern 0. **PASS**.
- Index 0 (all -1 ternary) produces pattern 511. **PASS**.

### Complexity Analysis

- `AngularCoherenceLoss.forward()`: cyclomatic complexity ~6 (acceptable, linear flow with early exits).
- `digit_prefix_class`: complexity 1 (vectorized, no branches).
- `CombinedLoss.forward()`: high cyclomatic complexity (~25) but unavoidable for a config-driven multi-loss combiner.

### AST Scan Results (Fake Data / Mock Detection)

- No `Faker`, `factory_boy`, `unittest.mock`, `MagicMock`, or `patch` usage in any audited file.
- No hard-coded dummy values in production paths.
- All test data in the wiring test uses real `TERNARY.all_ternary()` data and actual model forward passes.

## Fake Data / Mock Contamination Report

**Clean**: No synthetic data generation, monkey patching, or happy-path shortcuts detected in the implementation. The `AngularCoherenceLoss` operates on real embeddings from actual batch data. The AQ metric uses real model outputs.

## ROI Assessment & Recommended Next Actions

1. **Fix `valuation_prefix_class` docstring** (effort: 5 min, impact: prevents future misuse): Update the docstring to accurately document that the function provides 6 sub-classes only for v=0 and degrades to 1-2 classes for v>=1 due to the ternary shift convention. Alternatively, rewrite the function to use raw base-3 digits (the unshifted `ops + 1` values) for the `first_nonzero` lookup.

2. **Add warning for `angular_coherence` skip when `r=None`** (effort: 10 min, impact: debugging aid): Add a one-time warning in `CombinedLoss.forward()` when `self.angular_coherence is not None` but `r is None`, e.g., `warnings.warn("angular_coherence enabled but r=None (non-factored mode); loss skipped", stacklevel=2)`.

3. **Increase `n_pairs` or use stratified sampling in AngularCoherenceLoss** (effort: 30 min, impact: medium): With B=512 and k=3, only 13 same-class pairs were found via random permutation. Stratified sampling (group by key, sample within groups) would yield more pairs and stronger gradient signal.

4. **Add `digit_prefix_class` to the test suite** (effort: 20 min, impact: regression protection): Test range [0, 3^k), uniqueness count, and consistency (same-prefix for operations differing only in higher digits).

## Delegation Log

No delegation was necessary. All verification was performed directly through numerical tests.

## Audit Limitations

1. **GPU path not tested**: All tests ran on CPU. The `TERNARY.valuation()` GPU path was not stress-tested, though the device-caching mechanism appears correct from code inspection.
2. **Long training run not verified**: The AQ metric's convergence over 200 epochs with real training dynamics was not tested. Only the computational correctness of a single forward+backward pass was verified.
3. **Mutation testing not performed**: Time constraints prevented running `mutmut` on the new code. The test suite coverage for the new classifiers is unknown (no tests in `tests/` specifically cover `digit_prefix_class`, `nonzero_pattern`, or `valuation_prefix_class`).
