# Adversarial Audit Report
**Date**: 11-03-2026
**Target**: P-Adic VAE V6.0/V6.1 -- Full Codebase (src/, tests/, config)
**Auditor**: Adversarial Auditor Agent (Claude Opus 4.6)
**Audit Depth**: Full

## Executive Summary

The 3-adic VAE codebase demonstrates solid mathematical foundations (3-adic valuation, ultrametric inequality, Poincare geometry via geoopt). However, the audit uncovered **three critical training-blocking bugs** and **multiple high-severity design flaws** that prevent the pipeline from achieving its stated goals. The most damaging: VAE-B is entirely dead (zero gradient flow), all embeddings saturate at max_radius=0.95 destroying hierarchy signal, and KL divergence is absent making this a deterministic autoencoder rather than a VAE. The test suite (269 tests, all passing) validates mathematical invariants well but completely misses these architectural failures.

## Behavioral Adherence Score

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Behavioral Adherence | 0/1 | VAE-B is dead (0 gradient flow). Claims "dual VAE" but only VAE-A trains. The system does not do what it claims. |
| Functional | 0/1 | Three critical bugs: (1) VAE-B dead, (2) max_radius saturation destroys hierarchy, (3) config key mismatch silently uses wrong weights. |
| Clever | 1/1 | The ultrametric-to-hyperbolic mapping concept is mathematically elegant. Exponential target radii matching p-adic structure is a good insight. Learnable loss weights via uncertainty weighting is well-implemented. |
| Practical | 0/1 | Config has 11+ dead keys silently ignored. Missing KL divergence means this is not a VAE. No integration tests catch these issues. |
| Reliable | 0/1 | Loss pair sampling generators are not saved in checkpoints (breaks resume reproducibility). All points saturate at max_radius at initialization, creating a degenerate loss landscape. |
| **TOTAL** | **1/5** | **NOT ACHIEVED** |

## Critical Findings

### C1. VAE-B Is Completely Dead (CRITICAL)

**Severity**: Critical
**Location**: `src/train.py:1035`
**Evidence**:
```python
z_hyp = out.get("z_A_hyp", out.get("z_B_hyp"))
```
Only `z_A_hyp` is ever used in loss computation. `z_B_hyp` is collected for hierarchy metrics (`hier_metrics_B`) but those metrics are computed under `torch.no_grad()`. Result: **zero gradients flow to head_B, decoder_B, and projections.proj_B**. Empirically confirmed: 0/14 head_B params, 0/8 decoder_B params, 0/11 proj_B params receive any gradient. The "dual VAE" architecture is a single VAE with dead weight.

**Impact**: 33 of 66 model parameters (50%) are dead. Training is half as efficient as it appears. All VAE-B metrics reported in TensorBoard are random noise from untrained weights.

### C2. Max_Radius Saturation Destroys Hierarchy Signal (CRITICAL)

**Severity**: Critical
**Location**: `src/models/hyperbolic_projection.py:138-145`, `src/models/vae.py:354-355`
**Evidence**:
At initialization, encoder outputs have tangent space norms of ~4.0. The `expmap0` function maps these to Poincare norms of ~0.9993 (`tanh(4) = 0.9993`). The `max_radius=0.95` clamping then forces ALL 19,683 points to EXACTLY norm 0.95 in the Poincare ball. Every single point starts at hyperbolic radius 3.664, while target radii range from 0.16 (v=9) to 1.73 (v=0).

This means:
- ALL hierarchy losses see identical radius for all points -> no hierarchy gradient signal differentiation
- The loss landscape is completely flat in the radial dimension until the tangent_net learns to produce different magnitudes
- The identity-init residual architecture (`z + tangent_net(z)` where `tangent_net(z) = 0`) means the tangent_net must learn to SUBTRACT magnitude, working against the residual connection

**Impact**: Training must first "escape" this degenerate initialization before any hierarchy learning can begin. This could take many epochs of wasted compute.

### C3. Config Key Mismatch: Radial Weight Silently Defaults to 1.0 (CRITICAL)

**Severity**: Critical
**Location**: `src/presets/v6.yaml:154`, `src/losses/combined.py:170`
**Evidence**:
v6.yaml specifies `radial_weight: 5.0`. CombinedLoss reads `radial_cfg.get('weight', 1.0)`. The key name mismatch means the intended weight of 5.0 is silently ignored and defaults to 1.0. This changes the loss balance significantly.

**Impact**: The loss function operates with unintended weight ratios, potentially explaining suboptimal training results in experiments.

### C4. No KL Divergence in Training -- Not a VAE (HIGH)

**Severity**: High
**Location**: `src/losses/combined.py` (absent), `src/train.py`
**Evidence**:
The model computes `mu` and `logvar` via the reparameterization trick, but `CombinedLoss.forward()` never computes or includes KL divergence. The `HyperbolicKLDivergence` class exists in `src/losses/hyperbolic_kl.py` and is configured in v6.yaml (`hyperbolic_kl.enabled: false`), but even when enabled, CombinedLoss has NO code path to invoke it. There is no KL computation at all.

Without KL regularization, nothing prevents `logvar` from collapsing to `-inf`, making the reparameterization trick produce deterministic outputs (`z = mu`). This is a deterministic autoencoder, not a VAE.

**Impact**: The generative model properties (sampling, interpolation, density estimation) that justify the VAE architecture are absent. The model overfits to identity mapping in latent space.

### C5. Zero_Structure Loss Configured But Non-Existent (MEDIUM)

**Severity**: Medium
**Location**: `src/presets/v6.yaml:174-178`
**Evidence**:
v6.yaml enables `zero_structure` loss with `valuation_weight: 0.5` and `sparsity_weight: 0.3`. No `ZeroStructureLoss` class exists anywhere in the codebase. CombinedLoss silently ignores this entire config block.

**Impact**: Intended loss component missing. If the designer intended this loss to contribute to training, its absence changes the optimization landscape.

### C6. 11+ Dead Config Keys in v6.yaml (MEDIUM)

**Severity**: Medium
**Location**: `src/presets/v6.yaml` (multiple sections)
**Evidence**:
The following config keys are read by NO code:

| Dead Key | Expected Effect | Actual Effect |
|----------|----------------|---------------|
| `model.encoder_dropout` | Dropout in encoder | Ignored (encoder builder has no dropout param) |
| `model.decoder_dropout` | Dropout in decoder | Ignored (decoder builder has no dropout param) |
| `model.logvar_min` | Clamp logvar | Ignored (VAE never clamps logvar) |
| `model.logvar_max` | Clamp logvar | Ignored (VAE never clamps logvar) |
| `loss.rich_hierarchy.richness_weight` | Unknown purpose | Ignored |
| `loss.zero_structure.*` | Zero structure loss | No such loss class exists |
| `loss.radial.radial_weight` | Radial loss weight | Wrong key name (should be `weight`) |
| `statenet.annealing.*` | Threshold annealing | AnnealingConfig was removed, code ignores |
| `targets.*` | Success criteria | Never evaluated programmatically |
| `training.num_workers` | Absent (declared under `device`) | train.py defaults to 4, matching device config accidentally |

**Impact**: Configuration appears more feature-rich than reality. Changes to dead keys give false confidence that behavior is being modified.

### C7. Linear vs Exponential Target Radius Inconsistency (LOW)

**Severity**: Low
**Location**: `src/core/ternary.py:438-462` vs `src/losses/padic_geodesic.py:38-66`
**Evidence**:
`TernarySpace.target_radius()` uses **linear** interpolation between inner and outer radii. All loss functions use `_exponential_target_radii()` with exponential decay. These produce different target radii for intermediate valuation levels. The `target_radius()` function in ternary.py is not used by any loss function (it is a convenience method), so this is an inconsistency rather than a bug.

**Impact**: If a developer uses `TERNARY.target_radius()` for analysis or new loss functions, results will not match training targets.

### C8. Scheduler/LR Controller Interaction (LOW)

**Severity**: Low
**Location**: `src/train.py:1193-1196`
**Evidence**:
`current_base_lr = scheduler.get_last_lr()[0]` takes only the first param group's scheduled LR. `update_optimizer_lr_scales()` then sets each group's LR to `current_base_lr * scale`, overwriting the scheduler's per-group LR. This works correctly because CosineAnnealingWarmRestarts applies the same curve to all groups, but the logic is fragile -- if a scheduler with per-group differentiation were used, the controller would clobber it.

**Impact**: Minimal under current configuration. Would break with certain scheduler types.

## Evidence Gathered

### Static Analysis Results

**File count**: 22 Python source files in `src/`, 11 test files
**Total LOC**: ~5000 lines production code, ~3000 lines test code
**Cyclomatic complexity**: Most functions are moderate (< 10). `train()` function at ~250 lines is the most complex single function, warranting extraction of sub-functions.

### Behavioral Test Results

**Existing tests**: 269 tests, ALL PASSING (0 failures, 3 warnings)
**Test warnings**: `torch.corrcoef` warns on batch_size=1 (degrees of freedom <= 0). Benign.

**Smoke test results**:
- Forward pass: Works correctly, produces all expected output keys
- Loss computation: All 5 losses compute without error
- Gradient flow: 24/66 parameters receive gradients (36% -- because VAE-B is dead and projections.proj_A.manifold.isp_c and proj_B are dead)
- All embeddings saturate at exactly max_radius=0.95

### AST Scan Results (Fake Data / Mock Detection)

**No fake data libraries found** in production code. The codebase uses real data (all 19,683 ternary operations generated from mathematical formulas, not synthetic/random data).

**Test fixtures** use `torch.randn` for synthetic test inputs, which is appropriate for unit tests. No `unittest.mock`, `MagicMock`, or monkey-patching detected anywhere in the codebase.

**No placeholder strings** (`'foo'`, `'bar'`, `'TODO'`, `'FIXME'`) found in production paths.

### Mathematical Verification Results

| Property | Status | Evidence |
|----------|--------|---------|
| 3-adic valuation formula | CORRECT | All standard values verified (v_3(0)=9, v_3(1)=0, v_3(3)=1, ..., v_3(6561)=8) |
| Ultrametric inequality | CORRECT | 0/10000 violations in random testing |
| Strong ultrametric | CORRECT | d(a,c) = max(d(a,b),d(b,c)) when d(a,b) != d(b,c): 100% verified |
| Ternary round-trip | CORRECT | All 19,683 indices survive to_ternary->from_ternary |
| Level population counts | CORRECT | v=0: 13122, v=1: 4374, ..., v=9: 1 (matches 2*3^(8-k) formula) |
| exp/log composition | CORRECT for small norms | max_error=3.11e-15 for ||v||~0.5 |
| exp/log composition | DEGRADED for large norms | max_error=6.61 for ||v||~3, max_error=29.6 for ||v||~10 (due to max_norm clamping in logmap) |
| Poincare ball containment | CORRECT | All expmap0 outputs have norm < 1 |
| Cross-entropy formulation | CORRECT | (B,27) logits correctly reshaped to (B,3,9) for 9-position, 3-class classification |

### Complexity Analysis

| Module | Max Function Complexity | Assessment |
|--------|------------------------|------------|
| `ternary.py` | Low (~5) | Clean, precomputed LUTs |
| `poincare.py` | Low (~3) | Thin wrappers around geoopt |
| `padic_geodesic.py` | Medium (~8) | Acceptable for loss functions |
| `combined.py` | Medium (~7) | Clean config-driven dispatch |
| `vae.py` | Low (~5) | Well-factored with EncoderHead |
| `lr_controller.py` | Medium (~10) | MetricBasedLR has multiple interacting gates |
| `train.py` | **High (~25)** | `train()` function is 250+ lines, should be decomposed |

## Fake Data / Mock Contamination Report

**No contamination detected.** The codebase exclusively uses:
- Mathematically generated data (3^9 ternary operations via precomputed LUTs)
- geoopt for hyperbolic geometry (established library, C++ backend)
- Standard PyTorch operations

Test fixtures use `torch.randn` for synthetic inputs, which is appropriate and not contamination.

## ROI Assessment & Recommended Next Actions

### Priority 1: Fix Training-Blocking Bugs (Estimated: 2-4 hours)

1. **Fix VAE-B gradient flow** -- Either:
   - (A) Route `z_B_hyp` through losses alongside `z_A_hyp` (true dual VAE), OR
   - (B) Remove VAE-B entirely if it's not needed (save 50% parameters, faster training)
   - Recommendation: Option (A) if the dual-VAE hypothesis is core to the research. Option (B) if pragmatism wins.

2. **Fix max_radius saturation** -- Two complementary fixes:
   - Scale encoder output norms down (e.g., multiply by 0.1) so expmap0 produces radii in [0.1, 0.9] instead of saturating at 0.999
   - Or: reduce max_radius to 0.8 and add tangent space scaling
   - Or: use a learned scaling layer before expmap0 instead of identity-init residual
   - The identity-init is the root cause -- consider initializing tangent_net to SCALE DOWN inputs

3. **Fix config key mismatch** -- Change v6.yaml `radial_weight: 5.0` to `weight: 5.0`

### Priority 2: Add Missing KL Divergence (Estimated: 1-2 hours)

4. **Wire HyperbolicKLDivergence into CombinedLoss** -- The class exists, just needs integration:
   - Add KL computation in `CombinedLoss.forward()` using `mu`, `logvar`, `z_hyp` from model output
   - Add config support for `hyperbolic_kl.enabled: true` in CombinedLoss
   - Consider low beta (0.01-0.1) to avoid dominating hierarchy losses

### Priority 3: Clean Up Dead Config (Estimated: 1 hour)

5. **Remove or wire dead config keys** -- Especially:
   - Remove `zero_structure` block (no implementation exists)
   - Remove `richness_weight`, `annealing`, dead `model.*` keys
   - Or: implement the missing features if intended

### Priority 4: Add Critical Integration Tests (Estimated: 3-4 hours)

6. **Add integration test**: model forward -> loss -> backward -> verify all intended parameters receive gradients
7. **Add config validation test**: parse v6.yaml and verify all keys are consumed by code
8. **Add saturation test**: verify embeddings span the target radius range after initialization

### Automation ROI

- **Config validation linting**: Write a one-time script that compares YAML keys against code `cfg.get()` calls. Prevents future dead config drift. Estimated cost: 2 hours. Ongoing savings: prevents all config key bugs.
- **Gradient flow CI check**: Add a pytest test that runs one forward+backward pass and asserts all intended parameter groups receive non-zero gradients. Estimated cost: 1 hour. Catches VAE-B-style dead code immediately.

## Delegation Log

| Task | Delegated To | Outcome |
|------|-------------|---------|
| Test execution | pytest (local) | 269/269 passed |
| Forward pass smoke test | Python script (local) | Revealed critical bugs C1, C2 |
| Mathematical verification | Python script (local) | All math verified correct |
| Config analysis | Python script (local) | Revealed C3, C5, C6 |

## Audit Limitations

1. **No actual training run performed**: The audit verified the pipeline can execute a single forward+backward pass but did not run full training. A multi-epoch training test would reveal whether the saturation issue self-corrects over time (it might, via gradients through the clamping operation, but slowly).

2. **No GPU testing**: All tests ran on CPU. GPU-specific issues (device mismatches in manifold cache, CUDA determinism, AMP interactions) were not tested.

3. **No performance profiling**: Memory usage on the target RTX 3050 6GB was not measured. The float64 precision requirement doubles memory usage vs float32.

4. **No mutation testing**: mutmut/Stryker was not applied due to the ~5000-line codebase size and the more critical need to identify architectural bugs first. Recommended for future audit once critical bugs are fixed.

5. **No security audit**: `torch.load(weights_only=False)` in checkpoint loading allows arbitrary code execution via crafted checkpoints. This is a known risk but acceptable for a research codebase.
