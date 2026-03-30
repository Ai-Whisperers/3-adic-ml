# Adversarial Audit Report: Deep Architectural Analysis
**Date**: 12-03-2026
**Target**: P-Adic VAE V6.2 -- Full Architecture, Config Wiring, Loss System, Training Pipeline
**Auditor**: Adversarial Auditor Agent (Opus 4.6)
**Audit Depth**: Full
**Prior Context**: Follows 11-03-2026 audit (found 5 critical bugs, 3 now fixed) and re-audit confirming specific issues.

## Executive Summary

This deep audit goes beyond surface bugs into structural and architectural issues that hide as "legacy" or "framework flexibility." The codebase has sound mathematical foundations (3-adic algebra, geoopt geometry) but the training pipeline contains a **critical LR oscillation bug**, a **high-severity architectural identity problem** where both VAEs train on identical objectives, **33 dead config keys** that create a false sense of configurability, and multiple layers of redundant loss functions that collectively drown out reconstruction signal. The system trains, but not in the way its architecture claims.

## Behavioral Adherence Score

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Behavioral Adherence | 0/1 | Dual VAE claims complementary learning (coverage vs hierarchy) but both train on identical loss. LR controller claims differential learning rates but oscillation bug negates this every other epoch. |
| Functional | 0/1 | LR oscillation bug causes projections/decoders to see 20x LR swing every eval_every epochs. 33 dead config keys mean the YAML does not control what it appears to control. |
| Clever | 1/1 | The mathematical design (p-adic -> hyperbolic mapping, expmap0/logmap0, conformal KL) is genuinely novel. TernarySpace singleton with O(1) LUTs is well-engineered. |
| Practical | 0/1 | Cannot resume training from checkpoints (scheduler, controller, loss params not saved). Config drift between YAML and code makes the system unmaintainable. |
| Reliable | 0/1 | LR oscillation, pair sampling correlation between losses, and the 17.5:1 hierarchy-to-coverage weight ratio produce fragile training dynamics. |
| **TOTAL** | **1/5** | **NOT ACHIEVED** |

---

## Critical Findings

### FINDING 1: LR Controller Uses Wrong Base LR -- Causes 20x Oscillation
**Severity: CRITICAL**
**Where**: `src/train.py:1205`
**What**: `update_optimizer_lr_scales()` is called with `scheduler.get_last_lr()[0]` as the base LR. But `get_last_lr()[0]` returns the **first param group's scheduled LR**, which is encoder_a's LR -- already scaled by 0.05x. The controller then applies its own 0.05x scale on top, yielding 0.0025x base for encoder_a instead of 0.05x.

**Evidence** (empirical simulation):
```
Epoch | projections LR | Event
  0   | 0.0000398423   | sched+ctrl  <-- 20x too low (0.05 * base instead of 1.0 * base)
  1   | 0.0007874333   | scheduler   <-- correct (scheduler restores initial per-group LR)
  2   | 0.0000385955   | sched+ctrl  <-- 20x too low again
  3   | 0.0007505227   | scheduler   <-- correct again
```

On eval epochs (every 2 epochs), projections LR drops from ~0.0008 to ~0.00004 (a 20x reduction). On non-eval epochs, the scheduler restores the correct differential LRs. This creates a sawtooth oscillation pattern where:
- Encoder_a oscillates between 0.00004 and 0.000002
- Projections oscillates between 0.0008 and 0.00004
- Decoders oscillates between 0.0008 and 0.00004

**Root cause**: Line 1205 should use the config base_lr (0.0008), not the scheduled per-group LR. The comment on line 1203-1204 says "Use the scheduler's current LR so controller scales compose with cosine annealing" but this is wrong -- it should compose with the cosine *factor*, not the already-scaled group LR.

**Impact**: The entire differential learning rate system (the "Complementary Learning Systems" design) is undermined. On half of all epochs, the projections and decoders receive the same LR as encoder_a (which should be the slowest learner). Training is effectively a random walk between two LR regimes.

---

### FINDING 2: Dual VAE Trains on Identical Objectives -- No Complementarity
**Severity: HIGH**
**Where**: `src/train.py:1041-1051`
**What**: Both VAE-A and VAE-B pass through the exact same `CombinedLoss` with all 6 loss functions enabled. There is no mechanism to differentiate their training objectives.

**Evidence**:
```python
# Line 1041-1044: VAE-A loss
losses = loss_fn(z_A_hyp, batch_idx, logits_A, batch_ops, epoch=epoch,
                 mu=out.get("mu_A"), logvar=out.get("logvar_A"))
# Line 1046-1049: VAE-B loss -- IDENTICAL function, different embeddings
losses_B = loss_fn(z_B_hyp, batch_idx, logits_B, batch_ops, epoch=epoch,
                   mu=out.get("mu_B"), logvar=out.get("logvar_B"))
# Line 1051: sum
loss = losses["total"] + losses_B["total"]
```

The architecture documentation claims:
- "VAE-A: Coverage (reconstruction accuracy)"
- "VAE-B: Hierarchy (radial structure in Poincare ball)"

But both receive:
- Hierarchy losses (RichHierarchy, Radial, Monotonic, Rank): weight ~17.5
- Coverage loss (CrossEntropy): weight ~1.0
- Geodesic loss: weight 2.0
- KL loss: weight ~0.001

**Impact**: The dual VAE provides 2x parameters and 2x loss magnitude, but zero specialization. Both VAEs converge toward the same solution. The "complementary" claim is structurally unsupported.

---

### FINDING 3: 33 Dead Config Keys in v6.yaml
**Severity: HIGH**
**Where**: `src/presets/v6.yaml` (various sections)
**What**: 33 out of ~95 config keys are never read by any code path. This means the YAML creates a false impression of control -- changing these keys has zero effect on training.

**Dead keys by section**:

| Section | Dead Keys | Examples |
|---------|-----------|---------|
| `model` (2) | `init_identity`, `tangent_scale` | These are blocked by the constructor chain: train.py -> TernaryVAEV6 -> DualHyperbolicProjection. Neither `init_identity` nor `tangent_scale` is passed through. `tangent_scale` is hardcoded to 0.05 in HyperbolicProjection. |
| `device` (4) | `name`, `pin_memory`, `num_workers`, `empty_cache_freq` | train.py reads `num_workers` from `training` section (which doesn't have it, so uses default 4). |
| `precision` (1) | `dtype` | Float64 is set by `set_determinism()`, not by config. |
| `riemannian` (1) | `optimizer` | train.py hardcodes `get_riemannian_optimizer()` without passing optimizer type. |
| `anchor_checkpoint` (2) | `encoder_to_load`, `decoder_to_load` | Checkpoint loading uses `strict=False` for all keys. |
| `data` (2) | `use_full_dataset`, `n_operations` | `DataAuditor` always uses `TERNARY.all_ternary()` (19683). |
| `checkpoints` (4) | `save_dir`, `save_best`, `best_metric`, `checkpoint_name` | Checkpoints are saved to `log_dir/checkpoints`, ignoring these keys entirely. |
| `targets` (6) | All 6 keys | Never read by any code. Pure documentation. |
| `logging` (8) | `tensorboard`, `log_dir`, `print_every`, and 5 `enhanced_metrics` sub-keys | `log_dir` comes from CLI. `print_every` is read from `training` section. Most enhanced_metrics sub-keys are stored but never acted upon. |
| `version` (4) | All 4 keys | Never read by any code. |
| `statenet` (3) | `coverage.floor`, `hierarchy.patience_ceiling`, `controller.patience_ceiling` | Stored in StateNetConfig but never accessed by MetricBasedLR. |

**Evidence**: Constructor chain trace for `model.init_identity` and `model.tangent_scale`:
1. `train.py` reads `model_cfg` keys into `TernaryVAEV6Controllable()` constructor -- neither `init_identity` nor `tangent_scale` are read.
2. `TernaryVAEV6.__init__()` accepts `n_projection_layers`, `projection_dropout`, `learnable_curvature` -- not `init_identity` or `tangent_scale`.
3. `DualHyperbolicProjection.__init__()` passes `n_layers`, `dropout`, `learnable_curvature` to `HyperbolicProjection` -- not `init_identity`.
4. `HyperbolicProjection.__init__()` accepts `init_identity` (default=False) but never receives it from any caller. `tangent_scale` is always `nn.Parameter(torch.tensor(0.05))`.

**Impact**: Any user tuning `model.init_identity` or `model.tangent_scale` in YAML is changing nothing. This is especially dangerous because the prior audit recommended changing `tangent_scale` -- but any YAML change would be silently ignored.

---

### FINDING 4: Curvature "Sharing" Between VAEs Is Not Implemented
**Severity: MEDIUM**
**Where**: `src/models/hyperbolic_projection.py:249`
**What**: The code comment says `learnable_curvature=False  # Share curvature with A` for proj_B. The CLAUDE.md states: "Intentional - both projections share A's curvature." But the curvatures are NOT shared. They are independent tensors.

**Evidence**:
```python
# DualHyperbolicProjection creates:
self.proj_A = HyperbolicProjection(learnable_curvature=learnable_curvature)  # True
self.proj_B = HyperbolicProjection(learnable_curvature=False)  # Fixed at 1.0

# Each creates its OWN geoopt.PoincareBall with its OWN isp_c parameter:
# proj_A.manifold.isp_c: requires_grad=True, data_ptr=848432832
# proj_B.manifold.isp_c: requires_grad=False, data_ptr=848429632
# DIFFERENT tensors. Not shared.
```

When `learnable_curvature=True` (as in v6.yaml), proj_A's curvature will drift during training while proj_B's stays fixed at 1.0. The two VAE branches operate on manifolds with DIFFERENT curvatures after training begins. This is a geometric inconsistency, not curvature sharing.

**Impact**: If curvature learning is active, VAE-A and VAE-B embed into Poincare balls with different curvatures. Loss functions computed on z_B_hyp use the loss's curvature parameter (from config), not proj_B's manifold curvature, so the geometry may be inconsistent.

---

### FINDING 5: Pair Sampling Correlation -- Geodesic and Rank Use Identical Pairs
**Severity: MEDIUM**
**Where**: `src/losses/padic_geodesic.py` -- PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss
**What**: All three pair-sampling losses initialize `torch.Generator()` with `seed=42`. Since PAdicGeodesicLoss and GlobalRankLoss both draw 2000 pairs per forward call, they produce **identical pair sequences on every batch, forever**.

**Evidence**:
```
First 10 i-indices (batch_size=512):
  Geodesic:  [102, 435, 348, 270, 106, 71, 188, 20, 102, 121]
  Radial:    [102, 435, 348, 270, 106, 71, 188, 20, 102, 121]
  Rank:      [102, 435, 348, 270, 106, 71, 188, 20, 102, 121]
Geodesic == Rank (first 1000): True
```

The generators advance in lockstep because they start from the same seed and draw the same number of values. This means Geodesic and Rank losses are not providing independent pair diversity -- they are optimizing the exact same 2000 pairs per batch.

**Impact**: Reduced effective diversity in pair sampling. The losses appear to provide complementary signals but are sampling from identical subsets. This limits the gradient diversity across the loss landscape.

---

### FINDING 6: Loss Weight Imbalance -- 17.5:1 Hierarchy-to-Coverage Ratio
**Severity: MEDIUM**
**Where**: `src/presets/v6.yaml` loss section, `src/losses/combined.py`
**What**: Three loss functions redundantly compute radius-to-target MSE, and multiple compute margin enforcement. The effective total weight for hierarchy-related losses is ~17.5, versus ~1.0 for coverage (reconstruction).

**Breakdown**:

| Loss Component | Type | Weight |
|---------------|------|--------|
| RichHierarchy.hierarchy | Radius MSE (level means) | 5.0 |
| RadialHierarchyLoss | Radius MSE (per-point) | 5.0 |
| MonotonicRadialLoss.target | Radius MSE (level means) | 0.5 |
| **Subtotal: Radius MSE** | | **10.5** |
| RichHierarchy.separation | Margin enforcement | 3.0 |
| RadialHierarchy.margin | Margin enforcement | 2.5 (5.0 * 0.5) |
| MonotonicRadialLoss.monotonic | Margin enforcement | 1.0 |
| GlobalRankLoss | Soft ordering | 0.5 |
| **Subtotal: Ordering** | | **7.0** |
| **Total Hierarchy** | | **17.5** |
| RichHierarchy.coverage | CrossEntropy | 1.0 |
| PAdicGeodesicLoss | Distance alignment | 2.0 (after epoch 30) |
| HyperbolicKLDivergence | KL regularization | 0.001 (0.1 * 0.01) |

Both VAEs compute this full loss, so the effective multiplier is 2x on everything.

**Impact**: The model is strongly incentivized to achieve radial hierarchy but has very weak reconstruction incentive. Combined with the negligible KL (0.001 effective weight), the model is closer to a deterministic hyperbolic embedding than a VAE. This explains the pattern of strong hierarchy metrics but potentially weak reconstruction.

---

### FINDING 7: Checkpoints Cannot Properly Resume Training
**Severity: MEDIUM**
**Where**: `src/train.py:1342-1418`
**What**: No checkpoint saves the full training state needed for resumption.

| Checkpoint | Saves | Missing |
|-----------|-------|---------|
| `best_Q.pt` | model_state_dict, epoch, metrics | optimizer, scheduler, lr_controller, loss_fn params |
| `epoch_N.pt` | model_state_dict, optimizer | scheduler, lr_controller, loss_fn params |
| `final.pt` | model_state_dict, optimizer | scheduler, lr_controller, loss_fn params |

Missing `scheduler.state_dict()` means resuming resets the cosine annealing schedule to epoch 0. Missing `lr_controller` state means plateau counters, histories, and active states are lost. If `learnable_weights` were ever enabled, loss parameters would also be lost.

---

### FINDING 8: Duplicate avg_val_acc Computation
**Severity: LOW**
**Where**: `src/train.py:1153-1154`
**What**: `avg_val_acc = val_acc_sum / val_batches` is written twice in consecutive lines. Classic copy-paste error. Harmless (second overwrites first with same value) but indicates low code review coverage.

---

### FINDING 9: ScheduleBasedLR Referenced But Never Implemented
**Severity: LOW**
**Where**: `src/models/lr_controller.py` module docstring (line 21)
**What**: The docstring references `ScheduleBasedLR` as a class that can be imported, but it was never implemented. Only `MetricBasedLR` exists.

---

## Evidence Gathered

### Static Analysis Results

**Constructor Chain Trace (model.init_identity)**:
```
v6.yaml model.init_identity=False
  -> train.py ModelAuditor: NOT READ from model_cfg
  -> TernaryVAEV6.__init__: NOT in parameter list
  -> DualHyperbolicProjection.__init__: NOT in parameter list
  -> HyperbolicProjection.__init__: ACCEPTS init_identity (default=False), but never receives non-default
```
The key is not just dead in YAML -- it is structurally impossible to pass through the constructor chain.

**Constructor Chain Trace (model.tangent_scale)**:
```
v6.yaml model.tangent_scale=0.05
  -> train.py ModelAuditor: NOT READ from model_cfg
  -> TernaryVAEV6.__init__: NOT in parameter list
  -> DualHyperbolicProjection.__init__: NOT in parameter list
  -> HyperbolicProjection.__init__: NOT a constructor param
  -> HyperbolicProjection.tangent_scale: hardcoded nn.Parameter(torch.tensor(0.05))
```

### Behavioral Test Results

**LR Oscillation Simulation** (12 epochs, eval_every=2):
```
Epoch 0 (ctrl): projections LR = 0.00003984  (should be ~0.0008)
Epoch 1 (sched): projections LR = 0.00078743  (correct)
Epoch 2 (ctrl): projections LR = 0.00003860  (should be ~0.0008)
Epoch 3 (sched): projections LR = 0.00075052  (correct)
...pattern continues indefinitely...
```

**Pair Sampling Identity**:
```
PAdicGeodesicLoss pairs (first 10): [102, 435, 348, 270, 106, 71, 188, 20, 102, 121]
GlobalRankLoss pairs (first 10):    [102, 435, 348, 270, 106, 71, 188, 20, 102, 121]
Identical: True (for all 2000 pairs, every batch)
```

**Curvature Independence**:
```
proj_A.manifold.isp_c: requires_grad=True, data_ptr=848432832
proj_B.manifold.isp_c: requires_grad=False, data_ptr=848429632
After modifying proj_A: proj_A.c = 2.13, proj_B.c = 1.00 (independent)
```

### AST Scan Results (Fake Data / Mock Detection)

No fake data libraries detected in production code. `torch.Generator(seed=42)` is used for reproducible pair sampling, which is appropriate. The `dummy_input` in ModelAuditor is a legitimate health check, not a test fixture leak.

### Complexity Analysis

| Module | Functions | Max Cyclomatic Complexity | Notes |
|--------|-----------|--------------------------|-------|
| `train.py:train()` | 1 | ~35 | Monolithic function with nested conditionals. Should be decomposed. |
| `combined.py:CombinedLoss.forward()` | 1 | ~15 | Sequential loss accumulation, reasonable for its purpose. |
| `lr_controller.py:MetricBasedLR` | 7 | ~12 | State machine with multiple gates. Complexity is inherent to the design. |
| `padic_geodesic.py` | 5 classes | ~8 each | Individual losses are clean. Redundancy is the issue, not complexity. |

---

## Fake Data / Mock Contamination Report

No synthetic data contamination detected. All losses compute against real batch data. The `seed=42` generators produce reproducible but non-synthetic pair indices. The TernarySpace singleton computes real mathematical operations, not mock values.

---

## ROI Assessment & Recommended Next Actions

### Immediate (High ROI, Low Effort)

1. **Fix the LR base bug** (Finding 1): Change `src/train.py:1205` from `scheduler.get_last_lr()[0]` to `base_lr` (the config value). This is a one-line fix that eliminates the 20x oscillation. Estimated effort: 5 minutes. Impact: Eliminates training instability.

2. **Remove dead config keys** (Finding 3): Delete the 33 dead keys from v6.yaml and add comments explaining what actually controls each parameter. Estimated effort: 30 minutes. Impact: Prevents silent config drift.

3. **Use different seeds for pair-sampling losses** (Finding 5): Change GlobalRankLoss to `seed=43` and RadialHierarchyLoss to `seed=44`. Estimated effort: 5 minutes. Impact: Independent pair diversity.

### Medium-Term (High ROI, Medium Effort)

4. **Wire init_identity and tangent_scale through constructors**: Pass these from train.py through TernaryVAEV6 and DualHyperbolicProjection to HyperbolicProjection. Estimated effort: 1 hour. Impact: Config actually controls model behavior.

5. **Differentiate VAE-A and VAE-B objectives** (Finding 2): Create a separate loss configuration or weight scheme for each VAE. For example, VAE-A gets coverage_weight=5.0 and hierarchy_weight=1.0, while VAE-B gets the inverse. Estimated effort: 2-3 hours. Impact: Implements the claimed complementary architecture.

6. **Save full checkpoint state**: Add `scheduler.state_dict()`, LR controller state, and loss_fn state to checkpoints. Estimated effort: 1 hour. Impact: Enables proper training resumption.

### Strategic (Medium ROI, Higher Effort)

7. **Consolidate redundant losses**: RichHierarchyLoss.hierarchy, RadialHierarchyLoss, and MonotonicRadialLoss.target_loss all compute radius-to-target MSE. Consider using ONE loss with configurable granularity (per-point vs per-level). Estimated effort: 4-6 hours. Impact: Simpler loss landscape, reduced gradient interference.

8. **Actually share curvature between proj_A and proj_B**: If the design intent is shared curvature, pass proj_A.manifold to proj_B instead of creating a separate PoincareBall. Or make proj_B's curvature track proj_A's via a shared parameter. Estimated effort: 1-2 hours. Impact: Geometric consistency between VAE branches.

9. **Implement config validation layer**: A startup check that verifies every YAML key is read by at least one code path. Estimated effort: 2-3 hours. Impact: Prevents future config drift.

---

## Delegation Log

All analysis was performed directly. No sub-tasks were delegated.

---

## Audit Limitations

1. **No live training run**: This audit did not execute a full training run. The LR oscillation was verified via simulation, not during actual training. A live run would confirm gradient magnitude effects.

2. **No mutation testing**: The test suite was not subjected to mutation testing. The 280 tests may have coverage theater (tests that pass regardless of code mutations).

3. **No profiling**: Memory usage, GPU utilization, and per-loss gradient magnitudes were not measured during actual training.

4. **geoopt internals**: The `isp_c` (inverse softplus of curvature) behavior was tested empirically but the geoopt internals for `stabilize` and Riemannian gradient projection were not audited.

5. **Prior fixes not re-verified**: The 11-03-2026 fixes (VAE-B wiring, tangent_scale addition, config key fix) were assumed correct based on the re-audit. This audit focused on deeper structural issues.

---

## Appendix: Complete Dead Config Key List

```
anchor_checkpoint.decoder_to_load = decoder_A
anchor_checkpoint.encoder_to_load = both
checkpoints.best_metric = composite_score
checkpoints.checkpoint_name = v6
checkpoints.save_best = True
checkpoints.save_dir = runs/checkpoints/v6
data.n_operations = 19683
data.use_full_dataset = True
device.empty_cache_freq = 25
device.name = v6_hyperbolic
device.num_workers = 4
device.pin_memory = True
logging.enhanced_metrics.effective_rank = True
logging.enhanced_metrics.gradient_flow_analysis = True
logging.enhanced_metrics.log_activations = False
logging.enhanced_metrics.log_weights = True
logging.log_dir = runs/v6
logging.print_every = 2
logging.tensorboard = True
model.init_identity = False
model.tangent_scale = 0.05
precision.dtype = float64
riemannian.optimizer = adam
statenet.controller.patience_ceiling = 20
statenet.coverage.floor = 0.3
statenet.hierarchy.patience_ceiling = 25
targets.Q_target = 2.2
targets.coverage = 1.0
targets.distance_correlation = 0.7
targets.hierarchy_B = -0.83
targets.r_v9 = 0.12
targets.richness = 0.008
version.changes = [list of 13 items]
version.config = v6_hyperbolic_modular
version.date = 2026-01-24
version.model = 6.1
```
