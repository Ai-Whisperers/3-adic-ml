# Full Codebase Audit — 2026-03-24

**Scope**: All 27 Python files in `src/`, focused on `train.py` (2102 lines)
**Method**: Cross-referenced every function definition against all call sites; checked imports, exports, duplication, dead code, and wiring correctness.

---

## 1. Executive Summary

The codebase is **mature and well-structured** for a research project. The core pipeline (`train.py` → `CombinedLoss` → `padic_geodesic.py` losses → `geometry/poincare.py`) is battle-tested across 15+ training runs. Configuration is properly driven by YAML → Pydantic validation → dataclass injection.

**Findings**: 6 issues (0 critical, 2 medium, 4 low).

---

## 2. Function Call Audit — train.py

Every function defined or called in `train.py` was verified.

### 2.1 Functions Defined in train.py

| Function | Lines | Called From | Status |
|----------|-------|-------------|--------|
| `set_determinism` | 95-128 | `main()` line 2056 | **Active** |
| `get_timestamp` | 131-133 | `main()` line 2067 | **Active** |
| `DataAuditor.__init__` | 144-146 | `main()` line 2032 | **Active** |
| `DataAuditor.prepare_data` | 148-... | `main()` line 2033 | **Active** |
| `compute_accuracy` | 381-401 | Lines 1315, 1388 | **Active** (train + val) |
| `compute_coverage` | 404-425 | Line 1389 | **Active** (val only) |
| `compute_hyperbolic_coverage` | 428-440 | Line 1390 | **Active** (val only) |
| `compute_tree_coherence` | 443-498 | Line 593 (via `compute_hierarchy_metrics`) | **Active** (indirect) |
| `compute_level_stratified_hierarchy` | 501-537 | Line 594 (via `compute_hierarchy_metrics`) | **Active** (indirect) |
| `compute_hierarchy_metrics` | 540-615 | Lines 1410, 1413 | **Active** (both VAEs) |
| `GrokkingDetector` | 633-735 | Lines 1081-1083 | **Active** |
| `train` | 743-1934 | `main()` line 2072 | **Active** (main loop) |
| `main` | 1942-2102 | `__main__` | **Active** |

**All defined functions are called. No dead functions in train.py.**

### 2.2 Functions Called from train.py (External)

| Import | From | Call Sites | Status |
|--------|------|------------|--------|
| `compute_Q` | `src.models.lr_controller` | Line 590 | **Active** |
| `update_optimizer_lr_scales` | `src.models.lr_controller` | Lines 1130, 1207 | **Active** |
| `get_optimizer_grad_stats` | `src.models.lr_controller` | Line 1129 | **Active** |
| `MetricBasedLR` | `src.models.lr_controller` | Line 1098 | **Active** |
| `TrainingMetrics` | `src.models.lr_controller` | Line 1120 | **Active** |
| `TernaryVAEV6Controllable` | `src.models.vae` | Line 850 | **Active** |
| `StateNetConfig` | `src.config` | Line 1090 | **Active** |
| `validate_config` | `src.config.schema` | Line 1996 | **Active** |
| `CombinedLoss` | `src.losses` | Lines 942, 960 | **Active** (2 instances: A+B) |
| `LagrangianDualState` | `src.losses.lagrangian` | Line 1087 | **Active** |
| `get_riemannian_optimizer` | `src.geometry` | Line 900 | **Active** |
| `hyperbolic_radius` | `src.geometry` | Line 432 | **Active** |
| `poincare_distance` | `src.geometry` | Lines 497, 521, 560 | **Active** |
| `HardwareMonitor` | `src.utils` | Line 1036 | **Active** |
| `TensorBoardLogger` | `src.utils` | Line 988 | **Active** |
| `VisualizationPipeline` | `src.utils.visualization` | Lines 1035, 1762 | **Active** |
| `load_checkpoint_compat` | `src.utils.checkpoint` | Line 868 (guarded) | **Active** |
| `get_model_state_dict` | `src.utils.checkpoint` | Not called in train.py | **Unused import** *(low)* |
| `TERNARY` | `src.core` | Lines 465, 522, 561, 1427, 1454, 1497, etc. | **Active** (many sites) |

### 2.3 Unused Import in train.py

**`get_model_state_dict`** (line 87): Imported but never called. It's a utility for extracting state_dict from DataParallel wrappers — not needed since this project doesn't use DataParallel.

---

## 3. Duplication Analysis

### 3.1 Checkpoint Saving — DUPLICATED 3x (Medium)

The checkpoint payload construction is repeated at three locations:

| Location | Purpose | Lines |
|----------|---------|-------|
| Best Q checkpoint | `ckpt_dir / "best_Q.pt"` | 1782-1807 |
| Periodic checkpoint | `ckpt_dir / f"epoch_{epoch}.pt"` | 1850-1869 |
| Final checkpoint | `ckpt_dir / "final.pt"` | 1876-1898 |

All three repeat the same pattern:
```python
payload = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}
if lr_controller is not None:
    payload["controller_state"] = { ... same 5 fields ... }
if dual_state is not None:
    payload["lagrangian_state"] = dual_state.state_dict()
if loss_cfg.get("learnable_weights", False):
    payload["loss_fn_state_dict"] = ...
```

**Risk**: If a new field is added (e.g., visualization config), it must be added in 3 places. Already happened once (lagrangian_state was added to all 3).

**Recommendation**: Extract a `_build_checkpoint_payload()` helper. Not urgent since the code works, but prevents drift.

### 3.2 Radius Computation — DUPLICATED but Intentional

`hyperbolic_radius(z_hyp, c)` is computed in:
- `padic_geodesic.py`: Each of `RadialHierarchyLoss`, `MonotonicRadialLoss`, `RichHierarchyLoss`, `GlobalRankLoss` (4 independent calls)
- `train.py`: `compute_hierarchy_metrics` (1 call), `compute_hyperbolic_coverage` (1 call), `compute_level_stratified_hierarchy` (1 call, but through `poincare_distance` instead of `hyperbolic_radius` — equivalent but inconsistent pattern)

**Verdict**: The 4 loss calls are in separate `.forward()` methods that each need their own radii for gradient flow — this is **intentionally not shared** because they run in the same backward pass but with different computation graphs. The 3 train.py calls are all under `torch.no_grad()` (metrics only). No bug, just architectural consequence.

### 3.3 `compute_level_stratified_hierarchy` uses `poincare_distance(z, origin)` instead of `hyperbolic_radius(z)` (Low)

At line 520-521:
```python
origin = torch.zeros_like(z_hyp)
radii = poincare_distance(z_hyp, origin, c=curvature)
```

This is equivalent to `hyperbolic_radius(z_hyp, c=curvature)` but creates an unnecessary `origin` tensor. Same issue in `compute_hierarchy_metrics` line 559-560. The `hyperbolic_radius` function does the same thing internally but is the canonical API.

### 3.4 `_exponential_target_radii` vs `TERNARY.target_radius` — Separate but Not Duplicated

- `TERNARY.target_radius()` computes linear interpolation: `outer - (outer - inner) * v / max_v`
- `_exponential_target_radii()` computes exponential decay: `inner + (outer - inner) * exp(-alpha * v)`

These are **different formulas** for different use cases. `TERNARY.target_radius` is never called from losses — it's a utility. `_exponential_target_radii` is the actual loss target. Not duplication.

---

## 4. Dead Code Analysis

### 4.1 Dead Import: `exp_map_zero` in `hyperbolic_projection.py` (Low)

```python
# Line 33:
from src.geometry import ManifoldParameter, exp_map_zero
```

`exp_map_zero` is imported but never called. The module uses `self.manifold.expmap(origin, v)` instead (correct — uses the model's own manifold instance for learnable curvature gradient flow). The import has been dead since V6.0 when `self.manifold.expmap()` replaced the cached `exp_map_zero()`.

### 4.2 `StandardKLDivergence` — Never Instantiated (Low)

Defined in `hyperbolic_kl.py`, exported in `__init__.py`, but never instantiated anywhere. `CombinedLoss` always uses `HyperbolicKLDivergence`. This is a comparison baseline class from the paper reference — kept for potential future A/B testing.

### 4.3 `log_manifold_embedding` — Callable but Never Called (Low)

`TensorBoardLogger.log_manifold_embedding()` (line 128, 150 lines of code) logs Euclidean-space embeddings via `add_embedding`. The `VisualizationPipeline` now handles this with correct hyperbolic-metric UMAP 3D embeddings. The old method:
- Uses raw Euclidean coordinates (wrong metric)
- Is never called from `train.py`
- Does contain useful metadata computation (valuation, prefix classes)

**Verdict**: Not causing bugs since it's never called. Could be removed or marked deprecated. The visualization pipeline supersedes it.

### 4.4 Unused Geometry Functions (Intentional — Library Pattern)

The following are exported from `geometry/poincare.py` but never called in production code:
- `project_to_poincare` — available but models use `.clamp()` instead
- `mobius_add` — available for future Möbius-space operations
- `parallel_transport` — available for future transport-based losses
- `geodesic`, `geodesic_interpolation` — available for visualization
- `create_manifold_parameter`, `create_manifold_tensor` — available for future use

**Verdict**: These follow a library pattern — exported for potential use. Tests exist for all of them. Not dead code, just not actively used in the training pipeline.

---

## 5. Wiring Correctness Audit

### 5.1 Loss → CombinedLoss → train.py Wiring

| Loss Class | Enabled in v7_large.yaml | Wired in CombinedLoss | Called in forward() |
|------------|-------------------------|----------------------|---------------------|
| `RichHierarchyLoss` | `enabled: true` | Line 145 | Line 502 |
| `RadialHierarchyLoss` | `enabled: true` | Line 190 | Line 513 |
| `PAdicGeodesicLoss` | `enabled: true` | Line 161 | Line 481 |
| `GlobalRankLoss` | `enabled: true` | Line 204 | Line 523 |
| `MonotonicRadialLoss` | `enabled: true` | Line 214 | Line 533 |
| `HyperbolicKLDivergence` | `enabled: true` | Line 245 | Line 540 |
| `AngularCoherenceLoss` | `enabled: true` | Line 285 | Line 559 |
| `ValuationPriorLoss` | `enabled: false` | Line 263 (guarded) | Line 568 (guarded) |
| `WithinLevelContrastiveLoss` | `enabled: false` | Line 277 (guarded) | Line 578 (guarded) |

**All enabled losses are properly wired. Disabled losses are correctly guarded.**

### 5.2 train.py → Model Wiring

| Component | Created At | Used In Loop |
|-----------|-----------|--------------|
| `TernaryVAEV6Controllable` | Line 850 | Line 1240 (`model(batch_ops)`) |
| `DualHyperbolicProjection` | Created inside VAE | Called via `model.forward()` |
| `MetricBasedLR` | Line 1098 | Line 1119 (`lr_controller.update()`) |
| `RiemannianAdam` | Line 900 | Lines 1237 (`zero_grad`), 1288 (`step`) |
| `GradScaler` | Line 977 | Lines 1280, 1281, 1288, 1289 |
| `CosineAnnealingWarmRestarts` | Line 926 | Line 1346 (`scheduler.step()`) |
| `VisualizationPipeline` | Line 1035 | Line 1762 (`vis_pipeline.run()`) |

**All components properly created and used.**

### 5.3 Config Flow: YAML → Pydantic → Dataclass → Code

```
v7_large.yaml
  ↓ yaml.safe_load()
  ↓ validate_config() → TrainingConfigSchema (Pydantic)
  ↓ StateNetConfig.from_dict() → StateNetConfig (dataclass)
  ↓ MetricBasedLR(sn_config) → LR controller
  ↓ CombinedLoss(loss_cfg) → Loss composition
  ↓ TernaryVAEV6Controllable(model_cfg) → Model
```

**Config flow is correct and complete. All YAML keys map to code.**

---

## 6. Maturity Assessment

### Tier 1 — Battle-Tested (15+ training runs, 280 tests)

| Component | Lines | Tests | Confidence |
|-----------|-------|-------|------------|
| `src/core/ternary.py` | 852 | 57 | **Very High** — immutable singleton, mathematical invariants verified |
| `src/geometry/poincare.py` | 397 | 56 | **Very High** — geoopt backend, double precision, float64 throughout |
| `src/losses/padic_geodesic.py` | 1456 | 64 | **High** — all 8 loss classes tested, gradient flow verified |
| `src/losses/combined.py` | 803 | 37 | **High** — config-driven composition, learnable weights |
| `src/models/vae.py` | 563 | 11 | **High** — dual VAE architecture stable |
| `src/models/hyperbolic_projection.py` | 385 | 11 | **High** — factored mode verified, gradient isolation proven |
| `src/train.py` | 2102 | 0 (integration) | **High** — 15+ full runs, but no unit tests |

### Tier 2 — Mature but Less Exercised

| Component | Lines | Tests | Confidence |
|-----------|-------|-------|------------|
| `src/models/lr_controller.py` | 516 | 0 | **Medium** — works in training but no isolated tests |
| `src/losses/lagrangian.py` | ~200 | 0 | **Medium** — Lagrangian duals active in training, no isolated tests |
| `src/losses/hyperbolic_kl.py` | 204 | included in combined tests | **Medium-High** |
| `src/config/statenet_config.py` | 262 | 0 | **Medium** — dataclass, hard to break |
| `src/config/schema.py` | ~400 | 0 | **Medium** — Pydantic validation, called once at startup |

### Tier 3 — Newest / Least Exercised

| Component | Lines | Tests | Confidence |
|-----------|-------|-------|------------|
| `src/utils/visualization.py` | ~500 | 0 | **Low** — just created, smoke-tested, running in Run 15 |
| `src/utils/scatter_utils.py` | ~60 | 0 | **Low** — tested via loss tests but no isolated tests |
| `src/losses/radius_defaults.py` | ~130 | 0 | **Low** — used by CombinedLoss, no isolated tests |

---

## 7. Findings Summary

### F1: Checkpoint payload duplication (Medium)
- **What**: Checkpoint construction code repeated 3x (best_Q, periodic, final)
- **Where**: `train.py` lines 1782-1807, 1850-1869, 1876-1898
- **Risk**: Drift when adding new checkpoint fields
- **Fix**: Extract `_build_checkpoint_payload()` helper

### F2: Inconsistent radius API usage (Low)
- **What**: `compute_level_stratified_hierarchy` and `compute_hierarchy_metrics` use `poincare_distance(z, zeros_like(z))` instead of `hyperbolic_radius(z)`
- **Where**: `train.py` lines 520-521, 559-560
- **Risk**: None (equivalent computation), just inconsistent
- **Fix**: Replace with `hyperbolic_radius(z_hyp, c=curvature)`

### F3: Dead import `exp_map_zero` (Low)
- **What**: Imported but never called (replaced by `self.manifold.expmap()`)
- **Where**: `hyperbolic_projection.py` line 33
- **Risk**: None
- **Fix**: Remove from import

### F4: `StandardKLDivergence` never instantiated (Low)
- **What**: Defined, exported, never used — a comparison baseline
- **Where**: `hyperbolic_kl.py` line 135
- **Risk**: None (dead code, but intentionally kept)
- **Fix**: None needed (reference implementation)

### F5: `log_manifold_embedding` superseded (Low)
- **What**: `TensorBoardLogger` method that logs Euclidean-space embeddings, never called
- **Where**: `tensorboard_logger.py` line 128
- **Risk**: None (not called)
- **Fix**: Could add `@deprecated` or remove, but not urgent

### F6: Unused import `get_model_state_dict` in train.py (Low)
- **What**: Imported but never called
- **Where**: `train.py` line 87
- **Risk**: None
- **Fix**: Remove from import

---

## 8. Architecture Diagram (Verified Call Graph)

```
main()
 ├── validate_config(yaml) → Pydantic schema
 ├── DataAuditor.prepare_data() → train_ds, val_ds
 ├── TernaryVAEV6Controllable(model_cfg)
 │    ├── EncoderHead × 2 (A, B)
 │    └── DualHyperbolicProjection
 │         ├── HyperbolicProjection (proj_A)
 │         │    ├── tangent_net (nn.Sequential)
 │         │    ├── linear_r (factored mode)
 │         │    └── manifold (geoopt.PoincareBall)
 │         └── HyperbolicProjection (proj_B)
 ├── CombinedLoss(loss_cfg) × 2 (A, B)
 │    ├── RichHierarchyLoss
 │    ├── RadialHierarchyLoss
 │    ├── PAdicGeodesicLoss
 │    ├── GlobalRankLoss
 │    ├── MonotonicRadialLoss
 │    ├── HyperbolicKLDivergence
 │    ├── AngularCoherenceLoss
 │    ├── [disabled] ValuationPriorLoss
 │    └── [disabled] WithinLevelContrastiveLoss
 ├── LagrangianDualState
 ├── MetricBasedLR(StateNetConfig)
 ├── RiemannianAdam (geoopt)
 ├── TensorBoardLogger
 ├── VisualizationPipeline
 │    ├── UMAP 3D (precomputed Poincaré D)
 │    ├── PaCMAP 2D (hyperbolic kNN injection)
 │    ├── TriMAP 2D (precomputed distance matrix)
 │    ├── Poincaré 3D (logmap0 → PCA)
 │    └── Persistent Homology (ripser)
 ├── GrokkingDetector
 └── HardwareMonitor

train() loop:
 ├── Per batch:
 │    ├── model(batch_ops) → outputs dict
 │    ├── loss_fn(z_A, ...) + loss_fn_b(z_B, ...) → total loss
 │    ├── backward + clip_grad_norm_ + optimizer.step()
 │    └── Lagrangian dual violation accumulation
 ├── Per eval_every:
 │    ├── Validation: accuracy, coverage, hyperbolic_coverage
 │    ├── compute_hierarchy_metrics() → Q, hierarchy, dist_corr
 │    │    ├── compute_tree_coherence()
 │    │    └── compute_level_stratified_hierarchy()
 │    ├── ARI per level (K-means on direction vectors)
 │    ├── TensorBoard: 50+ scalar tags
 │    └── VisualizationPipeline.run()
 ├── Per save_every: periodic checkpoint
 └── Per epoch: LR controller update, scheduler step
```

---

## 9. Conclusion

The codebase is well-organized and correctly wired. The 6 findings are all low-to-medium severity — none affect training correctness. The core pipeline (`train.py` → losses → geometry) is the most mature code. The newest addition (`VisualizationPipeline`) is running in production (Run 15) and generating outputs but has no isolated tests yet.

**Priority order for cleanup** (optional):
1. F1: Checkpoint duplication → extract helper (prevents future drift)
2. F2: Inconsistent `poincare_distance(z, origin)` → `hyperbolic_radius(z)` (consistency)
3. F3, F6: Dead imports → remove (trivial)
4. F4, F5: Dead/superseded code → defer (no harm)

---

## 10. Post-Session Findings (2026-03-24)

All F1–F6 from the original audit have been resolved. The following additional findings were identified during scatter refactor and hygiene pass. Status indicated for each.

### Resolved Findings

| ID | File | Issue | Resolution |
|----|------|-------|-----------|
| G1 | `train.py:88` | Imported `level_scatter_mean`, `level_has_data` but only `level_scatter_std` is used in `train.py` | Removed unused imports |
| G2 | `hyperbolic_projection.py:31` | `import torch.nn.functional as F` unused — only appeared in docstring comments (`F.normalize`) | Removed import |
| G3 | `losses/combined.py:33` | `Union` imported from `typing` but never used in annotations | Removed from import |
| G4 | `losses/__init__.py` | `AngularCoherenceLoss` defined in `padic_geodesic.py`, used in `combined.py`, but not exported from `losses/__init__.py` | Added to import and `__all__` |
| G5 | `models/vae.py:313` | `encode()` return type was bare `-> tuple:` — imprecise, no type parameters | Fixed to `-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`; added `Tuple` to typing imports |
| G6 | `losses/radius_defaults.py:111,119,128` | Three `print()` calls fired on every `CombinedLoss.__init__()`: two informational (defaults used, one-source case), one warning (multiple conflicting sources) | Removed two info prints (silent defaults); converted WARNING case to `warnings.warn(..., UserWarning, stacklevel=2)` |
| G7 | `losses/padic_geodesic.py:1146` | `ValuationPriorLoss.forward()` used a 10-iteration Python mask loop to build per-level mean norms and gap tensors — ran inside the differentiable forward pass | Replaced with `level_scatter_mean` + vectorized gap computation; per-level dicts still populated from scatter results |
| G8 | `train.py:524` | `compute_level_stratified_hierarchy()` used per-level `radii[mask].std()` loop under `no_grad` | Replaced with `level_scatter_std` single scatter pass |
| G9 | `train.py:874` | `create_stratified_dataloader()` used per-level mask assignment loop for `sqrt(1/count)` weights | Replaced with vectorized `level_weights[train_valuations.long()]` indexing — zero Python-level iteration |

### Open Findings (Not Yet Fixed)

| ID | Severity | File | Issue | Recommendation |
|----|----------|------|-------|----------------|
| G10 | Low | `src/c/ternary_hash.c` + `.so` | C extension providing FNV-1a → balanced ternary hashing. Not imported anywhere in `src/`. Untracked in git. Purpose: bridge arbitrary byte inputs (strings, codons, integers) into the `{-1,0,1}^9` ternary space — intended for the `ultrametric-antigen-AI` sister project. Well-documented, ~116 lines, builds with `gcc -O2 -shared -fPIC`. | **Keep the `.c` source** — it is a future bridge module, not dead code. Add `src/c/*.so` to `.gitignore` (binary should not be committed). Track in git as `src/c/ternary_hash.c` only. Consider a thin `src/bridge/ternary_hash.py` ctypes wrapper when the antigen pipeline integration begins. |
| G13 | Info | `src/train.py` | 66+ `print()` calls throughout. Acceptable for a CLI training script, but inconsistent with Python's `logging` module idiom. | No immediate action required. If the project grows a library surface (e.g., `train()` called from notebooks or other scripts), migrate to `logging.getLogger(__name__)`. |

### Scatter Refactor Summary (Phase 1)

The scatter refactor is **functionally complete**. All four differentiable loop sites in the loss system now use vectorized operations:

| Site | Before | After |
|------|--------|-------|
| `MonotonicRadialLoss.forward()` | Python loop (pre-existing) | `level_scatter_mean` (was already done) |
| `RichHierarchyLoss.forward()` | Python loop (pre-existing) | `level_scatter_mean` + `scatter_add_` (was already done) |
| `ValuationPriorLoss.forward()` | 10-iter Python mask loop | `level_scatter_mean` vectorized |
| `compute_level_stratified_hierarchy()` | 10-iter `radii[mask].std()` | `level_scatter_std` |
| `create_stratified_dataloader()` | 10-iter weight assignment | `level_weights[vals.long()]` indexing |

`torch-scatter` is available as an optional accelerator (commented in `requirements.txt`). The pure-torch fallback in `scatter_utils.py` is differentiable and correct in both modes. Sites not refactored (`GlobalRankLoss`, `AngularCoherenceLoss`) do per-level subsampling/prefix-specific work that is inherently sequential.

---

## 11. Config and Visualization Hardening (2026-03-24, later pass)

### Additional Resolved Findings

| ID | File | Issue | Resolution |
|----|------|-------|-----------|
| G11 | `src/geometry/poincare.py:336` | `get_riemannian_optimizer()` lacked an explicit return type | Added `RiemannianOptimizer` alias and annotated the factory return type |
| G12 | `src/models/__init__.py` | `src.models` relied on implicit exports | Added explicit `__all__` for IDE support and cleaner public-surface documentation |
| G14 | `tests/test_visualization.py` | Visualization pipeline had no dedicated unit coverage | Added focused tests for stratified subsampling, hyperbolic distance matrix symmetry, tangent-space PCA output, runtime input validation, and pipeline step control flow |
| G15 | `src/config/schema.py` | Schema validation silently ignored live preset fields such as `visualization`, `anchor_checkpoint`, `version`, and several loss/training variants | Rebuilt the schema around the actual runtime config surface, including visualization, checkpoints, metadata, scheduler phases, grokking config, and legacy-compatible aliases |
| G16 | `src/train.py:2022` | `train.py` validated YAML but then continued with the raw dict, so defaults/normalization from Pydantic were never applied | Training now uses the validated+normalized config as the runtime source of truth |
| G17 | `src/utils/visualization.py` | VisualizationPipeline accepted an unvalidated dict and deferred shape/config failures into optional backend code | Added `VisualizationRuntimeConfig` plus explicit tensor-shape/runtime validation before backend execution |

### Recommended Non-Breaking Abstractions

These are not urgent fixes, but they are the most leverage-positive patterns for future maintainability:

| ID | Scope | Proposal | Benefit |
|----|-------|----------|---------|
| A1 | Entry points | Introduce a single `load_training_config(path)` helper that performs YAML load, schema validation, normalization, and emits one canonical dict/model | Makes config handling grep-able and prevents future scripts from reintroducing raw-dict drift |
| A2 | Runtime modules | Use small runtime config objects like `VisualizationRuntimeConfig` for subsystems with non-trivial local behavior (`TensorBoardLogger`, checkpoint/resume policy, hardware monitor) | Localizes validation and makes module-specific debugging possible without reading `train.py` end-to-end |
| A3 | Diagnostics | Add narrow `validate_*` helpers per subsystem (`validate_visualization_inputs`, `validate_resume_checkpoint`, `validate_loss_config`) | Turns late numeric/backend failures into early, source-local errors with actionable messages |
| A4 | Public surfaces | Keep explicit `__all__` and stable factory/type aliases for import hubs (`src.models`, `src.geometry`, `src.config`) | Lowers cognitive load when navigating the repo and makes the intended public API obvious |
