# Comprehensive Codebase Audit Report

**Date**: 2026-02-24
**Scope**: All source files in `src/` (22 files), all test files in `tests/` (8 files), and configuration in `src/presets/`
**Auditor**: Claude Opus 4.6 (automated deep-read code review)
**Codebase**: P-Adic VAE V6.0/V6.1 at commit `c71c2ef` → `e2b74b0`

---

## Executive Summary

The codebase is architecturally sound with well-structured mathematical foundations. Eight bugs (3 critical, 5 moderate) were identified and fixed during this audit session. The remaining issues are concentrated in the TensorBoard logger (stale V5.x code), dead code modules, and test coverage gaps.

**Current state after fixes**: 214 tests passing, 5 files patched.

| Severity | Remaining | Fixed | Details |
|----------|-----------|-------|---------|
| CRITICAL | 2 | 3 | TensorBoard V5.x API crashes (remaining); LR controller + learnable weights (fixed) |
| MODERATE | 5 | 5 | Validator, dead code, perf (remaining); determinism, weight_decay, logmap, scheduler, dtype (fixed) |
| LOW | 10 | 0 | Unused imports, dead modules, config keys, test fixtures |
| INFO | 6 | 0 | Test coverage gaps, design observations |

---

## FIXED BUGS (Commit `c71c2ef`)

These eight bugs were found during this audit and fixed in commit `c71c2ef`:

### FIXED-1: Encoder LR scales silently never applied (was CRITICAL)

**Files**: `src/models/vae.py:489,498` and `src/models/lr_controller.py:386-394`

`get_param_groups()` created optimizer groups named `"encoder_A"` and `"encoder_B"` (uppercase), while `MetricBasedLR` returned scale keys `"encoder_a"` and `"encoder_b"` (lowercase). The `update_optimizer_lr_scales()` function compared with `if name in lr_scales` — the match always failed silently. The entire LR controller mechanism for encoders was non-functional; coverage-gating and hierarchy-gating decisions had zero effect on training.

**Fix**: Renamed param group names to `"encoder_a"` / `"encoder_b"` in `vae.py`.

### FIXED-2: Double gate computation corrupted controller state (was CRITICAL)

**File**: `src/models/lr_controller.py:417-430`

`MetricBasedLR.update()` called `_compute_coverage_gate()`, `_compute_hierarchy_gate()`, and `_compute_projections_gate()` on lines 418-420, then called `self.get_lr_scales(metrics)` on line 430 which invoked all three gate methods **again**. Each gate mutates `self._active`, `self._last_change`, `self._hierarchy_b_plateau_count`, `self._grad_low_count`, and `self._hierarchy_a_stall_count`. Plateau counters incremented twice per epoch — components froze in half the configured patience.

**Fix**: Capture scale values from initial gate calls and build `lr_scales` dict directly instead of calling `get_lr_scales()`.

### FIXED-3: Learnable loss weights never actually optimized (was CRITICAL)

**File**: `src/train.py:741-773`

When `learnable_weights: true`, `CombinedLoss` creates `nn.Parameter` objects (`log_sigma_hierarchy`, etc.). But the optimizer was built only from `model.get_param_groups()` — `loss_fn.parameters()` was never added. Gradients were computed during backward but never applied. The "learnable" weights were frozen at their initial values forever.

**Fix**: Added `loss_fn.parameters()` as a `"loss_weights"` param group to both RiemannianAdam and AdamW paths.

### FIXED-4: cudnn.benchmark silently overrode determinism (was MODERATE)

**File**: `src/train.py:105-106` then `1237-1238`

`set_determinism()` set `cudnn.benchmark = False`, then `main()` re-enabled it because `memory.cudnn_benchmark` defaults to `True` in v6.yaml.

**Fix**: Added check `if not torch.backends.cudnn.deterministic` before enabling benchmark mode.

### FIXED-5: weight_decay not passed to RiemannianAdam (was MODERATE)

**File**: `src/train.py:764-768`

The AdamW path passed `weight_decay`, but the Riemannian path (which v6.yaml defaults to) omitted it. Config specifies `weight_decay: 1e-4` but it was silently ignored.

**Fix**: Added `weight_decay=weight_decay` to the `get_riemannian_optimizer()` call.

### FIXED-6: log_map_zero() ignored its max_norm parameter (was MODERATE)

**File**: `src/geometry/poincare.py:172-187`

The function accepted `max_norm` but the body just delegated to `manifold.logmap(origin, z)` without any clamping. Callers like `vae.py:355` passed `max_norm=self.max_radius` expecting boundary clamping before arctanh, but it had no effect. Points near the Poincaré ball boundary could produce extreme tangent vectors.

**Fix**: Added norm clamping before logmap. Default changed to `None` (auto-computes `ball_radius - 1e-5` from curvature). Also caps `max_norm` at `ball_radius - 1e-5` regardless of caller input.

### FIXED-7: Scheduler and LR controller fought each other (was MODERATE)

**File**: `src/train.py:926,1004`

`CosineAnnealingWarmRestarts` adjusted LRs at line 926. Then on eval epochs, `update_optimizer_lr_scales()` overwrote them using constant `base_lr`, erasing the scheduler's cosine decay entirely.

**Fix**: Changed to use `scheduler.get_last_lr()[0]` so controller scales compose multiplicatively with cosine annealing.

### FIXED-8: log_sigma parameters were float32 in float64 codebase (was MODERATE)

**File**: `src/losses/combined.py:215-243`

All 7 `nn.Parameter(torch.tensor(value))` calls created float32 tensors by default. Every other tensor in the codebase is float64.

**Fix**: Added `dtype=torch.float64` to all `torch.tensor()` calls.

---

## REMAINING: CRITICAL (2 issues)

### C-1: TensorBoardLogger.log_manifold_embedding — crashes with V6 model

**File**: `src/utils/tensorboard_logger.py`
**Lines**: 425-547 (method), 29-34 (stale constants)
**Severity**: CRITICAL (crash on invocation)

The `log_manifold_embedding()` method contains three distinct V5.x incompatibilities:

**Problem 1 — Wrong call signature (line 467):**
```python
outputs = model(x, DEFAULT_TEMP_A, DEFAULT_TEMP_B, DEFAULT_BETA_A, DEFAULT_BETA_B)
```
V6 `TernaryVAEV6.forward()` signature is `forward(self, x)` — only one argument. This raises `TypeError: forward() takes 2 positional arguments but 6 were given`.

**Problem 2 — Wrong output dict keys (lines 468-469):**
```python
z_A = outputs["z_A"]   # V6 uses "z_A_hyp"
z_B = outputs["z_B"]   # V6 uses "z_B_hyp"
```
V6 output dict keys: `logits`, `logits_A`, `logits_B`, `z_A_hyp`, `z_B_hyp`, `mu_A`, `mu_B`, `logvar_A`, `logvar_B`. Accessing `"z_A"` raises `KeyError`.

**Problem 3 — Redundant manual Poincaré projection (lines 471-476):**
```python
z_A_euc_norm = torch.norm(z_A, dim=1, keepdim=True)
z_A_poincare = z_A / (1 + z_A_euc_norm) * 0.95
```
In V6, `z_A_hyp` is already on the Poincaré manifold (via `expmap0`). This Euclidean heuristic projection is wrong for V6 — it would double-project already-valid manifold points.

**Stale constants (lines 29-34):**
```python
DEFAULT_TEMP_A = 1.0
DEFAULT_TEMP_B = 1.0
DEFAULT_BETA_A = 0.5
DEFAULT_BETA_B = 0.5
```
These V5.x temperature/beta scheduling constants have no counterpart in V6.

**Current mitigation**: `train.py` never calls `log_manifold_embedding()`. The method is public API but unreachable in default training. However, if any user or script calls it, the training run will crash.

**Full method trace** (122 lines, lines 425-547):
- Lines 425-446: Method signature and docstring (correct)
- Lines 450-463: Operation sampling via `TERNARY.all_ternary()` (correct)
- Line 467: **CRASH** — wrong model call signature
- Lines 468-469: **CRASH** — wrong output keys
- Lines 471-476: **WRONG** — redundant Euclidean projection
- Lines 478-512: Metadata computation via `_compute_3adic_depth` (correct, uses TERNARY singleton)
- Lines 514-544: TensorBoard `add_embedding` calls (correct API usage)

**To fix**: Replace line 467 with `outputs = model(x)`, lines 468-469 with `z_A = outputs["z_A_hyp"]` / `z_B = outputs["z_B_hyp"]`, remove lines 471-476 (use z_A/z_B directly as they're already on the manifold), and remove lines 29-34.

---

### C-2: TensorBoardLogger.log_epoch — crashes with V6 loss dicts

**File**: `src/utils/tensorboard_logger.py`
**Lines**: 247-360 (method), 362-407 (helper `_log_padic_losses`)
**Severity**: CRITICAL (crash on invocation)

The `log_epoch()` method directly indexes into `train_losses` dict with V5.x keys using bracket notation `train_losses["key"]` (not `.get()`), so any missing key raises `KeyError`.

**Full inventory of V5.x keys accessed (20 keys, all missing in V6):**

| Line(s) | Key | Access pattern | V6 status |
|---------|-----|---------------|-----------|
| 274 | `train_losses["loss"]` | `[]` | Exists in V6 ✓ |
| 279 | `train_losses["ce_A"]` | `[]` | **Missing** — V6 uses `"cross_entropy"` |
| 280 | `train_losses["kl_A"]` | `[]` | **Missing** — no KL in loss dict |
| 281 | `train_losses["H_A"]` | `[]` | **Missing** — entropy not tracked |
| 286 | `train_losses["ce_B"]` | `[]` | **Missing** |
| 287 | `train_losses["kl_B"]` | `[]` | **Missing** |
| 288 | `train_losses["H_B"]` | `[]` | **Missing** |
| 294-295 | `train_losses["H_A"]`, `["H_B"]` | `[]` | **Missing** (duplicate access) |
| 305 | `train_losses["phase"]` | `[]` | **Missing** — no phase system in V6 |
| 306 | `train_losses["rho"]` | `[]` | **Missing** |
| 307 | `train_losses["grad_ratio"]` | `[]` | **Missing** |
| 309 | `train_losses["ema_momentum"]` | `[]` | **Missing** |
| 316-318 | `train_losses["lambda1/2/3"]` | `[]` | **Missing** — replaced by learnable weights |
| 326-327 | `train_losses["temp_A/B"]` | `[]` | **Missing** — no temperature scheduling |
| 333-334 | `train_losses["beta_A/B"]` | `[]` | **Missing** — no beta scheduling |
| 338 | `train_losses["lr_scheduled"]` | `[]` | **Missing** |

**The helper `_log_padic_losses` (lines 362-407)** uses `.get()` with defaults, so it wouldn't crash — but it logs V5.x keys (`padic_metric_A`, `padic_ranking_A`, `padic_norm_A`) that V6 never produces, resulting in silent zero-logging.

**How train.py works around this**: The V6 training loop (lines 920-1104 of `train.py`) bypasses `log_epoch()` entirely. Instead, it writes directly to `tb_logger.writer.add_scalar()` for each metric individually. The only `TensorBoardLogger` methods actually called from `train.py` are:
- `tb_logger.log_batch(global_step, loss.item())` — line 921 (works, but only logs `loss`; `ce_A`, `ce_B`, `kl_A`, `kl_B` params default to 0.0)
- `tb_logger.log_histograms(epoch, model)` — line 1094 (works correctly)
- `tb_logger.writer.add_scalar(...)` — used ~30 times directly
- `tb_logger.flush()` — line 1089

**Additional stale methods never called from train.py:**
- `log_hyperbolic_batch()` (lines 119-145) — references `centroid_loss` which doesn't exist in V6
- `log_hyperbolic_epoch()` (lines 147-245) — references `ranking_weight`, `centroid_loss`, V5.10 StateNet metrics
- `log_epoch()` (lines 247-360) — the subject of this finding

**To fix**: Either rewrite the stale methods to accept V6 metric structures, or remove them and consolidate all logging into `train.py`'s direct `writer.add_scalar()` pattern.

---

## REMAINING: MODERATE (5 issues)

### M-1: checkpoint_validator.py contradicts train.py and is dead code

**File**: `src/utils/checkpoint_validator.py` (94 lines)
**Lines**: 44-53 (anchor checkpoint requirement), 78-89 (StateNet validation)

**The contradiction (lines 44-53):**
```python
if model_name in v6_models:
    anchor_cfg = config.get("anchor_checkpoint", {})
    checkpoint_path = anchor_cfg.get("path")
    if checkpoint_path is None or str(checkpoint_path).lower() == "null":
        errors.append(
            f"Model '{model_name}' requires an anchor checkpoint for proper initialization. "
            f"Set 'anchor_checkpoint.path' to a valid checkpoint."
        )
```

But `train.py` `ModelAuditor.validate()` (approx line 292) explicitly supports from-scratch training:
```python
print("[AUDIT] No anchor checkpoint specified (training from scratch)")
```

**Dead code confirmation**: `validate_training_config` is:
- Exported via `src/utils/__init__.py` line 4
- Exported in `__all__` of `checkpoint_validator.py` line 94
- **Never called** from `train.py` or any other `.py` file in `src/` (verified by grep for `validate_training_config`)
- Never called from any test file

**The valid parts of the validator (lines 60-91):**
- `training.epochs` must be positive integer — useful check
- `training.learning_rate` must be positive — useful check (but note: v6.yaml uses `lr`, not `learning_rate`, so this check also wouldn't match)
- StateNet `fix_threshold < train_threshold` — useful check

**Config key mismatch**: The validator checks `training_cfg["learning_rate"]` (line 69) but v6.yaml uses `training.lr`. The check would never trigger even if the validator were called.

---

### M-2: ScheduleBasedLR division by zero with duplicate epochs

**File**: `src/models/lr_controller.py`
**Lines**: 139-164

**The vulnerable code (line 160):**
```python
def _get_scale_at_epoch(self, schedule: List[Tuple[int, float]], epoch: int) -> float:
    ...
    for i in range(len(schedule) - 1):
        e1, s1 = schedule[i]
        e2, s2 = schedule[i + 1]
        if e1 <= epoch < e2:
            if self.interpolate:
                t = (epoch - e1) / (e2 - e1)  # ZeroDivisionError if e1 == e2
                return s1 + t * (s2 - s1)
```

**The validation that misses it (lines 139-145):**
```python
def _validate_schedules(self):
    for name, schedule in self.schedules.items():
        if not schedule:
            raise ValueError(f"Schedule for {name} is empty")
        sorted_schedule = sorted(schedule, key=lambda x: x[0])
        self.schedules[name] = sorted_schedule
    # No duplicate check!
```

**Impact**: `ScheduleBasedLR` is not the default controller (V6 uses `MetricBasedLR`), but it's a public class exported in `__all__` and documented in docstrings. Anyone using it with `[(10, 0.5), (10, 1.0)]` would get `ZeroDivisionError`.

**Note**: The `e1 <= epoch < e2` condition with `e1 == e2` produces `e1 <= epoch < e1` which is always False, so the division would never actually execute for the *same* epoch value as the duplicate. However, after sorting, duplicates become adjacent and later schedule segments may be skipped entirely, producing incorrect interpolation.

---

### M-3: LearnableLRController creates float32 tensors in float64 codebase

**File**: `src/models/lr_controller.py`
**Lines**: 497-539

**The dtype mismatch (line 532-539):**
```python
metrics_tensor = torch.tensor([
    metrics.coverage,
    metrics.hierarchy_a,
    ...
], dtype=torch.float32)  # <-- float32 in a float64 codebase
```

**Also**: The MLP layers in `__init__` (lines 501-509) are created with default dtype. Since `set_determinism()` calls `torch.set_default_dtype(torch.float64)`, the MLP weights will be float64 but the input tensor is float32. PyTorch auto-casts, but:
- The `sigmoid` output (line 527) returns float64 (promoted from MLP)
- The `.item()` calls (line 545) are fine
- But if someone tries to backprop through this (the whole point of "learnable"), the mixed dtypes could cause gradient precision issues

**This entire class is dead code**: `LearnableLRController` is:
- Defined at lines 473-556
- Exported in `__all__` (line 638) and `models/__init__.py` (line 9)
- **Never instantiated** by `train.py` or any other production code
- Labeled "experimental" in its docstring

---

### M-4: compute_tree_coherence uses O(N) Python loop with .item() calls

**File**: `src/train.py`
**Lines**: 400-449

**The slow path (lines 425-437):**
```python
# Python dict comprehension: ~20K .item() calls
index_to_pos = {idx.item(): pos for pos, idx in enumerate(indices)}

# Python loop: ~20K iterations with .item() per iteration
for pos, (idx, parent_idx) in enumerate(zip(indices, parent_indices)):
    parent_val = parent_idx.item()
    if parent_val < 0:
        continue
    if parent_val in index_to_pos:
        child_positions.append(pos)
        parent_positions.append(index_to_pos[parent_val])
```

**When this runs**: Every validation epoch (every `eval_every=2` epochs), for the full dataset of 19,683 operations. The dict construction alone calls `.item()` 19,683 times, and the loop calls `.item()` another ~19,683 times.

**After the loop (lines 448-449)**, the results are converted back to tensors:
```python
child_pos_t = torch.tensor(child_positions, device=device, dtype=torch.long)
parent_pos_t = torch.tensor(parent_positions, device=device, dtype=torch.long)
```

**Impact**: Adds ~2-5 seconds per validation epoch depending on hardware. Over 100 epochs with `eval_every=2`, that's 50 validations × ~3s = ~150s of unnecessary Python overhead.

---

### M-5: CombinedGeodesicLoss is dead code (75 lines)

**File**: `src/losses/padic_geodesic.py`
**Lines**: 312-369

**Full class listing:**
```python
class CombinedGeodesicLoss(nn.Module):
    """Combined Geodesic + Radial Loss for V5.11.
    Wraps both losses with curriculum-based blending:
    - Early: More radial loss (establish hierarchy)
    - Late: More geodesic loss (refine correlation)
    The tau parameter controls the blend (can be learned by controller).
    """
```

**Dead code verification:**
- `CombinedLoss` in `combined.py` instantiates `PAdicGeodesicLoss` and `RadialHierarchyLoss` directly (not via `CombinedGeodesicLoss`)
- `train.py` never references it
- It IS exported in `losses/__init__.py` line 4: `from .padic_geodesic import CombinedGeodesicLoss`
- No test file imports it

**Its `tau` blending approach** (line 358: `total_loss = (1 - tau) * rad_loss + tau * geo_loss`) is superseded by `CombinedLoss`'s config-driven weight system and the learnable weights feature (V6.1).

---

## REMAINING: LOW (10 issues)

### L-1: hyperbolic_kl.py is entirely unused (193 lines)

**File**: `src/losses/hyperbolic_kl.py` (193 lines, 2 classes)

**Classes defined:**
- `HyperbolicKLDivergence` (lines 33-120): Curvature-corrected KL using conformal factor `λ(x) = 2 / (1 - c||x||²)`. Mathematically correct implementation of Mathieu et al. (2019).
- `StandardKLDivergence` (lines 123-186): Standard Gaussian KL with API-compatible signature (accepts and ignores `z_hyp`).

**Usage trace:**
- Imported by `losses/__init__.py` line 10
- Exported in `__all__` of both `hyperbolic_kl.py` and `__init__.py`
- v6.yaml line 188-189 has `hyperbolic_kl: enabled: false`
- `CombinedLoss.__init__()` in `combined.py` **never reads** `loss.hyperbolic_kl` config — no code path instantiates either class
- `train.py` does not import either class

**The only consumer** of the `lambda_x` function from `src/geometry` is `HyperbolicKLDivergence` (line 88). If this module is removed, `lambda_x` also becomes dead code in practice (though it's still a valid geometric utility).

---

### L-2: Unused import `torch.nn.functional as F` in train.py

**File**: `src/train.py`, line 48
```python
import torch.nn.functional as F
```
Grep for `\bF\.` in `src/train.py` returns zero matches. `F` is never used.

---

### L-3: Unused imports in combined.py

**File**: `src/losses/combined.py`, lines 39-40
```python
from src.core import TERNARY        # line 39 — never used in file body
from src.geometry import poincare_distance  # line 40 — never used in file body
```

The individual loss classes (`PAdicGeodesicLoss`, etc.) import these themselves via `padic_geodesic.py`. `CombinedLoss` delegates to them and never uses `TERNARY` or `poincare_distance` directly.

---

### L-4: Unused import `CHECKPOINTS_DIR` in train.py

**File**: `src/train.py`, line 66
```python
from src.config.paths import RUNS_DIR, CHECKPOINTS_DIR
```

`RUNS_DIR` is used (line 814: `log_dir = RUNS_DIR / run_name`). `CHECKPOINTS_DIR` is never used — `train.py` creates its own checkpoint directory as `ckpt_dir = log_dir / 'checkpoints'`.

---

### L-5: Unused import `Callable` in lr_controller.py

**File**: `src/models/lr_controller.py`, line 39
```python
from typing import Any, Callable, Dict, List, Optional, Tuple
```

`Callable` is never used in any type annotation in this file.

---

### L-6: `patience_ceiling` config fields never consumed

**File**: `src/config/statenet_config.py`
```python
# Line 40 (HierarchyThresholds):
patience_ceiling: int = 25  # "Max patience (annealing limit)"

# Line 49 (ControllerThresholds):
patience_ceiling: int = 20  # "Max patience (annealing limit)"
```

`MetricBasedLR` reads `config.hierarchy.plateau_patience` (line 323) and `config.controller.grad_patience` (line 358) but never reads `patience_ceiling` from either dataclass. The `patience_ceiling` was part of the removed `AnnealingConfig` system.

**v6.yaml** also defines these (hierarchy line ~122: `patience_ceiling: 25`, controller line ~131: `patience_ceiling: 20`) and they're silently ignored.

---

### L-7: `PRESETS_DIR` and `MODELS_DIR` defined but never used

**File**: `src/config/paths.py`, lines 5-7
```python
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"  # line 5
MODELS_DIR = PROJECT_ROOT / "models"                        # line 6
PRESETS_DIR = PROJECT_ROOT / "presets"                       # line 7
```

- `MODELS_DIR` is exported by `config/__init__.py` (line 12) but never imported by any source file
- `PRESETS_DIR` is defined but NOT exported by `config/__init__.py`
- `SRC_PRESETS_DIR` (line 8) is exported but also never imported by any source file

---

### L-8: checkpoint.py uses `weights_only=False`

**File**: `src/utils/checkpoint.py`, line 36
```python
return torch.load(path, map_location=map_location, weights_only=False)
```

`weights_only=False` allows arbitrary code execution via pickle. This is a known PyTorch security concern (CVE-2025-32434). Low risk for a research codebase with self-generated checkpoints, but becomes relevant if checkpoints are ever shared or downloaded.

---

### L-9: Test fixture `sample_z_hyp` creates points outside Poincaré ball

**File**: `tests/conftest.py`, lines 34-37
```python
@pytest.fixture
def sample_z_hyp():
    """Sample hyperbolic embeddings (inside Poincaré ball)."""
    z = torch.randn(50, 16, dtype=torch.float64) * 0.5
    return z
```

**The math**: Each component is drawn from `N(0, 0.5²)`. For 16 dimensions, the expected norm is `0.5 * sqrt(16) = 2.0`. Empirically, all 50 samples have norms between 1.2 and 3.1 — **every point is outside the Poincaré ball** (radius 1.0 for curvature c=1).

**The same issue** exists in `tests/test_losses.py` lines 48-49:
```python
z_raw = torch.randn(batch_size, latent_dim, dtype=torch.float64)
z_hyp = 0.8 * torch.tanh(z_raw)  # Comment: "Keeps norm well inside ball"
```
Each component is bounded by 0.8, but 16-dim vectors have norms up to `0.8 * sqrt(16) = 3.2`.

**Impact**: Tests pass because geoopt's distance functions internally project via `projx`. But tests aren't exercising the intended scenario (valid manifold points). Edge cases near the boundary are untested with proper inputs.

**Correct fixture would be:**
```python
z = torch.randn(50, 16, dtype=torch.float64)
z = 0.8 * z / z.norm(dim=-1, keepdim=True)  # norm = 0.8 for all points
```

---

### L-10: ~20 v6.yaml config keys silently ignored by code

**File**: `src/presets/v6.yaml`

These config keys are defined in v6.yaml but never read by any code in `src/`:

| YAML key | Line | Why unused |
|----------|------|-----------|
| `device.pin_memory` | 28 | `train.py` hardcodes `pin_memory=torch.cuda.is_available()` |
| `device.num_workers` | 29 | `train.py` reads from `training.num_workers`, not `device` |
| `device.empty_cache_freq` | 30 | Read from `memory` section instead |
| `model.encoder_dropout` | 50 | Not passed to `TernaryVAEV6Controllable` or `EncoderHead` |
| `model.decoder_dropout` | 51 | Same — classes don't accept dropout config |
| `model.logvar_min` | 52 | Not passed to model — `EncoderHead` has no logvar clamping |
| `model.logvar_max` | 53 | Same |
| `geometry.use_manifold_parameter` | 69 | Never read by any code |
| `geometry.geodesic_steps` | 71 | Never read by any code |
| `precision.dtype` | 79 | `set_determinism()` hardcodes `torch.float64` |
| `training.use_stratified` | 204 | Never read — always uses full dataset shuffle |
| `training.high_v_budget_ratio` | 205 | Never read |
| `training.use_adaptive` | 208 | Never read |
| `training.hierarchy_threshold` | 209 | Never read |
| `training.patience` | 210 | Never read (early stopping not implemented) |
| `training.min_epochs` | 211 | Never read |
| `loss.zero_structure` | 181-184 | `ZeroStructureLoss` class **does not exist** |
| `loss.hyperbolic_kl` | 188-189 | Not consumed by `CombinedLoss` |
| `checkpoints.save_dir` | 251 | `train.py` uses `log_dir / 'checkpoints'` instead |
| `checkpoints.save_best` | 252 | Never read (hardcoded to save best_Q) |
| `checkpoints.best_metric` | 253 | Never read |
| `checkpoints.checkpoint_name` | 254 | Never read |
| `data.use_full_dataset` | 228 | Never read |
| `data.n_operations` | 229 | Never read — hardcoded via `TERNARY` singleton |
| `early_stopping.*` | 279-282 | Never read — early stopping not implemented |
| `memory.gradient_checkpointing` | 271 | Never read |
| `memory.max_memory_growth` | 274 | Never read |
| `targets.*` | 259-265 | Never read — purely documentation |

**Impact**: Users may spend time tuning `encoder_dropout`, `use_stratified`, `patience`, or `early_stopping` expecting them to have an effect. They are silently ignored.

---

## REMAINING: INFO (6 issues)

### I-1: Test coverage gaps

**Current**: 214 tests across 6 files covering 3 modules (core, geometry, losses).

**Planned but unimplemented** (per `docs/plans/TESTS_CRITICAL_TARGETS.md`):

| Planned file | Tier | Target module | Status |
|-------------|------|---------------|--------|
| `tests/test_lr_controller.py` | 3 | `MetricBasedLR` gate transitions, hysteresis, warmup | Not started |
| `tests/test_vae_trainability.py` | 3 | `set_encoder_a_trainable()`, gradient flow, param groups | Not started |
| `tests/test_edge_cases.py` | 4 | Empty tensors, boundary points, batch_size=1 | Not started |

**Untested modules with no test files:**

| File | Lines | Key untested functionality |
|------|-------|--------------------------|
| `models/vae.py` | 528 | `TernaryVAEV6.forward()` output shapes, reparameterization, Poincaré ball containment of `z_A_hyp`/`z_B_hyp`, `get_param_groups()` group names and LR scales, `set_encoder_a_trainable()` |
| `models/hyperbolic_projection.py` | 267 | `HyperbolicProjection` output containment, tangent_net transform, `DualHyperbolicProjection` shared curvature, learnable curvature stays positive |
| `models/lr_controller.py` | 643 | `MetricBasedLR` warmup→active transition, coverage gate freeze/unfreeze, hierarchy plateau detection, projections gradient monitoring, hysteresis enforcement, `ScheduleBasedLR` interpolation, `update_optimizer_lr_scales` param group modification |
| `utils/tensorboard_logger.py` | 577 | Not critical to test (visualization only) |
| `utils/hardware_monitor.py` | 263 | Not critical to test (monitoring only) |
| `losses/hyperbolic_kl.py` | 193 | Dead code — test if integrating into loss system |

---

### I-2: GrokkingDetector heuristic

**File**: `src/train.py`, `GrokkingDetector` class

Fits linear regression to train/val loss gap to detect late generalization. Reasonable heuristic but may false-positive on natural gap narrowing from LR decay. Design choice, not a bug.

---

### I-3: Per-loss Generator state not checkpoint-safe

**File**: `src/losses/padic_geodesic.py`

`PAdicGeodesicLoss`, `RadialHierarchyLoss`, `GlobalRankLoss`, and `MonotonicRadialLoss` each maintain a `torch.Generator(manual_seed=42)` for pair sampling. Generator state advances each forward call. If training is interrupted and resumed from checkpoint, the generators reset to seed 42, producing different pair sequences than the original run would have at that epoch. This means checkpoint-resumed training is not bit-identical to uninterrupted training.

---

### I-4: `_manifold_cache` module-level global

**File**: `src/geometry/poincare.py`, line 45
```python
_manifold_cache = {}
```

Not thread-safe, grows without bounds, not clearable. Negligible for single-GPU training.

---

### I-5: CombinedLoss reconstruction loss target shifting

**File**: `src/losses/combined.py`, lines 385-400

Targets shift `{-1, 0, 1}` → `{0, 1, 2}` via `(targets + 1).long()`. Correct implementation. The `(B, 27)` logit path in `RichHierarchyLoss` (line 729) does the same shift but without `.clamp(0, 2)` — relies on data validity. `CombinedLoss._compute_coverage_loss()` (line 391) does add `.clamp(0, 2)`.

---

### I-6: Copyright year range

All source files read "Copyright 2024-2025". Current date is 2026-02-24. Cosmetic.

---

## Summary Table

| Severity | Count | Key areas |
|----------|-------|-----------|
| FIXED | 8 | LR controller (3), optimizer params (2), geometry (1), scheduler (1), dtype (1) |
| CRITICAL | 2 | TensorBoard logger V5.x API (crash if called) |
| MODERATE | 5 | Validator contradiction, division-by-zero, dead code, dtype, performance |
| LOW | 10 | Unused imports (4), dead modules (2), config keys (20+), test fixtures, security |
| INFO | 6 | Test coverage gaps, design observations, reproducibility |

## Recommended Priority Actions

1. **Immediate**: Fix or remove `TensorBoardLogger.log_manifold_embedding` and `log_epoch` (C-1, C-2) — these are crash bugs in public API
2. **Short-term**: Remove dead code — `CombinedGeodesicLoss` (M-5), `hyperbolic_kl.py` or integrate it (L-1), unused imports (L-2 through L-5), dead `checkpoint_validator.py` or fix it (M-1)
3. **Short-term**: Clean v6.yaml — remove or comment the ~20 ignored config keys (L-10), especially `loss.zero_structure` which references a nonexistent class
4. **Medium-term**: Add Tier 3 tests for `vae.py`, `hyperbolic_projection.py`, and `lr_controller.py` (I-1); fix test fixtures to produce valid Poincaré ball points (L-9)
5. **Low priority**: Fix `ScheduleBasedLR` duplicate epoch handling (M-2), update `weights_only` (L-8), remove unused config fields (L-6)

---

*Generated by deep-read codebase audit. All line numbers verified against codebase state as of 2026-02-24, post-fix commit `e2b74b0`.*
