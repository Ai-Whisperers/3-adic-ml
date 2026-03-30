# Utils Module Audit: src/utils/

**Date**: 2025-01-23
**Scope**: `src/utils/__init__.py`, `checkpoint.py`, `checkpoint_validator.py`, `coverage_evaluator.py`, `tensorboard_logger.py`
**Lines of Code**: 57 (checkpoint.py) + 263 (checkpoint_validator.py) + 146 (coverage_evaluator.py) + 573 (tensorboard_logger.py) + 8 (__init__.py) = ~1047 total

---

## Executive Summary

The utils module provides **essential infrastructure utilities** for checkpoint management, configuration validation, coverage evaluation, and TensorBoard visualization. The code is well-organized with clear single-responsibility design. Notable strengths include proper geometry integration (using `poincare_distance` for hyperbolic radii) and graceful degradation when optional dependencies are missing.

**Verdict**: The module is **solid and production-ready** with good defensive programming practices.

---

## File Structure

```
src/utils/
├── __init__.py              # Clean re-exports (7 symbols)
├── checkpoint.py            # Checkpoint loading utilities (57 lines)
├── checkpoint_validator.py  # Config/checkpoint validation (263 lines)
├── coverage_evaluator.py    # VAE coverage evaluation (146 lines)
└── tensorboard_logger.py    # TensorBoard visualization (573 lines)
```

---

## Module Exports (__init__.py)

| Export | Type | Source | Purpose |
|--------|------|--------|---------|
| `evaluate_coverage` | Function | coverage_evaluator | Quick coverage eval |
| `CoverageEvaluator` | Class | coverage_evaluator | Configurable evaluator |
| `TensorBoardLogger` | Class | tensorboard_logger | Visualization logging |
| `load_checkpoint_compat` | Function | checkpoint | Safe checkpoint loading |
| `get_model_state_dict` | Function | checkpoint | State dict extraction |
| `CheckpointValidator` | Class | checkpoint_validator | Validation utilities |
| `CheckpointCompatibilityError` | Exception | checkpoint_validator | Custom exception |
| `validate_training_config` | Function | checkpoint_validator | Config validation |

**Assessment**: Clean, well-organized exports with clear responsibilities.

---

## Detailed Analysis

### 1. checkpoint.py - Checkpoint Loading

#### 1.1 load_checkpoint_compat (Lines 17-36)

```python
def load_checkpoint_compat(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=map_location, weights_only=False)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Path handling | ✅ Good | Uses pathlib |
| Existence check | ✅ Good | Clear error message |
| weights_only | ⚠️ Security | False allows arbitrary pickle |
| map_location | ✅ Good | Device flexibility |

**Security Note**: `weights_only=False` is necessary for complex checkpoints but allows arbitrary code execution from malicious files. Consider documenting this risk.

#### 1.2 get_model_state_dict (Lines 39-56)

```python
def get_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    else:
        return checkpoint  # Assume checkpoint itself is state dict
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Multiple formats | ✅ Good | Handles 3 common formats |
| Fallback | ✅ Good | Assumes raw state dict |
| Documentation | ✅ Good | Clear format descriptions |

---

### 2. checkpoint_validator.py - Validation Utilities

#### 2.1 CheckpointValidator Class

```python
class CheckpointValidator:
    RECOMMENDED_CHECKPOINTS = {
        "TernaryVAEV5_11": "models/checkpoints/v5_5/latest.pt",
        "TernaryVAEV5_11_PartialFreeze": "models/checkpoints/v5_5/latest.pt",
    }
```

| Method | Purpose | Status |
|--------|---------|--------|
| `validate_checkpoint_exists` | Check file exists | ✅ Good |
| `validate_checkpoint_dimensions` | Shape compatibility | ✅ Good |
| `fix_null_checkpoint_config` | Auto-fix null paths | ✅ Good |
| `get_checkpoint_info` | Extract metadata | ✅ Good |

#### 2.2 validate_checkpoint_dimensions (Lines 53-102)

```python
def validate_checkpoint_dimensions(cls, checkpoint_path, model):
    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Check for dimension mismatches
    for key in state_dict:
        if key in model_state:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state[key].shape
            if ckpt_shape != model_shape:
                errors.append(f"Dimension mismatch for '{key}': ...")
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Shape validation | ✅ Good | Catches dimension mismatches |
| CPU loading | ✅ Good | Avoids GPU memory issues |
| Error accumulation | ✅ Good | Reports all errors at once |

#### 2.3 validate_training_config (Lines 189-262)

```python
def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    # Check required sections
    if "model" not in config:
        errors.append("Missing 'model' section")

    # V5.11+ architecture checks
    if model_name in v5_11_models:
        if checkpoint_path is None:
            errors.append("Model requires frozen checkpoint...")

    # Training config validation
    if epochs < 1:
        errors.append("Invalid epochs...")
    if lr <= 0:
        errors.append("Invalid learning_rate...")

    # StateNet threshold validation
    if coverage_freeze >= coverage_unfreeze:
        errors.append("StateNet thresholds invalid...")
```

| Validation | Status | Notes |
|------------|--------|-------|
| Required sections | ✅ Good | model section required |
| V5.11 checkpoint | ✅ Good | Prevents 0% coverage issue |
| Hyperparameter ranges | ✅ Good | epochs, lr, batch_size |
| StateNet thresholds | ✅ Good | freeze < unfreeze |

---

### 3. coverage_evaluator.py - Coverage Evaluation

#### 3.1 evaluate_coverage Function (Lines 25-65)

```python
def evaluate_coverage(
    model: torch.nn.Module,
    num_samples: int,
    device: str,
    vae: str = "A",
    batch_size: int = 1000,
) -> Tuple[int, float]:
    model.eval()

    with torch.no_grad():
        all_samples_list = []
        for _ in range(num_batches):
            samples = model.sample(batch_size, device, vae)
            samples_rounded = torch.round(samples).long()
            all_samples_list.append(samples_rounded)

        all_samples = torch.cat(all_samples_list, dim=0)
        unique_samples = torch.unique(all_samples, dim=0)
        unique_count = unique_samples.size(0)

    coverage_pct = (unique_count / N_TERNARY_OPERATIONS) * 100
    return unique_count, coverage_pct
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Vectorized | ✅ Good | torch.unique instead of loop |
| GPU efficient | ✅ Good | Single sync per batch |
| Batched sampling | ✅ Good | Configurable batch_size |
| eval() mode | ✅ Good | Disables dropout |
| no_grad | ✅ Good | Saves memory |

#### 3.2 CoverageEvaluator Class (Lines 68-143)

```python
class CoverageEvaluator:
    def __init__(self, num_samples=100000, batch_size=1000):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def evaluate(self, model, device, vae="A", num_samples=None):
        samples = num_samples or self.num_samples
        return evaluate_coverage(model, samples, device, vae, self.batch_size)

    def evaluate_both(self, model, device, num_samples=None):
        unique_A, cov_A = self.evaluate(model, device, "A", num_samples)
        unique_B, cov_B = self.evaluate(model, device, "B", num_samples)
        return unique_A, cov_A, unique_B, cov_B
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Configurable defaults | ✅ Good | num_samples, batch_size |
| Override support | ✅ Good | Per-call num_samples |
| Dual VAE support | ✅ Good | evaluate_both() |

#### 3.3 Potential Issue: model.sample()

The function assumes `model.sample(batch_size, device, vae)` exists. If the model doesn't have this method, it will fail.

| Issue | Severity | Notes |
|-------|----------|-------|
| Duck typing | ⚠️ Medium | No type check for sample() method |

---

### 4. tensorboard_logger.py - Visualization

#### 4.1 TensorBoardLogger Class

```python
class TensorBoardLogger:
    def __init__(self, tensorboard_dir, experiment_name, log_callback=None):
        self.writer = None
        self.log_callback = log_callback or (lambda msg: None)

        if TENSORBOARD_AVAILABLE and tensorboard_dir is not None:
            log_path = Path(tensorboard_dir) / f"ternary_vae_{experiment_name}"
            self.writer = SummaryWriter(str(log_path))
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Optional dependency | ✅ Excellent | Graceful degradation |
| Null callback | ✅ Good | Default no-op |
| Path construction | ✅ Good | Uses pathlib |

#### 4.2 Logging Methods

| Method | Purpose | Metrics Logged |
|--------|---------|----------------|
| `log_batch` | Batch-level losses | loss, ce_A/B, kl_A/B |
| `log_hyperbolic_batch` | Hyperbolic batch metrics | ranking, radial, hyp_kl, centroid |
| `log_hyperbolic_epoch` | Hyperbolic epoch metrics | correlations, radii, StateNet |
| `log_epoch` | Full epoch metrics | losses, coverage, dynamics, lambdas |
| `log_histograms` | Weight histograms | params, gradients |
| `log_manifold_embedding` | Latent visualization | 3D embeddings with metadata |

#### 4.3 log_manifold_embedding (Lines 419-537)

```python
def log_manifold_embedding(self, model, epoch, device, n_samples=5000):
    # Get all ternary operations
    all_operations = TERNARY.all_ternary()

    # Sample or use all
    if include_all or n_samples >= total_ops:
        indices = list(range(total_ops))
    else:
        indices = sorted(random.sample(range(total_ops), n_samples))

    # Encode
    with torch.no_grad():
        outputs = model(x, 1.0, 1.0, 0.5, 0.5)
        z_A = outputs["z_A"]
        z_B = outputs["z_B"]

        # Project to Poincare ball
        z_A_poincare = z_A / (1 + z_A_euc_norm) * 0.95

        # V5.12.2: Compute hyperbolic radii
        hyp_radii_A = poincare_distance(z_A_poincare, origin_A, c=1.0)

    # Log embeddings with 3-adic metadata
    self.writer.add_embedding(z_A.cpu(), metadata=metadata, ...)
```

| Aspect | Status | Notes |
|--------|--------|-------|
| TERNARY integration | ✅ Good | Uses singleton |
| Hyperbolic projection | ✅ Good | tanh-style projection |
| poincare_distance | ✅ Correct | V5.12.2 fix applied |
| Metadata | ✅ Good | 3-adic prefix, depth, radii |

#### 4.4 _compute_3adic_depth (Lines 539-558)

```python
def _compute_3adic_depth(self, n: int) -> int:
    if n == 0:
        return 9
    depth = 0
    while n % 3 == 0:
        depth += 1
        n //= 3
    return depth
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Correctness | ✅ Correct | Same as TERNARY.valuation |
| Duplication | ⚠️ Redundant | Could use TERNARY.valuation |

**Recommendation**: Replace with `TERNARY.valuation(torch.tensor([n])).item()` for consistency.

---

## Integration Analysis

### Geometry Module Usage

| File | Uses src/geometry? | Notes |
|------|-------------------|-------|
| tensorboard_logger.py | ✅ Yes | `poincare_distance` for radii |
| coverage_evaluator.py | ❌ No | Coverage only |
| checkpoint.py | ❌ No | File I/O only |
| checkpoint_validator.py | ❌ No | Validation only |

### Core Module Usage

| File | Uses src/core? | Notes |
|------|---------------|-------|
| tensorboard_logger.py | ✅ Yes | `TERNARY.all_ternary()` |
| coverage_evaluator.py | ✅ Yes | `N_TERNARY_OPERATIONS` via constants |
| checkpoint.py | ❌ No | File I/O only |
| checkpoint_validator.py | ❌ No | Validation only |

---

## Issues Summary

### Critical (0)

None.

### High (0)

None.

### Medium (2)

| Issue | Location | Description |
|-------|----------|-------------|
| M1 | `checkpoint.py:36` | `weights_only=False` is security risk for untrusted files |
| M2 | `coverage_evaluator.py:55` | Duck typing assumes model.sample() exists |

### Low (3)

| Issue | Location | Description |
|-------|----------|-------------|
| L1 | `tensorboard_logger.py:551` | _compute_3adic_depth duplicates TERNARY.valuation |
| L2 | `tensorboard_logger.py:452` | `random.sample` not seeded for reproducibility |
| L3 | `checkpoint_validator.py:30` | Hardcoded checkpoint paths may become stale |

---

## Code Quality Assessment

| Metric | Score | Notes |
|--------|-------|-------|
| Correctness | 9/10 | All utilities work correctly |
| Defensive Programming | 9/10 | Good error handling, graceful degradation |
| Documentation | 8/10 | Good docstrings, clear purpose |
| Integration | 9/10 | Proper use of geometry/core modules |
| Maintainability | 8/10 | Well-organized, single responsibility |
| Security | 7/10 | weights_only=False is documented risk |

---

## File-by-File Summary

### checkpoint.py (57 lines)
- **Purpose**: Safe checkpoint loading with format flexibility
- **Rating**: 8/10
- **Strengths**: Simple, handles multiple formats
- **Concern**: Security note for weights_only=False

### checkpoint_validator.py (263 lines)
- **Purpose**: Config/checkpoint validation to prevent training issues
- **Rating**: 9/10
- **Strengths**: Comprehensive validation, auto-fix capability
- **Note**: Specifically prevents V5.11+ 0% coverage issue

### coverage_evaluator.py (146 lines)
- **Purpose**: VAE operation coverage evaluation
- **Rating**: 9/10
- **Strengths**: Vectorized, GPU-efficient, batched
- **Note**: Assumes model.sample() interface

### tensorboard_logger.py (573 lines)
- **Purpose**: TensorBoard visualization for training
- **Rating**: 8/10
- **Strengths**: Comprehensive metrics, 3D embeddings, optional dependency
- **Note**: Minor duplication of valuation computation

### __init__.py (8 lines)
- **Purpose**: Module exports
- **Rating**: 10/10
- **No issues**: Clean, minimal

---

## Recommendations

### Should Fix

1. **Add reproducibility to embedding sampling**:
   ```python
   def log_manifold_embedding(self, ..., seed: int = 42):
       rng = random.Random(seed)
       indices = sorted(rng.sample(range(total_ops), n_samples))
   ```

2. **Use TERNARY.valuation instead of _compute_3adic_depth**:
   ```python
   def _compute_3adic_depth(self, n: int) -> int:
       return TERNARY.valuation(torch.tensor([n])).item()
   ```

### Could Improve

1. **Document security risk in checkpoint loading**
2. **Add interface check for model.sample() in coverage evaluator**
3. **Make checkpoint paths configurable instead of hardcoded**

---

## Verdict

**The utils module provides essential infrastructure that is well-designed and production-ready.** It demonstrates good defensive programming with graceful degradation for optional dependencies, comprehensive validation to prevent common training issues, and proper integration with the geometry and core modules.

The TensorBoard logger is particularly well-done, providing rich visualization capabilities including 3D embedding projectors with 3-adic metadata.

**Rating**: 8.5/10 (Very Good)

---

**Audit completed**: 2025-01-23
**Auditor**: Claude Opus 4.5

---

## Addendum (2026-02-26 Audit)

**Auditor**: Claude Opus 4.6

### Stale Information Corrections

1. **`coverage_evaluator.py` no longer exists.** This audit references it throughout (sections 3, exports table, etc.). The file has been removed. `evaluate_coverage` and `CoverageEvaluator` are no longer in the codebase.

2. **Scope line is outdated.** Current files: `checkpoint.py` (56 lines), `checkpoint_validator.py` (94 lines), `hardware_monitor.py` (262 lines), `tensorboard_logger.py` (275 lines). Total ~700 lines.

3. **Export table is outdated.** `__init__.py` no longer exports `evaluate_coverage`, `CoverageEvaluator`, or `CheckpointCompatibilityError`. Current exports: `load_checkpoint_compat`, `get_model_state_dict`, `validate_training_config`, `TensorBoardLogger`, `HardwareMonitor`.

4. **Missing file: `hardware_monitor.py`** (262 lines). Not in original audit. Contains `HardwareMonitor` class for GPU/RAM monitoring with graceful fallbacks.

5. **TensorBoard logger was substantially rewritten.** Stale V5.x methods (`log_epoch`, `_log_padic_losses`, `log_hyperbolic_batch`, `log_hyperbolic_epoch`) were removed in commit `e2b74b0`. Current: 275 lines (was 573).

### New Issues Found (2026-02-26)

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| checkpoint_validator.py is dead code | MODERATE | entire file (94 lines) | `validate_training_config()` never called by train.py; contradicts train.py on anchor checkpoint requirement |
| No tests for any utils/ file | HIGH | — | Zero test coverage for checkpoint, validator, hardware_monitor, tensorboard_logger |
| `weights_only=False` security risk | LOW | checkpoint.py:36 | Allows arbitrary code execution from untrusted checkpoints |
| `_compute_3adic_depth` duplication | LOW | tensorboard_logger.py | Still duplicates `TERNARY.valuation` |

### Updated Rating

**Rating**: 6.5/10 (was 8.5/10 — downgraded for stale file references, zero test coverage, 94 lines dead code)
