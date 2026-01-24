# Terminal Monitoring & Logging Integration Audit

**Date**: 2026-01-23
**Status**: Pre-Implementation Analysis
**Related**: `tensorboard-logs.md`

---

## Executive Summary

The codebase has a comprehensive `TensorBoardLogger` class that is **not used** by the main training script. Instead, `train.py` duplicates TensorBoard setup with raw `SummaryWriter`. Additionally, there is no terminal-based progress monitoring, hardware utilization tracking, or OOM detection.

This audit documents the current architecture and provides a detailed integration plan.

---

## 1. Current Architecture

### 1.1 File Structure

```
src/
├── train.py                      # Main training script (uses raw SummaryWriter)
├── utils/
│   ├── __init__.py               # Exports TensorBoardLogger (unused)
│   ├── tensorboard_logger.py     # Comprehensive logger class (UNUSED)
│   ├── checkpoint.py
│   ├── checkpoint_validator.py
│   └── coverage_evaluator.py
```

### 1.2 Duplication Map

| Component | `train.py` | `tensorboard_logger.py` | Status |
|-----------|------------|-------------------------|--------|
| TensorBoard import | Lines 54-59 | Lines 31-37 | **DUPLICATE** |
| `TENSORBOARD_AVAILABLE` flag | Line 56 | Line 34 | **DUPLICATE** |
| `SummaryWriter` instance | Line 675 | Line 72 (inside class) | **DUPLICATE** |
| Epoch metric logging | Lines 797-809 | `log_epoch()` method | **DUPLICATE** |
| Writer close | Line 862 | `close()` method | **DUPLICATE** |

### 1.3 Current `train.py` TensorBoard Usage

```python
# Line 54-59: Duplicate import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Line 675: Raw writer creation
writer = SummaryWriter(str(log_dir)) if TENSORBOARD_AVAILABLE else None

# Lines 797-809: Manual scalar logging (epoch-level only)
if writer is not None:
    writer.add_scalars('Accuracy', {'train': avg_train_acc, 'val': avg_val_acc}, epoch)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Coverage', avg_val_coverage, epoch)
    writer.add_scalars('Hierarchy/corr', {...}, epoch)
    writer.add_scalars('Hierarchy/Q', {...}, epoch)
    writer.add_scalar('Hierarchy/dist_corr', hier_metrics_A['dist_corr'], epoch)

# Line 862: Close
if writer is not None:
    writer.close()
```

### 1.4 Unused `TensorBoardLogger` Capabilities

The existing class at `src/utils/tensorboard_logger.py` provides:

| Method | Purpose | Currently Used |
|--------|---------|----------------|
| `__init__()` | Creates writer with experiment naming | NO |
| `log_batch()` | Batch-level CE/KL losses | NO |
| `log_hyperbolic_batch()` | Ranking/radial/centroid per batch | NO |
| `log_hyperbolic_epoch()` | Full hyperbolic metrics + StateNet | NO |
| `log_epoch()` | Comprehensive epoch logging (20+ metrics) | NO |
| `_log_padic_losses()` | P-adic loss components | NO |
| `log_histograms()` | Weight/gradient distributions | NO |
| `log_manifold_embedding()` | 3D latent visualization | NO |
| `flush()` | Real-time TensorBoard updates | NO |
| `close()` | Clean shutdown | NO |
| `is_available` property | Null-safe availability check | NO |

---

## 2. Missing Features

### 2.1 Terminal Progress Monitoring

**Current State**: Only periodic `print()` statements every `print_every` epochs (default: 5)

```python
# Line 830-841: Current progress output
if epoch % print_every == 0 or epoch == epochs - 1:
    dt = time.time() - t0
    print(
        f"Ep {epoch:03d} | "
        f"Loss {avg_train_loss:.4f} | "
        ...
    )
```

**Problems**:
- No indication of training activity between prints
- No batch-level progress within epochs
- No way to know if process is still running vs. hung
- No real-time feedback during long epochs

### 2.2 Hardware Utilization Monitoring

**Current State**: None

**Missing Metrics**:
| Metric | PyTorch API | Purpose |
|--------|-------------|---------|
| GPU Memory Allocated | `torch.cuda.memory_allocated()` | Current tensor memory |
| GPU Memory Reserved | `torch.cuda.memory_reserved()` | Total CUDA memory |
| GPU Memory Max | `torch.cuda.max_memory_allocated()` | Peak usage |
| GPU Utilization % | `nvidia-smi` or `pynvml` | Compute utilization |
| RAM Usage | `psutil.virtual_memory()` | System memory |
| RAM Available | `psutil.virtual_memory().available` | Free memory |

### 2.3 OOM Detection & Handling

**Current State**: Silent crash on CUDA OOM

**Needed**:
- Try/except around forward/backward passes
- Memory state dump on OOM
- Graceful checkpoint save before exit
- Diagnostic output for debugging

---

## 3. Integration Points in `train.py`

### 3.1 Initialization Phase (Lines 670-684)

**Current**:
```python
# Line 675
writer = SummaryWriter(str(log_dir)) if TENSORBOARD_AVAILABLE else None
```

**Target**:
```python
from src.utils import TensorBoardLogger

# Replace line 675
tb_logger = TensorBoardLogger(
    tensorboard_dir=str(log_dir),
    experiment_name=run_name,
    log_callback=print,
)
```

### 3.2 Batch Loop (Lines 710-732)

**Current**: No batch-level logging

**Integration Points**:

| After Line | Add | Purpose |
|------------|-----|---------|
| 722 | `tb_logger.log_batch(global_step, loss.item(), ...)` | Batch loss tracking |
| 726 | Log gradient norm scalar | Gradient health |
| 728 | Progress bar update | Terminal feedback |
| 728 | Memory check | OOM prevention |

### 3.3 Epoch Validation (Lines 740-841)

**Current**: Minimal TensorBoard logging at lines 797-809

**Integration Points**:

| After Line | Add | Purpose |
|------------|-----|---------|
| 786 | `tb_logger.log_hyperbolic_epoch(...)` with StateNet metrics | Full metrics |
| 809 | `tb_logger.flush()` | Real-time updates |
| 841 | Hardware stats to terminal | Resource monitoring |

### 3.4 Periodic Operations

| Frequency | Location | Operation |
|-----------|----------|-----------|
| Every batch | Line 728 | Progress bar update, memory check |
| Every epoch | Line 734 | LR logging, epoch summary |
| Every `eval_every` | Line 809 | Full validation metrics |
| Every 10 epochs | After line 841 | Weight histograms |
| Every 50 epochs | After line 841 | Embedding visualization |

### 3.5 Cleanup (Lines 861-862)

**Current**:
```python
if writer is not None:
    writer.close()
```

**Target**:
```python
tb_logger.close()  # Null-safe
```

---

## 4. Proposed New Components

### 4.1 HardwareMonitor Class

**Location**: `src/utils/hardware_monitor.py` (new file)

**Responsibilities**:
- GPU memory tracking (allocated, reserved, peak)
- GPU utilization percentage
- RAM usage and availability
- Formatted status string for terminal
- Warning thresholds (e.g., >90% GPU memory)

**Interface**:
```python
class HardwareMonitor:
    def __init__(self, device: torch.device, warn_threshold: float = 0.9):
        ...

    def get_gpu_memory_mb(self) -> Dict[str, float]:
        """Returns {'allocated': X, 'reserved': Y, 'peak': Z}"""
        ...

    def get_gpu_utilization_pct(self) -> float:
        """Returns GPU compute utilization 0-100"""
        ...

    def get_ram_usage_mb(self) -> Dict[str, float]:
        """Returns {'used': X, 'available': Y, 'percent': Z}"""
        ...

    def get_status_string(self) -> str:
        """Returns formatted string: 'GPU: 2.1/6.0GB (35%) | RAM: 8.2/32GB (26%)'"""
        ...

    def check_memory_warning(self) -> Optional[str]:
        """Returns warning message if memory > threshold, else None"""
        ...

    def reset_peak_stats(self) -> None:
        """Reset peak memory tracking"""
        ...
```

**Dependencies**: `psutil` (add to requirements), `torch.cuda`

### 4.2 Training Progress Bar

**Option A: tqdm integration**

```python
from tqdm import tqdm

# Wrap epoch loop
for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
    # Wrap batch loop
    batch_pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)
    for batch_ops, batch_idx in batch_pbar:
        ...
        batch_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'gpu': hw_monitor.get_status_string(),
        })
```

**Option B: Simple status line (no dependencies)**

```python
def print_status(epoch, batch, total_batches, loss, gpu_mem):
    """Overwrite single line with current status"""
    print(f"\rEp {epoch:03d} | Batch {batch}/{total_batches} | "
          f"Loss {loss:.4f} | GPU {gpu_mem}", end='', flush=True)
```

**Recommendation**: Option A (tqdm) for better UX, but keep Option B as fallback

### 4.3 OOM Handler

**Location**: Inline in `train.py` or `src/utils/error_handlers.py`

**Pattern**:
```python
try:
    with torch.amp.autocast('cuda', enabled=use_amp):
        out = model(batch_ops, compute_control=False)
        ...
    scaler.scale(loss).backward()
    ...
except torch.cuda.OutOfMemoryError as e:
    # 1. Log current state
    print(f"\n[OOM] CUDA Out of Memory at epoch {epoch}, batch {n_batches}")
    print(f"[OOM] GPU Memory: {hw_monitor.get_status_string()}")

    # 2. Clear cache
    torch.cuda.empty_cache()

    # 3. Save emergency checkpoint
    emergency_path = ckpt_dir / f'emergency_oom_epoch_{epoch}.pt'
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, emergency_path)
    print(f"[OOM] Emergency checkpoint saved: {emergency_path}")

    # 4. Re-raise or graceful exit
    raise
```

---

## 5. Implementation Checklist

### Phase 1: Remove Duplication

- [ ] Remove lines 54-59 (duplicate TensorBoard import)
- [ ] Add `from src.utils import TensorBoardLogger` to imports
- [ ] Replace line 675 `SummaryWriter` with `TensorBoardLogger`
- [ ] Replace lines 797-809 with `tb_logger.log_*()` calls
- [ ] Replace line 862 with `tb_logger.close()`

### Phase 2: Add Hardware Monitoring

- [ ] Create `src/utils/hardware_monitor.py`
- [ ] Add `psutil` to dependencies
- [ ] Initialize `HardwareMonitor` in `train()` function
- [ ] Log hardware stats at epoch boundaries
- [ ] Add memory warning checks

### Phase 3: Add Terminal Progress

- [ ] Add `tqdm` to dependencies (or implement fallback)
- [ ] Wrap epoch loop with progress bar
- [ ] Wrap batch loop with nested progress bar
- [ ] Add postfix updates with loss/memory stats
- [ ] Ensure proper newline handling for clean output

### Phase 4: Add OOM Handling

- [ ] Wrap forward/backward in try/except
- [ ] Add emergency checkpoint save on OOM
- [ ] Add memory diagnostic output
- [ ] Add graceful degradation (reduce batch size suggestion)

### Phase 5: Enhanced TensorBoard Logging

- [ ] Add batch-level logging with `log_batch()`
- [ ] Add gradient norm logging after `clip_grad_norm_`
- [ ] Add LR logging after `scheduler.step()`
- [ ] Add StateNet metrics to `log_hyperbolic_epoch()`
- [ ] Add weight histograms every N epochs
- [ ] Add embedding visualization every M epochs
- [ ] Add `flush()` calls for real-time updates

---

## 6. Dependencies to Add

```
# requirements.txt additions
tqdm>=4.64.0        # Progress bars
psutil>=5.9.0       # System/RAM monitoring
```

**Optional** (for GPU utilization %):
```
pynvml>=11.0.0      # NVIDIA Management Library
```

---

## 7. Expected Terminal Output After Integration

### During Training (with tqdm):

```
Training:  45%|████████████                    | 45/100 [12:34<15:23, 16.8s/epoch]
Ep 045:  78%|███████████████████████          | 28/36 [00:12<00:03] loss=0.0234 GPU: 2.1/6.0GB (35%)
```

### At Epoch Boundaries:

```
Ep 045 | Loss 0.0234 | Acc T/V 0.987/0.982 | Cov 0.995 | Hier A/B -0.812/-0.798 | Q 1.847
       | GPU: 2.1/6.0GB (35%) peak=2.8GB | RAM: 8.2/32GB (26%) | 16.8s
```

### On OOM:

```
[OOM] CUDA Out of Memory at epoch 45, batch 28
[OOM] GPU Memory: allocated=5.8GB, reserved=6.0GB, peak=6.0GB
[OOM] RAM: used=12.4GB, available=19.6GB (39%)
[OOM] Emergency checkpoint saved: runs/.../checkpoints/emergency_oom_epoch_45.pt
[OOM] Suggestion: Reduce batch_size from 512 to 256
```

---

## 8. Config Integration

The preset YAML files already have logging configuration sections that are currently ignored:

```yaml
# From research_extended_grokking.yaml (lines 181-202)
logging:
  tensorboard: true
  log_dir: runs/v5_12_4_extended_grokking
  print_every: 2

  enhanced_metrics:
    enabled: true
    log_gradients: true
    log_weights: true
    ...
```

**Action**: Parse these config sections in `train()` to control:
- Histogram logging frequency
- Embedding visualization frequency
- Gradient logging enable/disable
- Terminal verbosity level

---

## 9. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Replace SummaryWriter | Low | TensorBoardLogger is already tested |
| Add tqdm | Low | Optional dependency with fallback |
| Add psutil | Low | Standard package, no GPU dependency |
| Add OOM handling | Low | Only affects error path |
| Add batch logging | Medium | May increase TensorBoard file size |
| Add embedding viz | Medium | Memory intensive, make periodic |

---

## 10. Testing Plan

1. **Smoke test**: Run `--validate-only` to ensure imports work
2. **Short run**: 5 epochs with `minimal_smoke_test.yaml`
3. **Progress test**: Verify terminal output updates correctly
4. **Memory test**: Monitor GPU/RAM during training
5. **OOM test**: Artificially trigger OOM (large batch) to verify handling
6. **TensorBoard test**: Verify all metrics appear in dashboard

---

## References

- `src/train.py` - Main training script
- `src/utils/tensorboard_logger.py` - Unused logger class
- `docs/audits/tensorboard-logs.md` - Previous TensorBoard audit
- `src/presets/research_extended_grokking.yaml` - Example config with logging section
