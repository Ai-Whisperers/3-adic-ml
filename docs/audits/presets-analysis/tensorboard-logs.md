# TensorBoard Real-Time Logging Analysis

**Last Updated**: 2026-01-24
**Status**: Pre-Implementation Analysis

---

## Current State

`src/train.py` uses TensorBoard directly via raw `SummaryWriter`:
- Lines 54-59: Duplicate TensorBoard import and availability check
- Line 648: Creates writer at run start
- Lines 769-782: Logs epoch-level metrics only (accuracy, loss, coverage, hierarchy, Q)
- Line 834: Closes writer at end

`src/utils/tensorboard_logger.py` has a comprehensive `TensorBoardLogger` class that is **NOT currently used** by train.py. It provides:
- `log_batch()` - batch-level loss components (CE, KL)
- `log_hyperbolic_batch()` - ranking/radial/centroid losses per batch
- `log_hyperbolic_epoch()` - full epoch metrics with StateNet
- `log_epoch()` - comprehensive epoch logging (20+ metrics)
- `log_histograms()` - weight/gradient distributions
- `log_manifold_embedding()` - 3D latent space visualization with p-adic metadata

---

## Integration Points for Real-Time Logging

### 1. Batch-Level Metrics (inside training loop)

**Location**: `src/train.py:683-705` (inside `for batch_ops, batch_idx` loop)
**Current**: Only accumulates loss/acc sums
**Enhancement**: Call `log_batch()` with loss components after each batch
**Benefit**: Real-time loss curve updates during training

### 2. Loss Component Breakdown

**Location**: `src/train.py:694-695`
**Current**: `losses['total']` extracted, components discarded
**Enhancement**: Log individual loss components (hierarchy, coverage, separation, radial, geodesic, etc.)
**Benefit**: Diagnose which loss is dominating/stuck

### 3. Gradient Statistics

**Location**: `src/train.py:699` (after `clip_grad_norm_`)
**Current**: Gradient norm computed but not logged
**Enhancement**: Log gradient norm per batch and per parameter group
**Benefit**: Detect gradient explosion/vanishing in real-time

### 4. Learning Rate Tracking

**Location**: `src/train.py:707` (after `scheduler.step()`)
**Current**: LR not logged
**Enhancement**: Log current LR from scheduler
**Benefit**: Verify warmup/decay behaving correctly

### 5. StateNet Decisions

**Location**: `src/train.py:751-758`
**Current**: StateNet state applied but not logged to TensorBoard
**Enhancement**: Log trainability events, threshold values, Q_delta, annealing state
**Benefit**: Understand when/why components freeze

### 6. Weight Histograms (end of epoch)

**Location**: After validation (~line 815)
**Current**: Not implemented
**Enhancement**: Call `log_histograms()` every N epochs
**Benefit**: Track weight distribution evolution, detect dead neurons

### 7. Embedding Visualization (periodic)

**Location**: After validation, every K epochs
**Current**: Not implemented
**Enhancement**: Call `log_manifold_embedding()` periodically
**Benefit**: Interactive 3D exploration of latent space with 3-adic coloring

---

## Recommended Changes (Summary)

| Location | Current | Enhancement | Flush Frequency |
|----------|---------|-------------|-----------------|
| Line 703 | None | `log_batch()` | Every batch |
| Line 699 | None | Gradient norm scalar | Every batch |
| Line 707 | None | LR scalar | Every epoch |
| Line 758 | None | StateNet metrics | Every eval |
| Line 815 | None | Weight histograms | Every 10 epochs |
| Line 815 | None | Embeddings | Every 50 epochs |

---

## Architecture Recommendation

Replace the direct `SummaryWriter` usage in train.py with the existing `TensorBoardLogger` class:

1. Remove lines 54-59 (duplicate TensorBoard import)
2. Add `from src.utils import TensorBoardLogger` to imports
3. Replace line 648 `SummaryWriter` with `TensorBoardLogger`
4. Replace lines 769-782 with `tb_logger.log_*()` calls
5. Replace line 834 with `tb_logger.close()`
6. Add `flush()` at end of each epoch for real-time updates

The `TensorBoardLogger` class already handles:
- Null safety (no-op when TensorBoard unavailable)
- Proper flushing for real-time updates
- Structured metric naming conventions
- P-adic specific metric groups

---

## Metrics to Log

### Batch-Level (Every Batch)
- `Batch/Loss` - total loss
- `Batch/CE_A`, `Batch/CE_B` - cross-entropy components
- `Batch/KL_A`, `Batch/KL_B` - KL divergence
- `Batch/GradNorm` - total gradient norm

### Epoch-Level (Every Eval)
- `Loss/train`, `Loss/val` - epoch averages
- `Accuracy/train`, `Accuracy/val`
- `Coverage` - perfect reconstruction rate
- `Hierarchy/corr/VAE_A`, `Hierarchy/corr/VAE_B` - radial hierarchy correlations
- `Hierarchy/Q/VAE_A`, `Hierarchy/Q/VAE_B` - structure capacity
- `Hierarchy/dist_corr` - distance correlation
- `LR/scheduled` - current learning rate

### StateNet-Level (When Changed)
- `StateNet/encoder_a_trainable` - 0/1
- `StateNet/encoder_b_trainable` - 0/1
- `StateNet/controller_trainable` - 0/1
- `StateNet/coverage_fix_threshold` - current threshold
- `StateNet/hierarchy_patience` - current patience
- `StateNet/Q_delta` - Q change from cycle start
- `StateNet/best_Q` - best Q achieved

### Loss Components (Every Batch)
- `Loss/rich_hierarchy` - weighted RichHierarchyLoss
- `Loss/radial` - RadialHierarchyLoss
- `Loss/geodesic` - PAdicGeodesicLoss
- `Loss/rank` - GlobalRankLoss
- `Loss/monotonic` - MonotonicRadialLoss

### Periodic (Every N Epochs)
- Weight histograms: `Weights/{param_name}`
- Gradient histograms: `Gradients/{param_name}`
- Embedding projections: `Embedding/VAE_A_Poincare`, `Embedding/VAE_B_Poincare`
