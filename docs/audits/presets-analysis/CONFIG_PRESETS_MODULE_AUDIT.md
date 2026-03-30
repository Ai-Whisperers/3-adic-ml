# Config & Presets Module Audit: src/config/ & src/presets/

**Date**: 2025-01-23
**Scope**: `src/config/__init__.py`, `constants.py`, `paths.py`, `src/presets/*.yaml`
**Lines of Code**: 17 (__init__.py) + 18 (constants.py) + 9 (paths.py) + 267 (preset YAML) = ~311 total

---

## Executive Summary

The configuration system is split into two complementary parts:
- **src/config/**: Python constants and path definitions (compile-time configuration)
- **src/presets/**: YAML-based experiment presets (runtime configuration)

This separation is **architecturally sound**: static constants in Python, dynamic hyperparameters in YAML. The preset file is comprehensive and well-documented, serving as both configuration and documentation.

**Verdict**: The configuration system is **well-designed** with clear separation of concerns.

---

## File Structure

```
src/config/
├── __init__.py      # Re-exports key constants (17 lines)
├── constants.py     # StateNet + general constants (18 lines)
└── paths.py         # Project path definitions (9 lines)

src/presets/
└── research_extended_grokking.yaml  # Training preset (267 lines)
```

---

## Part 1: src/config/ - Python Configuration

### 1.1 Module Exports (__init__.py)

```python
from .constants import (
    STATENET_COVERAGE_FREEZE_THRESHOLD,
    STATENET_COVERAGE_UNFREEZE_THRESHOLD,
    N_TERNARY_OPERATIONS,
)
from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR
```

| Export | Source | Used By |
|--------|--------|---------|
| `STATENET_COVERAGE_FREEZE_THRESHOLD` | constants | statenet.py |
| `STATENET_COVERAGE_UNFREEZE_THRESHOLD` | constants | statenet.py |
| `N_TERNARY_OPERATIONS` | constants | coverage_evaluator.py, data modules |
| `PROJECT_ROOT` | paths | Various |
| `RUNS_DIR` | paths | Training scripts |
| `CHECKPOINTS_DIR` | paths | Checkpoint utilities |

**Assessment**: Clean, minimal exports of key values.

### 1.2 constants.py - StateNet Configuration

```python
# General Constants
N_TERNARY_OPERATIONS = 19683

# StateNet Controller Constants
STATENET_ANNEALING_STEP = 0.005
STATENET_CONTROLLER_GRAD_PATIENCE = 10
STATENET_CONTROLLER_GRAD_THRESHOLD = 0.01
STATENET_CONTROLLER_PATIENCE_CEILING = 20
STATENET_COVERAGE_FLOOR = 0.95
STATENET_COVERAGE_FREEZE_THRESHOLD = 0.99
STATENET_COVERAGE_UNFREEZE_THRESHOLD = 0.999
STATENET_HIERARCHY_PATIENCE_CEILING = 30
STATENET_HIERARCHY_PLATEAU_PATIENCE = 15
STATENET_HIERARCHY_PLATEAU_THRESHOLD = 0.001
STATENET_HYSTERESIS_EPOCHS = 5
STATENET_WARMUP_EPOCHS = 10
STATENET_WINDOW_SIZE = 20
```

| Constant | Value | Purpose | Used In |
|----------|-------|---------|---------|
| `N_TERNARY_OPERATIONS` | 19683 | 3^9 total operations | Core, coverage |
| `STATENET_COVERAGE_FREEZE_THRESHOLD` | 0.99 | Freeze encoder when coverage drops below | StateNet |
| `STATENET_COVERAGE_UNFREEZE_THRESHOLD` | 0.999 | Unfreeze when coverage recovers above | StateNet |
| `STATENET_COVERAGE_FLOOR` | 0.95 | Minimum acceptable coverage | StateNet annealing |
| `STATENET_WARMUP_EPOCHS` | 10 | Epochs before StateNet activates | StateNet |
| `STATENET_HYSTERESIS_EPOCHS` | 5 | Minimum epochs between state changes | StateNet |
| `STATENET_WINDOW_SIZE` | 20 | Moving window for metric averaging | StateNet |
| `STATENET_HIERARCHY_PLATEAU_PATIENCE` | 15 | Epochs before declaring plateau | StateNet |
| `STATENET_HIERARCHY_PLATEAU_THRESHOLD` | 0.001 | Minimum change to avoid plateau | StateNet |
| `STATENET_CONTROLLER_GRAD_THRESHOLD` | 0.01 | Low gradient threshold | StateNet |
| `STATENET_CONTROLLER_GRAD_PATIENCE` | 10 | Epochs of low gradient before freeze | StateNet |
| `STATENET_ANNEALING_STEP` | 0.005 | Threshold adjustment step | StateNet |
| `STATENET_HIERARCHY_PATIENCE_CEILING` | 30 | Max patience for hierarchy | StateNet |
| `STATENET_CONTROLLER_PATIENCE_CEILING` | 20 | Max patience for controller | StateNet |

**Assessment**:
- ✅ All StateNet parameters centralized
- ✅ Consistent naming convention (STATENET_ prefix)
- ⚠️ `N_TERNARY_OPERATIONS` duplicates value from `src/core/ternary.py`

### 1.3 paths.py - Project Paths

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
PRESETS_DIR = PROJECT_ROOT / "presets"
SRC_PRESETS_DIR = PROJECT_ROOT / "src" / "presets"
```

| Path | Resolves To | Purpose |
|------|-------------|---------|
| `PROJECT_ROOT` | `/d1/VAEs/.../p-adic-vaes` | Repository root |
| `RUNS_DIR` | `{root}/runs` | Training run outputs |
| `CHECKPOINTS_DIR` | `{root}/models/checkpoints` | Saved checkpoints |
| `MODELS_DIR` | `{root}/models` | Model definitions |
| `PRESETS_DIR` | `{root}/presets` | Root-level presets (if any) |
| `SRC_PRESETS_DIR` | `{root}/src/presets` | Source presets |

**Assessment**:
- ✅ Uses pathlib (cross-platform)
- ✅ Relative to file location (portable)
- ⚠️ `PRESETS_DIR` vs `SRC_PRESETS_DIR` could cause confusion
- ⚠️ Not all paths are exported in `__init__.py`

---

## Part 2: src/presets/ - YAML Configuration

### 2.1 Preset File Analysis: research_extended_grokking.yaml

This is a comprehensive 267-line configuration file organized into logical sections.

#### Section Breakdown

| Section | Lines | Purpose |
|---------|-------|---------|
| Device | 14-20 | GPU, AMP, workers |
| Model | 22-44 | Architecture parameters |
| Option C | 46-50 | Partial freeze settings |
| Frozen Checkpoint | 52-56 | Pretrained weights |
| StateNet | 58-72 | Controller overrides |
| Progressive Unfreeze | 74-76 | Disabled |
| Loss | 78-118 | Loss function config |
| Riemannian | 120-122 | Optimizer type |
| Training | 125-174 | Epochs, LR, scheduler |
| Data | 176-179 | Dataset settings |
| Logging | 181-202 | TensorBoard, metrics |
| Checkpoints | 204-218 | Save settings |
| Targets | 220-227 | Success criteria |
| Memory | 229-234 | GPU memory settings |
| Early Stopping | 236-241 | Override settings |
| Analysis | 243-254 | Phase detection |
| Version | 256-267 | Tracking info |

### 2.2 Key Configuration Sections

#### Device Configuration
```yaml
device:
  name: "v5_12_4_extended_grokking"
  cuda_device: 0
  use_amp: false
  pin_memory: true
  num_workers: 4
  empty_cache_freq: 25
```

| Setting | Value | Notes |
|---------|-------|-------|
| `use_amp` | false | Mixed precision disabled for stability |
| `num_workers` | 4 | DataLoader parallelism |
| `empty_cache_freq` | 25 | GPU cache clearing frequency |

#### Model Configuration
```yaml
model:
  name: TernaryVAEV5_11_PartialFreeze
  latent_dim: 16
  hidden_dim: 64
  max_radius: 0.95
  curvature: 1.0
  learnable_curvature: true
  manifold_aware: true
  projection_layers: 2
  projection_dropout: 0.1
  encoder_type: improved
  decoder_type: improved
```

| Parameter | Value | Validated In |
|-----------|-------|--------------|
| `latent_dim` | 16 | vae.py |
| `max_radius` | 0.95 | hyperbolic_projection.py |
| `curvature` | 1.0 | geometry module |
| `learnable_curvature` | true | geoopt integration |

#### Loss Configuration
```yaml
loss:
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
    coverage_weight: 1.0
    separation_weight: 3.0

  radial:
    enabled: true
    inner_radius: 0.08
    outer_radius: 0.90
    weight: 1.0

  geodesic:
    enabled: true
    phase_start_epoch: 50
    weight: 0.4

  rank:
    enabled: true
    weight: 0.5
```

| Loss | Enabled | Weight | Phase |
|------|---------|--------|-------|
| `rich_hierarchy` | ✅ | 5.0/1.0/3.0 | Always |
| `radial` | ✅ | 1.0 | Always |
| `geodesic` | ✅ | 0.4 | After epoch 50 |
| `rank` | ✅ | 0.5 | Always |
| `zero_structure` | ✅ | 0.5/0.3 | Always |

**Note**: Multiple losses enabled simultaneously. The `CombinedLoss` class handles composition.

#### Training Configuration
```yaml
training:
  epochs: 100
  batch_size: 512
  lr: 8.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0

  scheduler:
    type: multi_phase_cosine
    phases:
      - name: exploration
        epoch_range: [0, 150]
        base_lr_scale: 1.0
      - name: grokking_search
        epoch_range: [150, 350]
        base_lr_scale: 0.3
      - name: fine_tuning
        epoch_range: [350, 500]
        base_lr_scale: 0.1
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 100 | Reduced for from-scratch |
| `batch_size` | 512 | Standard for 19683 samples |
| `lr` | 8e-4 | Moderate learning rate |
| `max_grad_norm` | 1.0 | Gradient clipping |

**Issue**: `epochs: 100` but scheduler phases go to 500. Phases 2-3 will never execute.

#### Grokking Detection
```yaml
grokking_detection:
  enabled: true
  monitor_window: 20
  plateau_threshold: 0.0001
  plateau_patience: 15
  accuracy_jump_threshold: 0.02
```

This is a **research feature** for detecting sudden generalization improvements after prolonged training.

#### Success Criteria
```yaml
targets:
  coverage: 1.0
  hierarchy_B: -0.83
  richness: 0.008
  r_v9: 0.12
  distance_correlation: 0.70
  Q_target: 2.2
```

| Target | Value | Meaning |
|--------|-------|---------|
| `coverage` | 1.0 | 100% of 19683 operations |
| `hierarchy_B` | -0.83 | Strong negative correlation |
| `r_v9` | 0.12 | v=9 radius near origin |
| `Q_target` | 2.2 | Structure capacity |

---

## Integration Analysis

### How Config is Consumed

```
src/config/constants.py
       │
       ├──→ src/models/statenet.py (default values)
       │
       └──→ src/config/__init__.py (re-export)
               │
               └──→ src/utils/coverage_evaluator.py
                      (N_TERNARY_OPERATIONS)

src/presets/*.yaml
       │
       └──→ Training scripts (loaded at runtime)
               │
               ├──→ Override statenet defaults
               ├──→ Configure losses
               ├──→ Set training hyperparameters
               └──→ Define success criteria
```

### Constants vs Preset Overrides

| Parameter | constants.py Default | YAML Override |
|-----------|---------------------|---------------|
| `STATENET_COVERAGE_FREEZE_THRESHOLD` | 0.99 | 0.995 |
| `STATENET_COVERAGE_UNFREEZE_THRESHOLD` | 0.999 | 1.0 |
| `STATENET_WARMUP_EPOCHS` | 10 | 10 |
| `STATENET_HYSTERESIS_EPOCHS` | 5 | 5 |
| `STATENET_HIERARCHY_PLATEAU_PATIENCE` | 15 | 10 |
| `STATENET_ANNEALING_STEP` | 0.005 | 0.002 |

**Design**: Python constants are defaults, YAML can override.

---

## Issues Summary

### Critical (0)

None.

### High (0)

None.

### Medium (2)

| Issue | Location | Description |
|-------|----------|-------------|
| M1 | `constants.py:2` | `N_TERNARY_OPERATIONS` duplicates `TernarySpace.N_OPERATIONS` |
| M2 | `research_extended_grokking.yaml:127,143-159` | epochs=100 but scheduler phases go to 500 |

### Low (4)

| Issue | Location | Description |
|-------|----------|-------------|
| L1 | `paths.py:7-8` | Both `PRESETS_DIR` and `SRC_PRESETS_DIR` exist (confusing) |
| L2 | `__init__.py` | Not all paths exported (MODELS_DIR, SRC_PRESETS_DIR missing) |
| L3 | `research_extended_grokking.yaml:54` | Checkpoint path may not exist |
| L4 | `research_extended_grokking.yaml` | Only one preset exists |

---

## Code Quality Assessment

| Metric | Score | Notes |
|--------|-------|-------|
| Organization | 9/10 | Clear separation of concerns |
| Naming | 9/10 | Consistent prefixes, descriptive names |
| Documentation | 8/10 | Good comments in YAML |
| Completeness | 7/10 | Only one preset, some missing exports |
| Consistency | 7/10 | Duplicate N_TERNARY_OPERATIONS |
| Usability | 8/10 | Well-structured YAML |

---

## YAML Schema Analysis

The preset file implicitly defines a configuration schema. Key sections:

```
Config
├── device
│   ├── name: str
│   ├── cuda_device: int
│   ├── use_amp: bool
│   └── ...
├── model
│   ├── name: str (enum: TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze)
│   ├── latent_dim: int
│   └── ...
├── loss
│   ├── rich_hierarchy
│   │   ├── enabled: bool
│   │   └── {component}_weight: float
│   └── ...
├── training
│   ├── epochs: int
│   ├── batch_size: int
│   ├── lr: float
│   └── scheduler: SchedulerConfig
└── targets
    ├── coverage: float [0, 1]
    └── hierarchy_B: float [-1, 0]
```

**Recommendation**: Consider adding a JSON Schema or Pydantic model for validation.

---

## Recommendations

### Should Fix

1. **Remove N_TERNARY_OPERATIONS duplication**:
   ```python
   # constants.py
   from src.core import TERNARY
   N_TERNARY_OPERATIONS = TERNARY.N_OPERATIONS
   ```
   Or import directly from core where needed.

2. **Fix scheduler/epochs mismatch**:
   ```yaml
   training:
     epochs: 500  # Match scheduler phases
   ```
   Or adjust phases to fit within 100 epochs.

### Could Improve

1. **Export all paths**:
   ```python
   # __init__.py
   from .paths import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR, MODELS_DIR, SRC_PRESETS_DIR
   ```

2. **Add more presets**:
   - `quick_test.yaml` (few epochs, small batch)
   - `production.yaml` (optimized defaults)
   - `debug.yaml` (verbose logging)

3. **Add config validation**:
   ```python
   from pydantic import BaseModel

   class TrainingConfig(BaseModel):
       epochs: int = Field(gt=0)
       batch_size: int = Field(gt=0)
       lr: float = Field(gt=0)
   ```

---

## Verdict

**The configuration system demonstrates good software engineering practices** with a clear separation between compile-time constants (Python) and runtime configuration (YAML). The preset file is comprehensive and well-documented, serving as both configuration and documentation for the experiment.

Minor issues include a duplicated constant, mismatched scheduler phases, and only one preset file. These are easily fixable.

**Rating**: 8/10 (Good, minor improvements needed)

---

**Audit completed**: 2025-01-23
**Auditor**: Claude Opus 4.5

---

## Addendum (2026-02-26 Audit)

**Auditor**: Claude Opus 4.6

### Stale Information Corrections

1. **Preset file has changed.** `research_extended_grokking.yaml` no longer exists. Current presets:
   - `src/presets/v6.yaml` (304 lines) — main V6.0 config
   - `src/presets/5.12.4.yaml` (188 lines) — extended grokking config

2. **`src/config/` has changed.** Current files:
   - `statenet_config.py` (233 lines) — 8 nested dataclasses for StateNet config
   - `constants.py` (2 lines) — Only `N_TERNARY_OPERATIONS = 19683`
   - `paths.py` (8 lines) — `PROJECT_ROOT`, `RUNS_DIR`, `CHECKPOINTS_DIR`, `MODELS_DIR`, `PRESETS_DIR`, `SRC_PRESETS_DIR`

3. **`constants.py` was gutted.** All `STATENET_*` constants were removed. They're now in `statenet_config.py` as dataclass defaults. Only `N_TERNARY_OPERATIONS` remains.

4. **Model name in YAML is now `TernaryVAEV6Controllable`**, not `TernaryVAEV5_11_PartialFreeze`.

### Critical Finding: 20+ Silently Ignored YAML Keys

This is the **most important training-readiness issue** in the codebase. See full list in MASTER_AUDIT.md.

Key categories:
- **V5 remnants** (encoder/decoder dropout, logvar clamping) — defined but code never reads them
- **Planned features never implemented** (stratified sampling, adaptive curriculum, early stopping, ZeroStructureLoss)
- **Config name mismatches** (GrokkingDetector params don't match YAML keys — will crash if grokking_detection section has keys)
- **5.12.4.yaml scheduler mismatch** — defines `multi_phase_cosine` but train.py only handles `cosine_warmup_restart` and `cosine`

### New Issues Found (2026-02-26)

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| 20+ silently ignored YAML keys | HIGH | v6.yaml | Users may tune parameters that have zero effect |
| GrokkingDetector param name mismatch | HIGH | v6.yaml vs train.py | YAML keys don't match constructor params — will crash |
| 5.12.4.yaml scheduler type unsupported | MODERATE | 5.12.4.yaml | `multi_phase_cosine` not handled by train.py |
| `patience_ceiling` fields orphaned | LOW | statenet_config.py:40,49 | Loaded but never consumed by MetricBasedLR |
| `PRESETS_DIR`, `MODELS_DIR`, `SRC_PRESETS_DIR` unused | LOW | paths.py | Defined but never imported |

### Updated Rating

**Rating**: 5/10 (was 8/10 — downgraded significantly for massive config drift: 20+ ignored keys, crash bug in grokking config)
