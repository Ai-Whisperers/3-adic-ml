# src/ - P-Adic VAE Source Code

**Last Updated**: 2026-02-24

---

## CRITICAL: StateNet Integration Guide (V6.0)

This section explains how to properly integrate the training controller with both VAEs trainable.

### Architecture Overview (Corrected)

**Important**: There is NO `src/models/statenet.py` file. The StateNet system is split across:

| File | Purpose |
|------|---------|
| `src/config/statenet_config.py` | **StateNetConfig** dataclass (configuration only) |
| `src/models/lr_controller.py` | **MetricBasedLR** class (decision logic) |
| `src/models/vae.py` | **TernaryVAEV6Controllable** (trainability methods) |
| `src/train.py` | Integration point (wires everything together) |

### Option C: LR-Based Trainability Control

The system uses "Option C" - **all trainability control happens via LR scales**:
- `LR = 0` → component is frozen (no gradient updates)
- `LR > 0` → component is trainable (with differential rates)

This is cleaner than setting `requires_grad=False` because:
1. Single source of truth (optimizer param groups)
2. Continuous control (soft freezing via small LR)
3. Easy to log and monitor

### How to Make Both VAEs Trainable

To train with BOTH encoder_A and encoder_B trainable from the start:

```yaml
# In your preset YAML config
statenet:
  enabled: true
  initial:
    encoder_a_trainable: true    # ← Set to true (default is false)
    encoder_b_trainable: true    # ← Already true by default
    projections_trainable: true  # ← Already true by default

option_c:
  enabled: true
  encoder_a_lr_scale: 0.05      # Still use slower LR for encoder_A
  encoder_b_lr_scale: 0.1       # Medium LR for encoder_B
  projections_lr_scale: 1.0     # Full LR for projections
```

### Integration Code Flow

```python
# 1. Load config from YAML
statenet_cfg = config.get('statenet', {})
option_c_cfg = config.get('option_c', {})

# 2. Create StateNetConfig from dict
sn_config = StateNetConfig.from_dict(statenet_cfg)

# 3. Merge option_c LR scales (if present)
if option_c_cfg.get('enabled', True):
    sn_config.lr_scales.encoder_a = option_c_cfg.get('encoder_a_lr_scale', 0.05)
    sn_config.lr_scales.encoder_b = option_c_cfg.get('encoder_b_lr_scale', 0.1)
    sn_config.lr_scales.projections = option_c_cfg.get('projections_lr_scale', 1.0)

# 4. Create LRController
lr_controller = MetricBasedLR(sn_config)

# 5. Create model with initial trainability from config
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    projections_trainable=sn_config.initial.projections_trainable,
    encoder_a_lr_scale=sn_config.lr_scales.encoder_a,
    encoder_b_lr_scale=sn_config.lr_scales.encoder_b,
    projections_lr_scale=sn_config.lr_scales.projections,
    # ... other args
)

# 6. In training loop (per epoch):
metrics = TrainingMetrics(
    epoch=epoch,
    coverage=avg_val_coverage,
    hierarchy_a=hier_metrics_A['hierarchy'],
    hierarchy_b=hier_metrics_B['hierarchy'],
    dist_corr_a=hier_metrics_A['dist_corr'],
    q_value=hier_metrics_A['Q'],
    grad_norm_projections=controller_grad_norm,
)
controller_state = lr_controller.update(metrics)

# 7. Apply LR scales to optimizer (THE KEY STEP)
update_optimizer_lr_scales(optimizer, base_lr, controller_state['lr_scales'])
```

### Key Classes

#### StateNetConfig (src/config/statenet_config.py)

```python
from src.config import StateNetConfig

# From YAML
config = StateNetConfig.from_dict(yaml_config.get('statenet', {}))

# Access nested config
print(config.initial.encoder_a_trainable)  # False by default
print(config.lr_scales.encoder_a)          # 0.05 by default
print(config.coverage.fix_threshold)       # 0.995 by default
```

#### MetricBasedLR (src/models/lr_controller.py)

```python
from src.models import MetricBasedLR, TrainingMetrics

# Create controller
controller = MetricBasedLR(sn_config)

# Update and get LR scales
metrics = TrainingMetrics(epoch=50, coverage=0.98, ...)
state = controller.update(metrics)

# state['lr_scales'] = {'encoder_a': 0.0, 'encoder_b': 0.1, 'projections': 1.0, 'decoders': 1.0}
# state['events'] = ['encoder_a frozen (coverage drop)']
```

#### TernaryVAEV6Controllable (src/models/vae.py)

```python
from src.models import TernaryVAEV6Controllable

model = TernaryVAEV6Controllable(
    encoder_a_trainable=True,   # Start trainable
    encoder_b_trainable=True,   # Start trainable
    projections_trainable=True, # Start trainable
)

# Manual control (rarely needed - use LRController instead)
model.set_encoder_a_trainable(False)
model.set_encoder_b_trainable(True)
model.set_projections_trainable(True)

# Get param groups for optimizer (respects current trainability)
param_groups = model.get_param_groups(base_lr=1e-3)
```

### StateNet Decision Logic

The `MetricBasedLR` makes decisions based on:

| Component | Frozen When | Unfrozen When |
|-----------|-------------|---------------|
| `encoder_a` | Coverage < 0.995 | Coverage ≥ 1.0 AND hierarchy_A stalled |
| `encoder_b` | Hierarchy_B plateaus for `patience` epochs | Hierarchy_B degrades |
| `projections` | Gradient norm < 0.005 for `patience` epochs | Gradient spike detected |

### Warmup and Hysteresis

- **Warmup**: During `timing.warmup_epochs`, no decisions are made (initial states preserved)
- **Hysteresis**: At least `timing.hysteresis_epochs` must pass between state changes

### Complete YAML Config Reference

```yaml
statenet:
  enabled: true

  initial:
    encoder_a_trainable: false    # Start frozen (coverage anchor)
    encoder_b_trainable: true     # Start trainable (hierarchy learner)
    projections_trainable: true   # Start trainable (fast adapter)

  coverage:
    fix_threshold: 0.35           # Freeze encoder_A when coverage drops below
    train_threshold: 0.45         # Unfreeze encoder_A when above (+ stall)
    floor: 0.3                    # Minimum threshold (annealing limit)

  hierarchy:
    plateau_threshold: 0.0005     # Improvement below this = plateau
    plateau_patience: 10          # Epochs before freezing encoder_B
    patience_ceiling: 25          # Max patience
    stall_patience: 5             # For encoder_A stall detection

  controller:
    grad_threshold: 0.005         # Freeze projections when grad norm below
    grad_patience: 5              # Epochs of low grad before freeze
    patience_ceiling: 20          # Max patience
    spike_multiplier: 2.0         # Unfreeze when grad > avg * this

  timing:
    warmup_epochs: 10             # Skip decisions during warmup
    hysteresis_epochs: 5          # Min epochs between state changes
    window_size: 10               # Moving window for metric history

option_c:
  enabled: true
  encoder_a_lr_scale: 0.05        # Coverage encoder: slowest
  encoder_b_lr_scale: 0.1         # Hierarchy encoder: medium
  projections_lr_scale: 1.0       # Projections: fastest
```

---

## Architecture Overview

```
src/
├── core/           # Foundation: 3-adic algebra (TernarySpace singleton)
├── geometry/       # Hyperbolic operations (Poincare ball)
├── losses/         # Training objectives (config-driven composition)
├── models/         # VAE architectures (encoder/decoder/projection)
├── config/         # Constants and paths
├── presets/        # YAML experiment configurations
└── utils/          # Checkpoints, coverage, TensorBoard
```

### Data Flow (True Hyperbolic - V6.0)

```
Input (ternary ops)
       │
       ▼
   ┌───────────┐
   │  Encoder  │ → μ, logvar (tangent space T₀M)
   └───────────┘
       │
       ▼ Reparameterization (in tangent space)
   ┌───────────┐
   │ z_tangent │ = μ + ε * σ (Euclidean - tangent space IS Euclidean)
   └───────────┘
       │
       ▼ expmap0 (true hyperbolic projection)
   ┌───────────┐
   │  z_hyp    │ (on Poincaré manifold)
   └───────────┘
       │
       ├──────────────────────────┐
       ▼                          ▼
   ┌───────────┐            ┌───────────┐
   │  logmap0  │            │  Losses   │ (hyperbolic distances)
   └───────────┘            └───────────┘
       │
       ▼
   ┌───────────┐
   │  Decoder  │ ← tangent space (Euclidean-compatible)
   └───────────┘
       │
       ▼
   Output (logits)
```

Key insight: The tangent space at origin T₀M IS Euclidean ℝⁿ. Standard MLPs work in tangent space.
The manifold operations (expmap0, logmap0) provide the non-Euclidean structure.

---

## Module Details

### src/core/ - Ternary Algebra Foundation

**Key File:** `ternary.py`

The `TernarySpace` singleton (`TERNARY`) is the single source of truth for all 3-adic operations:

#### Core Operations (O(1) lookups)

| Operation | Method | Description |
|-----------|--------|-------------|
| Valuation | `TERNARY.valuation(indices)` | 3-adic valuation v_3(n) via LUT |
| Distance | `TERNARY.distance(i, j)` | 3-adic metric d_3 = 3^(-v_3(\|i-j\|)) |
| Distance matrix | `TERNARY.distance_matrix(indices)` | Pairwise 3-adic distances |
| Target radius | `TERNARY.target_radius(indices)` | Map valuation → Poincaré radius |
| To ternary | `TERNARY.to_ternary(indices)` | Index → 9-trit representation |
| From ternary | `TERNARY.from_ternary(ternary)` | 9-trit → index |

#### Structured Properties (Option B)

Each index has precomputed algebraic properties accessible via O(1) lookup:

| Method | Description |
|--------|-------------|
| `digit_count(indices)` | Number of non-zero digits (0-9) |
| `digit_sum(indices)` | Sum of digits (-9 to +9) |
| `first_nonzero(indices)` | Position of first non-zero digit |
| `last_nonzero(indices)` | Position of last non-zero digit |
| `parent(indices)` | Parent in 3-adic tree (n // 3) |
| `level_rank(indices)` | Rank within same-valuation cohort |
| `level_count(level)` | Population at valuation level |
| `properties(indices)` | Dict of all properties |

**Constants:**
- `N_DIGITS = 9` (trits per operation)
- `N_OPERATIONS = 19683` (3^9 total operations)
- `MAX_VALUATION = 9`
- `N_PROPERTIES = 7` (structured property columns)

**Memory per device:**
- Valuation LUT: 157 KB
- Ternary LUT: 1.4 MB (float64)
- Properties LUT: 1.1 MB
- **Total: ~2.7 MB**

---

### src/geometry/ - Hyperbolic Operations

**Key File:** `poincare.py`

Provides numerically stable Poincaré ball operations via geoopt backend.
All functions automatically use the correct device based on input tensors.

#### Actively Used Functions

| Function | Usage |
|----------|-------|
| `hyperbolic_radius(z, c)` | Distance from origin (used by all hierarchy losses) |
| `poincare_distance(x, y, c)` | Distance between points (geodesic alignment) |
| `exp_map_zero(v, c)` | Tangent space → manifold (HyperbolicProjection) |
| `log_map_zero(z, c)` | Manifold → tangent space (VAE decoder) |
| `lambda_x(x, c)` | Conformal factor (HyperbolicKLDivergence) |
| `get_riemannian_optimizer()` | RiemannianAdam/SGD factory |

#### Available Utilities

| Function | Purpose |
|----------|---------|
| `mobius_add(x, y, c)` | Hyperbolic translation |
| `geodesic(x, y, t, c)` | Interpolation along geodesic |
| `parallel_transport(x, y, v, c)` | Transport tangent vectors |
| `poincare_distance_matrix(z, c)` | All pairwise distances |

**Important:** All losses use `hyperbolic_radius()` for radius computation (not Euclidean norm).
This is the canonical way to compute distances from origin in hyperbolic space.

---

### src/losses/ - Training Objectives

**Key Files:** `combined.py`, `padic_geodesic.py`

#### CombinedLoss (Config-Driven Factory)

Reads YAML `loss:` section and instantiates enabled losses:

```yaml
loss:
  learnable_weights: false  # Enable trainable loss weights (V6.1)
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
    coverage_weight: 1.0
    separation_weight: 3.0
  radial:
    enabled: true
    inner_radius: 0.08
    outer_radius: 0.90
    radial_weight: 1.0
  geodesic:
    enabled: true
    phase_start_epoch: 30
    weight: 0.4
  rank:
    enabled: true
    weight: 0.5
    temperature: 0.1
```

#### Loss Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `RichHierarchyLoss` | Unified hierarchy + coverage + separation | `separation_margin` |
| `RadialHierarchyLoss` | Direct radius enforcement per valuation | `valuation_weight_exponent`, `margin_step_factor` |
| `PAdicGeodesicLoss` | Poincare distance alignment | `max_target_distance`, `n_pairs` |
| `GlobalRankLoss` | Soft ranking violation | `temperature` |
| `MonotonicRadialLoss` | Level-wise ordering constraints | `target_loss_weight` |

**Hierarchy Principle:** High valuation (v_3) → small radius (near origin)

---

### src/models/ - VAE Architectures

**Key Files:** `vae.py`, `hyperbolic_projection.py`, `lr_controller.py`

> **Note**: There is NO `statenet.py` file. See the **StateNet Integration Guide** at the top of this document.

#### Model Variants (V6.0)

| Model | Description | Use Case |
|-------|-------------|----------|
| `TernaryVAEV6` | Dual VAE with true hyperbolic geometry | Base architecture |
| `TernaryVAEV6Controllable` | Dual VAE + StateNet trainability control | Production training |

#### Modular Architecture (V6.0)

```
TernaryVAEV6Controllable
├── head_A: EncoderHead              ← encoder_a_trainable (slow learner)
│   ├── backbone (9→128→128→64)
│   ├── fc_mu (64→16)
│   └── fc_logvar (64→16)
│
├── head_B: EncoderHead              ← encoder_b_trainable (medium learner)
│   ├── backbone (9→128→128→64)
│   ├── fc_mu (64→16)
│   └── fc_logvar (64→16)
│
├── projections: DualHyperbolicProjection  ← controller_trainable (fast adapter)
│   ├── proj_A.tangent_net + expmap0
│   └── proj_B.tangent_net + expmap0
│
├── decoder_A (16→64→128→27)         ← always trainable
└── decoder_B (16→64→128→27)         ← always trainable
```

#### EncoderHead Class

The `EncoderHead` class bundles encoder backbone + mu/logvar heads with trainability control:

```python
from src.models import EncoderHead

head = EncoderHead(hidden_dim=64, latent_dim=16, encoder_type="improved")
mu, logvar = head(x)           # Forward pass
head.set_trainable(False)      # Freeze all parameters
params = head.get_trainable_params()  # Get trainable params for optimizer
```

#### StateNet → VAE Mapping

| StateNet State | VAE Component | Learning Rate | Description |
|----------------|---------------|---------------|-------------|
| `encoder_a_trainable` | `head_A` | 0.05× base | Coverage encoder (slowest) |
| `encoder_b_trainable` | `head_B` | 0.1× base | Hierarchy encoder (medium) |
| `controller_trainable` | `projections` | 1.0× base | Tangent transform (fastest) |

#### Data Flow

1. **EncoderHeads**: Input → backbone → (μ, log_σ) in tangent space T₀M
2. **Reparameterization**: z_tangent = μ + σ * ε (Euclidean - tangent space IS Euclidean)
3. **Projections**: z_hyp = expmap0(transform(z_tangent)) → Poincaré manifold
4. **Losses**: Operate on z_hyp using true hyperbolic distances
5. **Decoder**: logmap0(z_hyp) → logits (9×3 for ternary)

---

### src/config/ - Configuration

**Files:** `constants.py`, `paths.py`

| Constant | Value | Location |
|----------|-------|----------|
| `N_TERNARY_OPERATIONS` | 19683 | `constants.py` |
| `PROJECT_ROOT` | Auto-detected | `paths.py` |
| `RUNS_DIR` | `PROJECT_ROOT / "runs"` | `paths.py` |
| `CHECKPOINTS_DIR` | `PROJECT_ROOT / "models" / "checkpoints"` | `paths.py` |

Coverage/hierarchy thresholds are now in `StateNetConfig` (not module-level constants):
- `config.coverage.fix_threshold` (default: 0.995)
- `config.coverage.train_threshold` (default: 1.0)

---

### src/presets/ - YAML Configurations

**File:** `v6.yaml` (and other versioned configs)

Sections: device, model, loss, training, scheduler, targets, logging, checkpoints, statenet

Example loss config flow:
```
YAML loss.radial.valuation_weight_exponent: 0.3
  → train.py loads config
  → CombinedLoss(loss_cfg)
  → RadialHierarchyLoss(valuation_weight_exponent=0.3)
```

---

### src/utils/ - Utilities

| File | Purpose |
|------|---------|
| `checkpoint.py` | Safe checkpoint loading |
| `checkpoint_validator.py` | Training config validation |
| `tensorboard_logger.py` | TensorBoard logging (batch/epoch metrics, histograms, embeddings) |
| `hardware_monitor.py` | GPU/RAM monitoring, OOM diagnostics |

---

## Training Features

### Progress Monitoring

The training script (`train.py`) includes real-time progress monitoring:

| Feature | Dependency | Fallback |
|---------|------------|----------|
| Progress bars | `tqdm` | Simple print statements |
| GPU memory tracking | PyTorch CUDA | Shows "N/A" on CPU |
| RAM monitoring | `psutil` | Shows "N/A" if not installed |

**With tqdm installed:**
```
Training:  45%|████████████                    | 45/100 [12:34<15:23, 16.8s/epoch]
Ep 045:  78%|███████████████████████          | 28/36 [00:12<00:03] loss=0.0234 GPU: 2.1/6.0GB
```

### Hardware Monitoring

The `HardwareMonitor` class tracks resource usage:

```python
from src.utils import HardwareMonitor

monitor = HardwareMonitor(device, warn_threshold=0.9)

# Get GPU memory (returns dict with allocated, reserved, peak, total)
gpu_mem = monitor.get_gpu_memory_gb()

# Get formatted status string
print(monitor.get_status_string())  # "GPU: 2.1/6.0GB (35%) | RAM: 8.2/32.0GB (26%)"

# Check for high memory warning
warning = monitor.check_memory_warning()
if warning:
    print(warning)
```

### OOM Handling

Training includes automatic OOM (Out of Memory) handling:

1. Catches `torch.cuda.OutOfMemoryError`
2. Logs diagnostic information (GPU/RAM usage)
3. Saves emergency checkpoint before exit
4. Suggests reduced batch size

**Example OOM output:**
```
[OOM] CUDA Out of Memory at epoch 45, batch 28
[OOM] GPU Memory: allocated=5.8GB, reserved=6.0GB, peak=6.0GB
[OOM] RAM: used=12.4GB, available=19.6GB (39%)
[OOM] Emergency checkpoint saved: runs/.../checkpoints/emergency_oom_epoch_45.pt
[OOM] Suggestion: Reduce batch_size from 512 to 256
```

### TensorBoard Logging

The `TensorBoardLogger` class provides comprehensive logging:

| Method | Logged Metrics |
|--------|----------------|
| `log_batch()` | Loss, CE, KL per batch |
| `log_hyperbolic_epoch()` | Correlations, radii, StateNet |
| `log_histograms()` | Weight/gradient distributions |
| `log_manifold_embedding()` | 3D latent space visualization |

**Config-driven logging:**
```yaml
logging:
  enhanced_metrics:
    enabled: true       # Enable batch-level logging
  histogram_every: 10   # Log weight histograms every N epochs
  embedding_every: 50   # Log embeddings every N epochs
```

---

## Key Concepts

### 3-Adic Valuation

v_3(n) = largest k such that 3^k divides n

| n | v_3(n) | Interpretation |
|---|--------|----------------|
| 1, 2, 4, 5, 7, 8 | 0 | Not divisible by 3 |
| 3, 6, 12, 15 | 1 | Divisible by 3 |
| 9, 18, 36 | 2 | Divisible by 9 |
| 27, 54 | 3 | Divisible by 27 |
| 0 | 9 | Convention (infinity) |

### Hierarchy Target

Operations with high valuation should map to small radii (near Poincare ball origin):
- v=0 → radius ≈ 0.85 (boundary)
- v=9 → radius ≈ 0.10 (origin)

### Reproducibility

All stochastic operations use seeded `torch.Generator`:
- Loss pair sampling
- TensorBoard embedding sampling
- `TERNARY.sample_indices(n, generator=gen)`

---

## Architecture (V6.0 - True Hyperbolic + Modular)

All architectural issues have been resolved with proper geoopt integration:

| Issue | Status | Solution |
|-------|--------|----------|
| **Decoder uses z_euc** | ✅ Fixed | Decoder receives `logmap0(z_hyp)` |
| **Euclidean reparameterization** | ✅ Fixed | Sample in tangent space (which IS Euclidean) |
| **Euclidean projection math** | ✅ Fixed | Use `expmap0` instead of direction × radius |
| **controller_trainable unused** | ✅ Fixed | Wired to `projections` component |
| **Encoder duplication** | ✅ Fixed | Modularized into `EncoderHead` class |

### How It Works

1. **EncoderHeads** output μ, logvar in tangent space T₀M at origin
2. **Reparameterization**: `z_tangent = μ + ε * σ` (valid - tangent space is Euclidean)
3. **Projections**: `z_hyp = expmap0(transform(z_tangent))` (true hyperbolic)
4. **Losses**: Operate on `z_hyp` using `poincare_distance`
5. **Decoder**: Receives `logmap0(z_hyp)` (back to tangent space)

### Complementary Learning Systems

The architecture implements CLS theory via StateNet:
- **Slow pathway** (encoders): Consolidate learned representations, fix when objectives met
- **Fast pathway** (projections): Continuously adapt to geometric structure
- **Q-metric**: `Q = dist_corr + 1.5 × |hierarchy|` guides threshold annealing

### Key Insight

The tangent space at the origin T₀M **IS** Euclidean ℝⁿ. This means:
- Standard MLPs work in tangent space
- Gaussian sampling is valid in tangent space
- `expmap0`/`logmap0` provide the bridge to/from the hyperbolic manifold

### Implementation Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | `EncoderHead`, `TernaryVAEV6`, `TernaryVAEV6Controllable` |
| `src/models/hyperbolic_projection.py` | `HyperbolicProjection`, `DualHyperbolicProjection` (expmap0) |
| `src/models/lr_controller.py` | `MetricBasedLR` (trainability controller; legacy schedulers archived in `archive-for-review/`) |
| `src/config/statenet_config.py` | `StateNetConfig` dataclass (centralized config) |
| `src/geometry/poincare.py` | `exp_map_zero`, `log_map_zero` via geoopt |

---

## Quick Reference

### Import Patterns

```python
# Core - basic operations
from src.core import TERNARY, valuation, distance, target_radius
valuations = TERNARY.valuation(indices)
radii = TERNARY.target_radius(indices, inner=0.1, outer=0.9)

# Core - structured properties (Option B)
from src.core import digit_count, parent, level_rank
dc = digit_count(indices)       # Number of non-zero digits
p = parent(indices)             # Parent in 3-adic tree
props = TERNARY.properties(indices)  # Dict of all properties

# Geometry
from src.geometry import hyperbolic_radius, poincare_distance
radii = hyperbolic_radius(z, c=1.0)  # Preferred for radius computation
dist = poincare_distance(z1, z2, c=1.0)  # Distance between points

# Losses
from src.losses import CombinedLoss
loss_fn = CombinedLoss(config['loss'], curvature=1.0)

# Config
from src.config import N_TERNARY_OPERATIONS, PROJECT_ROOT

# Hardware monitoring
from src.utils import HardwareMonitor
monitor = HardwareMonitor(device, warn_threshold=0.9)
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install monitoring extras
pip install tqdm psutil
```

### Running Training

```bash
# Production training (V6.0 true hyperbolic)
python src/train.py --config src/presets/v6.yaml

# Validate config only (no training)
python src/train.py --config src/presets/v6.yaml --validate-only

# With mixed precision (faster on compatible GPUs)
python src/train.py --config src/presets/v6.yaml --amp

# Custom seed for reproducibility
python src/train.py --config src/presets/v6.yaml --seed 123
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to YAML config file (required) |
| `--seed N` | Random seed (default: 42) |
| `--device cuda/cpu` | Training device (default: cuda) |
| `--validate-only` | Validate config and exit |
| `--force` | Continue even if validation fails |
| `--amp` | Use automatic mixed precision |
| `--name NAME` | Custom run name |

---

## Codebase Review Findings (2026-01-26)

This section documents findings from a comprehensive review of the entire `src/` codebase.

### Code Statistics

| Module | Files | Lines (approx) | Purpose |
|--------|-------|----------------|---------|
| `src/config/` | 3 | ~350 | Constants, paths, StateNetConfig dataclass |
| `src/core/` | 2 | ~550 | TernarySpace singleton, 3-adic operations |
| `src/geometry/` | 2 | ~400 | Poincaré ball operations via geoopt |
| `src/losses/` | 4 | ~800 | Hierarchy/geodesic losses, combined loss factory |
| `src/models/` | 4 | ~1200 | VAE architectures, projections, LR controller |
| `src/utils/` | 4 | ~450 | Checkpointing, logging, monitoring, validation |
| `src/train.py` | 1 | ~1000 | Training entry point (includes hierarchy metrics) |

**Total**: ~4500 lines across 19 files

### Confirmed Working Patterns

| Pattern | Location | Status |
|---------|----------|--------|
| Option C (LR-based trainability) | `train.py:636-651`, `lr_controller.py` | ✅ Implemented |
| StateNetConfig → MetricBasedLR flow | `train.py:863-899` | ✅ Wired correctly |
| True hyperbolic geometry | `vae.py`, `hyperbolic_projection.py` | ✅ Uses expmap0/logmap0 |
| Config-driven loss composition | `combined.py` | ✅ Factory pattern |
| Immutable TernarySpace singleton | `ternary.py` | ✅ Thread-safe LUTs |
| Differential LR per component | `vae.py:get_param_groups()` | ✅ Named groups |
| Riemannian optimizer support | `geometry/poincare.py` | ✅ geoopt integration |

### Dead Code Removed (2026-01-26, 2026-02-01)

The following dead code was removed after deep analysis:

| Item | Was In | Reason Removed |
|------|--------|----------------|
| `CheckpointCompatibilityError` | `utils/checkpoint_validator.py` | Exception defined but never raised |
| `AnnealingConfig` | `config/statenet_config.py` | Config loaded but logic never implemented |
| `src/metrics/` module | Entire directory | Orphaned code; useful parts integrated into train.py |
| `CoverageEvaluator` | `utils/coverage_evaluator.py` | Measured generative diversity, not training metric |
| `CheckpointValidator` class | `utils/checkpoint_validator.py` | Duplicated by ModelAuditor in train.py |

**Integrated into train.py:** `tree_coherence()` and `level_stratified_hierarchy()` metrics from the deleted metrics module are now computed inline in `compute_hierarchy_metrics()`.

### Design Decisions (Not Dead Code)

| Item | Location | Rationale |
|------|----------|-----------|
| `proj_B.learnable_curvature=False` | `hyperbolic_projection.py:238` | **Intentional**: Comment says "Share curvature with A" - both projections use same curvature |

### Module Summaries

#### src/config/

- **`constants.py`**: Dataset constants (`N_TERNARY_OPERATIONS=19683`), StateNet thresholds
- **`paths.py`**: `PROJECT_ROOT` auto-detection
- **`statenet_config.py`**: 8 nested dataclasses for complete StateNet configuration

#### src/core/

- **`ternary.py`**: `TernarySpace` singleton with precomputed LUTs
  - ~2.7 MB memory footprint per device
  - All operations O(1) via tensor indexing
  - Structured properties (digit_count, parent, level_rank, etc.)
- **`__init__.py`**: Exports core functions as module-level convenience

#### src/geometry/

- **`poincare.py`**: geoopt-backed hyperbolic operations
  - `exp_map_zero`, `log_map_zero` for manifold ↔ tangent
  - `hyperbolic_radius`, `poincare_distance` for metrics
  - `get_riemannian_optimizer()` factory
- **`__init__.py`**: Clean exports of commonly used functions

#### src/losses/

- **`padic_geodesic.py`**:
  - `RichHierarchyLoss` - unified hierarchy/coverage/separation
  - `PAdicGeodesicLoss` - Poincaré distance alignment
  - `RadialHierarchyLoss` - per-valuation radius targets
  - `GlobalRankLoss` - soft ranking violations
  - `MonotonicRadialLoss` - level-wise ordering
  - `CombinedGeodesicLoss` - wrapper for geodesic + radial (unused, prefer CombinedLoss)
- **`combined.py`**: `CombinedLoss` factory reads YAML, instantiates enabled losses
- **`hyperbolic_kl.py`**: `HyperbolicKLDivergence` with conformal factor correction (not currently used by CombinedLoss)

#### src/models/

- **`vae.py`**:
  - `EncoderHead` - modular encoder backbone + mu/logvar
  - `TernaryVAEV6` - dual VAE with true hyperbolic geometry
  - `TernaryVAEV6Controllable` - adds LR-based trainability control
- **`hyperbolic_projection.py`**:
  - `HyperbolicProjection` - tangent_net + expmap0
  - `DualHyperbolicProjection` - shared curvature, separate tangent_nets
- **`lr_controller.py`**:
  - `MetricBasedLR` - Q-gated threshold decisions
  - Legacy `ScheduleBasedLR`/`LearnableLRController` moved to `archive-for-review/dead_code/`
  - `TrainingMetrics` dataclass
  - `update_optimizer_lr_scales()` - applies scales to optimizer

#### src/utils/

- **`checkpoint.py`**: `safe_load_checkpoint()` with device handling
- **`checkpoint_validator.py`**: `validate_training_config()` for config sanity checks
- **`tensorboard_logger.py`**: `TensorBoardLogger` with batch/epoch/histogram logging
- **`hardware_monitor.py`**: `HardwareMonitor` for GPU/RAM tracking, OOM diagnostics

### New Feature: Learnable Loss Weights (V6.1)

Added trainable loss weights using **homoscedastic uncertainty weighting** (Kendall et al. 2018).

#### The Problem

The system has multiple competing objectives (hierarchy, coverage, separation, geodesic, rank). Fixed weights are guesses that may not be optimal throughout training:
- Early training: coverage matters more (establish reconstruction)
- Mid training: hierarchy matters more (establish structure)
- Late training: separation/geodesic matter more (refine distances)

#### The Solution

Instead of fixed weights, make them **trainable nn.Parameters**:

```yaml
loss:
  learnable_weights: true  # Enable trainable weights
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0   # Used as initial weight
    coverage_weight: 1.0
    separation_weight: 3.0
```

#### Mathematical Formulation

Each loss component `L_i` gets a learnable log-variance parameter `log_σ_i`:

```
effective_weight_i = 1 / (2 * exp(2 * log_σ_i))
loss_contribution_i = effective_weight_i * L_i - log_σ_i
```

The `-log_σ_i` term is **regularization** that prevents weights from collapsing to zero (if σ grows, the regularization term becomes more negative, penalizing the total loss).

#### Initialization

Initial `log_σ` is derived from config weights so training starts at the configured balance:

```
log_σ = -0.5 * log(2 * weight)
```

Example: `hierarchy_weight: 5.0` → `log_σ = -1.15` → `effective_weight = 5.0`

#### Why This Is Trainable (Not Heuristic)

Unlike the removed `AnnealingConfig` which adjusted thresholds based on lagging metrics:
- **Gradients flow through** the weight parameters
- Weights respond to **embedding dynamics**, not computed metrics
- Network **discovers** the optimal curriculum automatically
- No heuristics - the balance emerges from training

#### Integration with Training

The `log_sigma` parameters are part of `CombinedLoss.parameters()`:

```python
# In train.py - loss_fn parameters are automatically included
loss_fn = CombinedLoss(config['loss'], curvature=1.0)
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': loss_fn.parameters(), 'lr': base_lr * 0.1},  # Optional: slower LR for weights
])
```

Or simply let them train with the model:
```python
all_params = list(model.parameters()) + list(loss_fn.parameters())
optimizer = torch.optim.Adam(all_params, lr=base_lr)
```

#### Monitoring

```python
# Current effective weights (what the network learned)
loss_fn.get_learned_weights()
# {'hierarchy': 4.2, 'coverage': 1.8, 'separation': 2.5}

# Raw log_sigma values (for debugging)
loss_fn.get_log_sigmas()
# {'hierarchy': -1.07, 'coverage': -0.59, 'separation': -0.82}

# Log to TensorBoard
for name, weight in loss_fn.get_learned_weights().items():
    writer.add_scalar(f'loss_weights/{name}', weight, epoch)
```

#### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Exploring new loss combinations | **Enable** - let network find balance |
| Reproducing known-good results | **Disable** - use validated fixed weights |
| Long training runs | **Enable** - adapts to training phases |
| Debugging loss interactions | **Enable** - watch weights evolve |

---

**Maintainer:** Claude Opus 4.6
