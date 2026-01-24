# src/ - P-Adic VAE Source Code

**Last Updated**: 2026-01-24

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
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
    separation_margin: 0.01      # configurable
  radial:
    enabled: true
    valuation_weight_exponent: 0.25  # configurable
    margin_step_factor: 0.5          # configurable
  monotonic:
    enabled: true
    target_loss_weight: 0.5          # configurable
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

**Key Files:** `vae.py`, `hyperbolic_projection.py`

#### Model Variants (V6.0)

| Model | Description | Anchor Checkpoint |
|-------|-------------|-------------------|
| `TernaryVAEV6` | Dual VAE with true hyperbolic geometry | Recommended |
| `TernaryVAEV6Controllable` | Dual VAE + StateNet trainability control | Recommended |

#### Architecture Components

1. **Encoder**: Input → hidden → (μ, log_σ)
2. **Reparameterization**: z_euc = μ + σ * ε (Euclidean)
3. **Hyperbolic Projection**: z_hyp = project(z_euc) → Poincare ball
4. **Decoder**: z → hidden → logits (9×3 for ternary)

**Known Issue:** Decoder currently uses z_euc, not z_hyp. See Architecture Issues below.

---

### src/config/ - Configuration

**Files:** `constants.py`, `paths.py`

| Constant | Value | Usage |
|----------|-------|-------|
| `N_TERNARY_OPERATIONS` | 19683 | Coverage calculation |
| `STATENET_COVERAGE_FIX_THRESHOLD` | 0.99 | Fix encoder when coverage drops |
| `STATENET_COVERAGE_TRAIN_THRESHOLD` | 0.999 | Allow training when coverage above |
| `PROJECT_ROOT` | Auto-detected | Path resolution |

---

### src/presets/ - YAML Configurations

**File:** `5.12.4.yaml` (and other versioned configs)

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
| `checkpoint_validator.py` | Config/checkpoint validation |
| `coverage_evaluator.py` | VAE coverage evaluation |
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

## Architecture (V6.0 - True Hyperbolic)

All architectural issues have been resolved with proper geoopt integration:

| Issue | Status | Solution |
|-------|--------|----------|
| **Decoder uses z_euc** | ✅ Fixed | Decoder receives `logmap0(z_hyp)` |
| **Euclidean reparameterization** | ✅ Fixed | Sample in tangent space (which IS Euclidean) |
| **Euclidean projection math** | ✅ Fixed | Use `expmap0` instead of direction × radius |

### How It Works

1. **Encoder** outputs μ, logvar in tangent space T₀M at origin
2. **Reparameterization**: `z_tangent = μ + ε * σ` (valid - tangent space is Euclidean)
3. **Projection**: `z_hyp = expmap0(transform(z_tangent))` (true hyperbolic)
4. **Losses**: Operate on `z_hyp` using `poincare_distance`
5. **Decoder**: Receives `logmap0(z_hyp)` (back to tangent space)

### Key Insight

The tangent space at the origin T₀M **IS** Euclidean ℝⁿ. This means:
- Standard MLPs work in tangent space
- Gaussian sampling is valid in tangent space
- `expmap0`/`logmap0` provide the bridge to/from the hyperbolic manifold

### Implementation Files

- `src/models/hyperbolic_projection.py`: Uses `expmap0` for projection
- `src/models/vae.py`: Uses `logmap0` for decoder input
- `src/geometry/poincare.py`: Provides `exp_map_zero`, `log_map_zero` via geoopt

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
# Production training
python src/train.py --config src/presets/5.12.4.yaml

# Validate config only (no training)
python src/train.py --config src/presets/5.12.4.yaml --validate-only

# With mixed precision (faster on compatible GPUs)
python src/train.py --config src/presets/5.12.4.yaml --amp

# Custom seed for reproducibility
python src/train.py --config src/presets/5.12.4.yaml --seed 123
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

**Maintainer:** Claude Opus 4.5
