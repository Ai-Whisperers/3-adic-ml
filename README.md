# 3-Adic ML

**Deep learning training pipelines and mathematical foundations for p-adic variational autoencoders.**

This repository contains the core machine learning infrastructure for training VAEs with hyperbolic latent spaces aligned to p-adic (specifically 3-adic) ultrametric structures.

## Relationship to ultrametric-antigen-AI

This repository provides the **mathematical and deep learning foundations** for the [ultrametric-antigen-AI](https://github.com/Ai-Whisperers/ultrametric-antigen-AI) project.

| Repository | Focus | Content |
|------------|-------|---------|
| **3-adic-ml** (this repo) | Deep Learning & Mathematics | VAE architectures, hyperbolic geometry, training pipelines, loss functions |
| **ultrametric-antigen-AI** | Bioinformatics Application | Antigen analysis, protein/codon pipelines, immunological research |

The ultrametric-antigen-AI project uses the trained models and mathematical framework developed here to analyze hierarchical relationships in biological data (antigens, proteins, codons).

## Architecture Overview

**Dual VAE + True Hyperbolic Geometry + LR Controller**

```
Input (9 ternary values, {-1, 0, 1})
    |
+-- Encoder A --+    +-- Encoder B --+
|  9->128->64   |    |  9->128->64   |
|  mu_A, sig_A  |    |  mu_B, sig_B  |
+------+--------+    +------+--------+
       |                    |
   z_tangent (16-dim)   z_tangent
       |                    |
   +----------------------------+
   |  DualHyperbolicProjection  |
   |  tangent_net + expmap0     |
   +----------------------------+
       |                    |
   z_A_hyp              z_B_hyp       <- Poincare manifold points
       |                    |
   Decoder A            Decoder B
       |
   Reconstruction (27 logits -> 9 x 3 ternary)
```

### Core Components

| Component | Structure | Purpose |
|-----------|-----------|---------|
| **VAE-A** | Encoder 9->128->64, Decoder 16->64->27 | Coverage (reconstruction) |
| **VAE-B** | Same structure, independent weights | Hierarchy learning |
| **Hyperbolic Projection** | Tangent net + expmap0 -> Poincare ball | True hyperbolic mapping |
| **LR Controller** | MetricBasedLR with Q-gated thresholds | Dynamic LR scale control |

### What Makes It "P-Adic"

1. **Data**: All 19,683 ternary operations (3^9) with values {-1, 0, 1}
2. **3-adic valuation**: v_3(n) measures divisibility by powers of 3
3. **Geometric encoding**: High valuation -> near origin, low valuation -> near boundary
4. **Loss alignment**: Poincare distances aligned to 3-adic valuations (ultrametric -> hyperbolic)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ai-Whisperers/3-adic-ml.git
cd 3-adic-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install monitoring extras
pip install tqdm psutil
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on RTX 3050 6GB)

## Usage

### Training

```bash
# Run training with V6.0 configuration
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

### Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

Key metrics:
- **Q metric**: `Q = dist_corr + 1.5 * |hierarchy|` (composite quality)
- **Coverage**: Reconstruction accuracy
- **Hierarchy**: Spearman correlation between valuation and radius
- **LR scales**: Per-component learning rate multipliers

## Project Structure

```
src/
├── core/           # 3-adic algebra (TernarySpace singleton)
├── geometry/       # Hyperbolic operations (Poincare ball via geoopt)
├── losses/         # Training objectives (config-driven composition)
├── models/         # VAE architectures (encoder/decoder/projection)
├── config/         # Constants, paths, StateNetConfig
├── presets/        # YAML experiment configurations
├── utils/          # Checkpoints, TensorBoard, hardware monitoring
└── train.py        # Training entry point (includes hierarchy metrics)
```

### Key Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | TernaryVAEV6, TernaryVAEV6Controllable, EncoderHead |
| `src/models/lr_controller.py` | MetricBasedLR, TrainingMetrics, LR scale control |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/config/statenet_config.py` | StateNetConfig dataclass |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/core/ternary.py` | Immutable 3-adic field logic |
| `src/losses/combined.py` | Config-driven loss composition |

## Configuration

Training uses YAML configuration files. See `src/presets/v6.yaml` for a complete example.

### Key Configuration Sections

```yaml
# Model architecture
model:
  name: TernaryVAEV6Controllable
  hidden_dim: 128
  latent_dim: 16

# Training parameters
training:
  epochs: 200
  batch_size: 512
  lr: 1e-3

# Loss functions (config-driven)
loss:
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0
  radial:
    enabled: true
  monotonic:
    enabled: true

# LR Controller (differential learning rates)
option_c:
  enabled: true
  encoder_a_lr_scale: 0.05   # Coverage encoder (slowest)
  encoder_b_lr_scale: 0.1    # Hierarchy encoder (medium)
  projections_lr_scale: 1.0  # Projections (fastest)

# StateNet controller settings
statenet:
  enabled: true
  initial:
    encoder_a_trainable: false
    encoder_b_trainable: true
    projections_trainable: true
```

## Mathematical Background

### 3-Adic Valuation

The 3-adic valuation v_3(n) is the largest k such that 3^k divides n:

| n | v_3(n) | Interpretation |
|---|--------|----------------|
| 1, 2, 4, 5, 7, 8 | 0 | Not divisible by 3 |
| 3, 6, 12, 15 | 1 | Divisible by 3 |
| 9, 18, 36 | 2 | Divisible by 9 |
| 27, 54 | 3 | Divisible by 27 |
| 0 | 9 | Convention (infinity) |

### Hierarchy Encoding

Operations with high valuation map to small radii (near Poincare ball origin):
- v=0 -> radius ~ 0.85 (boundary)
- v=9 -> radius ~ 0.10 (origin)

This creates a natural hierarchical structure where "more fundamental" operations (higher valuation) are geometrically central.

### Loss Functions

| Loss | Purpose |
|------|---------|
| **RichHierarchyLoss** | Unified hierarchy + coverage + separation |
| **PAdicGeodesicLoss** | Poincare distance alignment |
| **RadialHierarchyLoss** | Direct radius enforcement per valuation |
| **GlobalRankLoss** | Soft ranking violations |
| **MonotonicRadialLoss** | Per-level ordering constraints |

## Documentation

- `CLAUDE.md` - Detailed architecture documentation
- `src/README.md` - Module documentation and integration guide
- `docs/FAQ.md` - Frequently asked questions
- `docs/audits/` - Codebase audit reports

## Hardware Requirements

Tested on:
- **GPU**: NVIDIA RTX 3050 (6GB VRAM)
- **RAM**: 16GB minimum recommended
- **Storage**: ~100MB for model checkpoints

The training pipeline includes:
- Automatic OOM (Out of Memory) handling
- Emergency checkpoint saving
- GPU/RAM monitoring via `HardwareMonitor`

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Acknowledgments

This work builds on:
- [geoopt](https://github.com/geoopt/geoopt) - Riemannian optimization in PyTorch
- Theoretical foundations from p-adic analysis and hyperbolic geometry
