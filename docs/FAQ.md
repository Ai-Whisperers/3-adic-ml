# Frequently Asked Questions

## Architecture

### Why is it called "LR Controller"?

It's called "LR Controller" because it controls component trainability through **Learning Rate scaling** rather than boolean freeze/unfreeze flags. Instead of setting `requires_grad=False` to freeze a component, it sets that component's learning rate to 0 (or near-zero), which achieves the same effect but through the optimizer.

This approach provides:
- **Single source of truth**: All control happens via optimizer param groups
- **Continuous control**: Soft freezing via small LR (e.g., 0.05×) vs hard freeze (0.0)
- **Easy monitoring**: LR scales are simple to log and visualize

Source: `src/models/lr_controller.py` line 6: `"""Learning Rate Controller - Unified Training Control via Optimizer.`

---

### Why is there no `statenet.py` file?

The "StateNet" concept is implemented across multiple files rather than a single monolithic file:

| File | Responsibility |
|------|----------------|
| `src/config/statenet_config.py` | Configuration dataclasses |
| `src/models/lr_controller.py` | Decision logic (`MetricBasedLR`) |
| `src/models/vae.py` | Trainability methods on model |
| `src/train.py` | Integration and wiring |

This separation follows single-responsibility principle: configuration is separate from logic, which is separate from the model it controls.

---

### Why a dual VAE architecture (VAE-A and VAE-B)?

The dual architecture serves different learning objectives:

| VAE | Primary Objective | Learning Rate |
|-----|-------------------|---------------|
| **VAE-A** | Coverage (reconstruction accuracy) | 0.05× (slowest) |
| **VAE-B** | Hierarchy (p-adic structure) | 0.1× (medium) |

This implements **Complementary Learning Systems** theory:
- **Slow pathway** (VAE-A): Consolidates stable representations, preserves reconstruction
- **Fast pathway** (VAE-B): Adapts quickly to geometric structure

The projections layer learns at 1.0× (full rate) as the fastest adapter.

---

### What makes this a "p-adic" VAE?

Three key aspects:

1. **Data**: All 19,683 ternary operations (3^9) with values {-1, 0, 1}
2. **3-adic valuation**: `v_3(n)` measures how many times 3 divides n
3. **Geometric encoding**: High valuation → near origin, low valuation → near boundary

The loss functions align Poincaré distances with 3-adic valuations, creating a bridge from discrete ultrametric space to continuous hyperbolic space.

---

### Why Poincaré ball / hyperbolic geometry?

Hyperbolic space naturally represents hierarchical structures because:
- **Exponential growth**: Volume grows exponentially with radius (like tree branching)
- **Ultrametric embedding**: p-adic ultrametrics embed naturally into hyperbolic space
- **Distance properties**: Points near the boundary are far from everything (isolation)

The Poincaré ball is a specific model of hyperbolic space that's bounded (radius < 1), making it practical for neural networks.

---

## Technical Decisions

### Why use float64 instead of float32?

Float64 is required for **numerical stability** in hyperbolic geometry operations near the Poincaré ball boundary (radius → 1.0). The geoopt library performs expmap/logmap calculations that lose precision in float32.

Files enforcing float64:
- `src/models/vae.py` (model init and forward)
- `src/models/hyperbolic_projection.py` (all Poincaré operations)
- `src/losses/*.py` (loss computations)

Note: The `LearnableLRController` uses float32 because it's a meta-controller (metrics → LR scales), not performing geometry operations.

---

### Why `expmap0`/`logmap0` instead of `direction * radius` projection?

The V6.0 architecture uses **true hyperbolic geometry**:

| Approach | Method | Correctness |
|----------|--------|-------------|
| Old (V5.x) | `direction * radius` | Euclidean approximation |
| New (V6.0) | `expmap0(tangent_vector)` | True hyperbolic projection |

`expmap0` (exponential map at origin) properly maps tangent vectors to the Poincaré manifold following geodesics. `logmap0` is its inverse, mapping manifold points back to tangent space for the decoder.

---

### What is the Q metric?

Q is a composite quality metric combining hierarchy and distance correlation:

```
Q = dist_corr + 1.5 × |hierarchy|
```

Where:
- `dist_corr`: Spearman correlation between pairwise radii differences and valuation differences
- `hierarchy`: Spearman correlation between valuation and radius (negative is good)

Q guides threshold annealing in the LR controller and is used for checkpointing (`best_Q.pt`).

---

### What is the TERNARY singleton?

`TERNARY` is a singleton instance of `TernarySpace` that provides O(1) lookups for all 3-adic operations via precomputed lookup tables:

```python
from src.core import TERNARY

valuations = TERNARY.valuation(indices)      # 3-adic valuation
distances = TERNARY.distance(i, j)           # 3-adic metric
radii = TERNARY.target_radius(indices)       # Valuation → target radius
```

Memory footprint: ~2.7 MB per device (valuation LUT + ternary LUT + properties LUT).

---

## Configuration

### How do I make both VAEs trainable from the start?

Set `encoder_a_trainable: true` in your config:

```yaml
statenet:
  initial:
    encoder_a_trainable: true   # Default is false
    encoder_b_trainable: true   # Default is true
    projections_trainable: true # Default is true
```

Or programmatically:

```python
from src.config import StateNetConfig

config = StateNetConfig.from_dict(yaml_cfg.get('statenet', {}))
config.initial.encoder_a_trainable = True
```

---

### What config file should I use?

Use `v6.yaml` for the current V6.0 architecture with true hyperbolic geometry:

```bash
python src/train.py --config src/presets/v6.yaml
```

Older configs (e.g., `5.12.4.yaml`) may reference deprecated architectures.

---

### How do I adjust learning rates per component?

Use the `option_c` section in your config:

```yaml
option_c:
  enabled: true
  encoder_a_lr_scale: 0.05    # Coverage encoder (slowest)
  encoder_b_lr_scale: 0.1     # Hierarchy encoder (medium)
  projections_lr_scale: 1.0   # Projections (fastest)
```

These are multipliers applied to the base learning rate.

---

## Troubleshooting

### Why am I getting NaN losses?

Common causes:
1. **Float32 precision**: Ensure the model uses float64 for geometry operations
2. **Radius > 1.0**: Points escaped the Poincaré ball (check `max_radius` constraint)
3. **Large learning rate**: Try reducing `training.lr`

---

### Why is coverage dropping during training?

The LR controller may have frozen encoder_A while encoder_B continues training. Check:
1. `statenet.coverage.fix_threshold` - encoder_A freezes when coverage drops below this
2. LR scales in TensorBoard: `LRController/encoder_a_lr_scale`

---

### How do I resume from a checkpoint?

Set the anchor checkpoint in your config:

```yaml
anchor_checkpoint:
  path: runs/checkpoints/v6/best_Q.pt
```

Or use a custom run name to continue from a previous run's final checkpoint.

---

## See Also

- `CLAUDE.md` - Architecture overview
- `src/README.md` - Module documentation and integration guide
- `src/presets/v6.yaml` - Reference configuration
