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

## Direction Geometry (V7+)

### What is `level_prefix_k` and why is it needed?

`level_prefix_k` gives `AngularCoherenceLoss` a per-valuation-level prefix depth for defining direction classes:

```yaml
angular_coherence:
  level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]
```

Without it, a single global `prefix_k=3` is used for all levels. This is insufficient because:
- v=0 has 18 distinct prefix_k=3 classes → AC works well
- v=1 has only 2 prefix_k=2 classes (2187 ops each) → AC has minimal leverage
- v=2 has only 1 prefix_k=2 class → AC has zero leverage

With `level_prefix_k`, deeper levels use deeper prefix splits (k=4, k=5), giving AC enough class granularity to sharpen direction clustering at every active level.

---

### What is `target_sim` (soft margin)?

`target_sim` sets per-level cosine similarity targets for `AngularCoherenceLoss`:

```yaml
angular_coherence:
  target_sim: [1.0, 0.85, 0.70, 0, 0, 0, 0, 0, 0, 0]
```

The loss becomes `F.relu(target_sim - cos_sim).mean()` — gradient stops once the pair similarity exceeds the target. This preserves reconstruction diversity at direction-diverse levels.

**Critical**: `target_sim[0]` must be `1.0`, not a lower value. At v=0, within-class cosine similarity is already ~0.981. Setting `target_sim=0.90` makes the loss identically zero (since 0.90 < 0.981), destroying the primary ARI driver. This was confirmed empirically: `target_sim[0]=0.90` caused ARI to regress from 0.844 to 0.716.

---

### What is the ARI metric and how is it computed?

ARI (Adjusted Rand Index) measures how well K-means clusters in direction space align with digit prefix classes. It is computed during training (every `eval_every` epochs):

1. Extract direction vectors for v=0 operations
2. Subsample to 5000 if needed (for speed)
3. Run K-means(k=15, n_init=3)
4. Compare cluster labels to `digit_prefix_class(k=3)` labels via `adjusted_rand_score`

**TensorBoard scalar**: `Direction/ARI_prefix3`

ARI=1.0 means perfect agreement; ARI=0.0 means random. Current best: 0.844 (V7.2 large architecture).

---

### What metrics are only available offline?

While most key metrics are logged to TensorBoard during training, some are only available via `diagnose_direction_geometry.py`:

| Available in training | Offline only |
|----------------------|-------------|
| AQ (intra-inter sim) | Per-level within-sim |
| ARI (prefix3, v=0) | kNN digit overlap |
| Intra/inter level sim | Multi-level ARI breakdown |

Per-level loss details (e.g., `r_v0..r_v9`, `angular_coherence_pairs`) are computed by loss classes but not currently logged to TensorBoard.

---

## See Also

- `CLAUDE.md` - Architecture overview
- `src/README.md` - Module documentation and integration guide
- `src/presets/v7_large.yaml` - Recommended V7.2 configuration
- `docs/SPECS.md` - Technical specifications
- `docs/audits/23-03-2026-LEVEL-PREFIX-AUDIT.md` - Level prefix audit details
