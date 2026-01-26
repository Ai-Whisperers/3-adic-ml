# P-Adic VAE Architecture (V6.0)

## Architecture Summary

**Dual VAE + True Hyperbolic Geometry + LR Controller**

### Core Components

| Component | Structure | Purpose |
|-----------|-----------|---------|
| **VAE-A** | Encoder 9→128→64, Decoder 16→64→27 | Coverage (reconstruction) |
| **VAE-B** | Same structure, independent weights | Hierarchy learning |
| **Hyperbolic Projection** | Tangent net + expmap0 → Poincaré ball | True hyperbolic mapping |
| **LR Controller** | MetricBasedLR with Q-gated thresholds | Dynamic LR scale control |

### What Makes It "P-Adic"

1. **Data**: All 19,683 ternary operations (3^9) with values {-1, 0, 1}
2. **3-adic valuation**: v_3(n) measures divisibility by powers of 3
3. **Geometric encoding**: High valuation → near origin, low valuation → near boundary
4. **Loss aligns**: Poincaré distances to 3-adic valuations (ultrametric → hyperbolic)

### Architecture Flow (V6.0 - True Hyperbolic)

```
Input (9 values, {-1,0,1})
    |
+-- Encoder A --+    +-- Encoder B --+
|  9->128->64   |    |  9->128->64   |
|  mu_A, sig_A  |    |  mu_B, sig_A  |
+------+--------+    +------+--------+
       |                    |
   z_tangent (16-dim)   z_tangent        <- Tangent space at origin (Euclidean)
       |                    |
   +----------------------------+
   |  DualHyperbolicProjection  |
   |  tangent_net + expmap0     |
   +----------------------------+
       |                    |
   z_A_hyp              z_B_hyp          <- Poincaré manifold points
       |                    |
   logmap0              logmap0          <- Back to tangent space
       |                    |
   Decoder A            Decoder B
       |
   Reconstruction logits
```

---

## Loss System (Config-Driven)

### Primary Losses

| Loss | Purpose | Implementation |
|------|---------|----------------|
| **RichHierarchyLoss** | Unified hierarchy + coverage + separation | Per-level mean radii with margins |
| **PAdicGeodesicLoss** | Poincaré distance alignment | Random pairs within batch |
| **RadialHierarchyLoss** | Direct radius enforcement | Weighted MSE to target radii |
| **GlobalRankLoss** | Soft ranking violations | Sigmoid-based differentiable ranking |
| **MonotonicRadialLoss** | Per-level ordering | Groups by valuation, enforces r[v] > r[v+1] |

### Design Decisions

**Pair Sampling**: Uses random within-batch sampling (not synthetic stratified). This is intentional:
- Uses **real embeddings** from actual batch data
- `MonotonicRadialLoss` handles per-valuation-level structure explicitly
- Avoids fabricating artificial index pairs

**Per-Level Metrics**: Tracked via `MonotonicRadialLoss`:
- Logs `r_v0`, `r_v1`, ..., `r_v9` (mean radius per level)
- Tracks `margin_violations` and `mean_violation_magnitude`
- In hyperbolic geometry, radial ordering implies distance ordering

---

## LR Controller (Option C)

Training uses **continuous LR scales** via `MetricBasedLR` (not boolean freeze/unfreeze):

| Component | LR Scale | Role |
|-----------|----------|------|
| `encoder_a` | 0.05× base | Coverage encoder (slowest learner) |
| `encoder_b` | 0.1× base | Hierarchy encoder (medium learner) |
| `projections` | 1.0× base | Hyperbolic projection (fastest adapter) |

**Q-Metric**: `Q = dist_corr + 1.5 × |hierarchy|` guides threshold annealing.

**How it works**: Each epoch, `MetricBasedLR.update(metrics)` returns LR scales, then `update_optimizer_lr_scales()` applies them to optimizer param groups. Setting LR=0 effectively freezes a component.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | `TernaryVAEV6`, `TernaryVAEV6Controllable`, `EncoderHead` |
| `src/models/lr_controller.py` | `MetricBasedLR`, `TrainingMetrics`, LR scale control |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/config/statenet_config.py` | `StateNetConfig` dataclass (configuration) |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/core/ternary.py` | Immutable 3-adic field logic |
| `src/losses/padic_geodesic.py` | All hierarchy/geodesic losses |
| `src/losses/combined.py` | Config-driven loss composition |
| `src/train.py` | Unified training entry point |

## Training

```bash
python src/train.py --config src/presets/v6.yaml
```

### Config Keys (V6.0)

| Key | Purpose |
|-----|---------|
| `anchor_checkpoint.path` | Pre-trained weights to start from |
| `option_c.encoder_a_lr_scale` | LR multiplier for coverage encoder (default: 0.05) |
| `option_c.encoder_b_lr_scale` | LR multiplier for hierarchy encoder (default: 0.1) |
| `option_c.projections_lr_scale` | LR multiplier for projections (default: 1.0) |
| `statenet.initial.encoder_a_trainable` | Initial trainability for encoder A |
| `model.name` | `TernaryVAEV6Controllable` |

### Training Loop Algorithm

```
For each epoch:
    1. Training phase:
       - Forward: out = model(batch_ops)
       - Loss: losses = loss_fn(z_hyp, batch_idx, logits, batch_ops, epoch)
       - Backward + gradient clipping
       - Optimizer step

    2. Validation phase:
       - Collect z_A_hyp, z_B_hyp from both VAEs
       - Compute hierarchy metrics (Spearman correlation, Q metric)

    3. LR Controller update:
       - metrics = TrainingMetrics(coverage, hierarchy_a/b, dist_corr, q_value, grad_norm)
       - controller_state = lr_controller.update(metrics)
       - update_optimizer_lr_scales(optimizer, base_lr, lr_scales)

    4. Checkpointing:
       - Save best_Q.pt when Q improves
       - Periodic checkpoints every save_every epochs
```

---

## P-Adic VAEs

- **Core idea**: Dual VAE + Controller where latents live in **ultrametric p-adic space (p=3)**, inducing hierarchy by construction
- **Geometry**: Discrete → continuous bridge via **p-adic → hyperbolic projections** (Poincaré ball with expmap0/logmap0)
- **Dynamics**: Dual-VAE (coverage/hierarchy) with LR controller; ELBO stability via geometry-aware optimization
- **Evidence**: Empirical correlations between ultrametric distance and semantic/functional similarity
- **Applications**: Hierarchical AI, neurosymbolic AI, semantic compression, protein/codon pipelines
- **Constraints**: RTX 3050 6GB compatible, aggressive memory discipline
- **Philosophy**: Meaning = geometry; hierarchy **emerges structurally**, not memorized

---

## Configuration Details

### Config Structure (Nested)

The YAML config uses **nested structure** (not flat keys):

```yaml
statenet:
  enabled: true
  initial:
    encoder_a_trainable: false  # Set to true for both VAEs trainable
    encoder_b_trainable: true
    projections_trainable: true
  coverage:
    fix_threshold: 0.995
    train_threshold: 1.0
  hierarchy:
    plateau_patience: 10
  timing:
    warmup_epochs: 10

option_c:
  enabled: true
  encoder_a_lr_scale: 0.05
  encoder_b_lr_scale: 0.1
  projections_lr_scale: 1.0
```

### Making Both VAEs Trainable

To train with BOTH encoders trainable from the start:

1. Set `statenet.initial.encoder_a_trainable: true` in YAML
2. Or programmatically:

```python
from src.config import StateNetConfig

config = StateNetConfig.from_dict(yaml_cfg.get('statenet', {}))
config.initial.encoder_a_trainable = True
config.initial.encoder_b_trainable = True
```

### Integration Pattern (train.py)

```python
from src.config import StateNetConfig
from src.models import (
    TernaryVAEV6Controllable,
    MetricBasedLR,
    TrainingMetrics,
    update_optimizer_lr_scales,
)

# 1. Load config
sn_config = StateNetConfig.from_dict(config.get('statenet', {}))

# 2. Create controller
lr_controller = MetricBasedLR(sn_config)

# 3. Create model
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    projections_trainable=sn_config.initial.projections_trainable,
)

# 4. In training loop
metrics = TrainingMetrics(epoch=epoch, coverage=cov, hierarchy_a=h_a, ...)
state = lr_controller.update(metrics)
update_optimizer_lr_scales(optimizer, base_lr, state['lr_scales'])
```

### StateNetConfig Dataclass Structure

```python
@dataclass
class StateNetConfig:
    enabled: bool = True
    coverage: CoverageThresholds      # fix_threshold, train_threshold, floor
    hierarchy: HierarchyThresholds    # plateau_threshold, plateau_patience, ...
    controller: ControllerThresholds  # grad_threshold, grad_patience, ...
    annealing: AnnealingConfig        # enabled, step, q_decrease_threshold
    timing: TimingConfig              # warmup_epochs, hysteresis_epochs, window_size
    lr_scales: LRScales               # encoder_a, encoder_b, projections, decoders
    initial: InitialStates            # encoder_a_trainable, encoder_b_trainable, ...
```

### See Also

- `src/README.md` - Full integration guide with code examples
- `src/presets/v6.yaml` - Reference V6.0 configuration
