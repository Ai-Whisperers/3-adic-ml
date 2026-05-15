# src/ - P-Adic VAE Source Code (V6.2 Modularized)

**Last Updated**: 2026-05-15

---

## Architecture Overview (V6.2)

The codebase has been modularized to improve maintainability and type safety. The previous monolithic `train.py` has been split into dedicated packages.

### Component Structure

| Package | Purpose |
|---------|---------|
| `src/core/` | 3-adic algebra and `TernarySpace` singleton (LUTs) |
| `src/geometry/` | Hyperbolic operations via `geoopt` (Poincaré ball) |
| `src/losses/` | Config-driven loss composition (Hierarchy, Geodesic, etc.) |
| `src/models/` | VAE architectures and LR controllers |
| `src/training/` | Modularized training engine, setup, and reporting |
| `src/config/` | Pydantic schemas and StateNet configurations |
| `src/utils/` | Checkpointing, logging, and hardware monitoring |

### Training Logic (V6.2)

The training process is now orchestrated across:
- `src/training/bootstrap.py`: Hardware init, data loading, and config auditing.
- `src/training/setup.py`: Component factory (optimizers, schedulers, losses).
- `src/training/engine.py`: Core training/validation loops.
- `src/training/reporting.py`: Checkpointing and metric logging.

## StateNet Integration

The training controller (StateNet) manages component trainability via learning rate scales ("Option C").

### Default Thresholds (V6.2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fix_threshold` | 0.35 | Freeze encoder_A if coverage drops below |
| `train_threshold` | 0.45 | Unfreeze encoder_A if above (+ stall detection) |
| `floor` | 0.30 | Minimum threshold for annealing |

### YAML Configuration

```yaml
statenet:
  enabled: true
  coverage:
    fix_threshold: 0.35
    train_threshold: 0.45
  hierarchy:
    plateau_patience: 10
  timing:
    warmup_epochs: 10
```

## Data Flow (True Hyperbolic)

1. **EncoderHeads**: Input → Tangent space $T_0M$ ($\mu, \sigma$).
2. **Reparameterization**: $z_{tangent} = \mu + \epsilon \cdot \sigma$ (Euclidean).
3. **Projections**: $z_{hyp} = \text{expmap}_0(z_{tangent})$ (Poincaré manifold).
4. **Losses**: Operate on $z_{hyp}$ using true hyperbolic distances.
5. **Decoder**: Receives $\text{logmap}_0(z_{hyp})$ (back to tangent space).

## Technical Standards

- **Type Safety**: Use Pydantic `schema.py` for config validation.
- **Geometry**: Always use `src/geometry/poincare.py` for manifold operations.
- **Logging**: Metrics are logged via `TensorBoardLogger` to `runs/`.
