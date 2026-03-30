# P-Adic VAE Architecture (V6.2)

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
2. **3-adic valuation**: v_3(n) measures divisibility of the **index integer n** by powers of 3
3. **Geometric encoding**: High valuation → near origin, low valuation → near boundary

> **Important**: `v_3(n)` is a property of the **index n as an integer**, not of the algebraic
> content of the 9-digit operation it represents. All hierarchy losses are indexing-derived.
> Only `AngularCoherenceLoss` uses intrinsic digit content (via `digit_prefix_class`).
> See `docs/DATA-SEMANTICS.md` for full analysis including the v=9 singleton convention
> and dataset expansion options.
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

### Training Loop Algorithm (V6.2)

```
For each epoch:
    1. Training phase (BOTH VAEs contribute to loss):
       - Forward: out = model(batch_ops)
       - Loss A: losses_A = loss_fn(z_A_hyp, batch_idx, logits_A, batch_ops, epoch,
                                     mu=mu_A, logvar=logvar_A)
       - Loss B: losses_B = loss_fn(z_B_hyp, batch_idx, logits_B, batch_ops, epoch,
                                     mu=mu_B, logvar=logvar_B)
       - Total: loss = losses_A["total"] + losses_B["total"]
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

---

## Critical Fixes (V6.2 - 2026-03-11)

### Adversarial Audit Results & Fixes Applied

Full adversarial audit revealed 3 critical bugs + 2 design flaws. All fixed:

| Bug | Severity | Fix |
|-----|----------|-----|
| **VAE-B dead**: z_B_hyp never passed to loss | CRITICAL | Both z_A_hyp and z_B_hyp now go through CombinedLoss in train.py |
| **max_radius saturation**: all points clamped to 0.95 | CRITICAL | Added learnable `tangent_scale` (init=0.1) in HyperbolicProjection |
| **Config key mismatch**: `radial_weight` vs `weight` | CRITICAL | Fixed v6.yaml to use `weight: 5.0` |
| **No KL divergence**: model was a deterministic AE | HIGH | Wired HyperbolicKLDivergence into CombinedLoss.forward() |
| **Dead config keys**: 11+ unused YAML keys | MEDIUM | Removed zero_structure, richness_weight, annealing, encoder/decoder_dropout, logvar_min/max |

### Post-Fix Training Results (20 epochs, CPU)

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Initial radii | All 0.95 (saturated) | 0.23-0.57 (spread) |
| VAE-B gradients | 0/14 params | 14/14 params |
| Hierarchy corr | N/A (flat) | 0.83 from epoch 0 |
| Accuracy | N/A | 55% at epoch 20 |
| Loss curve | N/A | 15.6 → 2.1 (converging) |

### tangent_scale Parameter

`HyperbolicProjection` now has a learnable `tangent_scale` parameter:
- Initialized to 0.1 (prevents expmap0 saturation)
- Encoder outputs (~4.0 norm) are scaled to ~0.4, giving expmap0 radii ~0.38
- Learned during training; adapts to optimal magnitude for hierarchy

---

## Codebase Review Summary (2026-01-26)

### Architecture Verification

The entire `src/` codebase (~5000 lines, 22 files) has been reviewed. Key findings:

**Confirmed Working:**
- Option C (LR-based trainability) is fully implemented
- True hyperbolic geometry via geoopt expmap0/logmap0
- StateNetConfig → MetricBasedLR → train.py integration is correct
- All losses use proper hyperbolic distances
- TernarySpace singleton is immutable and thread-safe

**No statenet.py**: The "StateNet" system is distributed across:
- `src/config/statenet_config.py` - Configuration dataclass
- `src/models/lr_controller.py` - Decision logic (MetricBasedLR)
- `src/models/vae.py` - Component trainability
- `src/train.py` - Integration point

### Dead Code Removed (2026-01-26)

| Item | Reason |
|------|--------|
| `CheckpointCompatibilityError` | Never raised, validation not called |
| `AnnealingConfig` | Heuristic meta-control, logic never implemented |

### Design Decisions (Not Issues)

| Item | Location | Rationale |
|------|----------|-----------|
| `proj_B.learnable_curvature=False` | `hyperbolic_projection.py:238` | Intentional - both projections share A's curvature |

### New: Learnable Loss Weights (V6.1)

Loss weights can now be **trainable** using homoscedastic uncertainty weighting (Kendall et al. 2018):

```yaml
loss:
  learnable_weights: true  # Enable
  rich_hierarchy:
    enabled: true
    hierarchy_weight: 5.0   # Initial weight
    coverage_weight: 1.0
    separation_weight: 3.0
```

**How it works:**
- Each loss gets `nn.Parameter` log_sigma (initialized from config weights)
- Effective weight = `0.5 * exp(-2 * log_sigma)`
- Regularization `-log_sigma` prevents collapse to zero
- Gradients flow through → network learns optimal balance

**Key difference from removed AnnealingConfig:**
- AnnealingConfig was heuristic (adjusted thresholds based on Q metric)
- Learnable weights are **trainable** (gradients flow, network learns)

**Usage:**
```python
loss_fn = CombinedLoss(config['loss'], curvature=1.0)

# Include loss_fn parameters in optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=base_lr
)

# Monitor learned weights
print(loss_fn.get_learned_weights())  # {'hierarchy': 4.2, 'coverage': 1.8, ...}
```

**When to use:** Enable for long training runs or when exploring new loss combinations. The network will discover the optimal curriculum (e.g., coverage → hierarchy → separation).

### Quick Reference

```python
# Core imports
from src.core import TERNARY, valuation, distance, target_radius
from src.geometry import hyperbolic_radius, poincare_distance, exp_map_zero
from src.losses import CombinedLoss
from src.models import TernaryVAEV6Controllable, MetricBasedLR, TrainingMetrics
from src.config import StateNetConfig, N_TERNARY_OPERATIONS

# Make both VAEs trainable
sn_config = StateNetConfig.from_dict(yaml_cfg.get('statenet', {}))
sn_config.initial.encoder_a_trainable = True  # Default is False
sn_config.initial.encoder_b_trainable = True  # Default is True

# Integration pattern
model = TernaryVAEV6Controllable(
    encoder_a_trainable=sn_config.initial.encoder_a_trainable,
    encoder_b_trainable=sn_config.initial.encoder_b_trainable,
    ...
)
controller = MetricBasedLR(sn_config)

# Training loop
state = controller.update(TrainingMetrics(...))
update_optimizer_lr_scales(optimizer, base_lr, state['lr_scales'])
```

### File Map

| Category | Files |
|----------|-------|
| **Entry** | `train.py` |
| **Models** | `vae.py`, `hyperbolic_projection.py`, `lr_controller.py` |
| **Config** | `config/statenet_config.py`, `config/constants.py` |
| **Core** | `core/ternary.py` (TernarySpace singleton) |
| **Geometry** | `geometry/poincare.py` (geoopt backend) |
| **Losses** | `losses/combined.py`, `losses/padic_geodesic.py` |
| **Utils** | `utils/checkpoint.py`, `utils/tensorboard_logger.py` |
| **Tests** | `tests/` (280 tests across 8 files) |

---

## Test Suite (280 tests)

Comprehensive test coverage following `docs/plans/TESTS_CRITICAL_TARGETS.md`.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run by tier
pytest tests/test_core_ternary.py tests/test_geometry_poincare.py -v  # Tier 1
pytest tests/test_losses.py tests/test_losses_combined.py -v          # Tier 2

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

| File | Tests | Coverage |
|------|-------|----------|
| `test_core_ternary.py` | 28 | Valuation formula, ultrametric inequality, round-trip |
| `test_core_ternary_extended.py` | 29 | Distance formula, tree structure, level consistency |
| `test_geometry_poincare.py` | 29 | exp/log composition, ball containment, triangle inequality |
| `test_geometry_poincare_extended.py` | 27 | Möbius add, geodesics, distance matrix, formula verification |
| `test_losses.py` | 64 | Gradient flow, non-negativity, monotonicity, edge cases |
| `test_losses_combined.py` | 37 | CombinedLoss, phase gating, learnable weight formula |
| `test_gradient_flow.py` | 11 | Full pipeline gradient flow, radius spread, KL integration |

### What's Tested

**Tier 1 - Mathematical Invariants (113 tests):**
- 3-adic valuation: `v_3(0)=9, v_3(1)=0, v_3(3)=1, v_3(9)=2`
- Ultrametric inequality: `d(a,c) ≤ max(d(a,b), d(b,c))`
- exp/log composition: `log_map_zero(exp_map_zero(v)) ≈ v`
- Ball containment: `||exp_map_zero(v)|| < 1`

**Tier 2 - Loss Correctness (101 tests):**
- Gradient flow through all 6 loss classes
- Loss non-negativity (all losses ≥ 0)
- Target distance monotonicity (`d(v) > d(v+1)`)
- Learnable weight formula: `w = 0.5 * exp(-2 * log_sigma)`
- Phase gating for geodesic loss

### Test Philosophy

Tests validate **mathematical invariants** and **actual computation**, not:
- Shape assertions (PyTorch guarantees these)
- Import success (Python handles this)
- Constructor return values (can't be None)

See `docs/plans/TESTS_CRITICAL_TARGETS.md` for full testing strategy.

---

## Direction Geometry — Level Prefix & Soft Margin (V7.2+ - 2026-03-23)

### ARI Ceiling & Root Cause

Four V7 runs established an ARI ceiling at 0.844. Root cause: `AngularCoherenceLoss` with global `prefix_k=3` has leverage only at v=0 (18 classes). At v=1 (2 classes) and v=2 (1 class), AC has minimal/zero leverage.

### level_prefix_k — Per-Level Prefix Depth

```yaml
angular_coherence:
  level_prefix_k: [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]  # v=0→k=3, v=1→k=4, v=2→k=5, v=3+→skip
  target_sim: [1.0, 0.85, 0.70, 0, 0, 0, 0, 0, 0, 0]  # Soft-margin targets
  n_pairs: 3000  # ~1000 per active level
```

When `level_prefix_k` is set, `AngularCoherenceLoss.forward()` processes levels independently. When `None`, falls back to global `prefix_k` (backward compatible).

### target_sim Constraint

**`target_sim[0]` MUST be 1.0** — setting it to 0.90 caused ARI regression from 0.844 → 0.716 because `F.relu(0.90 - 0.981) = 0` (v=0 within-class sim is already 0.981, so the loss becomes identically zero). With `target_sim[0]=1.0`, `F.relu(1.0 - cos_sim)` is equivalent to the original `(1.0 - cos_sim)` formula.

### Live ARI in Training Loop

ARI is now computed during training (every `eval_every` epochs) in `src/train.py`:
- Extracts v=0 direction vectors, subsamples to 5000
- K-means(k=15, n_init=3) → compared to `digit_prefix_class(k=3)` via `adjusted_rand_score`
- Logged as `Direction/ARI_prefix3` in TensorBoard
- ~50ms per eval, zero GPU impact (CPU-only on detached tensors)

### Metrics Blind Spots (Audit Finding)

Per-level loss details (`r_v0..r_v9`, `angular_coherence_pairs`) are computed by loss classes but not logged to TensorBoard — only aggregate values are logged. Per-level within-sim and kNN digit overlap remain offline-only via `diagnose_direction_geometry.py`.

### Implementation Files

| File | Change |
|------|--------|
| `src/losses/padic_geodesic.py` | `AngularCoherenceLoss`: `level_prefix_k`, `target_sim` params, per-level forward |
| `src/losses/combined.py` | Passes `level_prefix_k` and `target_sim` from YAML |
| `src/presets/v7_large.yaml` | Config for level_prefix_k, target_sim, n_pairs |
| `src/train.py` | Live ARI computation in eval block |
