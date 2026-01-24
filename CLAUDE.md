# P-Adic VAE Architecture (V6.0)

## Architecture Summary

**Dual VAE + True Hyperbolic Geometry + StateNet Controller**

### Core Components

| Component | Structure | Purpose |
|-----------|-----------|---------|
| **VAE-A** | Encoder 9→128→64, Decoder 16→64→27 | Coverage (reconstruction) |
| **VAE-B** | Same structure, independent weights | Hierarchy learning |
| **Hyperbolic Projection** | Tangent net + expmap0 → Poincaré ball | True hyperbolic mapping |
| **StateNet** | Q-gated trainability controller | Dynamic component control |

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

## StateNet Controller

The StateNet manages component **trainability** (positive logic, not "freeze"):

| State | Initial | Trigger to Change |
|-------|---------|-------------------|
| `encoder_a_trainable` | `False` (fixed) | Hierarchy stalls at 100% coverage |
| `encoder_b_trainable` | `True` | Hierarchy plateaus for patience epochs |
| `controller_trainable` | `True` | Gradient norm stabilizes |

**Q-Metric**: `Q = dist_corr + 1.5 × |hierarchy|` guides threshold annealing.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | `TernaryVAEV6`, `TernaryVAEV6Controllable` |
| `src/models/statenet.py` | Q-gated trainability controller |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/core/ternary.py` | Immutable 3-adic field logic |
| `src/losses/padic_geodesic.py` | All hierarchy/geodesic losses |
| `src/losses/combined.py` | Config-driven loss composition |
| `src/train.py` | Unified training entry point |

## Training

```bash
python src/train.py --config src/presets/5.12.4.yaml
```

### Config Keys (V6.0)

| Key | Purpose |
|-----|---------|
| `anchor_checkpoint.path` | Pre-trained weights to start from |
| `statenet.coverage_fix_threshold` | Fix encoder when coverage drops below |
| `statenet.coverage_train_threshold` | Allow training when coverage above |
| `model.name` | `TernaryVAEV6Controllable` |

---

## P-Adic VAEs

- **Core idea**: Dual VAE + Controller where latents live in **ultrametric p-adic space (p=3)**, inducing hierarchy by construction
- **Geometry**: Discrete → continuous bridge via **p-adic → hyperbolic projections** (Poincaré ball with expmap0/logmap0)
- **Dynamics**: Dual-VAE (coverage/hierarchy) with StateNet controller; ELBO stability via geometry-aware optimization
- **Evidence**: Empirical correlations between ultrametric distance and semantic/functional similarity
- **Applications**: Hierarchical AI, neurosymbolic AI, semantic compression, protein/codon pipelines
- **Constraints**: RTX 3050 6GB compatible, aggressive memory discipline
- **Philosophy**: Meaning = geometry; hierarchy **emerges structurally**, not memorized
