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
|  mu_A, sig_A  |    |  mu_B, sig_B  |
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

### Loss System (Config-Driven)

- **Coverage**: CrossEntropy reconstruction
- **Hierarchy**: MSE(radius, target_radius) based on 3-adic valuation
- **Separation**: Margin between valuation levels
- **Geodesic**: Poincaré distance alignment to p-adic metric
- **Rank**: Soft ordering violations

### StateNet Controller

The StateNet manages component **trainability** (not "freeze" - we use positive logic):

- `encoder_a_trainable`: Starts `False` (fixed), becomes `True` when hierarchy stalls
- `encoder_b_trainable`: Starts `True`, becomes `False` when hierarchy plateaus
- `controller_trainable`: Gradient-gated stability control

**Q-Metric**: `Q = dist_corr + 1.5 × |hierarchy|` guides threshold annealing.

### Key Innovation

**True hyperbolic geometry** via expmap0/logmap0 with **p-adic structure as the organizing principle** - points divisible by high powers of 3 cluster near the origin, creating a natural tree-like hierarchy in the Poincaré ball.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/vae.py` | TernaryVAEV6Controllable - main model |
| `src/models/statenet.py` | Q-gated trainability controller |
| `src/models/hyperbolic_projection.py` | expmap0/logmap0 projections |
| `src/geometry/poincare.py` | Riemannian backend (geoopt) |
| `src/core/ternary.py` | Immutable 3-adic field logic |
| `src/losses/padic_geodesic.py` | Hierarchy enforcement |
| `src/train.py` | Unified training entry point |

## Training

```bash
python src/train.py --config src/presets/research_extended_grokking.yaml
```

### Config Keys (V6.0)

- `anchor_checkpoint`: Pre-trained weights to start from
- `coverage_fix_threshold`: Fix encoder when coverage drops below
- `coverage_train_threshold`: Allow training when coverage above

## P-Adic VAEs

- **Core idea**: Dual VAE + Controller where latents live in **ultrametric p-adic space (p=3)**, inducing hierarchy by construction
- **Geometry**: Discrete → continuous bridge via **p-adic → hyperbolic projections** (Poincaré ball with expmap0/logmap0)
- **Dynamics**: Dual-VAE (coverage/hierarchy) with StateNet controller; ELBO stability via geometry-aware optimization
- **Evidence**: Empirical correlations between ultrametric distance and semantic/functional similarity
- **Applications**: Hierarchical AI, neurosymbolic AI, semantic compression, protein/codon pipelines
- **Constraints**: RTX 3050 6GB compatible, aggressive memory discipline
- **Philosophy**: Meaning = geometry; hierarchy **emerges structurally**, not memorized
