# P-Adic VAE Architecture

## Architecture Summary

**Dual VAE + Hyperbolic Projection + StateNet Controller**

### Core Components

| Component | Structure | Purpose |
|-----------|-----------|---------|
| **VAE-A** | Encoder 9→128→64, Decoder 16→64→27 | Coverage (reconstruction) |
| **VAE-B** | Same structure, independent weights | Hierarchy learning |
| **Hyperbolic Projection** | Direction net + Radius net → Poincare ball | Maps to hyperbolic geometry |
| **StateNet** | Threshold-based freeze controller | Dynamic freeze/unfreeze |

### What Makes It "P-Adic"

1. **Data**: All 19,683 ternary operations (3^9) with values {-1, 0, 1}
2. **3-adic valuation**: v_3(n) measures divisibility by powers of 3
3. **Geometric encoding**: High valuation → near origin, low valuation → near boundary
4. **Loss aligns**: Poincare distances to 3-adic valuations (ultrametric → hyperbolic)

### Architecture Flow

```
Input (9 values, {-1,0,1})
    |
+-- Encoder A --+    +-- Encoder B --+
|  9->128->64   |    |  9->128->64   |
|  mu_A, sig_A  |    |  mu_B, sig_B  |
+------+--------+    +------+--------+
       |                    |
   z_A (16-dim)         z_B (16-dim)     <- Euclidean latents
       |                    |
   +----------------------------+
   |  DualHyperbolicProjection  |
   |  direction * radius        |
   +----------------------------+
       |                    |
   z_A_hyp              z_B_hyp          <- Poincare ball embeddings
       |
   Decoder A (16->64->27)
       |
   Reconstruction logits
```

### Loss System (Config-Driven)

- **Coverage**: CrossEntropy reconstruction
- **Hierarchy**: MSE(radius, target_radius) based on 3-adic valuation
- **Separation**: Margin between valuation levels
- **Geodesic**: Poincare distance alignment to p-adic metric
- **Rank**: Soft ordering violations

### Key Innovation

**Decoupled direction/radius learning** in hyperbolic space with **p-adic structure as the organizing principle** - points divisible by high powers of 3 cluster near the origin, creating a natural tree-like hierarchy in the Poincare ball.

---

## Project Goals

1. **Pre-Audit Analysis**: Analyze codebase to identify improvements (see `docs/audits/`)

2. **Training Refinement**: Achieve scientifically rigorous, reproducible checkpoints

   Key files:
   - `src/models/statenet.py` - Prevents manifold collapse during training
   - `src/core/ternary.py` - Immutable finite field logic
   - `src/geometry/poincare.py` - Riemannian backend (geoopt)
   - `src/data/generation.py` - Data source (19,683 operations)
   - `src/losses/padic_geodesic.py` - Hierarchy enforcement via 3-adic geometry

3. **Scientific Rigor Requirements**:
   - Audit-Then-Execute protocol (data integrity, model health)
   - Explicit targets: Hierarchy < -0.80, Richness > 0.008
   - Q-Metric optimization, Stratified Sampling
   - v5.5 anchor uses Euclidean embeddings (standard architecture)

## P-Adic VAEs

- **Core idea**: Dual VAE + Controller where latents live in **ultrametric p-adic space (p=3)**, inducing hierarchy by construction
- **Geometry**: Discrete → continuous bridge via **p-adic → hyperbolic projections** (Poincare ball)
- **Dynamics**: Dual-VAE (explore/exploit) with StateNet controller; ELBO stability via geometry-aware optimization
- **Evidence**: Empirical correlations between ultrametric distance and semantic/functional similarity
- **Applications**: Hierarchical AI, neurosymbolic AI, semantic compression, protein/codon pipelines
- **Constraints**: RTX 3050 6GB compatible, aggressive memory discipline
- **Philosophy**: Meaning = geometry; hierarchy **emerges structurally**, not memorized
