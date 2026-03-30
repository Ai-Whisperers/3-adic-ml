# Diagnostics Scripts

Offline analysis and debugging tools for trained models.

These scripts do NOT modify the model or training loop — they are purely for understanding what the model has learned.

## Script Descriptions

### `check_encoder_norm.py`
Quick check of encoder output norms on a random batch.

**Output:**
```
Encoder A z_tangent norm: 4.1234
Encoder B z_tangent norm: 4.1567
...
```

**When to use:** Initial sanity check, or when investigating encoder output scales.

---

### `check_actual_encoder_norms.py`
Detailed analysis of encoder outputs through the HyperbolicProjection, with distribution statistics across stochastic samples vs deterministic means.

**Output:**
- Encoder A/B mean and std of tangent space norms
- Poincaré ball norms (mean, std, % near boundary >0.9)
- Comparison of deterministic (mu) vs stochastic (mu + sigma*eps) outputs

**When to use:** Verifying that points are properly distributed in the Poincaré ball, not all collapsed at boundary.

---

### `check_tangent_net_weights.py`
Inspect the weight matrices of the tangent_net (the residual MLP inside HyperbolicProjection).

**Output:**
```
Tangent net architecture:
  Layer 0: Linear(...)
  Layer 1: ReLU()
  ...

Weight statistics:
  Layer 0 weight: mean=0.001234, std=0.042567
  ...

Final layer weight norm: 0.123456
Is weight essentially zero? False  ✓ (non-identity initialization)
```

**When to use:**
- Verifying the `init_identity=False` fix is applied (final layer should NOT be zeroed)
- Checking weight initialization magnitudes

---

### `diagnose_direction_geometry.py`
Comprehensive direction geometry audit. Loads a checkpoint and analyzes the z_A_hyp embeddings.

**Questions answered:**
1. **Q1:** Do operations with the same valuation cluster angularly?
2. **Q2:** Do operations sharing a digit value at position i cluster?
3. **Q3:** 2D UMAP of all 19,683 direction vectors
4. **Q4:** Do kNN@5 neighbors in direction space share more digits than random?
5. **Step 5:** K-means(15) on v=0 directions — do clusters align with digit patterns?

**Output:**
```
Q1: Intra-level cosine similarity vs random baseline
  Level      |   N   | intra_sim | random_sim | delta  | ratio
  ────────────┼───────┼───────────┼────────────┼────────┼──────
  0          | 13122 |   +0.9812 |    +0.0234 | +0.958 | 41.88  ✓ cluster
  1          |  4374 |   +0.7234 |   +0.0154  | +0.708 | 47.01  ✓ cluster
  ...

Adjusted Rand Index — K-means(15) vs:
  digit_prefix_class(k=3):   ARI=0.6234  ★ prefix explains islands
  ...

Cluster UMAP saved → runs/v7_.../direction_umap_v0_clusters.png
```

**When to use:** After training a model to understand what structure has been learned in direction space.

**Configuration:** Edit CHECKPOINT, CONFIG, OUT_DIR at the top of the script.

---

## Common Modifications

All scripts import from `src.` using relative paths, so they can be run from anywhere:

```bash
# From project root
python scripts/diagnostics/check_encoder_norm.py

# From scripts/diagnostics/
python check_encoder_norm.py
```

To point at a different checkpoint in `diagnose_direction_geometry.py`:

```python
CHECKPOINT = "path/to/your/checkpoint.pt"
CONFIG     = "src/presets/your_config.yaml"
OUT_DIR    = Path("runs/your_experiment")
```

---

## Output Artifacts

- **Plots:** `direction_umap.png`, `direction_umap_v0_clusters.png` (saved to OUT_DIR)
- **Console:** Tables with per-level statistics, ARI scores, kNN analysis
- **No model state changes:** These scripts are read-only on the checkpoint
