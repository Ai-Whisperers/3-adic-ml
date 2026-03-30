# Scripts — Diagnostics & Validation

This directory contains utility scripts for analyzing trained models and validating architectural changes.

## Organization

### `diagnostics/`
Analysis and debugging scripts for trained models:

- **`check_encoder_norm.py`** — Check the actual norm of encoder outputs on sample batches
- **`check_actual_encoder_norms.py`** — Detailed analysis of encoder outputs through HyperbolicProjection with distribution statistics
- **`check_tangent_net_weights.py`** — Inspect tangent_net weight initialization (checking for init_identity fix)
- **`diagnose_direction_geometry.py`** — Full direction geometry diagnostic:
  - Q1: Intra-level angular clustering (cosine similarity per valuation level)
  - Q2: Digit-position grouping (which digit positions encode structure?)
  - Q3: UMAP 2D visualization of direction vectors
  - Q4: kNN@5 digit pattern overlap vs random baseline
  - Step 5: K-means sub-island analysis at v=0 with ARI metrics

**Usage:**
```bash
python scripts/diagnostics/check_encoder_norm.py
python scripts/diagnostics/diagnose_direction_geometry.py
```

### `validation/`
Test and validation scripts for architecture fixes:

- **`validate_fix.py`** — Comprehensive test for embedding space collapse fixes:
  - Test 1: `init_identity=False` produces non-zero tangent_net output
  - Test 2: `tangent_scale=0.05` gives appropriate Poincaré ball distribution
  - Test 3: Points don't saturate at ball boundary

- **`validate_v7_concerns.py`** — Pre-training validation covering 6 items (A–F):
  - Item A: Concern 1–3 with variance_only=False fix
  - Item B: Hierarchy (Spearman), dist_corr, Q metric, reconstruction accuracy
  - Item C: Full-dataset hierarchy (includes rare v=8, v=9)
  - Item D: Decoder reliance ratio (z_r vs z_θ gradients)
  - Item E: StateNet plateau detection simulation
  - Item F: Within-level scatter statistics vs V6 baseline

- **`test_init_identity.py`** — Isolated test of the `init_identity=False` fix
- **`test_tangent_scale_effect.py`** — Sweep over tangent_scale values to find optimal magnitude

**Usage:**
```bash
python scripts/validation/validate_fix.py
python -u scripts/validation/validate_v7_concerns.py | tee validate_v7_results.txt
```

## Running Diagnostics on Checkpoints

To diagnose a specific checkpoint:

```bash
# Update CHECKPOINT and CONFIG paths in diagnose_direction_geometry.py, then:
python scripts/diagnostics/diagnose_direction_geometry.py
```

Output includes direction geometry plots saved to `runs/v7_*/direction_umap*.png`.

## When to Use Each Script

| Scenario | Script(s) |
|----------|-----------|
| Verify encoder normalization | `check_encoder_norm.py` |
| Debug embedding space collapse | `validate_fix.py`, `check_tangent_net_weights.py` |
| Test architectural changes | `validate_v7_concerns.py` |
| Analyze direction geometry | `diagnose_direction_geometry.py` |
| Check tangent_scale tuning | `test_tangent_scale_effect.py` |
