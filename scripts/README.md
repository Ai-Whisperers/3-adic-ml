# Scripts — Diagnostics & Validation

This directory contains utility scripts for analyzing trained models and validating architectural changes.

**Note (2026-07-15):** this file historically documented only `diagnostics/`
and `validation/`. Two more directories have since grown without matching
documentation updates — `analysis/` (~27 files) and `data/` (3 files), plus a
single `applications/` script. Rather than retrofit a full per-script table
(most of `analysis/` are point-in-time investigation scripts tied to a
specific past architecture question — e.g. `review_phase_17.py`,
`compare_v11_v17.py`, `pre_transition_audit.py` — not living tools meant to be
re-run), this note documents the categories honestly. See each script's own
docstring/comments for what it does; many don't have one, which is itself a
sign they were one-shot investigations rather than reusable tooling.

### `analysis/`
Investigation and evaluation scripts, largely tied to specific historical
questions (algebraic consistency probes, human-genome/proteome scans,
clinical benchmark validation, checkpoint audits). `project_audit.py` (1026
lines) is the largest and most general-purpose — checkpoint-backed
feasibility review, still likely relevant. Most others are narrower and
tied to a specific past run or hypothesis.

### `data/`
Data preparation for real (non-synthetic) inputs, distinct from the
synthetic ternary operation set `src/core/ternary.py` generates:
- **`prepare_codon_data.py`** — `seq_to_ternary_index()`: maps a 9-nucleotide
  window to a ternary operation index (A=0,C=1,G=2,T=2 — G/T merged to fit
  3 states). **Fixed 2026-07-15:** previously hand-rolled the base-3
  conversion with the opposite digit order from `TERNARY.from_ternary`, so
  indices decoded backwards via `TERNARY.to_ternary()` (the function the
  model actually uses internally) — silently scrambling nucleotide-position
  semantics for every consumer (`probe_codon_geometry.py`,
  `validate_human_anomalies.py`, `scan_human_proteome.py`, and 5 others).
  Now delegates to `TERNARY.from_ternary` directly.
- **`prep_human_tp53.py`** — reads a TP53 reference FASTA, slides a 9-nt
  window across it, saves indices via the function above.
- **`prepare_rosetta_dataset.py`** — builds the "Rosetta" custom-indices
  dataset consumed by `training.data.indices_path` in config.

### `applications/`
- **`symbolic_calculator.py`** — interactive/scripted algebraic latent
  calculator built on the trained model.

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
