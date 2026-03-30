# Validation Scripts

Test and verification scripts for architectural changes and fixes. Unlike diagnostics, these scripts actively train or test the model to verify correctness.

## Script Descriptions

### `validate_fix.py`
Comprehensive test suite for embedding space collapse fixes. Runs 3 test functions.

**Tests:**
1. **`test_tangent_net_output()`** — Verify `init_identity=False` produces non-zero residuals
2. **`test_expmap0_saturation()`** — Verify points don't all collapse to boundary with `tangent_scale=0.05`
3. **`test_identity_vs_non_identity()`** — Compare identity vs non-identity init; verify fix improves output variation

**Output:**
```
Testing tangent_net output...
✓ tangent_scale correctly set to 0.05
✓ Residual is non-zero: 0.142345

Testing expmap0 saturation...
Scale 4.0: hyp norm mean=0.3856, std=0.1234, max=0.8932
  Points near boundary (>0.9): 2.1%
  ✓ Good distribution for typical encoder scale

Comparing identity vs non-identity initialization...
Identity residual norm: 0.000001
Fixed residual norm: 0.154230
✓ Fix verified: identity residual ~0, fixed residual > 0
✓ Fixed version has better output variation than identity

✅ All validation tests passed!
```

**When to use:** After implementing a fix to verify it works before training.

**Runtime:** ~5 seconds (CPU/GPU, no training).

---

### `validate_v7_concerns.py`
Pre-training validation covering **6 items (A–F)** + original concerns:

**Items validated:**
- **A:** Concerns 1–3 with `variance_only=False` KL fix
- **B:** Actual hierarchy (Spearman), dist_corr, Q metric, reconstruction accuracy
- **C:** Full-dataset hierarchy (includes rare v=8/v=9)
- **D:** Decoder reliance ratio (z_r vs z_θ gradients) — watches for decoder ignoring z_θ
- **E:** StateNet plateau detection simulation — would encoder_b freeze too early?
- **F:** Within-level scatter of r vs V6 baseline

**Output:** Epoch-by-epoch table showing all 13 metrics + final summary:

```
Ep | Loss  | A:Spear4 | A:rsep | A:mu4n | B:hier | B:dco | B:Q  | B:acc | ... | C2:tsc | C3:KL | C3:mu+n
──────────────────────────────────────────────────────────────────────────────────────────────────────
 1 |  5.23 |    -0.52 |     ✓  |  0.234 |  0.012 | -0.05 | 0.03 | 0.15  | ... | 0.0500 |  8.23 | 0.1234
 5 |  2.15 |    -0.67 |     ✓  |  0.156 |  0.234 | +0.12 | 0.43 | 0.34  | ... | 0.0502 |  2.11 | 0.0856
...
40 |  0.89 |    -0.78 |     ✓  |  0.089 |  0.456 | +0.34 | 0.99 | 0.67  | ... | 0.0501 |  0.54 | 0.0234

─────────────────────────────────────────────────────────────────────────────────────────────────────

LEGEND:
  A:Spear4  Spearman(||mu[:,:4]||, valuation) — negative=good (concern 1 + variance_only fix)
  B:hier    -Spearman(valuation, r)  |  B:dco = dist_corr  |  B:Q = dco + 1.5*hier
  ...
  Item E: No early plateau firing in 40 epochs  ✓
  Item D: final decoder grad ratio z_r/z_θ = 0.32  ✓ decoder using z_θ as expected
  Item F: v=0 scatter = 0.0145  ✓ near-zero, scatter_weight=0.8 may be excessive
```

**When to use:** Before committing a new architecture variant to ensure it trains correctly.

**Configuration:** Edit top-level variables:
```python
EPOCHS = 40         # How many epochs to run
EVAL_EVERY = 5      # Evaluation frequency
DEVICE = "cuda"     # or "cpu"
```

**Runtime:** ~3–5 minutes on GPU, logs every EVAL_EVERY epochs.

**Output files:**
- Console table + final summary printed to stdout
- Optionally save output: `python scripts/validation/validate_v7_concerns.py | tee results.txt`

---

### `test_init_identity.py`
Isolated test of the `init_identity=False` fix.

**What it does:**
- Creates two HyperbolicProjection instances: one with `init_identity=True` (original), one with `False` (fixed)
- Passes the same encoder output through both
- Compares Poincaré ball norms and boundary saturation

**Output:**
```
Testing HyperbolicProjection with init_identity=False but tangent_scale=0.1
tangent_scale: 0.0500

Input z_tangent norm: 4.2134

Scaled input norm: 0.2107
Residual norm: 0.1456
Transformed norm: 0.3563
Poincaré ball norm: 0.3123
Poincaré ball norm std: 0.0567
Points near boundary (>0.9): 0.0%

--- For comparison, init_identity=True (original) ---
Original scaled input norm: 0.2107
Original transformed norm: 0.2107  ← Same as scaled (no residual)
Original Poincaré ball norm: 0.2106
Original Poincaré ball norm std: 0.0002  ← Near-zero std! All points collapse
Original points near boundary (>0.9): 0.0%
```

**When to use:** Quick validation that the fix is present.

**Runtime:** ~1 second.

---

### `test_tangent_scale_effect.py`
Sweep over different `tangent_scale` values to find the effect on Poincaré ball distribution.

**What it does:**
- Uses a real encoder to generate 16 samples with norm ~4.0
- Tests tangent_scale ∈ {0.01, 0.02, ..., 0.10}
- Reports Poincaré ball norm mean/std for each scale

**Output:**
```
Encoder A z_tangent norm: 4.1234
tangent_scale=0.01: scaled norm=0.0412, transformed norm=0.1862, Poincaré norm mean=0.1234, std=0.0156
tangent_scale=0.02: scaled norm=0.0824, transformed norm=0.2456, Poincaré norm mean=0.2012, std=0.0245
tangent_scale=0.03: scaled norm=0.1236, transformed norm=0.3145, Poincaré norm mean=0.2934, std=0.0312
tangent_scale=0.04: scaled norm=0.1648, transformed norm=0.3823, Poincaré norm mean=0.3745, std=0.0401
tangent_scale=0.05: scaled norm=0.2060, transformed norm=0.4512, Poincaré norm mean=0.4234, std=0.0456  ← Good spread
tangent_scale=0.06: scaled norm=0.2472, transformed norm=0.5123, Poincaré norm mean=0.4956, std=0.0512
tangent_scale=0.08: scaled norm=0.3296, transformed norm=0.6234, Poincaré norm mean=0.6123, std=0.0634
tangent_scale=0.10: scaled norm=0.4123, transformed norm=0.7145, Poincaré norm mean=0.7234, std=0.0756
```

**When to use:** Tuning `tangent_scale` for a new latent dimension or architecture.

**Runtime:** ~3 seconds.

---

## Quick Start

```bash
# Test the fixes work
python scripts/validation/validate_fix.py
# Should print: ✅ All validation tests passed!

# Validate V7 architecture before committing
python scripts/validation/validate_v7_concerns.py | tee v7_validation.txt
# Should show: all items A–F passing, Item E no plateau, Item D ratio < 0.5
```

## Common Issues

**Issue:** `ModuleNotFoundError: No module named 'src'`
- **Fix:** Run from project root: `python scripts/validation/validate_fix.py`

**Issue:** CUDA out of memory
- **Fix:** Edit `DEVICE = "cpu"` in the script (slower but works)

**Issue:** Validation fails with unusual metrics
- **Fix:** Check that the config file matches the model being created
