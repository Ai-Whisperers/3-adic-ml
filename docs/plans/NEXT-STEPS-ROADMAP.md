# Next Steps Roadmap
**Date:** 2026-03-22
**Status:** Post-V6 ceiling analysis. All individual targets met. Q=2.163 confirmed hard limit for V6.

See `docs/audits/22-03-2026-Q-CEILING-ANALYSIS.md` for the full mathematical proof of the ceiling.

---

## Phase Execution Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | ValuationPriorLoss (valuation-conditioned mean prior) | ✅ Done |
| Phase 2 | Validation + ceiling diagnosis | ✅ Done — Q=2.163, all individual targets met |
| Phase 3A | Per-valuation σ targets in tangent space | ❌ Skipped |
| Phase 3B | Lagrangian dual adaptive weighting | ✅ Done — no Q gain (ceiling confirmed) |
| Phase 4A | Positional significance encoding (18-dim input) | ❌ Not started |
| Phase 4B | FT-Transformer encoder | ❌ Not started |
| **Phase 4C** | **Factored latent z_radial ⊕ z_identity (V7)** | ✅ Done — implemented 2026-03-22, awaiting training validation |

---

## Step 1 — Fix the Geodesic Loss Objective ~~(1 hour, potentially +0.04 dist_corr)~~

**STATUS: ATTEMPTED AND CLOSED. Does not help. Reason documented below.**

**The attempted fix:** use `|val_i − val_j|` as the pair signal with linear target
`d_target = max_dist * |val_diff| / max_valuation`. Implemented via
`use_individual_valuation=True` config key. Ran 62 epochs. Q stayed at 2.157.

**Why it failed (post-mortem):**

1. **Wrong target direction for dominant pairs.** Linear target gives (v=0, v=1)
   target=0.33, but actual level gap=1.06. The geodesic loss *compressed* the most
   frequent cross-level pairs by 0.73 units, actively opposing the hierarchy losses
   (weight 5.0 vs 2.0 — losses cancelled, no net Q change).

2. **Wrong abstraction level.** `dist_corr` measures `Spearman(|radius_i − radius_j|,
   |val_i − val_j|)` — pure RADIAL differences from origin. The geodesic loss pushes
   POINCARÉ distances between pairs (radial + angular). These are decoupled once
   within-level scatter is small: Poincaré compression happens via angular reduction,
   which has zero effect on |radius_i − radius_j|. The radial hierarchy losses already
   control individual radii. **The geodesic loss cannot directly improve dist_corr.**

3. **Correct formula would still be redundant.** Even with `d_target = |r_target(v_i) −
   r_target(v_j)|` (actual radius gaps from LUT — avoids compression), the loss would
   be redundant with RadialHierarchyLoss + MonotonicRadialLoss which already enforce
   individual radii at their targets.

**Conclusion:** The geodesic loss objective mismatch is not the bottleneck for dist_corr.
dist_corr is bounded by the same group-size structural limit as hierarchy. Step 1 is closed.
Proceed to Step 3 (V7 factored latent).

---

## Step 2 — Phase 3A: Per-Valuation σ Targets (3 hours, cleanup)

Currently `HyperbolicKLDivergence(variance_only=True)` applies a uniform variance penalty.
The correct prior is `N(target_tangent[v], σ_target[v]²)` where σ decays with valuation
(high-valuation points near origin should have tighter posterior variance).

```python
# In ValuationPriorLoss or a new VariancePriorLoss:
sigma_target[v] = sigma_base * exp(-v * sigma_scale)  # e.g. base=0.5, scale=0.3
var_loss = mse(logvar_i.exp(), sigma_target[v_i]**2)
```

**Expected outcome:** tighter posterior variance per level, potentially improving
radial clustering. May marginally improve hierarchy. Low effort, principled completion
of the prior design.

---

## Step 3 — V7 Factored Latent Architecture ✅ IMPLEMENTED 2026-03-22

**This is the only change that can push Q beyond 2.2. Now implemented — run `python src/train.py --config src/presets/v7.yaml` to train.**

### The problem (proven)
V6 uses one shared latent code for both hierarchy (radius) and reconstruction (direction).
The encoder must give each of the 13,122 v=0 operations a unique angular direction for
reconstruction, creating irreducible within-level radial scatter that limits Spearman
hierarchy to ~0.839 regardless of loss engineering.

### The solution: factored latent

```
encoder(x) → z_tangent (D dims)
                  │
          ┌───────┴───────┐
     z_r (k dims)    z_θ (D-k dims)
     "valuation"     "identity"
          │                │
     r = sigmoid(linear(z_r)) * max_radius
     dir = normalize(z_θ)
          │                │
          └───────┬────────┘
             z_hyp = r * dir   ← Poincaré ball point
```

- `k = 4` recommended (enough capacity for 10 valuation levels, small enough to not dominate)
- Hierarchy/radial/monotonic losses → operate on `r` ONLY (no gradient to `z_θ`)
- Reconstruction loss → gradient flows through `z_θ` (and through `r` via decoder if needed)
- `z_r` and `z_θ` can share encoder backbone or have separate heads

### Implementation (completed 2026-03-22)

Files changed:
- `src/models/hyperbolic_projection.py`: `HyperbolicProjection(factored=True, radial_dims=4)`,
  `DualHyperbolicProjection` passes params, returns 4-tuple `(z_A_hyp, z_B_hyp, r_A, r_B)`
- `src/models/vae.py`: `TernaryVAEV6(factored=True, radial_dims=4)`, exposes `r_A`/`r_B` in output
- `src/train.py`: reads `model.factored` and `model.radial_dims` from config
- `src/presets/v7.yaml`: new V7 config
- `tests/test_models_vae.py`: updated for new output keys

Key finding: **No changes needed to CombinedLoss**. Since `||z_hyp|| = r` with
`d(||r*dir||)/d(z_θ) = 0` (F.normalize Jacobian orthogonality), hierarchy losses
automatically have zero gradient to `z_θ` without any explicit routing change.
Empirically verified: max grad on z_θ = 1e-10 when differentiating r.sum() w.r.t. z_tangent.

### Expected outcome
- hierarchy: 0.839 → ~0.95 (radial scatter eliminated by construction)
- dist_corr: 0.903 → ~0.94 (improved inter-level separation)
- Q: 2.163 → ~2.37 (well above 2.2 target)
- Reconstruction accuracy: maintained at ~100% (z_θ handles within-level identity)

---

## Step 4 — Phase 4A: Positional Significance Encoding (2 hours, faster convergence)

Concatenate `pos_weight = [1, 1/3, 1/9, ..., 1/3^8]` to the 9-dim input → 18-dim input.
The encoder currently has no structural prior about which digit positions matter most for
v₃. This gives it that information for free.

```python
# In TernaryVAEV6:
pos_weights = torch.tensor([1/3**k for k in range(9)], dtype=torch.float64)
x_aug = torch.cat([x, x * pos_weights], dim=-1)  # 18-dim augmented input
```

Change `nn.Linear(9, 128)` → `nn.Linear(18, 128)`. No other changes.

**Expected outcome:** faster convergence (5–10 fewer epochs to reach Q=2.0). Marginal
ceiling improvement. Worth doing as a free initialization prior before V7.

---

## Step 5 — Phase 4B: FT-Transformer Encoder (1–2 weeks, research direction)

Replace the 9→128→64 MLP encoder with a `rtdl.FTTransformer`. Treat 9 trits as tokens
with learned positional significance embeddings. Self-attention can learn the prefix
structure of v₃ (the leading zero pattern) directly.

This requires a new `EncoderHead` variant — do not touch existing V6 encoder weights.
Implement as `TernaryVAEV7Transformer`, warm-start from V7 factored latent checkpoint.

**Expected outcome:** Q > 2.5 (the transformer can learn to encode valuation level
structurally, removing the per-sample encoder variance that limits Spearman). Significant
engineering effort — defer until V7 factored latent is validated.

---

## Recommended Execution Order

```
DONE:               Step 1 — geodesic fix (closed, not the bottleneck)
DONE:               Step 3 — V7 factored latent implemented
Now:                Train V7: python src/train.py --config src/presets/v7.yaml
Optional:           Step 2 — Phase 3A σ targets (principled, low effort)
Future:             Steps 4, 5 — architectural improvements beyond V7
```

---

## Q Trajectory Projections

| Milestone | Change | Projected Q |
|-----------|--------|-------------|
| V6 current | — | 2.163 (confirmed ceiling) |
| Step 1 | Geodesic loss objective fix | ~2.20 (estimate) |
| Step 2 | Per-valuation σ targets | ~2.20 (marginal) |
| Step 3 | V7 factored latent | ~2.37 |
| Step 4 | Positional encoding | ~2.38 (faster, same ceiling) |
| Step 5 | FT-Transformer | ~2.5+ |

---

## V7 Pre-Training Validation Checklist (2026-03-22, results 2026-03-22)

Before declaring V7 a success, three architectural concerns must be corroborated
during early training (epochs 10–30). Each has a specific diagnostic and a
resolution if it fails.

---

### Concern 1 — fc_mu[:, :4] pollution by reconstruction gradients  ✅ CLEARED

**Risk**: The decoder receives full z_tangent (including z_r = mu[:, :4]).
Reconstruction loss therefore pushes `fc_mu[:, :4]` in directions that may
conflict with the hierarchy gradient signal from `linear_r`.

**Empirical result (40 epochs, validate_v7_concerns.py)**:
- Spearman(||mu[:,:4]||, valuation) = −0.81 from epoch 5, stable throughout.
- r separation is monotonic by epoch 10 and hits targets by epoch 40:
  v=0→0.857 (≈outer_radius=0.85), v=1→0.622, ..., v=7→0.074 (≈inner_radius=0.08).
- No pollution. z_r successfully encodes valuation despite reconstruction access.

---

### Concern 2 — tangent_scale semantic shift  ✅ CLEARED

**Risk**: In V6, `tangent_scale=0.1` prevented expmap0 saturation (critical).
In V7, there is no expmap0 — `tangent_scale` only scales z_θ before the
direction residual net. If it collapses toward 0, `dir_unnorm ≈ tiny_noise`
and `F.normalize(tiny_noise)` produces numerically noisy random unit vectors,
destroying angular discriminability within valuation levels.

**Empirical result (40 epochs, validate_v7_concerns.py)**:
- tangent_scale drops to 0.010 at epoch 5 (transient pressure), then recovers
  and stabilizes at 0.075 by epoch 10+. Never falls below threshold.
- Within-v=0 cosine similarity decreases from 0.952 → 0.865 over 40 epochs —
  directions are diverging, not collapsing. Adequate diversity for reconstruction.

---

### Concern 3 — mu mean unconstrained with variance_only=True  ⚠️ PARTIAL → FIXED

**Finding**: `HyperbolicKLDivergence(variance_only=True)` drops the `||mu||²`
mean penalty. In V6 this was safe because `ValuationPriorLoss` handled the mean
target for z_r. In V7, `ValuationPriorLoss` is disabled. Therefore:
- `mu[:, :4]` (z_r part): constrained indirectly by hierarchy losses through
  `r = sigmoid(linear_r(mu[:, :4]))`. Strong enough.
- `mu[:, 4:]` (z_θ part): **no mean constraint at all**. Only reconstruction
  pressure and `free_bits=0.5` prevent collapse.

**KL dimension mismatch: resolved**. `lambda_x(z_hyp)` computes
`2/(1 - c * ||z_hyp||²) = 2/(1 - c * r²)` — a scalar function of radius only,
correctly applied to all 32 variance dims. The 28-dim z_hyp vs 32-dim mu
mismatch is harmless because lambda is a function of the norm, not the dimension.

**Empirical result (40 epochs, validate_v7_concerns.py)**:
- KL loss = 7.1 at epoch 40: finite, positive, non-collapsed. ✓
- Conformal factor range [3.2, 7.9]: bounded by ball constraint (max ≈ 20.5). ✓
- `||mu[:, 4:]||` grows 4.1 → 6.9 over 40 epochs with decelerating rate
  (increments: 0.72→0.46→0.37→0.31→0.12). Likely converges ~7–8 but unverified.
- **Root cause**: z_θ drifts toward deterministic encoding without mean constraint.

**Fix applied (2026-03-22)**: Set `variance_only: false` in v7.yaml.
Re-enables `||mu||²/2` penalty on all 32 dims. KL weight=0.01 vs hierarchy
weight=5.0 → hierarchy wins for mu[:, :4]. Mean drift on mu[:, 4:] is bounded.

---

## V7 Pre-200-Epoch Checklist (outstanding items as of 2026-03-22)

Four items identified before committing to the full 200-epoch run.

---

### Item A — Re-run diagnostic with variance_only:false  ⬜ TODO

The `variance_only: false` fix (Concern 3) adds back the `||mu||²/2` penalty
on ALL 32 dims, including mu[:, :4] (z_r). This directly opposes hierarchy
losses for v=0 samples (which need large `linear_r(mu[:, :4])` ≈ 2.14 to
reach r=0.85). The 40-epoch diagnostic was run BEFORE this fix. Must re-run
`validate_v7_concerns.py` with the corrected config and confirm:
- C1 Spear still ≈ −0.81 (KL pull on z_r doesn't overwhelm hierarchy)
- r separation still monotonic by epoch 10
- mu[:, 4:] norm stays bounded (fix worked)

---

### Item B — Add actual Spearman hierarchy / Q / accuracy to diagnostic  ⬜ TODO

`validate_v7_concerns.py` measures r-separation as a proxy, but the training
run will ultimately be judged on `hierarchy = -Spearman(valuation, radius)`
and `Q = dist_corr + 1.5 * hierarchy`. These were never computed in the
diagnostic. Need to add to the eval loop:
- `spearmanr(val_vals, r_A)` — the true hierarchy metric
- `dist_corr` via pairwise |r_i − r_j| vs |val_i − val_j|
- Per-digit reconstruction accuracy (coverage)
A 40-epoch preview of Q would confirm whether V7 is on track to exceed 2.163
before committing to 200 epochs of GPU time.

---

### Item C — v=8 and v=9 invisible in val set  ⬜ TODO

The diagnostic showed `nan` for v=8 and v=9 in the per-level r table.
v=9 has 1 sample (n=0), v=8 has 3 samples — with a 10% random val split
they rarely land in val. The train.py validation loop uses the same random
split, so hierarchy metrics computed during training will also miss these
levels in validation. This is not catastrophic (they are <0.02% of data),
but it means the Spearman metric in training logs is computed on at most
v=0…v=7, which slightly inflates the measured hierarchy. Verify that
train.py's validation split is deterministic (seed=42) and check whether
stratified sampling could be added.

---

### Item D — Decoder reliance on z_r dims over long training  ⬜ TODO

Over 200 epochs, the decoder may learn to use z_r (4 dims) as a shortcut
for coarse valuation-level discrimination, since z_r encodes a clean radial
hierarchy signal. If decoder reliance on z_r grows, reconstruction gradients
through z_r could start competing with hierarchy losses. No mechanism currently
prevents this. Mitigation: monitor `||grad_decoder_A||` vs `||grad_linear_r||`
during training to detect if decoder is "leaning on" the radial dims.
If detected: detach z_r before passing to decoder (pass `z_r.detach() ⊕ z_θ`
to decoder), keeping decoder fully blind to the radial scalar path.

---

### Item E — StateNet plateau detection calibrated for V6 hierarchy ≈ 0.84  ⬜ TODO

`statenet.hierarchy.plateau_threshold: 0.0005` and `plateau_patience: 10` were
tuned when hierarchy was plateauing near 0.839. In V7 hierarchy is expected to
rise to ~0.95. The plateau detector may fire early (e.g., during the initial
fast climb through 0.84→0.88) and freeze encoder_b before it finishes learning.
Review whether these thresholds need to be relaxed for V7's steeper trajectory.

---

### Item F — scatter_weight=0.8 tuned for V6's irreducible scatter  ⬜ TODO

`loss.rank.scatter_weight: 0.8` was increased from 0.3 specifically because
within-level radial scatter was the bottleneck in V6 (though ultimately
ineffective). In V7, within-level scatter should be near zero by construction
(r = f(z_r) is tightly controlled). Keeping scatter_weight=0.8 may be harmless,
but could add unnecessary gradient noise on z_r dims for same-valuation samples
that are already well-clustered. Consider reducing to 0.3 (V6 default) for V7.

---

### What NOT to Try Again

These have been exhausted and will not improve Q:

- Lagrangian dual weight increases (tried 0.01, 0.1; no gain)
- WithinLevelContrastiveLoss (WLC reduced scatter 3×; Spearman insensitive past a point)
- Geodesic loss weight changes (2.0→4.0; zero effect)
- encoder_a LR tuning (coverage was dead, now fixed; Q formula doesn't include coverage)
- More training epochs from current checkpoint (convergence confirmed)
