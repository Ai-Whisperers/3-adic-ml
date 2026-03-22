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

## What NOT to Try Again

These have been exhausted and will not improve Q:

- Lagrangian dual weight increases (tried 0.01, 0.1; no gain)
- WithinLevelContrastiveLoss (WLC reduced scatter 3×; Spearman insensitive past a point)
- Geodesic loss weight changes (2.0→4.0; zero effect)
- encoder_a LR tuning (coverage was dead, now fixed; Q formula doesn't include coverage)
- More training epochs from current checkpoint (convergence confirmed)
