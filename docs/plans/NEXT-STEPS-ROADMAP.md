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
| **Phase 4C** | **Factored latent z_radial ⊕ z_identity (V7)** | ❌ Not started — **highest priority** |

---

## Step 1 — Fix the Geodesic Loss Objective (1 hour, potentially +0.04 dist_corr)

**The problem:** `PAdicGeodesicLoss` targets `v₃(|index_i − index_j|)` — the 3-adic
valuation of the *difference* between operation indices. But `dist_corr` evaluates
`|valuation(i) − valuation(j)|` — the absolute difference of *individual* valuations.
These are different signals, and the mismatch means the geodesic loss is not directly
optimizing the metric it's supposed to improve.

**The fix:** change `PAdicGeodesicLoss.forward()` to use individual valuation differences:

```python
# Before (current):
diff = torch.abs(batch_indices[i_idx].long() - batch_indices[j_idx].long())
valuation = TERNARY.valuation(diff).double()

# After:
v_i = TERNARY.valuation(batch_indices[i_idx]).double()
v_j = TERNARY.valuation(batch_indices[j_idx]).double()
valuation = torch.abs(v_i - v_j)  # direct individual valuation difference
```

**Expected outcome:** dist_corr improves from 0.903 toward ~0.93–0.94, potentially
pushing Q from 2.163 to ~2.20. Low risk change — the old signal was correct for the
3-adic ultrametric interpretation, but misaligned with the evaluation metric.

**Note:** This changes the semantic: the geodesic loss will now directly optimize
what dist_corr measures, at the cost of no longer encoding the true 3-adic ultrametric
structure. Whether that matters depends on research goals (pure 3-adic fidelity vs Q score).

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

## Step 3 — V7 Factored Latent Architecture (2–3 days, breaks Q ceiling)

**This is the only change that can push Q beyond 2.2.**

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

### Implementation steps

1. **`HyperbolicProjection`**: add `factored=True` mode:
   - Split `z_tangent` into `z_r[:k]` and `z_theta[k:]`
   - Compute `r = sigmoid(linear_r(z_r)) * max_radius`
   - Compute `dir = F.normalize(z_theta, dim=-1)`
   - Return `z_hyp = r.unsqueeze(-1) * dir`, plus expose `r` for loss routing

2. **`CombinedLoss`**: route hierarchy losses to `r`, not `hyperbolic_radius(z_hyp)`:
   - `radial_loss(r, valuations)` — direct MSE to target radius, no expmap needed
   - `monotonic_loss(r, valuations)` — same
   - `geodesic_loss(z_hyp, ...)` — unchanged (operates on Poincaré distances)

3. **`VAE`**: expose `r` tensor in forward output dict

4. **No changes needed to**: decoder, KL loss, coverage loss, training loop

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
Today (1–2 hours):  Step 1 — fix geodesic loss objective → quick Q check
This week:          Step 3 — V7 factored latent → break Q ceiling
Optional cleanup:   Step 2 — Phase 3A σ targets (principled, low effort)
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
