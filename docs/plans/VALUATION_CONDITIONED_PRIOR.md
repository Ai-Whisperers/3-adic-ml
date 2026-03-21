# Valuation-Conditioned Prior: Full Architecture Plan

**Date:** 2026-03-20
**Status:** Design — awaiting implementation
**Prerequisite:** Fixes 5A/5B/5C (variance_weight, geodesic reweight, scatter penalty) already merged

---

## 0. The Problem in One Number

**80.0% of the KL gradient per batch actively fights hierarchy.**

```
KL gradient burden analysis (per batch of 512, c=1, target radii exponential):
  v=0: 341 ops × ||μ||=1.256 → burden=428.8  WRONG (pulling toward origin)
  v=1: 114 ops × ||μ||=0.725 → burden= 82.5  wrong (target r=0.62, not 0)
  v=2:  38 ops × ||μ||=0.492 → burden= 18.7  wrong (target r=0.46, not 0)
  ...
  v=9:   0 ops × ||μ||=0.080 → burden=  0.0  ok (target r=0.08 ≈ 0)

Total KL burden: 535.8 | Wrong-direction: 428.8 (80.0%)
```

The standard prior `p(z) = N(0, I)` is flat in valuation — it knows nothing about
3-adic structure. Replacing it with `p(z|v₃(x))` is not an optional refinement. It is
**the correct generative assumption** for this model.

---

## 1. The Six Spaces and Their Problems

The V6 pipeline transforms data through six distinct representation spaces.
Each space has a different geometric contract. The prior problem propagates differently
through each.

```
Input (ℤ₃⁹)
    ↓ encoding: position k = (i // 3^k) % 3 - 1, LSB first
Input space: 9-dim discrete {-1, 0, 1}
    ↓ nn.Linear(9, 128) + SiLU + LayerNorm
IR-1: 128-dim Euclidean ℝ¹²⁸        ← intermediate representation, not embedding
    ↓ nn.Linear(128, 64) + SiLU + LayerNorm
IR-2: 64-dim Euclidean ℝ⁶⁴          ← intermediate representation, not embedding
    ↓ nn.Linear(64, 16) × 2 (mu/logvar heads)
Tangent space: 16-dim ℝ¹⁶           ← approximate posterior parameters live here
    ↓ reparameterize: z = μ + σ·ε, ε~N(0,I)
Sample space: 16-dim ℝ¹⁶            ← z_tangent, still Euclidean
    ↓ expmap0: tanh(√c·‖v‖)·v/(√c·‖v‖)  via HyperbolicProjection
Poincaré ball: 16-dim 𝔹¹⁶(c)        ← the only geometric embedding space
```

### Space-by-Space Problem Analysis

| Space | Current Problem | Severity |
|-------|-----------------|----------|
| **Input** | All 9 positions treated symmetrically; v₃ ≡ trailing −1 count, but MLP has no structural bias | Medium |
| **IR-1 / IR-2** | No problem per se — IRs exist to compress, not to embed | None |
| **Tangent space** | `p(μ) = N(0,I)` — origin-pulling prior ignores valuation | **Critical** |
| **Sample space** | Reparameterization `μ + σε` is geometrically incoherent (wrong for curved space) | Low (tolerable approximation) |
| **Poincaré ball** | Correct space, but receives corrupted μ from wrong prior | Downstream of critical |

The **tangent space** is where the prior is enforced. Everything else is downstream.

---

## 2. The KL Decomposition

Standard `HyperbolicKLDivergence` computes:

```
KL = 0.5 * (λ(z_hyp)² * σ² + μ² − log(σ²) − d)
```

This has two separable forces:

### Force 1: Mean Term `μ²` → **Must be replaced**

- Pulls `‖μ‖ → 0` for every sample
- For v=0 (66.7% of data): target `‖μ‖ = arctanh(0.85) ≈ 1.256`
- KL gradient on μ is `μ` — pointing directly away from where 66.7% of ops should be
- **This term has no correct form given our model.** The right thing is `(‖μ‖ − target_tangent[v])²`

### Force 2: Variance Term `λ²σ² − log(σ²) − 1` → **Must be kept, but reshaped**

- Penalizes σ→0 (posterior collapse) and σ→∞ (undefined reparameterization)
- Conformal factor `λ(z_hyp)` inflates the penalty near the boundary — exactly correct
  (points near boundary are in high-curvature region; large σ there is geometrically severe)
- This term is geometrically sound. Keep it with `free_bits=0.5` to prevent collapse

### Force 3: The conformal factor `λ(z_hyp)` → **Keep, but note the level dependency**

- `λ(x) = 2/(1 − c‖x‖²)` — grows as ‖x‖→1
- v=0 points (at r≈0.85) have `λ≈3.5`; v=9 point (r≈0.08) has `λ≈2.01`
- This means the variance KL is ~3× stronger for v=0 than v=9 — correct behavior
  (v=0 points near boundary need tighter σ to avoid overlapping across the 13,122 ops)

### Summary

| KL Term | Action | Reason |
|---------|--------|--------|
| `μ²` | **Remove** | Fights hierarchy for 80% of gradient |
| `λ²σ²` | **Keep** | Geometry-aware variance penalty |
| `−log(σ²) − 1` | **Keep** | Prevents collapse |
| `free_bits=0.5` | **Keep** | Minimum KL floor |

---

## 3. The Valuation-Conditioned Prior

### 3.1 Correct Target Radii (Euclidean coordinates)

Use the exponential decay already implemented in `_exponential_target_radii()`:

```
r_target(v) = inner + (outer − inner) × (exp(−v/s) − exp(−M/s)) / (1 − exp(−M/s))
```

With `inner=0.08, outer=0.85, scale=3.0, M=9`:

```
v=0: r_euclid=0.850  v=1: 0.620  v=2: 0.456  v=3: 0.338
v=4: 0.253           v=5: 0.193  v=6: 0.149  v=7: 0.118
v=8: 0.096           v=9: 0.080
```

**NOT** `torch.linspace(0.9, 0.1, 10)`. That's off by up to 0.30 at v=3.

### 3.2 Correct Tangent-Space Targets

The prior acts on `‖μ‖` in tangent space. After `expmap0`, the Poincaré radius is:

```
r_euclid = tanh(√c · ‖μ‖) / √c   (for c=1: tanh(‖μ‖))
```

So to target `r_euclid[v]` in hyperbolic space, we need:

```
target_tangent[v] = arctanh(r_euclid[v]) / √c   (for c=1: arctanh(r_euclid[v]))
```

Numerically for c=1:
```
v=0: target_tangent=1.256  v=1: 0.726  v=2: 0.492  v=3: 0.352
v=4: 0.259                  v=5: 0.196  v=6: 0.150  v=7: 0.119
v=8: 0.096                  v=9: 0.080
```

These must be **recomputed every time curvature changes** (since curvature is learnable).

### 3.3 The New Prior Loss

```python
# For each sample in batch:
# v = TERNARY.valuation(batch_indices[i])
# target_tangent_norm = arctanh(target_r_euclid[v]) / sqrt(c)
# hierarchy_pull = (||mu|| - target_tangent_norm)^2

hierarchy_pull = mean_over_batch((||mu|| − target_tangent[v₃(i)])²)
```

This replaces the `μ²` term. It's purely radial — no angular constraint.

### 3.4 Angular Scatter (Complement)

`hierarchy_pull` only constrains `‖μ‖`. Within a valuation level, direction is free.
For v=0 (13,122 ops), points must also spread angularly to avoid overlap.
Fix 5C (`scatter_weight` in `GlobalRankLoss`) handles this separately.

---

## 4. Space-Specific Interventions

### Space A: Input Space — Structural Asymmetry (Long Term)

**Problem:** Position 0 (least significant base-3 digit) alone determines whether v₃=0.
If position 0 ∈ {−1, 1}, then v₃=0 regardless of positions 1–8 (66.7% of all ops).
The MLP treats all 9 positions with equal initial weight.

**Intervention options (ranked by impact/effort):**

1. **Explicit positional significance encoding** — concatenate `[x; pos_significance]`
   where `pos_significance[k] = 1/(3^k)`, giving the network a prior on which positions matter.
   Simple: 9 extra scalars, no architecture change. LOW EFFORT.

2. **FT-Transformer / Token attention** — treat each of the 9 trits as a token.
   Self-attention with `significance_embedding[k]` as positional encoding.
   Prefix-predicate structure of v₃ is causal — attention over position 0 dominates.
   MEDIUM EFFORT, likely necessary to break Q=2.2 ceiling long-term.

3. **Explicit v₃ auxiliary head** — add auxiliary loss `CE(predicted_v₃, true_v₃)` on encoder.
   Forces encoder to explicitly compute v₃. MEDIUM EFFORT, can be removed after convergence.

**Decision:** Start with (1), plan (2) for after prior fix validates.

### Space B: IR-1 / IR-2 — No Intervention Needed

These are Euclidean IRs. Their job is compression. The hierarchy signal arrives
as a downstream gradient via z_hyp → logmap0 → decoder → reconstruction loss.
Adding constraints here is premature optimization. Skip.

### Space C: Tangent Space — Core Intervention

This is where the prior problem lives. Three changes:

**C1: ValuationPriorLoss** (new loss class)
- Replaces `μ²` term in KL
- Targets `‖μ‖` (not `‖z_hyp‖`) — before expmap, where gradient is clean
- Recomputes `target_tangent[v]` using current curvature each forward pass
- Weight: start at 1.0, tune empirically

**C2: Variance-only HyperbolicKL** (modify existing `HyperbolicKLDivergence`)
- Set `mu=0` when computing KL (pass `mu=torch.zeros_like(mu)`)
- Keeps conformal factor `λ(z_hyp)` for geometry-aware variance penalty
- OR: add a `variance_only: bool` flag that zeroes the `mu.pow(2)` term
- Keep `free_bits=0.5`

**C3: Per-valuation σ targets** (future, after C1+C2 validate)
- High-valuation ops (v≥6) need small σ: only 18+6+2+1=27 ops total,
  must cluster tightly (small bands in radius)
- Low-valuation ops (v=0): 13,122 ops, large hyperbolic volume at boundary;
  σ can be larger
- Implement as `target_sigma[v] = sigma_base * exp(-v * sigma_scale)` — KL against `N(0, sigma_target²)`

### Space D: Sample Space — Reparameterization

**Problem:** `z = μ + σε` is correct in Euclidean tangent space, but:
- After `expmap0`, this is NOT equivalent to sampling from a wrapped normal on 𝔹
- Proper form: `exp_p(μ_point)(σ·PT(ε))` where PT = parallel transport
- Current form = `expmap0(μ + σε)` ≠ `expmap_μ_point(σε)`

**Decision:** Tolerable approximation for now. The tangent space at origin IS Euclidean,
so Euclidean reparameterization is exact there. The geometric incoherence only matters
if we want to compute proper ELBO in manifold coordinates. Defer.

### Space E: Poincaré Ball — Downstream

Once the prior is valuation-conditioned, z_hyp will naturally distribute to the
correct radial bands. The existing `MonotonicRadialLoss`, `RichHierarchyLoss`, and
`RadialHierarchyLoss` continue to push z_hyp into correct positions.

**No additional intervention needed in the Poincaré ball itself.**

The conformal factor `λ(z_hyp)` in the variance KL already adapts to position on the ball.

---

## 5. Dual VAE Differentiation

Currently both VAE-A and VAE-B receive identical loss functions. The roles differ:

| | VAE-A | VAE-B |
|--|-------|-------|
| **Role** | Coverage (reconstruction) | Hierarchy learning |
| **Priority** | All 19683 ops covered | Valuation → radius alignment |
| **KL strategy** | Keep some μ-pulling (coverage) | Valuation-conditioned prior only |
| **Recommended prior** | Softened valuation prior (low weight) | Full valuation prior (high weight) |

**Implementation:** Two `ValuationPriorLoss` instances with different weights,
or a `vae_role: coverage|hierarchy` config parameter passed to `CombinedLoss`.

This differentiation should come **after** the single-prior fix validates — adding
it too early confounds the signal.

---

## 6. Implementation Plan

### Phase 1: Prior Replacement (Next Training Run)

**P1.1 — New `ValuationPriorLoss` class in `padic_geodesic.py`**

```python
class ValuationPriorLoss(HierarchyLossBase):
    """Replaces mean term of KL with valuation-conditioned radial prior.

    Loss = mean((||mu_i|| - target_tangent[v_3(i)])^2)

    target_tangent[v] = arctanh(target_r_euclid[v]) / sqrt(c)

    This is NOT the same as RadialHierarchyLoss — that operates on z_hyp after
    expmap0. This operates on mu (tangent space), so gradients are clean (no
    expmap Jacobian) and the prior has direct access to the approximate posterior mean.
    """
    def __init__(
        self,
        curvature: float = 1.0,
        inner_radius: float = 0.08,
        outer_radius: float = 0.85,
        scale: float = 3.0,
        max_valuation: int = 9,
    ): ...

    def forward(
        self,
        mu: Tensor,           # (B, d) tangent space means
        batch_indices: Tensor, # (B,) operation indices
        curvature: float,      # current (possibly learned) curvature
        **kwargs: Any,
    ) -> Tuple[Tensor, MetricsDict]: ...
```

**P1.2 — Modify `HyperbolicKLDivergence` to support variance-only mode**

Add parameter `mean_only: bool = False`. When `True`, zero out `mu.pow(2)`:

```python
if self.variance_only:
    kl_per_dim = 0.5 * (conf_factor.pow(2) * var - logvar - 1.0)
else:
    kl_per_dim = 0.5 * (conf_factor.pow(2) * var + mu.pow(2) - logvar - 1.0)
```

Parameter name: `variance_only: bool = False` (backward compatible).

**P1.3 — Wire in `combined.py`**

```python
# New section in CombinedLoss.__init__:
if cfg.get('valuation_prior', {}).get('enabled', False):
    prior_cfg = cfg['valuation_prior']
    self.valuation_prior = ValuationPriorLoss(
        curvature=curvature,
        inner_radius=...,  # shared from rich_hierarchy
        outer_radius=...,
        scale=prior_cfg.get('scale', 3.0),
    )
else:
    self.valuation_prior = None

# In CombinedLoss.forward, must receive mu and curvature:
if self.valuation_prior is not None:
    val_prior_loss, val_prior_metrics = self.valuation_prior(
        mu=mu, batch_indices=batch_indices, curvature=current_curvature
    )
    ...
```

**Note:** `CombinedLoss.forward` currently takes `(z_hyp, indices, logits, targets, epoch, mu, logvar)`.
`mu` is already passed (for hyperbolic_kl). No signature change needed.

**P1.4 — Update `v6.yaml`**

```yaml
loss:
  valuation_prior:
    enabled: true
    weight: 1.0
    scale: 3.0   # matches exponential target decay

  hyperbolic_kl:
    enabled: true
    beta: 0.1
    weight: 0.01
    free_bits: 0.5
    variance_only: true   # NEW: disable mu² term, keep σ penalty
```

**P1.5 — Update `train.py` to pass curvature to loss**

```python
# In training loop:
current_curvature = model.projection.get_curvature()  # or dual proj
losses_A = loss_fn(z_A_hyp, batch_idx, logits_A, batch_ops, epoch,
                   mu=mu_A, logvar=logvar_A, curvature=current_curvature)
```

### Phase 2: Validation (After Phase 1)

Run 50 epochs. Expected changes:
- `r_v0` should rise from current plateau toward 0.82–0.88
- `dist_corr` should rise from ~0.40 toward 0.65+
- `hierarchy_A` should remain ≥ 0.83
- Total KL should decrease (prior loss replaces the fighting term)

**If r_v0 does not rise**: valuation prior weight too low, or curvature conversion wrong.
Debug: log `target_tangent[v]` and compare to `||mu||.mean()` per valuation level.

**If hierarchy_A collapses**: prior weight too high, overpowering MonotonicRadialLoss.
Reduce `valuation_prior.weight` or increase `monotonic.weight`.

### Phase 3: Adaptive Geometry Constraints (After Phase 2)

**Context (2026-03-21 update):** Phase 1 completed. Empirical results show dist_corr
plateaus at ~0.885, not 0.40 as originally projected. Theoretical maximum with perfect
radii is 0.9925 (confirmed by simulation). The gap (0.885→0.9925) is entirely within-level
radial variance — points at the same valuation level are spread instead of tightly clustered.

The Q=2.2 target requires dist_corr≥0.944. This is achievable, but manual weight tuning
(scatter_weight=0.3→0.8, radial_weight=0.5→1.5) is a brittle proxy for what the system
actually needs: **self-regulating geometric constraints per valuation level**.

---

#### Phase 3A: Per-Valuation σ Targets in Tangent Space

Complete the prior: `ValuationPriorLoss` currently constrains `‖μ‖` but not `σ`.
`HyperbolicKLDivergence(variance_only=True)` keeps a uniform variance penalty.
The correct prior is `N(target_tangent[v], σ_target[v]²)` — variance should also
decay with valuation (high-v points near origin need tighter σ).

```python
# In ValuationPriorLoss or a new VariancePriorLoss:
sigma_target[v] = sigma_base * exp(-v * sigma_scale)  # e.g. base=0.5, scale=0.3
var_target = sigma_target ** 2

# Replace variance_only KL with:
# KL_var = 0.5 * (λ² * σ² / var_target - log(σ² / var_target) - 1)
# KL_mean = (‖μ‖ - target_tangent[v])² / var_target   ← already ValuationPriorLoss
```

This completes the prior: both posterior mean and variance are valuation-conditioned.

**Files:** `src/losses/hyperbolic_kl.py` (new `per_valuation_variance` mode),
`src/losses/padic_geodesic.py` (`ValuationPriorLoss.forward` — add σ term).

---

#### Phase 3B: Lagrangian Dual Adaptive Weighting (Replaces Manual Tuning)

**The root problem with fixed weights:** every weight in v6.yaml is a manually tuned
proxy for a constraint. When the system drifts (e.g. within-level variance increases after
an LR restart), no automatic mechanism corrects it. We tune once, observe, tune again —
this loop is the bottleneck.

**The correct formulation is constrained optimisation:**

```
minimise  reconstruction(θ) + hierarchy(θ)
subject to  scatter_v(θ) ≤ ε_scatter    for each v = 0..9
            margin_v(θ) ≥ min_margin     for each v = 0..8
            ‖μ‖_v ≈ target_tangent[v]    for each v = 0..9
```

The Lagrangian turns each constraint into a self-scaling penalty. The dual variables
λ_v **automatically increase** when a constraint is violated and decrease when satisfied —
replacing the manual weight tuning loop entirely.

**Dual ascent update rule (gradient ascent on λ):**

```python
# After each epoch's validation metrics:
for v in range(10):
    scatter_violation_v = max(0, within_level_std_v - eps_scatter)
    margin_violation_v  = max(0, min_margin - level_gap_v)
    prior_violation_v   = abs(mean_mu_norm_v - target_tangent_v)

    lambda_scatter[v] += lr_dual * scatter_violation_v
    lambda_margin[v]  += lr_dual * margin_violation_v
    lambda_prior[v]   += lr_dual * prior_violation_v

    lambda_scatter[v] = max(0, lambda_scatter[v])  # KKT: λ ≥ 0
```

**Implementation architecture:**

```
New module: src/losses/lagrangian.py
─────────────────────────────────────
class LagrangianDualState:
    """Stores and updates dual variables for per-level geometric constraints."""

    def __init__(self, n_levels=10, lr_dual=0.01):
        self.lambda_scatter = torch.zeros(n_levels)   # within-level radial tightness
        self.lambda_margin  = torch.zeros(n_levels)   # inter-level separation
        self.lambda_prior   = torch.zeros(n_levels)   # tangent-space mean targets
        self.lr_dual = lr_dual

    def update(self, scatter_violations, margin_violations, prior_violations):
        """Called once per epoch after validation metrics. Not differentiable."""
        self.lambda_scatter = (self.lambda_scatter + self.lr_dual * scatter_violations).clamp(min=0)
        self.lambda_margin  = (self.lambda_margin  + self.lr_dual * margin_violations).clamp(min=0)
        self.lambda_prior   = (self.lambda_prior   + self.lr_dual * prior_violations).clamp(min=0)

    def get_weights(self) -> Dict[str, Tensor]:
        """Returns current effective weights for each loss component."""
        return {
            'scatter_v': self.lambda_scatter,  # shape (10,)
            'margin_v':  self.lambda_margin,
            'prior_v':   self.lambda_prior,
        }
```

**Where violation signals come from (all already exist in the codebase):**

| Violation | Source | Already computed? |
|-----------|--------|-------------------|
| `scatter_v` (within-level std) | `GlobalRankLoss.metrics['scatter_loss']` | Partially — global only. Needs per-v split. |
| `margin_v` (inter-level gap) | `MonotonicRadialLoss.metrics['r_v0'..'r_v9']` | Yes — level means logged every epoch |
| `prior_v` (μ norm vs target) | `ValuationPriorLoss.metrics['vp_mu_norm_v0'..'v4']` | Partial — first 5 levels only |
| `level_hierarchy_v` | `compute_hierarchy_metrics()['level_hierarchy']` | Yes — full per-v dict |

**Changes required per component:**

1. **`GlobalRankLoss`**: split `scatter_loss` by valuation level → return `scatter_v[0..9]`
   - Currently: single scalar. Needs: per-level `(r_i - r_j)²` mean for each same-v group.
   - One new inner loop, ~15 lines.

2. **`MonotonicRadialLoss`**: already returns `r_v0..r_v9`. Add per-level violation array.
   - Compute `gap_v = r[v] - r[v+1]` and `violation_v = max(0, min_margin - gap_v)`.
   - Already ~90% there.

3. **`ValuationPriorLoss`**: extend per-level metrics from 5 to all 10 levels.
   - Trivial: change `range(min(5, ...))` to `range(self.max_valuation + 1)`.

4. **`LagrangianDualState`** (new): ~50 lines, no PyTorch autograd needed (dual updates
   are not differentiated through; they are outer-loop parameter updates like meta-learning).

5. **`train.py`**: after validation, extract per-level violations → call `dual_state.update()`
   → pass `dual_state.get_weights()` into loss_fn on next epoch.
   - Requires `CombinedLoss.set_dual_weights(weights)` or passing λ_v directly to loss calls.

6. **`CombinedLoss`**: add optional `dual_weights` parameter to `forward()`. When present,
   scale per-level components. When absent, fall back to fixed weights (backward compatible).

**Why not use `learnable_weights: true` (Kendall uncertainty weighting)?**

The existing system learns `log_sigma` per loss by gradient of the primal objective. It
minimises total weighted loss, not constraint violations. Two critical differences:

| | Kendall (existing) | Lagrangian dual (proposed) |
|---|---|---|
| **Update rule** | Gradient of primal loss | Gradient of dual (violation magnitude) |
| **Granularity** | One σ per loss function | One λ per constraint per level |
| **Behaviour** | Reduces weight on hard losses (escape) | Increases weight on violated constraints (enforce) |
| **Guarantee** | Minimises weighted sum | Converges to constrained optimum (under convexity) |
| **Failure mode** | σ→∞ collapses any loss | λ→∞ if constraint is fundamentally infeasible |

The Lagrangian dual is adversarial to the primal: it actively increases pressure on
violated constraints instead of learning to ignore them.

**When to implement:** After Phase 2 validation confirms Q>2.15 but dist_corr still
below 0.944. Current manual weight tuning (radial 1.5, scatter 0.8) is the interim fix.
Phase 3B removes the need for that tuning entirely.

---

### Phase 4: Architectural (Long Term)

**4A: Positional significance encoding**
Concatenate `pos_weight = [1, 1/3, 1/9, ..., 1/3^8]` to input → 18-dim input.
No architecture change beyond first linear layer. Expected gain: faster convergence.

**4B: FT-Transformer encoder (replaces 9→128→64 MLP)**
Treat 9 trits as tokens with learned significance embeddings.
Self-attention learns the prefix-predicate structure of v₃ directly.
Requires new `EncoderHead` variant — do not touch existing V6 encoder until V7.

**4C: Dual VAE prior differentiation**
Separate `ValuationPriorLoss` weights for A vs B.
VAE-A: low weight (coverage priority). VAE-B: full weight (hierarchy priority).
With Phase 3B in place, this becomes: separate `lambda_prior_A[v]` and `lambda_prior_B[v]`
dual variables, each adapting independently to their VAE's constraint violations.

---

## 7. What NOT to Change

| Item | Reason |
|------|--------|
| `_exponential_target_radii()` | Already correct — use it from `ValuationPriorLoss` |
| `expmap0` in `HyperbolicProjection` | Correct geoopt implementation |
| `MonotonicRadialLoss` | Works on z_hyp — orthogonal to prior fix, complements it |
| `scatter_weight` in `GlobalRankLoss` | Now 0.8 (raised from 0.3 for within-level tightness) |
| `RichHierarchyLoss` with `variance_weight=0.5` | Already added (Fix 5A), keep |
| `geodesic.weight` | Now 2.0 (raised from 0.5); primary driver of dist_corr |
| `tangent_scale` parameter | Learnable, working — leave alone |
| `learnable_curvature` | Required for `target_tangent` recalculation to matter |
| VAE-B loss wiring | Already fixed in V6.2, don't touch |

---

## 8. Why Existing RadialHierarchyLoss Is Not Sufficient

`RadialHierarchyLoss` operates on `z_hyp` (after expmap0). It pushes the actual
Poincaré-ball radius toward `target_r_euclid[v]`.

`ValuationPriorLoss` operates on `μ` (before expmap0, before reparameterization).
It pushes the **approximate posterior mean** to the correct radius.

These are complementary, not redundant:

```
RadialHierarchyLoss: minimize (r(z_hyp) − target_r)²       ← corrects samples
ValuationPriorLoss:  minimize (||μ|| − target_tangent)²     ← corrects the distribution
```

Without `ValuationPriorLoss`, KL is still fighting the posterior mean toward 0.
`RadialHierarchyLoss` can win this fight (and currently does, partially), but the
gradient conflict wastes training signal. With `ValuationPriorLoss`, the KL prior
**agrees** with the hierarchy losses — they all pull in the same direction.

The KL gradient currently provides 428.8 units of force against hierarchy per batch.
Turning it into 428.8 units of force *for* hierarchy is a ~2× effective gradient boost
on the most important objective.

---

## 9. Success Criteria

*Updated 2026-03-21: Phase 1 completed. Actual results vs. original projections.*

| Metric | Before Phase 1 | After Phase 1 (actual) | Phase 3 target | Theoretical max |
|--------|---------------|----------------------|----------------|-----------------|
| `dist_corr` | 0.882 (plateau) | 0.885 (marginal gain) | ≥ 0.944 | 0.9925 |
| `hierarchy_A` | 0.836 | 0.838 (saturated) | ≥ 0.838 | ~0.84 |
| `Q` | 2.141 | 2.145 (marginal gain) | ≥ 2.20 | ~2.49 |
| `KL wrong-direction` | 100% | 0% | 0% | 0% |
| within-level std (v=0) | unknown | unknown | ≤ 0.05 | 0 |
| convergence speed | Q=2.0 at ep≈60 | Q=1.67 at ep20 (faster) | — | — |

**Key revision from original projections:**
- dist_corr was NOT at 0.40 — it was already at 0.882. The original projection was wrong.
- Phase 1 (ValuationPriorLoss) improved convergence speed but did not raise the ceiling.
- The Q=2.141 ceiling is a within-level radial variance problem, not a prior direction problem.
- Theoretical max confirmed at 0.9925 — Q=2.2 is achievable, Q≥2.4 is achievable.
- Phase 3B (Lagrangian dual) is the principled path to Q≥2.2 without further manual tuning.

Q=2.2 requires dist_corr≥0.944 (holding hier=0.838). Currently 0.059 below that.
Q≥2.4 requires dist_corr≥0.962 — achievable with tight within-level clustering.
Q≥2.5 likely requires Phase 4A or 4B (structural inductive bias in encoder).

---

## 12. Per-Space Code Audit vs. PyTorch Patterns (2026-03-21)

Deep read of each space against the actual source, compared to canonical PyTorch
patterns and available external libraries. Findings are ordered by severity.

---

### Space A — Input (9-dim discrete {-1, 0, 1})

**Source:** `TernarySpace._ternary_lut` (ternary.py:141–151), `_build_encoder_backbone`
(vae.py:142–184), first Linear in the chain.

#### A1. Discrete input treated as continuous — no nn.Embedding

**Issue:** The 9 input dimensions are discrete trits: each position takes values
`{-1, 0, 1}` (3 classes). The encoder passes them directly to `nn.Linear(9, 128)`,
treating them as continuous reals. This is not wrong — PyTorch uses this pattern
widely for integer-encoded categoricals — but it forfeits structure.

**What `nn.Embedding` would give:**
```python
# Current (implicit continuous encoding)
nn.Linear(9, 128)  # each position's value is a scalar

# Better (explicit discrete token encoding)
self.trit_embed = nn.Embedding(3, embed_dim)  # 3 classes per position
# x in {0,1,2} (remap from {-1,0,1}), shape (B, 9)
# → embed: (B, 9, embed_dim) → reshape: (B, 9*embed_dim)
```

`nn.Embedding` learns a separate vector for each class, not a linear interpolation
between -1, 0, +1. This matters because the mapping from trit value to geometry is
not linear — the zero trit at position 0 has a categorical meaning (determines v₃=0).

**External tool:** `rtdl.FTTransformer` (Gorishniy et al., 2021 — "Revisiting Deep
Learning Models for Tabular Data"). Treats each feature as a token via
`FeatureTokenizer`, then applies standard transformer attention. Available via `pip
install rtdl`. For this use case: 9 tokens × 3-class embedding → cross-token attention
learns the prefix-predicate structure of v₃ without manual positional encoding.

**Effort:** Phase 4B. Breaking encoder architecture change.

#### A2. No positional significance weighting at source

The LUT produces the raw digits `(i // 3^k) % 3 - 1` with equal weight for all k.
Position 0 (LSB) alone determines whether v₃=0 (for 66.7% of data). The network has
no structural prior about this asymmetry. **Phase 4A** (concatenate `1/3^k` weights)
addresses this at low cost.

---

### Space B/C — IR-1 (128-dim) and IR-2 (64→128→128, corrected)

**Source:** `_build_encoder_backbone` (vae.py:165–175).

Actual architecture (improved encoder):
```python
Linear(9, 128) → LayerNorm(128) → SiLU()
Linear(128, 128) → LayerNorm(128) → SiLU()
Linear(128, 64) → SiLU()           ← no LayerNorm here
```

#### B1. No residual connections — gradient path through 3 layers is unprotected

**Issue:** Three `Linear → LN → SiLU` layers with no skip connections. For a 3-layer
MLP this isn't catastrophic, but the final layer `Linear(128, 64) → SiLU` has no
LayerNorm and no skip. The gradient from mu/logvar heads flows back through 3 linear
layers without shortcut.

**Standard PyTorch pattern for MLPs:** Pre-norm residual blocks:
```python
# Pre-norm residual (transformer-style)
h = x + SiLU(LayerNorm(Linear(x, hidden)))
```

`torch.nn.functional.layer_norm` + residual is ubiquitous in deep MLPs since ResNet.
Without it, the information bottleneck at the 128→64 layer is less efficiently trained.

#### B2. Final encoder layer (128→64) lacks LayerNorm before mu/logvar heads

`Linear(128, 64) → SiLU()` feeds directly into `fc_mu` and `fc_logvar`. No LayerNorm
means the output distribution of this layer is uncontrolled. If the 64-dim hidden
state has high variance, the mu/logvar heads produce high-variance outputs from random
init — amplifying the cold-start KL-vs-hierarchy conflict.

**Fix:** Add `LayerNorm(64)` between `Linear(128, 64)` and `SiLU()`, or wrap the
final layer in a pre-norm residual block. This stabilizes initialization.

#### B3. `torch.compile` opportunity

The encoder backbone is a pure PyTorch MLP — no custom C++ ops, no dynamic shapes.
`torch.compile(model.head_A)` with `mode="reduce-overhead"` would give ~20–30%
speedup on the encoder forward/backward passes (the most-called path per batch).

**Blocker:** `torch.compile` + geoopt may have issues in the projection layers.
Recommendation: compile only `head_A` and `head_B` (encoder heads), leave
`projections` and losses uncompiled.

---

### Space D — Tangent Space (μ, logvar, 16-dim)

**Source:** `EncoderHead.forward` (vae.py:99–111), `fc_mu` and `fc_logvar`.

#### D1. No logvar clamping — unconstrained output can explode

**Issue:** `fc_logvar` outputs unconstrained values. `std = exp(0.5 * logvar)`.
If logvar → +10, std → 148 — the reparameterized sample `z = μ + 148·ε` is 148×
farther from μ than the geometry intends. The only protection is the KL `free_bits`
floor, which doesn't cap the ceiling.

**Standard VAE pattern:** Clamp logvar at the source:
```python
logvar = self.fc_logvar(h).clamp(-10.0, 2.0)
# std in [exp(-5), exp(1)] = [0.007, 2.72]
```

This is done in essentially every production VAE implementation (e.g., JAX reference
implementation, HuggingFace VQVAE). The range `[-10, 2]` is a loose bound —
`[-4, 4]` is tighter. For this model, `[-4, 2]` is appropriate: min σ≈0.14 (tight
cluster), max σ≈2.7 (generous sample spread).

**Location to add:** `EncoderHead.forward` (vae.py:109), after `logvar = self.fc_logvar(h)`.

#### D2. Cold-start ‖μ‖ is below v=0 target

Kaiming initialization of `fc_mu` (fan-in=64, gain for SiLU≈1.0) gives output std
≈ sqrt(2/64) ≈ 0.177 per dimension. For 16-dim μ: `E[‖μ‖] ≈ 0.177 × sqrt(16) = 0.71`.

After the valuation prior is added, v=0 target is `target_tangent[0] = 1.256`.
Cold-start ‖μ‖ ≈ 0.71 means v=0 operations start at ~56% of their target — the prior
loss immediately applies a strong upward force, competing with KL (before it's disabled).

**Better initialization:** `nn.init.normal_(fc_mu.weight, 0, 0.3)` gives initial
`E[‖μ‖] ≈ 0.3 × sqrt(16) = 1.2` — closer to v=0 target from epoch 0.
Or: Xavier with `gain=2.0` for the mu head specifically.

This is a minor win but reduces the warm-up cost of prior training.

---

### Space E — Sample Space (z_tangent = μ + σε, 16-dim)

**Source:** `TernaryVAEV6.reparameterize` (vae.py:322–340).

#### E1. σε can dominate μ — prior constraint is on μ, not z_tangent

`ValuationPriorLoss` constrains `‖μ‖ → target_tangent[v]`. But the actual sample is
`z_tangent = μ + σε`. If σ is large (σ > 0.5), `‖z_tangent‖` can deviate significantly
from `‖μ‖`, meaning the expmap0 receives a sample at the wrong radius.

**This is not a bug** — it's the correct VAE behavior (the prior constrains the mean,
not the sample). But it means `ValuationPriorLoss` alone does not guarantee that
z_hyp is at the right radius — it only constrains where the MEAN is. `RadialHierarchyLoss`
and `MonotonicRadialLoss` operating on z_hyp handle the sample-level constraint.

**Interaction:** With logvar clamping added (D1), max σ ≈ 2.72. At target ‖μ‖=1.256
(v=0), the sample radius `E[‖z_tangent‖] ≈ sqrt(‖μ‖² + σ²·d) = sqrt(1.58 + 7.4·d)`
for d-dim σ. For d=1 (isotropic) this is sqrt(9.0)=3.0 — far above target. This is
why free_bits + variance KL is critical: controlling σ is as important as controlling μ.

#### E2. Wrapped normal reparameterization (deferred, acknowledged)

As noted in Section 4 (Space D), Euclidean reparameterization at the origin tangent
space is the standard approximation for Poincaré VAEs (Mathieu et al. 2019 also does
this). The wrapped normal distribution (Nagano et al. 2019) is the correct form, but
it requires parallel transport and is computationally expensive.

**Available implementation:** `geoopt` has `WrappedNormal` distribution in development
(not stable as of 2024). The `hyperspherical-vae` library (Davidson et al.) provides
von Mises-Fisher reparameterization for sphere geometry — not directly applicable here.

**Decision:** Defer to Phase 5+ (not in current roadmap).

---

### Space F — Poincaré Ball (z_hyp, 16-dim, 𝔹¹⁶(c))

**Source:** `HyperbolicProjection.forward` (hyperbolic_projection.py:139–177),
`exp_map_zero` (poincare.py:155–169), `get_manifold` (poincare.py:48–86).

#### F1. **CRITICAL: `exp_map_zero` bypasses model's learnable curvature**

`HyperbolicProjection` owns a learnable manifold:
```python
self.manifold = geoopt.PoincareBall(c=curvature, learnable=learnable_curvature)
self.curvature = self.manifold.c  # learnable Parameter
```

In `forward`, it calls:
```python
c = self.curvature.item()   # ← .item() DETACHES from computation graph
z_hyp = exp_map_zero(z_transformed, c=c)  # uses float, not Parameter
```

`exp_map_zero` calls `get_manifold(c=1.001, device=...)` — creates or retrieves a
**new manifold** with static curvature 1.001, NOT `self.manifold` with its learnable c.

**Consequence:**
- The expmap formula `tanh(√c·‖v‖)·v/(√c·‖v‖)` uses a static float for c
- ∂z_hyp/∂c = 0 in the computation graph — curvature gets no gradient from expmap
- Curvature can only be updated via gradients that reach `self.manifold.c` directly
  through other paths (there are almost none)
- The manifold cache at key `(1.001, 'cuda:0')` never gets freed — with learnable
  curvature changing slowly, this accumulates O(epochs) stale manifold entries

**Correct pattern:**
```python
# In HyperbolicProjection.forward — use model's own manifold, not global cache
origin = torch.zeros_like(z_transformed)
z_hyp = self.manifold.expmap(origin, z_transformed)
# Gradient flows through self.manifold.c (the learnable Parameter)
```

This requires removing the `c = self.curvature.item()` extraction and calling
`self.manifold.expmap` directly. The `forward_with_components` method has the same bug.

**Impact level:** HIGH. Learnable curvature is currently not being learned properly.

#### F2. Global manifold cache leaks memory for learnable curvature

`_manifold_cache` in `poincare.py` is a module-level dict keyed by `(c_float, device_str)`.
With learnable curvature, `c.item()` returns slightly different floats as training
progresses (e.g., 1.000000, 1.000023, 1.000041...). Each unique float creates a new
`geoopt.PoincareBall` instance in the cache. Over 200 epochs with ~10 updates/epoch,
this could accumulate ~2000 dead manifold objects.

**Fix for cache:** Key by `(round(c, 4), device_str)` to bucket nearby curvature values,
and/or use `weakref.WeakValueDictionary` so stale manifolds are GC'd.
**Better fix:** Use `self.manifold.expmap` (F1 fix) so the cache is bypassed entirely
for learnable curvature.

#### F3. `hyperbolic_radius` allocates a zero tensor every call

```python
def hyperbolic_radius(z, c=1.0):
    origin = torch.zeros_like(z)   # allocation per call
    return poincare_distance(z, origin, ...)
```

This is called inside every loss that uses radii (all 5 losses). Per batch:
- 5 losses × 1 allocation each = 5 × (B × d × 8 bytes) = 5 × 512 × 16 × 8 = 320 KB/batch

**Pattern:** geoopt's `PoincareBall` has `dist0(x)` or `norm0(x)` methods that
compute distance from origin without allocating a zero tensor. Check:
```python
manifold.dist0(x, keepdim=False)  # geoopt PoincareBall method
```

If available in the installed geoopt version, this is both faster and more idiomatic.

#### F4. Max-radius clamp in HyperbolicProjection uses a non-geoopt projection

The max_radius clamp after expmap:
```python
norm = torch.norm(z_hyp, dim=-1, keepdim=True)
scale = torch.where(norm > self.max_radius, self.max_radius/(norm+eps), ones)
z_hyp = z_hyp * scale
```

This clamps in Euclidean norm space, which is the right thing for the Poincaré ball
(the Euclidean norm IS the coordinate radius). But `geoopt.PoincareBall.projx(x)`
already does this clamp (to `1 - eps` by default). The `projx` + custom max_radius
clamp is redundant but not wrong.

However, this clamp is applied AFTER expmap. If z_transformed is large, expmap could
saturate at r≈1.0, then the clamp rescales to 0.95. **The clamp is the last safeguard
against boundary explosions** — keep it. But it might be cleaner to clamp the tangent
vector before expmap: `z_transformed = z_transformed.clamp(max=...)`.

---

### External Libraries: Applicability Matrix

| Library | Space | What it provides | Priority |
|---------|-------|-----------------|----------|
| `rtdl.FTTransformer` | A (Input) | Proper tabular tokenizer + attention for discrete features | Phase 4B |
| `einops` | A, B/C | Cleaner `rearrange`/`repeat` for trit tokenization | Any phase, low cost |
| `torch.compile` | B/C | ~20-30% speedup on encoder MLP (pure PyTorch, no C++ ops) | Near term |
| `geoopt.WrappedNormal` | E | Correct manifold reparameterization (when stable) | Phase 5+ |
| `geoopt.dist0` | F | Allocation-free distance-from-origin | Near term |
| `grokfast` | Training | Amplifies slow-gradient components to break plateaus | Near term |
| `torchmetrics.SpearmanCorrCoef` | Metrics | Differentiable Spearman (for trainable dist_corr loss) | Phase 3 |
| `mup` (μP) | B/C, D | Maximal Update Parameterization — correct LR scaling across widths | Phase 4+ |

#### `grokfast` (Kang et al., 2024) — Most Relevant Near Term

`pip install grokfast`. Wraps optimizer to apply EMA filter to gradients:
```python
from grokfast import gradfilter_ema
grads = None
# In training loop, after loss.backward():
grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
optimizer.step()
```

This amplifies slow-moving gradient components (the ones that signal generalization
structure) relative to fast-oscillating ones (noise). Given that hierarchy_A plateaus
at 0.836 and dist_corr is stuck at 0.40 — both showing the "plateau before grokking"
pattern the project is tracking — this could help break the Q ceiling without architectural
changes. Low risk, reversible, 3 lines of code.

#### `torchmetrics.SpearmanCorrCoef` — For Differentiable dist_corr Loss

Currently `dist_corr` is computed with `scipy.spearmanr` in numpy (train.py:577) —
no gradient. If `dist_corr` were a trainable objective, it would need a differentiable
surrogate. `torchmetrics.functional.spearman_corrcoef` is differentiable:

```python
from torchmetrics.functional import spearman_corrcoef
# dist_corr_loss = 1.0 - spearman_corrcoef(r_dists.flatten(), v_dists.flatten())
```

This would make `dist_corr` directly optimizable, complementing `scatter_weight` in
`GlobalRankLoss`. Medium priority — only needed if Phase 1+2 don't push dist_corr
past 0.70 on their own.

---

### Priority Order for Space Fixes

| Fix | Space | Severity | Effort |
|-----|-------|----------|--------|
| F1: Use `self.manifold.expmap` directly | F | HIGH (gradient broken) | 3 lines |
| D1: Clamp logvar at encoder source | D | MEDIUM (numerical safety) | 1 line |
| F2: Fix manifold cache leak | F | LOW (memory, not correctness) | 5 lines |
| B2: Add LayerNorm before mu/logvar | B/C | LOW (stability) | 1 line |
| D2: Better mu head initialization | D | LOW (warm-up cost) | 1 line |
| B1: Add residual connections | B/C | LOW (not blocking) | Refactor |
| A1: nn.Embedding for discrete input | A | MEDIUM (long term) | Phase 4B |
| F3: Use `dist0()` for radii | F | LOW (performance) | 3 lines |

F1 (manifold expmap) should be fixed before or alongside the valuation prior —
it affects the entire geometric pipeline and the learnable curvature claim.

---

## 11. Corrections from Code Audit (2026-03-21)

The following errors and missing constraints were found by reading the actual source
before Section 10 was written as assumptions. All corrections are canonical.

---

### 11.1 Section 0: KL Burden Numbers Were Wrong

**Error:** The original burden table used **natural batch distribution** (66.7% v=0),
but `train.py` already has a `WeightedRandomSampler` with `weight ∝ 1/√count` since
before this plan was written (lines 787–826).

**Corrected stratified distribution (512 batch):**

```
v=0: 217/batch (was 341) | KL burden = 272.7  (64.1% of total)
v=1: 125/batch (was 114) | KL burden =  91.0  (21.4%)   ← MORE than natural
v=2:  72/batch (was  38) | KL burden =  35.6
v=3:  42/batch (was  13) | KL burden =  14.7
...
v=9:   2/batch (was   0) | KL burden =   0.2  (0.0%)  ← appears reliably now

Total burden: 425.2 | Wrong-direction (v=0..8): 425.1 (100.0%)
```

**More important correction:** The original table labeled v=1 as "ok (→0)" — that was
wrong. v=1 target radius is 0.62 in Euclidean, target tangent norm 0.726.
KL pulling toward 0 is wrong for v=1, v=2, ..., v=8. Only v=9 (target r≈0.08) is
close enough to 0 that KL is approximately correct.

**Revised summary:**
- With stratification: absolute burden reduced 535→425 (v=0 appears less)
- But more levels now appear in every batch, all pulling wrong
- **100% of KL gradient is wrong-direction** (only v=9 is ok, contributes 0.0%)
- Stratification makes rare levels appear 11–125× more often than natural — they all fight hierarchy now too

The "80% wrong-direction" claim understated the problem. The correct figure is 100%.

---

### 11.2 Wrong Attribute: `model.projection` → `model.projections`

**Error in P1.5:** The plan says:
```python
current_curvature = model.projection.get_curvature()
```

**Correct attribute name:**
```python
current_curvature = model.projections.get_curvature()  # plural
```

`TernaryVAEV6` stores the `DualHyperbolicProjection` as `self.projections` (vae.py:287).
`DualHyperbolicProjection.get_curvature()` delegates to `self.proj_A.get_curvature()` (hyperbolic_projection.py:296).

---

### 11.3 `CombinedLoss.forward` Has No `curvature` Parameter

**Error in P1.3:** The plan says "No signature change needed" for `CombinedLoss.forward`.
That's true for `mu` (already present), but **not for `curvature`**.

**Actual signature (combined.py:371):**
```python
def forward(self, z_hyp, indices, logits, targets, epoch=0,
            mu=None, logvar=None) -> Dict[str, Any]:
```

**Problem:** `CombinedLoss` stores `self.curvature = curvature` at init time (line 108),
set from `config.get("model", {}).get("curvature", 1.0)` in train.py (line 836).
Since curvature is **learnable**, `self.curvature` goes stale after epoch 0.
`ValuationPriorLoss` needs the **current** curvature to compute correct tangent targets.

**Required change — add to `CombinedLoss.forward` signature:**
```python
def forward(self, z_hyp, indices, logits, targets, epoch=0,
            mu=None, logvar=None,
            curvature: Optional[float] = None) -> Dict[str, Any]:
    # Inside: curvature_to_use = curvature if curvature is not None else self.curvature
```

**Required change — two call sites in train.py (lines 1088–1098):**
```python
_c = model.projections.get_curvature()

losses = loss_fn(
    z_A_hyp, batch_idx, logits_A, batch_ops, epoch=epoch,
    mu=out.get("mu_A"), logvar=out.get("logvar_A"),
    curvature=_c,
)
losses_B = loss_fn_b(
    z_B_hyp, batch_idx, logits_B, batch_ops, epoch=epoch,
    mu=out.get("mu_B"), logvar=out.get("logvar_B"),
    curvature=_c,
)
```

One `get_curvature()` call per batch (not two) — both `loss_fn` and `loss_fn_b` use
the same curvature since they share the same underlying geometry.

---

### 11.4 Two CombinedLoss Instances — Both Need Updating

**Not mentioned in the original plan:** `train.py` creates **two** CombinedLoss instances:
- `loss_fn` (full config — coverage + hierarchy for VAE-A)
- `loss_fn_b` (hierarchy only — `coverage_weight=0.0` overridden for VAE-B)

The `loss_fn_b` config is a shallow copy (train.py:844):
```python
loss_cfg_b = {k: dict(v) if isinstance(v, dict) else v for k, v in loss_cfg.items()}
loss_cfg_b["rich_hierarchy"]["coverage_weight"] = 0.0
```

The `valuation_prior` section is a new top-level dict — the shallow copy will clone it
correctly (`dict(v)` for any dict value). So both `loss_fn` and `loss_fn_b` will
instantiate `ValuationPriorLoss` with identical config. **This is correct**: both
VAE-A and VAE-B need the valuation prior on their respective `mu_A` and `mu_B`.

**The Phase 4C dual-VAE differentiation** (separate weights for A vs B) will require
separate `valuation_prior` sections in the config or a `vae_role` parameter — that's
explicitly out of scope for Phase 1.

---

### 11.5 Encoder Architecture Is 9→128→128→64 (Not 9→128→64)

**Error in diagram (Section 1):** The pipeline diagram shows `IR-1: 128-dim` and
`IR-2: 64-dim`, suggesting 9→128→64. The actual `improved` encoder (vae.py:165):

```python
nn.Linear(9, hidden_dim*2),    # 9 → 128
nn.LayerNorm(hidden_dim*2),
nn.SiLU(),
nn.Linear(hidden_dim*2, hidden_dim*2),  # 128 → 128  ← missing from diagram
nn.LayerNorm(hidden_dim*2),
nn.SiLU(),
nn.Linear(hidden_dim*2, hidden_dim),   # 128 → 64
nn.SiLU(),
```

There are **three** linear layers, not two. The IR is actually:
```
Input (9) → IR-1a (128) → IR-1b (128) → IR-2 (64) → mu/logvar (16)
```

**Corrected pipeline diagram:**
```
Input space: 9-dim {-1, 0, 1}
    ↓ Linear(9, 128) + LN + SiLU
IR-1a: 128-dim ℝ¹²⁸
    ↓ Linear(128, 128) + LN + SiLU
IR-1b: 128-dim ℝ¹²⁸
    ↓ Linear(128, 64) + SiLU
IR-2: 64-dim ℝ⁶⁴
    ↓ Linear(64, 16) × 2 (mu, logvar heads)
Tangent space: 16-dim ℝ¹⁶
```

**Impact on Section 4A (positional significance encoding):**
The `pos_weight` concatenation adds 9 scalars: input becomes 18-dim.
The first `Linear(9, 128)` must be changed to `Linear(18, 128)`.
All other layers are unaffected. This is still LOW EFFORT.

---

### 11.6 Learnable Weights: `_init_learnable_weights` Needs a Branch for ValuationPriorLoss

**Not addressed in the original plan:** `CombinedLoss` has a `learnable_weights` mode
(Kendall et al. uncertainty weighting). When enabled, each loss gets a `log_sigma`
parameter (combined.py:280–339).

If `valuation_prior` is added and `learnable_weights: true` is set, the implementation
must add:
```python
# In CombinedLoss._init_learnable_weights():
if self.valuation_prior is not None:
    self.log_sigma_valuation_prior = nn.Parameter(
        torch.tensor(weight_to_log_sigma(self.valuation_prior_weight), dtype=torch.float64)
    )
```

And in `forward`, the weighted contribution:
```python
if self.use_learnable_weights and hasattr(self, 'log_sigma_valuation_prior'):
    total = total + self._weighted_loss(val_prior_out, self.log_sigma_valuation_prior)
else:
    total = total + self.valuation_prior_weight * val_prior_out
```

**Recommendation:** For Phase 1, set `learnable_weights: false` (it's already off in
v6.yaml). Add the `log_sigma` branch as part of implementation but don't enable until
the prior is validated.

---

### 11.7 `P1.2` Flag Name Inconsistency

**Error in P1.2:** The text says "Add parameter `mean_only: bool = False`" but the
code block shows `if self.variance_only`. These are inconsistent.

**Correct parameter name:** `variance_only: bool = False`
- When `True`: `kl_per_dim = 0.5 * (conf_factor² * var − logvar − 1.0)` (no `mu²`)
- When `False` (default): current behavior unchanged (backward compatible)

The name `variance_only` is semantically accurate: the flag makes the KL operate on
variance only, dropping the mean term. `mean_only` would mean the opposite.

---

### 11.8 `decode_b=False` in Training — Implications for ValuationPriorLoss

**Training loop (train.py:1081):**
```python
out = model(batch_ops, decode_b=False)
```

`decode_b=False` skips `decoder_B` — `logits_B = None`. However, `mu_B` and `logvar_B`
are still fully computed (the decoder is skipped, not the encoder). From vae.py:376–388,
the output dict always includes `mu_B`, `logvar_B`, `z_B_hyp`.

`ValuationPriorLoss` only needs `mu_B` — not `logits_B`. No impact.

---

### Summary of Changes Required vs. Original Plan

| Item | Original Plan | Corrected |
|------|--------------|-----------|
| Section 0 burden numbers | 80% wrong, natural dist | 100% wrong, 425.2 total, stratified |
| `model.projection` | Wrong attribute | `model.projections` (plural) |
| `CombinedLoss.forward` signature | "no change needed" | Must add `curvature: Optional[float] = None` |
| `train.py` call site | One example | Two calls (loss_fn + loss_fn_b), one `get_curvature()` |
| Encoder shape | 9→128→64 | 9→128→128→64 |
| Learnable weights | Not mentioned | Needs `log_sigma_valuation_prior` branch |
| `P1.2` flag name | `mean_only` typo | `variance_only` |
| `decode_b=False` impact | Not mentioned | No impact (mu_B still computed) |

---

## 10. Relationship to `HierarchicalVaeEnergy` (User Prototype)

The user's prototype captured the right spirit:
- Valuation-conditioned prior on ‖μ‖ ✓
- Decoupled variance-only KL ✓

Problems in the prototype (do not copy):
- `torch.linspace` targets: off by up to 0.30 at v=3 (use `_exponential_target_radii`)
- `‖μ‖` used directly as Euclidean radius target: should be `arctanh(target_r)/√c`
- Simplified expmap `tanh(‖z‖)·z/‖z‖` ignores learnable curvature: use geoopt
- No integration with existing `CombinedLoss` / config system

This plan is the correct version of that prototype's idea.
