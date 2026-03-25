# 2026-03-25 Project Feasibility Review

## Executive Verdict

This repository already demonstrates one real technical achievement: a dual-VAE system can learn a very accurate hyperbolic embedding of the **entire closed domain** `{-1,0,1}^9`, with near-perfect reconstruction and strong radial ordering by 3-adic valuation. The strongest saved checkpoint I recomputed on the full 19,683-state domain reached:

- per-digit reconstruction accuracy: **0.999458**
- perfect reconstruction coverage: **0.995275**
- hierarchy: **0.832050**
- distance correlation: **0.897258**
- `Q = dist_corr + 1.5 * hierarchy`: **2.145334**

That is materially better than I expected before reading the code and rerunning the checkpoint. The core embedding/reconstruction system is not vaporware.

The main skeptical conclusion is equally important: the project does **not** currently prove disruptive external prediction, commercial inference superiority, neurosymbolic reasoning, SIMD/runtime breakthroughs, or any task that traditional exact methods cannot already solve better on the native domain. The repository trains on the full fixed state space itself, not on an external phenomenon, and almost all supervisory structure comes from the integer indexing scheme rather than from external labels or measured outcomes.

## What The Project Actually Is

### Native data domain

The training domain is the full discrete set of `3^9 = 19,683` balanced-ternary vectors. The canonical constants are defined in `src/core/ternary.py:80-84`, and the training data construction simply materializes every operation via `TERNARY.all_ternary()` inside `DataAuditor.prepare_data()` in `src/train.py:167-213`.

This means:

- the project is **not** learning from a sampled real-world dataset
- the project is **not** forecasting an external target
- the project is **not** doing open-domain sequence modeling
- the project is learning a geometry over a fully enumerated symbolic state space

`docs/DATA-SEMANTICS.md:15-72` is the most important conceptual document in the repository. It states plainly that the hierarchy used by the losses is largely **indexing-derived**, not an intrinsic semantic property of the ternary vector contents.

### Indexing-derived target structure

The main hierarchy signal is based on the 3-adic valuation of integer indices and their differences:

- `PAdicGeodesicLoss` trains pairwise distances from `v_3(|i-j|)` or `|v_i - v_j|` in the alternate mode: `src/losses/padic_geodesic.py:89-255`
- `RadialHierarchyLoss` maps valuation levels to target hyperbolic radii: `src/losses/padic_geodesic.py:257-438`
- `RichHierarchyLoss` enforces per-level mean radius, reconstruction, and separation: `src/losses/padic_geodesic.py:840-1053`
- `compute_hierarchy_metrics()` recomputes `hierarchy`, `dist_corr`, and `Q`: `src/train.py:541-615`
- `compute_Q()` is explicitly `dist_corr + 1.5 * hierarchy`: `src/models/lr_controller.py:46-55`

The direct implication is that the system is best described as a **closed-domain hyperbolic geometry learner over an imposed ultrametric structure**.

## Architecture Grounded In Source

### Core algebra and lookup layer

`TernarySpace` is the single source of truth for the state space. It precomputes lookup tables for valuation and other properties and exposes exact algebraic helpers such as `parent()` and `digit_prefix_class()` in `src/core/ternary.py:583-675`.

This matters because many tasks one might describe as “reasoning over the space” are already available **exactly** from the LUT layer without ML.

### Model

The main model is `TernaryVAEV6Controllable`:

- encoder heads output `mu` and clamped `logvar`: `src/models/vae.py:99-113`
- the encoder backbone is a small MLP over 9 input dimensions: `src/models/vae.py:144-186`
- the decoder is a small MLP outputting 27 logits, interpreted as `9 x 3`: `src/models/vae.py:189-229`
- the forward pass samples tangent latents, projects them hyperbolically, but decodes from tangent space directly: `src/models/vae.py:349-400`

The decoder choice is important. `src/models/vae.py:375-380` explicitly says the decoder consumes `z_tangent` directly rather than `log_map_zero(z_hyp)`. That stabilizes training, but it also weakens the purity of the generative story: decoder-prior sampling is a practical diagnostic, not a strongly principled generative interface.

### Hyperbolic projection modes

There are two distinct projection regimes in `src/models/hyperbolic_projection.py:111-249`:

- non-factored V6: residual tangent transform plus `expmap0`
- factored V7: split latent into `z_r` and `z_theta`, predict radius from `z_r`, normalize direction from `z_theta`, and construct `z_hyp = r * dir`

The V7 factored mode is architecturally clean for radial/directional decoupling, but it has a subtle caveat: the presets themselves acknowledge that `learnable_curvature` is effectively a no-op there because `expmap0` is not used in factored mode. See `src/presets/v7_large.yaml:35-41` and the V7 comments at `src/presets/v7.yaml:270-273`.

### Dual-objective training

Training runs two parallel latent channels:

- VAE-A receives reconstruction plus hierarchy
- VAE-B is geometry-only, with coverage intentionally disabled

That routing is visible in `src/train.py:1295-1325`.

### Loss composition

`CombinedLoss` dynamically assembles the training objective and supports uncertainty weighting, phase-gated losses, Lagrangian penalties, contrastive within-level losses, and optional angular coherence: `src/losses/combined.py:135-428` and `src/losses/combined.py:430-642`.

This is a real engineering system, not a toy single-loss script.

## What The Model Can Consume

### What it was trained to consume

The trained domain is exact balanced ternary vectors of shape `(B, 9)` with values in `{-1, 0, 1}`. That is the domain created by `TERNARY.all_ternary()` in `src/train.py:167-213`.

### What it will numerically accept

At runtime, there is no hard input validation at the model boundary. The forward pass simply casts the input to `float64`: `src/models/vae.py:362-363`.

Practical interpretation:

- trained support: exact ternary 9-vectors
- accepted tensor format: any float tensor with shape `(B, 9)`
- unsupported but not rejected: arbitrary real-valued 9-vectors

So the model can **consume** arbitrary real vectors numerically, but it is only empirically validated on the discrete ternary support.

### External inputs

There is no active ingestion path in `src/` for text, images, tabular business data, biological sequences, logs, graphs, or sensor streams.

There is one future-oriented bridge candidate in `src/c/ternary_hash.c:1-115`, which hashes arbitrary bytes into the native 9-trit space. That is potentially useful as a deterministic adapter, but it is not wired into the training or inference path in `src/`.

## What The Model Can Output

The main forward output dictionary includes:

- reconstruction logits `logits_A`, `logits_B`: `src/models/vae.py:386-399`
- tangent latents `z_A_tangent`, `z_B_tangent`
- hyperbolic embeddings `z_A_hyp`, `z_B_hyp`
- explicit radii `r_A`, `r_B` in factored mode

On the best V7-large checkpoint, the output surface is:

- input shape: `(B, 9)`
- logits shape: `(B, 27)`
- decoded symbol shape: `(B, 9)` after argmax
- hyperbolic embedding shape: `(B, 60)`
- explicit radius present: yes
- parameter count: `400,316`

Those values were regenerated with `scripts/analysis/project_audit.py`.

## Empirical Findings From Saved Checkpoints

### `results.json` leaderboard

The saved run logs rank V7-large marginally above V7 and V6, but the gains are small:

| Family | Representative run | Logged best Q |
|---|---|---:|
| V6 | `runs/v6_20260321_210840` | 2.163081 |
| V7 | `runs/v7_20260322_180254` | 2.163347 |
| V7-large | `runs/v7_large_20260324_013725` | 2.164360 |

These are run-log values, not full-domain proofs.

### Full-domain checkpoint recomputation

I reran the checkpoints directly on all 19,683 states using the current code.

| Run | Factored | Params | Accuracy | Coverage | Q | Hierarchy | Dist Corr | Curvature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v6_20260321_210840` | no | 107,002 | 0.999548 | 0.995936 | 2.141094 | 0.832050 | 0.893018 | 1.190439 |
| `v7_20260322_180254` | yes | 105,980 | 0.999605 | 0.996494 | 2.144486 | 0.832050 | 0.896410 | 1.000000 |
| `v7_large_20260324_013725` | yes | 400,316 | 0.999458 | 0.995275 | 2.145334 | 0.832050 | 0.897258 | 1.000000 |

Conclusions:

- the project **does** learn a strong closed-domain representation
- V7 and V7-large improve `dist_corr` and `Q` only slightly over V6
- hierarchy is nearly unchanged across these families
- larger size does not obviously improve reconstruction quality
- factored V7-large did not move curvature away from `1.0`, consistent with the preset comment that learnable curvature is inactive there

### Why logged metrics differ from full-domain recomputation

`compute_hierarchy_metrics()` samples up to 1,000 points for the distance-correlation component rather than using all pairwise distances: `src/train.py:570-590`. It also operates during validation rather than on the entire state space. That explains why the logged `best_Q` values are a bit higher than the full-domain recomputed values.

## Direction Geometry Review

This is the main area where skepticism is warranted.

`AngularCoherenceLoss` is explicitly designed to sharpen direction clusters by digit prefix within valuation levels: `src/losses/padic_geodesic.py:1284-1443`. The idea is technically coherent.

The empirical evidence, however, is mixed:

| Run | Level 0 ARI | Level 1 ARI | Interpretation |
|---|---:|---:|---|
| `v7_20260322_180254` | 0.712 | 0.511 | substantial low-level directional structure |
| `v7_large_20260324_013725` | 0.122 | 0.097 | weak low-level directional structure |

For levels `v >= 2`, both checkpoints are weak to near-random in the direct clustering test.

Conclusion:

- the codebase does support a real directional-structure hypothesis
- at least one saved V7 checkpoint shows nontrivial low-level directional clustering
- the stronger V7-large checkpoint by `Q` does **not** preserve that effect strongly
- therefore directional semantic structure is **not yet stable enough** to treat as a dependable commercial primitive

## Generation Review

### What generation currently means here

There is no first-class generative API in `src/`.

The practical generation test is decoder-prior sampling: draw Gaussian tangent vectors, feed them into `decoder_A`, and decode ternary symbols. That is exactly what `scripts/analysis/project_audit.py` measures.

### Empirical behavior

For the V7-large checkpoint, 5,000 decoder-prior samples produced:

- `2,452` unique indices
- `0.4904` unique-per-sample fraction
- `0.1246` coverage of the full 19,683-state domain

The samples are valid native-domain states, but generation is still bounded by the same closed support the model was trained on.

### Practical conclusion

Generation is useful for:

- probing which native states the decoder prefers
- stress-testing reconstruction manifolds
- synthetic closed-domain sampling

Generation is **not yet evidence** of:

- open-ended symbolic invention
- external-world simulation
- likelihood-calibrated forecasting
- stronger-than-classical generative performance on real data

## Commercial And Research Application Assessment

### What is realistically validated today

These are plausible and evidence-backed:

1. Closed-domain ultrametric embedding benchmark for geometry research.
2. Learned compression/indexing of the 19,683-state ternary space.
3. Research tooling for comparing hyperbolic, Euclidean, and exact symbolic organization of finite algebraic spaces.
4. A controlled environment for studying radial/directional latent factorization, hyperbolic losses, and dual-objective curriculum control.

### What is not validated today

These claims are not supported by the current repository:

1. Predictive superiority on external business, science, or market datasets.
2. Inference impossible for traditional AI or exact algorithmic systems.
3. Commercially disruptive decision intelligence.
4. Neurosymbolic reasoning over real symbolic knowledge bases.
5. SIMD/compiler/runtime breakthrough performance.

The direct reason is simple: there is no external benchmark, no serving path, no baseline comparison, and no validated adapter from real-world raw data into a semantically faithful ternary code.

### Important baseline reality check

For the tasks defined natively by this repository, exact methods already dominate many operations:

- valuation lookup is exact from `TernarySpace`
- parent and level-rank are exact from `TernarySpace`
- digit-prefix classes are exact from `TernarySpace`
- dataset enumeration is exact and complete

So if the business question is “can this beat traditional methods on the native symbolic domain?”, the answer is often **no**, because the exact LUT is already the correct and cheaper solution for many analytic queries.

The learned model only becomes strategically differentiated if it can transfer useful structure to **new data mapped into this space** or support approximate retrieval/organization tasks that exact rules alone do not solve.

## Monte Carlo Feasibility Scenarios

I added a transparent scenario model in `scripts/analysis/project_audit.py`. It is deliberately conservative and assumption-driven rather than promotional.

For the best V7-large checkpoint, using 50,000 trials:

- probability of continued value as a **closed-domain engine**: **0.8534**
- probability of value as **research tooling**: **0.6814**
- probability of a credible **external prediction pilot**: **0.00224**
- probability of current-state **commercial disruption**: **0.00004**

Interpretation:

- best case: the project becomes a strong research platform and a specialized internal engine for finite ultrametric state modeling
- middle case: it remains a technically interesting geometry system with narrow applied value
- worst case: it stays a well-engineered closed-world demo with no external transfer signal

These numbers should be treated as structured skepticism, not market forecasting.

## Main Risks

1. **No external target signal.** The repository proves self-organization over a synthetic complete state space, not prediction of real outcomes.
2. **Indexing-derived supervision.** Most hierarchy is imposed by enumeration rather than learned from intrinsic semantics: `docs/DATA-SEMANTICS.md:28-72`.
3. **Generation is weakly principled.** Decoder sampling operates from tangent Gaussian draws while the decoder is trained directly on tangent latents: `src/models/vae.py:375-380`.
4. **Directional structure is unstable across checkpoints.** V7 shows promise, V7-large does not preserve it strongly.
5. **Config drift risk existed.** `ModelConfig.factored` defaults to `True` in `src/config/schema.py:57-83`, so legacy presets needed explicit pinning.

## Changes Added In This Audit

### Reproducible audit tooling

Added `scripts/analysis/project_audit.py` plus `scripts/analysis/README.md`.

The script:

- ranks saved runs from `results.json`
- loads checkpoints safely
- reevaluates them on the full domain
- reports input/output surface and parameter count
- measures decoder-prior sampling diversity
- recomputes direction clustering
- produces a transparent Monte Carlo feasibility summary

### Regression tests

Added tests in `tests/test_scripts_project_audit.py` and extended `tests/test_config_schema.py`.

These tests now verify:

- audit helper determinism
- run sorting logic
- explicit legacy preset handling
- prevention of silent V6-to-V7 architecture drift through schema defaults

### Legacy preset hardening

I explicitly pinned non-factored legacy presets:

- `src/presets/v6.yaml:35-48`
- `src/presets/5.12.4.yaml:20-32`

This prevents schema normalization from silently mutating old checkpoints into a factored architecture that their weights do not match.

## Recommended Next Steps

1. Add one external dataset with a real target and a transparent ternary adapter.
2. Benchmark against exact symbolic baselines and Euclidean neural baselines.
3. Decide whether direction geometry is a real product asset or just an unstable research effect.
4. Build a proper inference API only after an external task is validated.
5. If external adapters matter, formalize the bridge layer instead of leaving it as an isolated C utility.

## Bottom Line

This project is already a strong **closed-domain hyperbolic geometry learner** over the full balanced-ternary state space. That is real, reproducible, and more technically solid than I initially expected.

It is **not yet** evidence of disruptive external statistical prediction or a commercially superior AI system. To become that, it needs an external task, a faithful input bridge, comparative baselines, and repeatable wins outside the native 19,683-state sandbox.
