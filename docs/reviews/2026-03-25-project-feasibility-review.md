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

## How The Main Metrics Are Actually Computed

This matters because several metric names sound broader than they really are.

### Reconstruction metrics

- `Accuracy/val` is **per-trit accuracy**, not sample-level exact match. It is computed by argmax over each of the 9 ternary positions and averaging correctness over all trits: `src/train.py:382-402`.
- `Coverage` is **perfect-sample reconstruction rate**, meaning a sample counts only if all 9 positions are correct: `src/train.py:405-426`.
- The LR controller does **not** use perfect-sample coverage. It passes `avg_val_acc` into `TrainingMetrics.coverage`, so the controller’s “coverage” signal is actually per-trit accuracy: `src/train.py:1584-1592`.

This distinction is easy to miss and matters operationally.

### Geometry metrics

- `hierarchy` is the **negated Spearman correlation** between valuation and hyperbolic radius, so positive is better: `src/train.py:563-567`.
- `dist_corr` is Spearman correlation between pairwise radius differences and pairwise valuation differences over a sampled subset: `src/train.py:570-590`.
- `Q` is `dist_corr + 1.5 * hierarchy`: `src/models/lr_controller.py:46-55`.
- `tree_coherence` is the mean parent-child geodesic distance, so **lower is better**: `src/train.py:444-500`.
- `mean_level_hierarchy` is a level-wise scatter proxy built from per-level radius standard deviation, where values closer to `-1` are better: `src/train.py:502-538` and `src/train.py:596-615`.

### Direction metrics

The live ARI metrics are computed by:

1. Running a full-domain forward pass,
2. Normalizing direction vectors from `z_hyp / r`,
3. Splitting by valuation level,
4. Clustering each level with K-means,
5. Comparing cluster labels to `digit_prefix_class()` using ARI.

That logic lives in `src/train.py:1515-1560`. This is a reasonable research diagnostic, but it is still a **proxy for internal geometric organization**, not proof of external semantic reasoning.

### Hyperbolic coverage caveat

There is also a `compute_hyperbolic_coverage()` function based on entropy over radius bins: `src/train.py:429-441`. It is accumulated during evaluation, but it is not the `Coverage` metric shown in TensorBoard, and it is not the signal fed into the LR controller in the current training path.

## Training-State Review Of The Latest V7-Large Run

The latest fully instrumented run is `runs/v7_large_20260324_013725`. Its TensorBoard logs give a better picture of what training actually achieved and what degraded.

### Stable wins

These metrics stayed strong late in training:

| Metric | Peak | Final | Last-20 Mean ± Std | Interpretation |
|---|---:|---:|---:|---|
| Val accuracy | 1.000000 @ epoch 1230 | 0.999831 @ 1499 | 0.999839 ± 0.000090 | reconstruction remained saturated |
| Perfect reconstruction coverage | 1.000000 @ epoch 1230 | 0.998476 @ 1499 | 0.998603 ± 0.000769 | almost all samples reconstructed exactly |
| Hierarchy | 0.839463 @ epoch 585 | 0.839463 @ 1499 | 0.839463 ± 0.000000 | radial ordering fully plateaued |
| Q | 2.164360 @ epoch 810 | 2.161561 @ 1499 | 2.158395 ± 0.001819 | high and stable, with mild late drift |

### Weak or decaying signals

These are the metrics that undermine strong disruption claims:

| Metric | Peak | Final | Last-20 Mean ± Std | Skeptical read |
|---|---:|---:|---:|---|
| ARI v0 | 0.195761 @ epoch 355 | 0.079427 @ 1499 | 0.078713 ± 0.006376 | direction clustering stayed weak |
| ARI v1 | 0.126743 @ epoch 190 | 0.084199 @ 1499 | 0.081462 ± 0.002829 | weak |
| Composite ARI | 0.141148 @ epoch 355 | 0.064635 @ 1499 | 0.064242 ± 0.003959 | not commercially persuasive |
| AQ | 0.028817 @ epoch 5 | ~0.0 @ 1499 | ~0.0 | direction separation mostly disappeared |

This is the strongest evidence so far that the current V7-large run is **primarily a radial hierarchy engine**, not a strong direction-semantic engine.

## Research-Grade Metric Augmentation

To get beyond repo-native scalar names, I recomputed a small set of more standard metrics on the full domain.

### V7-large: best-Q checkpoint vs final checkpoint

| Checkpoint | Accuracy | Coverage | Cross-Entropy | ECE-15 | Brier | Q | Tree Coherence | Mean Level Hierarchy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_Q.pt` | 0.999441 | 0.995021 | 0.018978 | 0.017424 | 0.002716 | 2.145318 | 0.582617 | -0.962275 |
| `final.pt` | 0.999819 | 0.998374 | 0.012349 | 0.011723 | 0.001174 | 2.144204 | 0.593941 | -0.979512 |

Interpretation:

- `best_Q.pt` is the best checkpoint only in the narrow sense of the repo’s `Q` objective.
- `final.pt` is actually **better** on reconstruction, calibration, and level consistency.
- the project needs multi-objective checkpoint selection if it wants to claim broad model quality rather than just best-`Q`.

### Cross-family comparison on battle-tested metric families

Using `best_Q.pt` for each family:

| Family | Cross-Entropy | ECE-15 | Tree Coherence | Skeptical take |
|---|---:|---:|---:|---|
| V6 | 0.058462 | 0.054593 | 3.044381 | weakest reconstruction confidence and weak parent-child locality |
| V7 | 0.019499 | 0.018223 | 2.564976 | big reconstruction/calibration gain over V6 |
| V7-large | 0.018978 | 0.017424 | 0.582617 | marginal reconstruction gain, but **major** tree-coherence gain |

This is one place where the larger model actually does something materially interesting: it dramatically improves parent-child compactness relative to V6/V7 even though its `Q` improvement is small.

### Retrieval-style diagnostics

On a 2,000-point sampled retrieval evaluation for V7-large:

| Checkpoint | Valuation NN-1 Accuracy | Same-Valuation Precision@10 | Parent Hit@10 |
|---|---:|---:|---:|
| `best_Q.pt` | 0.9995 | 0.9978 | 0.0052 |
| `final.pt` | 1.0000 | 0.9988 | 0.0157 |

Interpretation:

- valuation cohorts are extremely well clustered
- parent-child retrieval is still poor
- so the embedding separates **levels** much better than it preserves local tree adjacency

That limits any claim that the model has learned a rich executable symbolic tree, even though the radial hierarchy itself is strong.

### Generative distribution fidelity

I also computed Jensen-Shannon divergence between the generated valuation histogram and the true domain valuation histogram:

| Checkpoint | Valuation JSD |
|---|---:|
| `best_Q.pt` | 0.2034 |
| `final.pt` | 0.2313 |

That is not catastrophic, but it is far from “distribution-faithful generation.” The decoder prior is clearly biased relative to the true domain frequency profile.

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

## Benchmark Gap Against Research And Industry Standards

If the goal is to claim disruptive value outside this sandbox, the repository still lacks the benchmark layer that serious research and production review would require.

### Metrics that should gate external claims

For future external-task evaluations, I would require:

1. Predictive quality: negative log-likelihood, AUROC/PR-AUC, macro-F1, calibration error, Brier score.
2. Retrieval quality: Recall@K, Precision@K, MRR, NDCG.
3. Representation quality: k-NN probe, linear probe, trustworthiness, ablations against Euclidean baselines.
4. OOD robustness: corruption/OOD AUROC, confidence calibration under shift.
5. Generative quality: held-out NLL or likelihood proxy, support coverage, conditional fidelity, diversity metrics that match the real task.

### Dataset suites that would count as serious evidence

No results currently exist on these, but these are the kinds of benchmark families that would matter:

1. Tabular benchmark suites such as OpenML-CC18 for structured prediction.
2. Clinical or operational tabular/time-series datasets such as MIMIC-IV or eICU if the project wants high-stakes inference relevance.
3. Graph or hierarchical benchmarks such as OGB if the project wants to argue that hyperbolic structure helps on real relational data.
4. Taxonomy-heavy vision or multimodal datasets such as iNaturalist or WordNet-linked image corpora if the claim is “learned hierarchy with real semantics”.

Until the model wins on at least one external benchmark family with those metric classes, “disruption” remains a hypothesis, not a demonstrated result.

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
