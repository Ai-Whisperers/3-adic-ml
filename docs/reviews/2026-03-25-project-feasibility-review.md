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

## Downstream Probe Benchmark

The most useful next standard test was not another internal geometry scalar. It was a downstream representation probe with explicit raw-input baselines and class-imbalance-aware reporting.

Why this test matters:

1. Linear probes are a standard way to ask whether the learned representation makes a target easier to decode than the original input.
2. k-NN probes test whether local neighborhood structure improves even when linear separability does not.
3. Trustworthiness checks whether the embedding preserves local neighborhoods instead of merely collapsing them into a useful-but-distorted code.
4. Reporting raw-input baselines is mandatory here because the ternary digits may already encode the label strongly enough to make the learned representation unnecessary.

### Protocol

The new audit path in `scripts/analysis/project_audit.py` now runs `representation_probe_suite(...)` directly from the checkpointed latent states.

- Probe target: valuation level derived from the state index.
- Included levels: `0..6`.
- Excluded levels: `7..9`, because their support is too sparse for a defensible stratified train/test split.
- Sample size: `2,034` states.
- Class counts: `{0: 1334, 1: 445, 2: 148, 3: 49, 4: 20, 5: 20, 6: 18}`.
- Evaluation seed: `42`, fixed in the audit path so repeated checkpoint evaluations stay reproducible even if the model forward pass samples latent noise.
- Metrics: accuracy, balanced accuracy, macro-F1, and trustworthiness@15.

This is still an internal-label benchmark, not an external task benchmark. That limitation matters: passing this test does not prove commercial predictive value. It only tells us whether the learned latent space improves access to a label the repository already defines exactly.

### Results

For `runs/v7_large_20260324_013725`:

| Checkpoint | Probe | Accuracy | Balanced Acc. | Macro-F1 | Skeptical interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `best_Q.pt` | Linear on embedding | `0.8845` | `0.4286` | `0.3172` | Looks decent by raw accuracy, but class imbalance exposes weak minority-level decoding. |
| `best_Q.pt` | Linear on raw ternary input | `1.0000` | `1.0000` | `1.0000` | The original digits already make valuation linearly trivial. |
| `best_Q.pt` | 15-NN on embedding | `0.9951` | `0.9905` | `0.9821` | Very strong local organization for valuation neighborhoods. |
| `best_Q.pt` | 15-NN on raw ternary input | `0.9165` | `0.6666` | `0.6674` | Raw space is materially worse as a neighborhood retrieval space. |
| `best_Q.pt` | Trustworthiness@15 | `0.6365` | - | - | Moderate neighborhood preservation, not elite manifold quality. |
| `final.pt` | Linear on embedding | `0.7396` | `0.4286` | `0.2476` | Late training still leaves minority-level linear decoding weak despite higher aggregate accuracy. |
| `final.pt` | Linear on raw ternary input | `1.0000` | `1.0000` | `1.0000` | No learned advantage over the native digits on this target. |
| `final.pt` | 15-NN on embedding | `1.0000` | `1.0000` | `1.0000` | Neighborhood ordering remains extremely strong for this internal label. |
| `final.pt` | 15-NN on raw ternary input | `0.9165` | `0.6666` | `0.6674` | Same weak baseline as above. |
| `final.pt` | Trustworthiness@15 | `0.6152` | - | - | Slightly worse than `best_Q.pt`; still moderate. |

### What this benchmark actually proves

This is the strongest new skeptical result from the current repository state.

What survived scrutiny:

1. The latent space is genuinely useful as a valuation-aware neighborhood index.
2. On a k-NN probe, the learned embedding is much better than the raw ternary coordinates.
3. That suggests a real retrieval or nearest-neighbor organization benefit inside the native finite state space.

What did **not** survive scrutiny:

1. The latent space is **not** a better linear feature space than the raw input for valuation prediction.
2. The high raw-input baseline means the project cannot honestly market this result as “the model discovered a hidden predictive factor unavailable to simple methods.”
3. Trustworthiness around `0.62-0.63` is useful but not strong enough to claim exceptionally faithful manifold preservation.
4. Because the label is internally derived, this still does not count as evidence for external forecasting, industrial prediction, or neurosymbolic reasoning superiority.

### Most defensible value statement after the probe benchmark

The disruptive angle, if one emerges, is narrower than broad predictive AI claims:

1. The model may become valuable as a hyperbolic retrieval/indexing layer over algebraically structured discrete state spaces.
2. It may support compression or neighborhood search where raw-coordinate nearest-neighbor structure is poor.
3. It is **not yet** validated as a superior general predictor, feature extractor, or commercially deployable inference engine on real datasets.

That is a real and useful result, but it is not yet the sort of asymmetrical evidence that would justify aggressive external-performance claims.

## Hyperbolic Vs Euclidean Baseline Ablation

The next wise benchmark was the one the previous section still lacked: an actual Euclidean baseline ablation on the same checkpointed latent sample.

This matters because the architecture does **not** use the hyperbolic code for decoding. In `src/models/vae.py`, the model samples `z_A_tangent`, projects it to `z_A_hyp` for the geometry losses, but still feeds `z_A_tangent` directly into `decoder_A`. So if Euclidean tangent features outperform hyperbolic features on reconstruction-adjacent linear probes, that is not surprising. The question is whether the hyperbolic projection adds a different kind of value anyway.

### Protocol

The audit now compares three representations on the same split:

1. Raw ternary input.
2. Euclidean tangent latent `z_A_tangent`.
3. Hyperbolic latent `z_A_hyp`.

For each checkpoint, I measured:

1. Linear probe on valuation levels `0..6`.
2. 15-NN probe on valuation levels `0..6`.
3. Trustworthiness@15 relative to raw-input neighborhoods.
4. Retrieval metrics on a 2,000-state sample using the correct metric for each space:
   - Poincaré distance for `z_A_hyp`
   - Euclidean distance for `z_A_tangent`

### Results

For `runs/v7_large_20260324_013725`:

| Checkpoint | Representation | Linear Acc. | Linear Bal. Acc. | Linear Macro-F1 | 15-NN Acc. | 15-NN Bal. Acc. | 15-NN Macro-F1 | Trustworthiness@15 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_Q.pt` | Hyperbolic | `0.8845` | `0.4286` | `0.3172` | `0.9951` | `0.9905` | `0.9821` | `0.6365` |
| `best_Q.pt` | Tangent Euclidean | `0.9361` | `0.6324` | `0.6581` | `0.6757` | `0.2017` | `0.2067` | `0.9648` |
| `best_Q.pt` | Raw input | `1.0000` | `1.0000` | `1.0000` | `0.9165` | `0.6666` | `0.6674` | - |
| `final.pt` | Hyperbolic | `0.7396` | `0.4286` | `0.2476` | `1.0000` | `1.0000` | `1.0000` | `0.6152` |
| `final.pt` | Tangent Euclidean | `0.9558` | `0.7110` | `0.7514` | `0.6560` | `0.1429` | `0.1134` | `0.9692` |
| `final.pt` | Raw input | `1.0000` | `1.0000` | `1.0000` | `0.9165` | `0.6666` | `0.6674` | - |

Retrieval on a 2,000-state sample:

| Checkpoint | Representation | Valuation NN@1 | Same-Valuation P@10 | Parent-Hit@10 |
| --- | --- | ---: | ---: | ---: |
| `best_Q.pt` | Hyperbolic | `1.0000` | `0.99835` | `0.00524` |
| `best_Q.pt` | Tangent Euclidean | `0.9980` | `0.98855` | `0.0000` |
| `final.pt` | Hyperbolic | `1.0000` | `0.99880` | `0.01571` |
| `final.pt` | Tangent Euclidean | `0.9920` | `0.98180` | `0.0000` |

### What this ablation proves

This ablation is more valuable than another high-level slogan because it isolates what the hyperbolic projection is actually doing.

What the Euclidean baseline wins:

1. Tangent Euclidean features are better for linear valuation decoding than hyperbolic features.
2. Tangent Euclidean features preserve raw-input neighborhoods much more faithfully (`trustworthiness ≈ 0.965-0.969` vs `0.615-0.637`).
3. So the hyperbolic projection is **not** a free lunch and should not be described as a universal representation improvement.

What the hyperbolic projection wins:

1. Hyperbolic features are dramatically better for valuation-aware k-NN organization than tangent Euclidean features.
2. Hyperbolic retrieval is better on NN@1, same-valuation precision, and parent-hit@10.
3. The Euclidean baseline never retrieved a parent in the top 10 on this sample, while the hyperbolic space did.

### Skeptical interpretation

The honest interpretation is:

1. The tangent latent remains the more decoder-friendly and more locally faithful Euclidean representation.
2. The hyperbolic projection is a task-specific geometric transform that sacrifices local faithfulness to create a stronger hierarchy-aware retrieval metric.
3. That is a real differentiator, but it is still an internal-domain differentiator, not an external predictive breakthrough.

This is probably the strongest argument currently available for the project’s future commercial direction: not “better generic AI,” but “better metric-space organization for hierarchical discrete state retrieval.” If that idea is going to survive outside the sandbox, it now needs an external dataset where hierarchical retrieval or approximate lookup matters operationally.

## Symbolic Engine And Hidden-Capability Tests

The next question was whether there is any hidden, more general-purpose capability in the current model that is **not** visible from valuation-only benchmarks.

The most defensible way to test that, without inventing fake external evidence, was to add a small symbolic engine that generates exact transformation orbits and feedback pairs over the current domain.

### Scope of the symbolic engine

The new module is intentionally narrow:

1. It is a finite transformation-group engine over `{-1,0,1}^9`.
2. Its generators are cyclic rotation, reflection, and global sign flip.
3. Together they define a closed 36-element group action over the current ternary word space.

This is **not** a full implementation of a 3-adic field, ring extension, or homeomorphic algebraic closure. Claiming that from the current codebase would be false. What it does provide is something much more useful right now: exact, compositional symbolic programs, orbit canonicalization, and positive/negative feedback pairs that can be used to audit or later train a neurosymbolic loop.

### What hidden capability is actually testable now

Given the current architecture, the only realistic hidden capabilities worth testing are:

1. Whether the learned space supports transformation-orbit retrieval better than raw digits.
2. Whether the learned space can verify exact symbolic equivalence classes better than raw digits.
3. Whether the geometry captures algebraic invariances that were **not** explicit in the current training losses.

High-dimensional external prediction is still **not** directly testable here, because the model still consumes fixed 9-trit inputs. There is no validated high-dimensional bridge or adaptive front-end in the repository yet.

### Standard tests added

Using the symbolic engine, I added two standard metric families:

1. **Orbit retrieval benchmark**:
   - task: query with a non-identity symbolic transform and retrieve the correct orbit representative
   - metrics: Recall@1, Recall@10, MRR
2. **Pair verification benchmark**:
   - task: distinguish exact same-orbit positive pairs from valuation-matched different-orbit negatives
   - metrics: ROC-AUC, Average Precision

Both are standard metric-learning / retrieval diagnostics and are much harder to game than another internal scalar.

### Results

For `runs/v7_large_20260324_013725`:

Orbit retrieval (`512` sampled orbits, group size `36`):

| Checkpoint | Representation | Recall@1 | Recall@10 | MRR |
| --- | --- | ---: | ---: | ---: |
| `best_Q.pt` | Hyperbolic | `0.0020` | `0.0293` | `0.0168` |
| `best_Q.pt` | Tangent Euclidean | `0.0039` | `0.0430` | `0.0216` |
| `best_Q.pt` | Raw input | `0.0059` | `0.0469` | `0.0246` |
| `final.pt` | Hyperbolic | `0.0059` | `0.0273` | `0.0206` |
| `final.pt` | Tangent Euclidean | `0.0039` | `0.0449` | `0.0206` |
| `final.pt` | Raw input | `0.0059` | `0.0469` | `0.0246` |

Pair verification (`2,048` positive + `2,048` negative pairs):

| Checkpoint | Representation | ROC-AUC | Average Precision |
| --- | --- | ---: | ---: |
| `best_Q.pt` | Hyperbolic | `0.2467` | `0.3753` |
| `best_Q.pt` | Tangent Euclidean | `0.2951` | `0.4111` |
| `best_Q.pt` | Raw input | `0.4101` | `0.4860` |
| `final.pt` | Hyperbolic | `0.2360` | `0.3727` |
| `final.pt` | Tangent Euclidean | `0.2999` | `0.4145` |
| `final.pt` | Raw input | `0.4101` | `0.4860` |

### What these results mean

These new tests are valuable precisely because the answer is mostly negative.

What the results rule out:

1. The current learned spaces are **not** secretly strong transformation-invariant symbolic feature extractors.
2. The current hyperbolic embedding does **not** outperform simple raw digits on these symbolic-equivalence tasks.
3. There is no evidence yet that the model has uncovered a latent algebraic closure structure beyond the valuation hierarchy it was trained to optimize.

What the results still justify:

1. The symbolic engine is a useful exact benchmark and future feedback-loop generator.
2. It gives the project a rigorous neurosymbolic interface for future losses, self-supervised pairs, or orbit-consistency objectives.
3. It clarifies the present boundary: the model is currently a strong hierarchy-aware retrieval learner, not a general symbolic reasoner.

### Most honest systemic conclusion

If this project is going to evolve into something commercially broader, the path is now clearer:

1. Keep the hyperbolic retrieval strength.
2. Add a real symbolic supervision channel using exact group-action or orbit-consistency losses.
3. Only after that should the project attempt claims about general symbolic reasoning or high-dimensional external prediction.

Right now, the symbolic-engine benchmarks are best interpreted as a map of what is **missing**, not as proof that the broader neurosymbolic goal has already been achieved.

One concrete engineering implication from the codebase matters here: the decoder still consumes `z_A_tangent`, while the hierarchy losses shape `z_A_hyp`. So a future unsupervised neurosymbolic path should probably stay split:

1. Tangent-space objectives for local algebraic consistency, reconstruction-faithful symbolic invariance, and Euclidean verification tasks.
2. Hyperbolic objectives for orbit retrieval, coarse hierarchy organization, and 3-adic tree structure.
3. A thin optional symbolic subsystem around exact radix-3 / 3-adic transforms, rather than hard-wiring symbolic logic directly into the baseline training path.

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

## Codebase Continuation Plan (March 26, 2026)

I rechecked the live repository state against the harsher codebase critique and the result is: the critique is directionally right, but a few specifics need updating.

### What is still worth continuing

These parts still look like the right foundation to preserve:

1. `src/core/ternary.py` as the canonical radix-3 / 3-adic kernel.
2. The encoder / projection / decoder split in `src/models/vae.py`.
3. The experimental audit surface in `scripts/analysis/project_audit.py`.
4. The explicit contract style in `src/config/statenet_config.py`.

Those are the pieces that still justify continuing the project.

### Verified remaining issues

These issues are still real in the current tree:

1. **`src/train.py` is still a god file.**
   It still mixes setup, training loop, validation, metric computation, hierarchy diagnostics, checkpointing, CLI handling, and controller logic in one file. Even where the code is technically correct, the file layout raises maintenance risk.
2. **Legal / provenance drift is real.**
   The root repo advertises MIT, but source headers in `src/train.py`, `src/losses/combined.py`, `src/core/ternary.py`, `src/models/vae.py`, `src/models/hyperbolic_projection.py`, and `src/config/statenet_config.py` still reference PolyForm Noncommercial and commercial-licensing language.
3. **README drift is real.**
   `README.md:10-18` still overstates what is proven, and `README.md:18` still labels “Coverage” as per-digit reconstruction accuracy even though the audit and the code distinguish per-digit accuracy from perfect-reconstruction coverage.
4. **`v7_large.yaml` is still partly a lab notebook.**
   `src/presets/v7_large.yaml:1-15` and the loss section still contain hypotheses, expectations, root-cause narratives, and historical notes mixed into the executable preset.
5. **The “single source of truth” story is still weakened by duplicated constants.**
   `src/core/ternary.py` presents itself as canonical, but `src/config/constants.py:2` still hardcodes `N_TERNARY_OPERATIONS = 19683`.
6. **`CombinedLoss` is still strategically overloaded.**
   It is useful, but it remains the convergence point for too many configuration branches and behavior switches. That is acceptable for research code today, but it is the wrong place to keep adding complexity.

### One earlier issue that does NOT reproduce anymore

As of **March 26, 2026**, I do **not** reproduce the exact duplicate immediate overwrite of `level_pfx` that appeared in the earlier critique. In the current working tree I only see one active `level_pfx` assignment in `src/train.py:1540`. So that specific residue appears to have been removed already, even though `train.py` remains overgrown overall.

### Highest-value refactor sequence

If the goal is to keep only what is worth continuing, the next sequence should be subtractive and credibility-oriented:

1. **Legal and provenance cleanup first.**
   Before more experimentation, align the root LICENSE, file headers, and any stale commercial-license notices. This is external-credibility debt, not cosmetic debt.
   Deeper detail from the live tree: as of March 26, 2026, I count **18 Python files under `src/`** still carrying stale PolyForm/commercial-era headers:
   - `src/config/__init__.py`
   - `src/config/schema.py`
   - `src/config/statenet_config.py`
   - `src/core/ternary.py`
   - `src/geometry/poincare.py`
   - `src/losses/base.py`
   - `src/losses/combined.py`
   - `src/losses/hyperbolic_kl.py`
   - `src/losses/lagrangian.py`
   - `src/losses/padic_geodesic.py`
   - `src/models/hyperbolic_projection.py`
   - `src/models/lr_controller.py`
   - `src/models/vae.py`
   - `src/train.py`
   - `src/utils/checkpoint.py`
   - `src/utils/hardware_monitor.py`
   - `src/utils/tensorboard_logger.py`
   - `src/utils/visualization.py`
   Two of those (`src/core/ternary.py`, `src/models/hyperbolic_projection.py`) still also carry direct “commercial licensing inquiries” language. The safest implementation order is:
   1. normalize `src/` headers only
   2. add a regression test that forbids stale PolyForm/commercial strings in `src/`
   3. clean `tests/` and generated docs in a separate pass
2. **Split `src/train.py` without changing behavior.**
   First move code, not logic. The best near-term split is:
   - `src/training/metrics.py`
   - `src/training/validation.py`
   - `src/training/checkpoints.py`
   - `src/training/loop.py`
   - `src/training/cli.py`
   The requirement is strict behavior preservation, not redesign.
3. **Freeze new losses and giant preset growth until the split lands.**
   Right now the expected value of a new loss is lower than the expected value of reducing orchestration entropy.
4. **Rewrite `README.md` to match the audit, not the aspiration.**
   Claims should be bounded by what the current audit actually verifies: closed-domain hierarchy learning, not general symbolic intelligence or external prediction.
5. **Convert research journaling in YAML into docs.**
   Keep presets executable and minimal. Move hypotheses, VRAM notes, expected ARI ceilings, and root-cause narratives into `docs/experiments/` or `docs/audits/`.

### What should be frozen immediately

To keep the project on a productive path, I would freeze these categories for now:

1. New loss classes.
2. New large presets with embedded experiment diaries.
3. README-level performance claims stronger than the audit.
4. More direction-geometry tuning unless it is attached to a real downstream benchmark.

### What the codebase itself suggests for future neurosymbolic work

Reading the actual architecture suggests a narrower but more defensible path:

1. Keep `src/core/ternary.py` as the exact symbolic substrate.
2. Keep tangent-space and hyperbolic-space objectives split, because the decoder still consumes tangent latents while the hierarchy objectives act on hyperbolic latents.
3. Keep symbolic machinery optional and isolated until there is evidence that orbit-consistency or other exact symbolic objectives improve real downstream behavior.
4. Treat the current symbolic engine as a benchmark generator and future supervision source, not yet as proof of neurosymbolic capability.

That is the part of the project that is still worth continuing.

## Bottom Line

This project is already a strong **closed-domain hyperbolic geometry learner** over the full balanced-ternary state space. That is real, reproducible, and more technically solid than I initially expected.

It is **not yet** evidence of disruptive external statistical prediction or a commercially superior AI system. To become that, it needs an external task, a faithful input bridge, comparative baselines, and repeatable wins outside the native 19,683-state sandbox.
