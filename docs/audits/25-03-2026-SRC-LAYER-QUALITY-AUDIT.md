# `src/` Layered Quality Audit

Date: 2026-03-25  
Scope: `/d1/VAEs/3-adic-ml/src/`  
Method: architectural review of the 5 practical abstraction layers already present in the codebase

## Scoring Rubric

- `90-100`: excellent, strong local design and strong integration
- `80-89`: good, robust overall but with notable maintainability or integration debt
- `70-79`: workable, but complexity or coupling is actively slowing development
- `<70`: needs structural attention

## Executive Summary

| Layer | Score | Short Verdict |
|------|------:|---------------|
| 1. Configuration / Contract | 88 | Strong after the schema hardening pass; still has dual-config-system drift risk |
| 2. Mathematical Primitives | 91 | Best layer in the repo: cohesive, well-factored, low ambiguity |
| 3. Model Components | 85 | Solid model/control split, but some files are getting too broad |
| 4. Objectives / Optimization Policy | 82 | Scientifically rich and fairly disciplined, but too much logic is concentrated in large files |
| 5. Runtime / Instrumentation | 78 | Functional and feature-rich, but `train.py` is a large orchestration monolith |

**Overall architectural quality: 85/100**

This is a good codebase with real engineering intent, not a throwaway experiment.  
The strongest parts are the algebra/geometry foundation and the recent config validation work.  
The weakest part is the runtime layer, mainly because too much end-to-end responsibility lives in one training script.

---

## 1. Configuration / Contract Layer

**Score: 88/100**

### Main anchor objects

- `src/config/schema.py:28` — `StrictConfigModel`
- `src/config/schema.py:57` — `ModelConfig`
- `src/config/schema.py:178` — `StateNetConfigSchema`
- `src/config/schema.py:515` — `VisualizationConfig`
- `src/config/schema.py:555` — `TrainingConfigSchema`
- `src/config/schema.py:592` — `normalize_config`
- `src/config/statenet_config.py:100` — `StateNetConfig`
- `src/config/statenet_config.py:135` — `StateNetConfig.from_dict`

### Why this layer scores well

- The schema is now explicit and strict by default via `StrictConfigModel` at `src/config/schema.py:28`.
- The top-level contract is centralized in `TrainingConfigSchema` at `src/config/schema.py:555`, which is the right place for cross-section validation.
- `normalize_config` at `src/config/schema.py:592` is a strong addition because it turns validation into a runtime boundary instead of a passive check.
- `StateNetConfig.from_dict` at `src/config/statenet_config.py:135` rejects unknown keys, which is good defensive behavior for handwritten YAML.

### Why it is not higher

- There are still **two config systems**: Pydantic in `schema.py` and dataclass-driven `StateNetConfig` in `statenet_config.py`. They are coherent enough right now, but this is still a drift vector.
- `src/config/schema.py` is already about `600` lines. That is acceptable, but it is approaching the point where sub-schema extraction would improve readability.
- Some compatibility aliases and legacy preset support are necessary, but they also make the contract more complex than a greenfield schema.

### Net assessment

This layer is in good shape and is substantially stronger than it would have been before the recent normalization fix.  
If the repo had one unified config model instead of one-and-a-half, this could reach the low 90s.

---

## 2. Mathematical Primitive Layer

**Score: 91/100**

### Main anchor objects

- `src/core/ternary.py:54` — `TernarySpace`
- `src/core/ternary.py:153` — `TernarySpace._build_properties_lut`
- `src/core/ternary.py:238` — `TernarySpace.valuation`
- `src/core/ternary.py:754` — module-level `valuation`
- `src/geometry/poincare.py:51` — `get_manifold`
- `src/geometry/poincare.py:92` — `poincare_distance`
- `src/geometry/poincare.py:110` — `hyperbolic_radius`
- `src/geometry/poincare.py:175` — `log_map_zero`
- `src/geometry/poincare.py:335` — `get_riemannian_optimizer`
- `src/geometry/poincare.py:360` — `poincare_distance_matrix`

### Why this layer scores well

- `TernarySpace` at `src/core/ternary.py:54` is a real single-source-of-truth object, not just a bag of utilities.
- The precomputed LUT design in `src/core/ternary.py:153` is pragmatic and appropriate for a fixed-size 19,683-element space.
- `TernarySpace.valuation` at `src/core/ternary.py:238` offering `strict=True` is a good example of operational ergonomics.
- `poincare.py` is cohesive: it owns manifold creation, distance, maps, optimizer factory, and pairwise matrix generation without leaking responsibilities outward.
- The geometry layer has strong naming and the functions are easy to compose.

### Why it is not higher

- `TernarySpace` does a lot in one singleton; that is a good tradeoff here, but it still concentrates multiple concerns.
- The manifold cache is documented as not thread-safe. That is not a current blocker, but it is a ceiling on how “production-grade” this layer is.
- A few routines remain intentionally precomputed/loop-based for clarity, not for absolute elegance.

### Net assessment

This is the cleanest layer in the repository.  
It has the strongest “small API, clear semantics, low ambiguity” profile.

---

## 3. Model Component Layer

**Score: 85/100**

### Main anchor objects

- `src/models/hyperbolic_projection.py:34` — `HyperbolicProjection`
- `src/models/hyperbolic_projection.py:284` — `DualHyperbolicProjection`
- `src/models/vae.py:60` — `EncoderHead`
- `src/models/vae.py:237` — `TernaryVAEV6`
- `src/models/vae.py:349` — `TernaryVAEV6.forward`
- `src/models/vae.py:412` — `TernaryVAEV6Controllable`
- `src/models/vae.py:474` — `TernaryVAEV6Controllable.apply_statenet_state`
- `src/models/vae.py:497` — `TernaryVAEV6Controllable.get_param_groups`
- `src/models/lr_controller.py:58` — `TrainingMetrics`
- `src/models/lr_controller.py:101` — `LRController`
- `src/models/lr_controller.py:130` — `MetricBasedLR`

### Why this layer scores well

- `HyperbolicProjection` at `src/models/hyperbolic_projection.py:34` validates its invariants early and clearly.
- The V6/V7 bridge is handled with discipline: factored and non-factored paths live in the same conceptual module but are still understandable.
- `TernaryVAEV6.forward` at `src/models/vae.py:349` returns a rich structured dict instead of a positional tuple mess, which helps downstream integration.
- `TernaryVAEV6Controllable` at `src/models/vae.py:412` cleanly exposes trainability control without leaking optimizer details into the core VAE path.
- `TrainingMetrics` at `src/models/lr_controller.py:58` and `MetricBasedLR` at `src/models/lr_controller.py:130` show good defensive validation and explicit controller state.

### Why it is not higher

- `src/models/vae.py` is `563` lines. That is not catastrophic, but it is drifting from “model definition” toward “model system”.
- `src/models/lr_controller.py` mixes good control logic with some documentation drift and a fairly broad scope for one file.
- There is still some conceptual coupling between model return structure, controller expectations, and training-loop loss wiring.

### Net assessment

This layer is well above average and clearly engineered.  
It loses points mostly on breadth and coupling pressure, not on correctness or clarity of intent.

---

## 4. Objective / Optimization Policy Layer

**Score: 82/100**

### Main anchor objects

- `src/losses/base.py:43` — `HierarchyLossBase`
- `src/losses/base.py:121` — `RichHierarchyLossBase`
- `src/losses/hyperbolic_kl.py:33` — `HyperbolicKLDivergence`
- `src/losses/lagrangian.py:49` — `LagrangianDualState`
- `src/losses/combined.py:57` — `CombinedLoss`
- `src/losses/combined.py:134` — `CombinedLoss._init_losses`
- `src/losses/combined.py:429` — `CombinedLoss.forward`
- `src/losses/padic_geodesic.py:89` — `PAdicGeodesicLoss`
- `src/losses/padic_geodesic.py:257` — `RadialHierarchyLoss`
- `src/losses/padic_geodesic.py:441` — `GlobalRankLoss`
- `src/losses/padic_geodesic.py:644` — `MonotonicRadialLoss`
- `src/losses/padic_geodesic.py:840` — `RichHierarchyLoss`
- `src/losses/padic_geodesic.py:1056` — `ValuationPriorLoss`
- `src/losses/padic_geodesic.py:1284` — `AngularCoherenceLoss`

### Why this layer scores well

- `HierarchyLossBase` and `RichHierarchyLossBase` in `src/losses/base.py` are good contracts and clearly separate scalar-loss vs component-loss behavior.
- `HyperbolicKLDivergence` at `src/losses/hyperbolic_kl.py:33` is self-contained and comparatively clean.
- `CombinedLoss` at `src/losses/combined.py:57` gives the repo one place to translate config into actual optimization behavior, which is the right abstraction.
- The loss layer is scientifically expressive: it captures multiple independent structural pressures without collapsing them into one opaque term.

### Why it is not higher

- `src/losses/padic_geodesic.py` is `1455` lines and contains too many distinct objectives in one file.
- `src/losses/combined.py` is `803` lines; `_init_losses` at `src/losses/combined.py:134` and `forward` at `src/losses/combined.py:429` are doing a lot of orchestration.
- This layer has the highest “blast radius per edit” risk after `train.py`. Small changes can ripple across config, metrics, and optimizer behavior.
- Some policy decisions are still embedded in code branches rather than isolated into smaller composable builders.

### Net assessment

This is a strong research-grade layer, but not yet a low-friction maintenance layer.  
The issue is not that the abstractions are wrong; it is that too many good abstractions live inside too few large files.

---

## 5. Runtime / Instrumentation Layer

**Score: 78/100**

### Main anchor objects

- `src/utils/checkpoint.py:18` — `load_checkpoint_compat`
- `src/utils/checkpoint.py:40` — `get_model_state_dict`
- `src/utils/tensorboard_logger.py:38` — `TensorBoardLogger`
- `src/utils/visualization.py:85` — `VisualizationRuntimeConfig`
- `src/utils/visualization.py:620` — `VisualizationPipeline`
- `src/utils/visualization.py:672` — `VisualizationPipeline.run`
- `src/utils/visualization.py:735` — `VisualizationPipeline._validate_inputs`
- `src/train.py:142` — `DataAuditor`
- `src/train.py:221` — `ModelAuditor`
- `src/train.py:743` — `_build_checkpoint_payload`
- `src/train.py:807` — `train`
- `src/train.py:1972` — `main`

### Why this layer scores well

- The runtime layer has real operational features: data auditing, model auditing, checkpointing, visualization, and TensorBoard support.
- `VisualizationRuntimeConfig` at `src/utils/visualization.py:85` and `VisualizationPipeline` at `src/utils/visualization.py:620` are good examples of runtime abstraction done correctly.
- `_build_checkpoint_payload` at `src/train.py:743` is a worthwhile refactor that reduces duplication in a high-risk area.
- `DataAuditor` and `ModelAuditor` in `src/train.py` show that the repo values pre-flight validation, not just “launch and hope”.

### Why it is not higher

- `src/train.py` is `2133` lines and combines CLI parsing, config loading, runtime validation, data prep, optimizer/scheduler setup, metrics, visualization, checkpointing, resume logic, and final reporting.
- `load_checkpoint_compat` at `src/utils/checkpoint.py:18` uses `weights_only=False` unconditionally. That may be pragmatically necessary, but it is a trust-boundary weakness.
- `TensorBoardLogger` at `src/utils/tensorboard_logger.py:38` is intentionally minimal, but the runtime layer overall still leans heavily on `print()`-driven observability.
- This is the layer where understanding one bug is most likely to require reading many unrelated concerns.

### Net assessment

This layer works, but it is the least decomposed part of the system.  
It is the main reason the repo feels more like a powerful research platform than a low-ceremony library.

---

## Highest-Value Improvements

If only three structural improvements were made next, these would give the best return:

1. Split `src/train.py` into `config/bootstrap`, `training_loop`, `evaluation_metrics`, and `runtime/reporting`.
2. Split `src/losses/padic_geodesic.py` into smaller objective-focused modules without changing public exports.
3. Unify the remaining dual-config surface so `schema.py` becomes the single canonical contract and `statenet_config.py` becomes either a thin adapter or disappears.

## Final Verdict

The codebase is **architecturally good** and **scientifically serious**.  
Its main weakness is not bad code; it is **concentration of good code into a few oversized orchestration files**.

If judged as a research codebase: **very good**.  
If judged as a long-lived platform that multiple engineers need to debug quickly: **good, but ready for the next refactor step**.

---

## Presets Appendix

### Chosen preset: `src/presets/v7_large.yaml`

### Why this preset was selected

`v7_large.yaml` is the best candidate under both interpretations of "last and best":

1. **Latest preset file in `src/presets/`**  
   By file modification time, `src/presets/v7_large.yaml` is newer than `v6.yaml`, `v7.yaml`, and `5.12.4.yaml`.

2. **Best observed run outcome among preset-backed runs in `runs/`**  
   The highest `best_Q` found in the local run artifacts is:

   - `runs/v7_large_20260324_013725/results.json`
   - `best_Q = 2.1643595617395928`

This is slightly above the best observed `v6` and `v7` runs, so `v7_large.yaml` is the correct choice as the current **latest and best-performing preset** in the workspace.

### Preset anchors

- `src/presets/v7_large.yaml:1` — preset identity and hypothesis
- `src/presets/v7_large.yaml:31` — model block
- `src/presets/v7_large.yaml:87` — `statenet`
- `src/presets/v7_large.yaml:116` — `loss`
- `src/presets/v7_large.yaml:168` — `angular_coherence`
- `src/presets/v7_large.yaml:202` — `training`
- `src/presets/v7_large.yaml:228` — `logging`
- `src/presets/v7_large.yaml:244` — `checkpoints`
- `src/presets/v7_large.yaml:253` — `targets`

### Comparison to prior presets

Relative to `src/presets/v7.yaml`, `v7_large.yaml` makes the following most meaningful changes:

- `latent_dim: 32 -> 64` at `src/presets/v7_large.yaml:33`
- `hidden_dim: 64 -> 128` at `src/presets/v7_large.yaml:34`
- `max_radius: 0.95 -> 0.99` at `src/presets/v7_large.yaml:35`
- stronger radial tightening via `variance_weight: 2.0` at `src/presets/v7_large.yaml:122`
- geodesic target mode changed to `use_individual_valuation: true` at `src/presets/v7_large.yaml:139`
- more aggressive angular coherence schedule at `src/presets/v7_large.yaml:168-183`

Relative to `src/presets/v6.yaml`, it also codifies the V7 factored-latent shift:

- `factored: true` at `src/presets/v7_large.yaml:40`
- `valuation_prior.enabled: false` at `src/presets/v7_large.yaml:194-197`
- `hyperbolic_kl.variance_only: false` at `src/presets/v7_large.yaml:159`

### How `v7_large.yaml` interacts with the 5 abstraction layers

#### 1. Configuration / Contract Layer

**Interaction quality: strong**

The preset is a good stress test for the config layer because it uses most of the modern surface:

- model contract at `src/presets/v7_large.yaml:31-50`
- controller contract at `src/presets/v7_large.yaml:87-111`
- multi-loss contract at `src/presets/v7_large.yaml:116-197`
- runtime/reporting contract at `src/presets/v7_large.yaml:202-259`

Why this matters:

- It exercises the schema breadth that now lives in `src/config/schema.py:555`.
- It uses fields that previously would have been easy to silently ignore, such as `targets`, `checkpoints`, and `logging`.
- It is aligned with the stricter validation layer added through `normalize_config` at `src/config/schema.py:592`.

Main caveat:

- It still coexists with the separate `StateNetConfig` dataclass path in `src/config/statenet_config.py:100`, so the preset benefits from the improved schema layer, but also exposes the remaining dual-config design.

#### 2. Mathematical Primitive Layer

**Interaction quality: very strong**

This preset is tightly coupled to the primitive layer in a coherent way:

- `curvature: 1.0` at `src/presets/v7_large.yaml:36`
- `precision.dtype: float64` at `src/presets/v7_large.yaml:63-64`
- fixed full dataset shape at `src/presets/v7_large.yaml:221-223`

Why this is good:

- It respects the assumptions of `TernarySpace` in `src/core/ternary.py:54`.
- It respects the numerical expectations of `poincare.py`, especially `get_manifold` at `src/geometry/poincare.py:51`, `log_map_zero` at `src/geometry/poincare.py:175`, and `poincare_distance_matrix` at `src/geometry/poincare.py:360`.
- It does not ask the primitive layer to support contradictory numerical modes.

Main caveat:

- The preset comment at `src/presets/v7_large.yaml:37` explicitly notes a semantic mismatch: `learnable_curvature: true` is effectively a no-op in factored mode. That is honest and useful, but it also shows the config surface is slightly broader than the actual active mathematical path.

#### 3. Model Component Layer

**Interaction quality: excellent**

This preset is most naturally aligned with the model layer:

- `latent_dim: 64`, `hidden_dim: 128`, `factored: true`, `radial_dims: 4` at `src/presets/v7_large.yaml:33-41`
- projection settings at `src/presets/v7_large.yaml:43-47`
- StateNet-compatible initial trainability at `src/presets/v7_large.yaml:90-93`
- differential LR scales at `src/presets/v7_large.yaml:69-73`

Why this is good:

- It is clearly designed for `HyperbolicProjection` in factored mode at `src/models/hyperbolic_projection.py:34`.
- It matches the V7 model path in `TernaryVAEV6.forward` at `src/models/vae.py:349`.
- It is compatible with `TernaryVAEV6Controllable` at `src/models/vae.py:412` and `MetricBasedLR` at `src/models/lr_controller.py:130`.

Main caveat:

- This preset also exposes how much system behavior is encoded in comments and conventions rather than first-class model presets or builders. It works, but understanding why it works still requires reading several model files.

#### 4. Objective / Optimization Policy Layer

**Interaction quality: strong but dense**

This is the layer most heavily exercised by `v7_large.yaml`:

- Rich hierarchy tuning at `src/presets/v7_large.yaml:117-124`
- Geodesic target-mode shift at `src/presets/v7_large.yaml:131-139`
- Rank + monotonic constraints at `src/presets/v7_large.yaml:141-152`
- KL policy at `src/presets/v7_large.yaml:154-161`
- Angular coherence specialization at `src/presets/v7_large.yaml:168-183`
- Lagrangian activation at `src/presets/v7_large.yaml:185-189`

Why this is good:

- It is a scientifically explicit preset, not a generic bag of defaults.
- It maps cleanly into `CombinedLoss` at `src/losses/combined.py:57` and the specialized losses in `src/losses/padic_geodesic.py`.
- The comments record actual experimental reasoning, especially around `variance_weight`, `use_individual_valuation`, and `target_sim`.

Why this also exposes weakness:

- The preset demonstrates how much of the optimization policy surface now exists. That is powerful, but it also reflects the density problem in `src/losses/combined.py:134` and `src/losses/padic_geodesic.py:89`.
- This preset is interpretable only if the reader already understands several interacting losses. That is not the preset’s fault, but it does reveal the abstraction pressure in the loss layer.

#### 5. Runtime / Instrumentation Layer

**Interaction quality: good**

`v7_large.yaml` uses the runtime layer thoroughly:

- training cadence at `src/presets/v7_large.yaml:202-216`
- logging at `src/presets/v7_large.yaml:228-239`
- checkpoints at `src/presets/v7_large.yaml:244-248`
- targets at `src/presets/v7_large.yaml:253-259`

Why this is good:

- It is aligned with the modern runtime path in `src/train.py:807` and `src/train.py:1972`.
- It uses TensorBoard, checkpoints, and success targets in a way that matches the project’s actual workflow.
- It is the preset most compatible with the newer visualization/instrumentation additions, even if the visualization block itself lives elsewhere in active runs.

Main caveat:

- The runtime layer is still centralized in `src/train.py`, so this preset is only easy to reason about if the reader already knows how `train.py` interprets it.
- The preset is clean; the runtime consumer is what remains heavy.

### Preset verdict

`src/presets/v7_large.yaml` is the **best current preset** because it is:

- the latest preset definition
- the richest expression of the current architecture
- the most complete interaction test of the 5 abstraction layers
- the source of the best observed quantitative run in local artifacts

### Final preset assessment

| Dimension | Verdict |
|-----------|---------|
| Architectural fit | Excellent with the model and config layers |
| Scientific clarity | High; rationale is embedded directly in the preset |
| Runtime compatibility | Good, but still mediated by a large `train.py` |
| Maintainability | Good for a research preset, but dense because the loss surface is dense |
| Chosen status | **Latest and best preset in the repository** |

---

## Future-Proofing and Resilience Appendix

### Short answer

**Yes, it is possible to make this codebase substantially more future-proof, resilient, modular, and machine-checkable.**  
**No, it is not possible to eliminate semantic analysis of interactions entirely.**

The right goal is not "never read internals again".  
The right goal is: **make most failures impossible, make many regressions machine-detectable, and make the remaining semantic issues much smaller and more local**.

### What is realistically possible

The following can be enforced well:

- config shape and config cross-field invariants
- module input/output types
- public API contracts between layers
- allowed metric names / loss names / checkpoint fields
- expected tensor rank, dtype, and key presence at boundaries
- file- and symbol-level refactors driven by AST tools
- deterministic unit/property/integration checks that run without LLM interpretation

The following cannot be fully enforced by schema/type tools alone:

- whether a loss combination is scientifically correct
- whether a metric plateau is data-derived or architecture-derived
- whether a new model path preserves the intended geometry semantics
- whether two individually valid changes interact badly at optimization time

That means:

- **mechanical correctness can be automated heavily**
- **semantic correctness still needs careful engineering review**

### Core conclusion

If the repo remains in its current shape, then **many future changes will still require reading multiple internals carefully**, especially around:

- `src/train.py`
- `src/losses/combined.py`
- `src/losses/padic_geodesic.py`
- model/loss/runtime coupling across returned dict keys

However, this is **not an unavoidable property of the domain**.  
It is mostly a property of the current architecture concentrating too much interaction logic into a few broad files.

### Is there one tool that solves this?

**No single tool solves it.**  
There is no magic "token-free root-cause fixer" that understands scientific ML semantics automatically.

What does work is a **stack**:

1. strict schemas for config and serialized artifacts
2. strict static typing for Python interfaces
3. runtime contract validation at subsystem boundaries
4. AST-based codemods for safe structural refactors
5. deterministic automated tests at multiple layers
6. small, explicit intermediate data models between layers

That stack can reduce LLM/manual debugging work dramatically.

### Recommended future-proofing stack

#### 1. Make every layer talk through typed contracts

The biggest structural improvement is to stop passing wide, weakly-typed dicts between major subsystems.

Current examples of weak boundaries:

- model `forward()` returning a broad dict in `src/models/vae.py:349`
- training/runtime code pulling many optional keys from config dicts in `src/train.py:807`
- loss aggregation consuming heterogeneous output dictionaries in `src/losses/combined.py:429`

Recommended replacement pattern:

- `ModelOutputs` dataclass or `TypedDict`
- `LossOutputs` dataclass or `TypedDict`
- `EvalMetrics` dataclass
- `CheckpointPayload` dataclass / Pydantic model
- `RunConfig` top-level runtime object instead of ad hoc dict access

Effect:

- static analyzers can catch missing fields
- refactors become symbol-aware instead of string-key fragile
- AST tools can update field names safely

#### 2. Make `schema.py` the only external configuration contract

The repo already moved in this direction. Finish it.

Recommended end state:

- all YAML enters through one loader
- that loader returns one validated runtime object
- `StateNetConfig` becomes:
  - either a thin adapter over the schema object
  - or removed entirely

Why:

- dual config systems always drift eventually
- config drift is one of the highest-value problems to kill completely

#### 3. Add strict static typing and fail CI on type regressions

Recommended tools:

- `pyright --strict` as the primary type checker
- optionally `mypy` if you want dual coverage, but one strict checker is enough

Recommended focus areas first:

- `src/train.py`
- `src/losses/combined.py`
- `src/models/vae.py`
- `src/utils/visualization.py`

Key requirement:

- replace broad `Dict[str, Any]` boundaries with `TypedDict`, dataclasses, or Pydantic models

Without that, strict typing will stall.

#### 4. Add runtime contracts for tensor semantics

Types alone do not express tensor semantics well.

Recommended boundary checks:

- shape
- dtype
- device compatibility
- finite/non-NaN checks where needed
- required key presence

Good current example:

- `VisualizationPipeline._validate_inputs()` in `src/utils/visualization.py:735`

Recommended expansion:

- model output validation
- loss input validation
- checkpoint payload validation
- resume-checkpoint compatibility validation

Useful libraries:

- plain explicit validators are fine
- `beartype` or `typeguard` can help at runtime
- `jaxtyping`-style tensor annotations are useful conceptually, though explicit checks may be simpler here

#### 5. Use AST tooling for structural changes

For "AST debuggable" and token-free bulk fixes, use syntax-aware tools rather than LLM search/replace.

Recommended tools:

- `ruff` for lint and many auto-fixes
- `libcst` for codemods
- `bowler` if you prefer query-style AST refactors
- `ast-grep` for repository-wide structural matching

These are good for:

- renaming keys/symbols safely
- migrating call signatures
- detecting anti-patterns structurally
- enforcing repository conventions

This is one of the best ways to reduce manual patching risk.

#### 6. Build a deterministic machine-checkable test pyramid

The repo already has a decent start. The next step is a more deliberate pyramid:

**Layer tests**

- core algebra properties
- geometry invariants
- model output contract tests
- loss contract tests

**Property tests**

- Hypothesis-based checks for ternary/index round-trips
- monotonicity and symmetry properties
- checkpoint round-trip properties
- config normalization invariants

**Contract tests**

- every preset can instantiate model/loss/runtime objects
- every enabled loss block in every preset matches a real code path
- every checkpoint payload validates against a schema

**Golden integration tests**

- one tiny deterministic training step
- one resume-from-checkpoint cycle
- one visualization pipeline cycle

This is the main path to "token-free" confidence.

#### 7. Introduce a machine-readable interaction graph

This codebase would benefit from an explicit "interaction map" instead of leaving it implicit in broad files.

Recommended artifacts:

- `ModelOutputs`
- `LossInputs`
- `LossOutputs`
- `TrainingState`
- `CheckpointPayload`

And then:

- generate docs or validation from these models
- use them in tests
- use them in codemods

This does not eliminate all reasoning, but it turns "read everything" into "read the boundary models first".

### What cannot be fully automated

Even after all of the above, changes in these areas will still require careful human analysis:

- geometry semantics
- coupled optimization objectives
- tradeoffs between hierarchy, reconstruction, and direction structure
- experimental interpretation of metric ceilings

Why:

- these are not only software problems
- they are also scientific-modeling problems

So the future-proofing target should be:

- **automate mechanics**
- **localize semantics**
- **minimize the surface that still needs expert judgment**

### Practical recommendation order

If this were my codebase, I would do the following in order:

1. Split `src/train.py` into smaller runtime modules with typed interfaces.
2. Introduce typed boundary models for model outputs, loss outputs, and checkpoint payloads.
3. Make `schema.py` the only config contract and remove the remaining dual-config drift.
4. Turn on strict static checking in CI.
5. Add AST-based codemod tooling and repo rules.
6. Expand property and contract tests so most regressions are machine-caught.

### Final judgment

**Yes, this can become much more future-proof and resilient.**

But not by a single tool, and not by pretending semantic interactions can be fully schema-solved.

The realistic win is this:

- today, many changes require broad internal reading
- after the refactor stack above, **most changes would require reading only one module boundary and one local implementation**

That is the correct target.  
Not "zero reasoning", but **far less global reasoning per change**.
