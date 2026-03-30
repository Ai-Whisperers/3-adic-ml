# 28-03-2026 Config Drift, Type Brittleness, and Repo Slop Audit

## Scope

This audit focuses on three areas across the current repository state:

1. configuration drift,
2. brittle type enforcement,
3. broader repo slop / maintainability debt.

The goal is to record only evidence-backed findings from the current codebase.

## Acceptance Criteria

This audit is considered complete when the report contains:

- concrete findings for config drift,
- concrete findings for brittle type enforcement,
- concrete findings for broader repo slop,
- file-path evidence for each major claim,
- a final prioritized remediation list.

## Audit Notes

- Report path: `docs/audits/28-03-2026-CONFIG-DRIFT-TYPE-SLOP-AUDIT.md`
- Method: direct code reading, targeted grep, diagnostics, and command verification where useful.
- This file is being expanded iteratively as new evidence is gathered.

---

## Pass 1 — High-Confidence Findings

### 1. StateNet has conflicting default surfaces

**Severity:** High  
**Category:** Configuration drift

There are at least three conflicting StateNet/default narratives live at the same time:

- `src/config/statenet_config.py` defines runtime dataclass defaults for coverage thresholds as `0.35 / 0.45 / 0.3`.
- `src/config/schema.py` defines schema defaults for StateNet coverage as `0.995 / 1.0 / 0.95`.
- `src/README.md` still documents the old `0.995 / 1.0 / 0.95` values in its “Complete YAML Config Reference”.

This means the repo has multiple “sources of truth” for the same controller surface.

**Evidence**

- `src/config/statenet_config.py:34-36`
- `src/config/schema.py:121-123`
- `src/README.md:177-180`

### 2. Schema accepts StateNet fields that runtime config object does not model anymore

**Severity:** High  
**Category:** Configuration drift

The schema still accepts legacy StateNet fields such as:

- `annealing`
- `initial.decoders_trainable`

But `StateNetConfig` in `src/config/statenet_config.py` no longer models `annealing` or `decoders_trainable` at all. The decoder LR scale is still present and live; the drift is specifically around the removed annealing/initial-trainability fields, not around `lr_scales.decoders` itself.

This is explicit schema/runtime drift. Validation accepts a larger surface than the runtime controller object actually represents.

**Evidence**

- `src/config/schema.py:144-188`
- `src/config/schema.py:169-176`
- `src/config/statenet_config.py:91-132`
- `src/config/statenet_config.py:144-202`

### 3. `--force` bypasses the schema and falls back to raw-YAML runtime behavior

**Severity:** High  
**Category:** Configuration drift / brittleness

`src/train.py` validates configs with `normalize_config(raw_config)`, but if validation fails and `--force` is supplied, the code falls back to raw YAML and keeps running. The schema failure is printed first, but the subsequent runtime path still consumes the raw config through repeated `.get(..., default)` lookups.

That means typos or invalid config structure can still be masked after the initial warning in the exact mode intended to continue execution.

**Evidence**

- `src/train.py:2024-2033`
- `src/train.py:240-278`
- `src/train.py:833-848`
- `src/train.py:1038-1113`

### 4. Train/runtime only consumes a subset of the validated config surface

**Severity:** High  
**Category:** Configuration drift

The schema validates many keys that the runtime training path does not appear to consume.

High-value examples already confirmed:

- `data.train_split` / `data.val_split` are validated, but training actually uses `training.val_frac`.
- `checkpoints.save_dir`, `checkpoints.best_metric`, and `checkpoints.checkpoint_name` are present in presets/schema, but `train.py` writes checkpoints to `log_dir / "checkpoints"` directly.
- `logging.embedding_every` is validated but not consumed in the searched runtime path.
- `device.pin_memory` exists in schema/presets, but dataloaders use `torch.cuda.is_available()` directly instead.

**Evidence**

- `src/config/schema.py:497-509`
- `src/train.py:2070-2071`
- `src/config/schema.py:536-546`
- `src/train.py:1116-1117`
- `src/train.py:1844-1859`
- `src/train.py:1901-1928`
- `src/config/schema.py:524-533`
- `src/train.py:1044-1106`
- no read of `embedding_every` found in the searched runtime path during this audit
- `src/config/schema.py:39-49`
- `src/train.py:893-906`

---

## Open Follow-Up Areas

The next append passes will drill further into:

- legacy compatibility keys still accepted by schema but ignored by runtime,
- type-checker findings and weakly typed interfaces,
- repo-level test/config/dependency drift.

---

## Pass 2 — Command-Verified Runtime Brittleness

### 5. The recommended `v7_large.yaml` preset passes schema tests but fails `train.py --validate-only`

**Severity:** Critical  
**Category:** Configuration drift / brittle runtime wiring

Two validations disagree right now:

- `python3 -m pytest tests/test_config_schema.py -q` passes in the current repo state (`42 passed` during this audit session).
- `python3 src/train.py --config src/presets/v7_large.yaml --validate-only` fails during model validation with:

```text
[ERROR] Model validation failed: 'NoneType' object has no attribute 'get'
```

The root cause is the `anchor_checkpoint` surface:

- schema normalization allows `anchor_checkpoint: null`,
- `ModelAuditor.create_and_validate_model()` does `anchor_cfg = self.config.get("anchor_checkpoint", {})`,
- if the normalized config contains `anchor_checkpoint: None`, the later `anchor_cfg.get("path")` dereference crashes.

So the repo currently has a concrete config/runtime mismatch on the recommended preset.

**Evidence**

- command: `python3 -m pytest tests/test_config_schema.py -q` → passed locally during this audit session
- command: `python3 src/train.py --config src/presets/v7_large.yaml --validate-only` → fails with `NoneType` error
- `src/train.py:287-288`
- `src/train.py:2024-2027`
- `src/config/schema.py:626-627`

### 6. Schema compatibility blocks are still live even when code comments say they were removed

**Severity:** High  
**Category:** Configuration drift / slop

The repository contains multiple “kept for compatibility” blocks that directly contradict removal language elsewhere:

- schema still keeps `zero_structure` (`src/config/schema.py` explicitly says it is a legacy block kept for preset compatibility),
- `5.12.4.yaml` still uses `zero_structure`, `annealing`, `use_stratified`, `high_v_budget_ratio`, `use_adaptive`, and `grokking_detection`,
- `v6.yaml` comments say several of these keys were removed as dead config.

This creates an ambiguous maintenance state: removed in comments, preserved in schema, partially present in legacy presets, and mostly ignored at runtime.

**Evidence**

- `src/config/schema.py:207-209`
- `src/config/schema.py:392`
- `src/config/schema.py:417`
- `src/presets/5.12.4.yaml:59-61`
- `src/presets/5.12.4.yaml:106-109`
- `src/presets/5.12.4.yaml:125-132`
- `src/presets/5.12.4.yaml:159-165`
- `src/presets/v6.yaml:68-69`
- `src/presets/v6.yaml:224-226`
- `src/presets/v6.yaml:288-293`

---

## Pass 3 — Brittle Type Enforcement Findings

### 7. `train.py` alone still exhibits multiple concrete static-type and contract failures

**Severity:** High  
**Category:** Brittle type enforcement

Direct diagnostics on `src/train.py` surface several distinct classes of brittle typing problems in the main runtime path:

- unresolved symbol: `get_model_state_dict` is called but not imported,
- repeated optional-member access on `tb_logger.writer.add_*`,
- history/state type mismatches around `MetricBasedLR` resume restoration,
- iterator/typing mismatches around tqdm/range and DataLoader objects,
- argument/type mismatches around `WeightedRandomSampler`, scatter helpers, and KMeans construction,
- attribute/call confusion on objects such as `model.projections` under current type inference.

This makes the type-health concern concrete without relying on a repo-wide numeric summary.

**Evidence**

- `lsp_diagnostics(/d1/VAEs/3-adic-ml/src/train.py)`
- representative diagnostics at `src/train.py:295`, `1090-1092`, `1224-1234`, `1401`, `1558`, `1886`, `1950`

### 8. `train.py` bypasses the TensorBoard logger abstraction and directly dereferences an optional writer dozens of times

**Severity:** High  
**Category:** Brittle type enforcement / abstraction slop

`TensorBoardLogger` correctly models `self.writer` as optional (`Optional[SummaryWriterType]`). But `train.py` repeatedly accesses `tb_logger.writer.add_*` directly instead of routing through safe methods. This is the exact pattern that static analysis flags as optional-member-access risk.

This is both a type-health problem and an encapsulation problem.

**Evidence**

- `src/utils/tensorboard_logger.py:62`
- `src/train.py:1090-1092`
- `src/train.py:1401`
- `src/train.py:1427`
- `src/train.py:1663-1813`
- `src/train.py:1837`
- `src/train.py:1950`

### 9. `TernaryVAEV6.forward()` promises `Dict[str, Tensor]` but returns nullable fields

**Severity:** Medium  
**Category:** Brittle type contracts

The forward signature says it returns `Dict[str, torch.Tensor]`, but the returned dictionary includes nullable values:

- `logits_B = None` when `decode_b=False`
- `r_A` / `r_B` can be `None` in non-factored mode according to the inline comment

So the function’s annotation is stricter than the actual runtime contract.

**Evidence**

- `src/models/vae.py:349-351`
- `src/models/vae.py:381-399`

### 10. Optional dependency guards leave several modules with possibly-unbound imports

**Severity:** Medium  
**Category:** Brittle type enforcement

Several modules use `try/except ImportError` to set availability flags, but the current structure still leaves static analysis with possibly-unbound imported names.

The structural hot spots are:

- `matplotlib`, `umap_lib`, `pacmap_lib`, `trimap_lib`, `_ripser`, and `go` in `src/utils/visualization.py`
- `psutil` in `src/utils/hardware_monitor.py`
- `_ts_scatter_mean` / `_ts_scatter_std` in `src/utils/scatter_utils.py`

These guards are runtime-friendly, but the current structure is brittle from both static-analysis and maintenance perspectives.

**Evidence**

- `src/utils/visualization.py:37-75`
- `src/utils/hardware_monitor.py:23-29`
- `src/utils/scatter_utils.py:11-16`

### 11. Type-check suppression strategy is broad enough to hide real problems

**Severity:** High  
**Category:** Brittle type enforcement / process debt

The repo’s mypy configuration suppresses or weakens checking in exactly the places where the diagnostics are most valuable:

- global `ignore_missing_imports = true`
- global `warn_return_any = false`
- global `disallow_untyped_defs = false`
- `ignore_errors = true` for `src.train`
- `ignore_errors = true` for `src.losses.padic_geodesic`
- `ignore_errors = true` for `src.utils.tensorboard_logger`

That means the repo has type tooling configured, but current enforcement is intentionally soft around high-risk modules.

**Evidence**

- `pyproject.toml:120-146`

### 12. `train.py` has a real symbol-level bug that type tooling already detects

**Severity:** High  
**Category:** Brittle type enforcement / correctness

`train.py` calls `get_model_state_dict(ckpt)` inside `ModelAuditor.create_and_validate_model()`, but the file only imports `load_checkpoint_compat` from `src.utils.checkpoint`.

`get_model_state_dict` is exported from `src.utils.__init__`, but `train.py` does not import it.

This is not just stylistic type noise; it is a concrete unresolved symbol path that static analysis already caught.

**Evidence**

- `src/train.py:87`
- `src/train.py:295`
- `src/utils/checkpoint.py:40-57`
- `src/utils/__init__.py:1-15`

### 12b. The missing `get_model_state_dict` import is command-verifiable on preset validation

**Severity:** Critical  
**Category:** Brittle type enforcement / runtime correctness

This is not only a static-analysis finding. It is reachable through the shipped preset flow:

- `python3 src/train.py --config src/presets/v6.yaml --validate-only` fails with `name 'get_model_state_dict' is not defined`
- `python3 src/train.py --config src/presets/5.12.4.yaml --validate-only` fails with the same error

So multiple shipped presets currently fail runtime validation on the unresolved symbol path.

**Evidence**

- command: `python3 src/train.py --config src/presets/v6.yaml --validate-only`
- command: `python3 src/train.py --config src/presets/5.12.4.yaml --validate-only`
- `src/train.py:295`
- `src/utils/checkpoint.py:40-57`

---

## Pass 4 — Repo Slop / Quality Debt Findings

### 13. Pytest configuration is duplicated, and `pytest.ini` shows no observable effect in current pytest behavior

**Severity:** High  
**Category:** Repo slop / test-surface drift

The repo currently has pytest configuration in both:

- `pyproject.toml`
- `pytest.ini`

But `pytest.ini` uses `[tool.pytest.ini_options]`, which is the `pyproject.toml` section name, not the standard `pytest.ini` section name. In the observed pytest behavior during this audit, the `integration` marker declared only in `pytest.ini` does not appear in `python3 -m pytest --markers`, while the markers added by `pyproject.toml`/`tests.conftest.py` do. That supports the conclusion that the standalone ini file currently has no observable effect.

The marker surface is also inconsistent:

- `pyproject.toml` defines `slow` and `gpu`
- `pytest.ini` defines `slow` and `integration`
- `tests/conftest.py` programmatically adds `slow` and `gpu`

This is configuration slop around the test runner itself.

**Evidence**

- `pyproject.toml:59-68`
- `pytest.ini:1-21`
- `tests/conftest.py:48-54`
- command: `python3 -m pytest --markers` (shows `slow` and `gpu`, but not `integration`)

### 14. Dependency metadata is drifting across `requirements.txt`, `requirements-dev.txt`, and `pyproject.toml`

**Severity:** High  
**Category:** Repo slop / packaging drift

`requirements-dev.txt` says runtime dependencies are declared in `pyproject.toml`, but `pyproject.toml` is missing a large set of packages that `requirements.txt` and the docs treat as required or actively used.

Examples present in `requirements.txt` but absent from `pyproject.toml [project.dependencies]`:

- `pydantic`
- `yamllint`
- `scikit-learn`
- `matplotlib`
- `umap-learn`
- `pacmap`
- `trimap`
- `ripser`
- `persim`
- `plotly`
- `pillow`
- `pandas`

That means editable/package installs and requirements-based installs do not describe the same environment.

**Evidence**

- `requirements.txt:16-48`
- `requirements-dev.txt:15-18`
- `pyproject.toml:26-35`
- `pyproject.toml:37-50`

### 15. CI gates YAML lint and pytest, but not type health

**Severity:** Medium  
**Category:** Repo slop / process gap

The CI workflow currently runs:

- `yamllint`
- `pytest tests/ -v --tb=short`

It does not run mypy, Pyright, or any equivalent static-type gate, even though the repo advertises type-aware schema/config discipline and the current source tree still emits a large non-zero static-diagnostics set.

**Evidence**

- `.github/workflows/ci.yml:10-47`
- `pyproject.toml:117-146`
- `lsp_diagnostics(/d1/VAEs/3-adic-ml/src, extension=.py)`

### 16. Test/schema coverage creates a false sense of config safety

**Severity:** Medium  
**Category:** Repo slop / coverage blind spot

The schema tests are good at validating Pydantic rules, but they do not guarantee that normalized config objects survive the runtime path through `train.py` and `ModelAuditor`.

That gap is demonstrated by the current state where:

- `tests/test_config_schema.py` passes fully,
- yet `src/train.py --config src/presets/v7_large.yaml --validate-only` still fails.

This is not a unit-test failure; it is a missing integration contract between “schema-valid” and “runtime-valid”.

**Evidence**

- `tests/test_config_schema.py:25-174`
- command: `python3 -m pytest tests/test_config_schema.py -q`
- command: `python3 src/train.py --config src/presets/v7_large.yaml --validate-only`

### 17. Status/docs overstate runtime health relative to current preset validation behavior

**Severity:** Medium  
**Category:** Repo slop / stale health narrative

`docs/STATUS.md` currently describes the training loop and config validation as fixed/validated, but multiple shipped presets still fail when driven through the actual `train.py --validate-only` runtime path.

That does not make the status document useless, but it does mean the repo’s health narrative is ahead of the current executable reality.

**Evidence**

- `docs/STATUS.md:12-16`
- `docs/STATUS.md:22`
- command: `python3 src/train.py --config src/presets/v7_large.yaml --validate-only`
- command: `python3 src/train.py --config src/presets/v6.yaml --validate-only`
- command: `python3 src/train.py --config src/presets/5.12.4.yaml --validate-only`

### 18. The repo claims “single source of truth” for config while exporting two parallel config systems

**Severity:** Medium  
**Category:** Repo slop / configuration narrative drift

`src/config/__init__.py` describes the config package as centralized and a “single source of truth”, but the package simultaneously exports:

- the dataclass-based `StateNetConfig` system, and
- the Pydantic `TrainingConfigSchema` system.

Given the already-confirmed drift between `statenet_config.py` and `schema.py`, that narrative is currently aspirational rather than true.

**Evidence**

- `src/config/__init__.py:8-31`
- `src/config/__init__.py:49-67`
- `src/config/statenet_config.py`
- `src/config/schema.py:609-627`

### 19. `src/README.md` still documents removed StateNet annealing fields as if they were active

**Severity:** Medium  
**Category:** Configuration drift / stale docs

The StateNet YAML reference in `src/README.md` still documents:

- `statenet.annealing.enabled`
- `statenet.annealing.step`
- `statenet.annealing.q_decrease_threshold`

But the runtime dataclass no longer models an `annealing` section. This compounds the schema/runtime drift by keeping legacy configuration live in user-facing docs.

**Evidence**

- `src/README.md:194-198`
- `src/config/statenet_config.py:91-132`
- `src/config/statenet_config.py:144-202`

### 20. `statenet_config.py` is internally self-contradictory about its own defaults

**Severity:** Medium  
**Category:** Configuration drift / stale docs

The same file both defines the live runtime defaults and documents an older threshold narrative:

- runtime `CoverageThresholds` defaults are `0.35 / 0.45 / 0.3`,
- but the `StateNetConfig` class doc/example still shows `0.995 / 1.0`.

So drift is not only between files; it also exists inside the StateNet dataclass module itself.

**Evidence**

- `src/config/statenet_config.py:34-36`
- `src/config/statenet_config.py:106-111`

---

## Prioritized Remediation Order

1. **Fix the runtime crash on normalized optional config blocks** (`anchor_checkpoint` / similar `None`-vs-dict surfaces in `train.py`).
2. **Collapse config truth into one authoritative model** for StateNet defaults and remove schema-only compatibility fields that are no longer meant to work.
3. **Remove or explicitly quarantine unused validated keys** (`precision.dtype`, `data.train_split`, `data.shuffle`, `targets.*`, `checkpoints.save_dir`, `logging.embedding_every`, `device.pin_memory`, etc.).
4. **Add an integration test that runs `src/train.py --validate-only` on every maintained preset**, not just schema validation.
5. **Reduce type suppressions and add a static-type CI gate** for at least `src/train.py`, `src/models/vae.py`, and `src/utils/visualization.py`.
6. **Stop direct access to `tb_logger.writer` from `train.py`** and move those operations behind `TensorBoardLogger` methods.
7. **Unify pytest configuration into one live config surface** and delete or repair inert config files.
8. **Reconcile package metadata** so `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt` describe the same install story.

---

## Appendix — Additional Confirmed Drift Surfaces

These are lower-level but still concrete surfaces confirmed during the audit.

### A. Duplicated worker / loader config

- `device.num_workers` exists in schema.
- `training.num_workers` also exists in schema.
- `train.py` uses `training.num_workers`, not `device.num_workers`.

**Evidence**

- `src/config/schema.py:39-49`
- `src/config/schema.py:476-482`
- `src/train.py:858`

### B. `device.pin_memory` is decorative in current runtime

The schema/presets expose `device.pin_memory`, but the data loaders hardcode `pin_memory=torch.cuda.is_available()`.

**Evidence**

- `src/config/schema.py:45`
- `src/presets/v6.yaml:28`
- `src/presets/v7.yaml:25`
- `src/presets/v7_large.yaml:24`
- `src/train.py:893-906`

### C. `data.use_full_dataset` and `data.n_operations` are validated but not consumed by training

`DataAuditor.prepare_data()` always builds from `TERNARY.all_ternary()` and the runtime path does not read `data.use_full_dataset` or `data.n_operations`.

**Evidence**

- `src/config/schema.py:497-504`
- `src/train.py:167-170`
- `src/train.py:2068-2071`

### D. `logging.print_every` exists, but training progress uses `training.print_every`

This creates two similarly named config knobs, with only one actually used for epoch printing.

**Evidence**

- `src/config/schema.py:476-480`
- `src/config/schema.py:524-533`
- `src/train.py:848`
- `src/train.py:1868`

### E. The `checkpoints` block is mostly metadata in current training runtime

The schema/presets expose checkpoint destination and metric selection fields, but `train.py` currently derives the checkpoint directory from `log_dir / "checkpoints"` and does not read the configured `save_dir`, `best_metric`, or `checkpoint_name` values in the searched runtime path.

**Evidence**

- `src/config/schema.py:536-546`
- `src/presets/v6.yaml:265-269`
- `src/presets/v7.yaml:237-241`
- `src/presets/v7_large.yaml:244-248`
- `src/train.py:1116-1117`
- `src/train.py:1844-1859`
- `src/train.py:1901-1928`

### F. `grokking_detection` silently drops some accepted keys

The schema accepts:

- `gradient_norm_track`
- `representation_analysis`

But `train.py` explicitly ignores unknown or unmapped grokking keys when building `GrokkingDetector` params.

**Evidence**

- `src/config/schema.py:453-463`
- `src/train.py:1122-1145`
- inline comment at `src/train.py:1144`

### G. `logging.tensorboard` and `logging.log_dir` are validated but ignored by runtime

The logging schema exposes explicit switches for TensorBoard enablement and log directory selection, but the runtime path currently always creates a run directory under `RUNS_DIR` and always passes that directory into `TensorBoardLogger`.

**Evidence**

- `src/config/schema.py:524-533`
- `src/train.py:1044-1048`
- `src/train.py:2057-2064`

### H. `device.empty_cache_freq` is exposed in presets but runtime reads `memory.empty_cache_freq`

The active presets place `empty_cache_freq` under `device`, but the training loop reads it from the `memory` block. This is a live config-placement drift.

**Evidence**

- `src/config/schema.py:47-49`
- `src/config/schema.py:560-566`
- `src/presets/v7.yaml:21-27`
- `src/presets/v7_large.yaml:20-26`
- `src/train.py:1112-1113`
- `src/train.py:2047-2052`

### I. The `checkpoints` block exposes more inert fields than originally called out

In the current runtime path, the checkpoint block appears to expose several fields that are not consumed, including not only `save_dir`, `best_metric`, and `checkpoint_name`, but also `save_best` and `save_freq`.

**Evidence**

- `src/config/schema.py:536-546`
- `src/presets/v6.yaml:265-269`
- `src/presets/v7.yaml:237-241`
- `src/presets/v7_large.yaml:244-248`
- `src/train.py:1116-1117`
- `src/train.py:1844-1859`
- `src/train.py:1901-1928`

### J. `schema.py` overstates runtime fidelity in its own module-level contract

`schema.py` says the schema models the live preset fields that the training script actually consumes and can normalize configs without silently dropping sections. The ignored-key set and the `anchor_checkpoint: None` runtime crash both undermine that claim.

**Evidence**

- `src/config/schema.py:8-11`
- `src/config/schema.py:609-627`
- `src/train.py:287-288`
- `src/train.py:2024-2033`

### K. `device.name` is validated and present in active presets but not consumed by runtime

The schema and active presets expose `device.name`, but the runtime device-selection path only reads `cuda_device` and `use_amp` from `device_cfg` in the searched training path.

**Evidence**

- `src/config/schema.py:39-45`
- `src/presets/v7.yaml:21-24`
- `src/presets/v7_large.yaml:20-23`
- `src/train.py:2036-2045`

### L. `memory.empty_cache` and `memory.gradient_checkpointing` are exposed but not consumed in the searched runtime path

The memory schema exposes `empty_cache` and `gradient_checkpointing`, but the runtime path currently reads `memory.empty_cache_freq` and unconditionally calls `torch.cuda.empty_cache()` on that schedule. No runtime read of `empty_cache` or `gradient_checkpointing` was found in the searched training path.

**Evidence**

- `src/config/schema.py:560-566`
- `src/train.py:1112-1113`
- `src/train.py:1911-1913`

### M. The entire `targets` block is validated and present in active presets but unused in the searched runtime path

The schema defines a dedicated `targets` block and all maintained presets populate it, but no `config.get("targets")` read was found in `src/train.py` during this audit.

**Evidence**

- `src/config/schema.py:549-557`
- `src/presets/v6.yaml:274-280`
- `src/presets/v7.yaml:246-252`
- `src/presets/v7_large.yaml:253-259`
- no matches for `targets` reads in `src/train.py` during this audit

### N. `data.shuffle` is validated but not used for dataset splitting/loading in the searched runtime path

The schema exposes `data.shuffle`, but `DataAuditor.prepare_data()` always permutes deterministically from a seeded RNG and the runtime loaders do not consult a config-driven shuffle flag in the searched path.

**Evidence**

- `src/config/schema.py:497-503`
- `src/train.py:172-178`
- `src/train.py:893-905`

### O. `precision.dtype` is validated and present in maintained presets but not consumed in the searched runtime path

The schema and maintained presets expose `precision.dtype`, but no `config.get("precision")` read was found in `src/train.py` during this audit. Instead, training setup relies on the `use_float64` function parameter and hard-sets the default dtype when that parameter is true.

**Evidence**

- `src/config/schema.py:98-101`
- `src/presets/v6.yaml:73-76`
- `src/presets/v7.yaml:64-65`
- `src/presets/v7_large.yaml:63-64`
- `src/train.py:97-104`
- `src/train.py:112-114`
- no matches for `precision` reads in `src/train.py` during this audit

---

## Pass 5 — Per-Item Deep Dives and False-Positive Review

This pass re-reads the origin files for the already-listed findings to add more implementation context, reduce false-positive risk, and clarify what each fix would actually need to touch.

### Config drift items — deeper context

**Item 1 — conflicting StateNet default surfaces: confirmed and broader than initially stated.**  
Direct code reads show the drift is not just between `src/config/statenet_config.py`, `src/config/schema.py`, and `src/README.md`; it also exists *inside* `statenet_config.py` itself. The `CoverageThresholds` runtime defaults are `0.35 / 0.45 / 0.3`, while the class doc/example on `StateNetConfig` and the `src/README.md` integration guide still describe `0.995 / 1.0 / 0.95`. That means a maintainer can read two contradictory threshold stories without ever leaving the StateNet-related files.

**Item 2 — schema/runtime StateNet surface mismatch: confirmed, with one important non-false-positive boundary.**  
The original concern remains valid for `annealing` and `initial.decoders_trainable`: `StateNetConfig.from_dict()` rejects unknown keys entirely and the dataclass does not model those fields, while the Pydantic schema still accepts them. The important false-positive check here is that `lr_scales.decoders` is *not* part of this mismatch: it is modeled in `LRScales`, returned by `MetricBasedLR.get_lr_scales()`, applied by `update_optimizer_lr_scales()`, and exposed by `TernaryVAEV6Controllable.get_param_groups()` under the `decoders` param-group name.

**Item 3 — `--force` raw-YAML fallback: confirmed, but the risk is post-warning masking, not silent entry.**  
Direct reads of `main()` show the failure path is: schema exception → printed error → optional `--force` fallback → raw config consumed through `.get()` across device, model, training, logging, checkpoint, and memory sections. So the operational risk is that raw YAML continues through a permissive runtime path after the warning, not that validation failure is completely invisible.

**Item 4 — “validated surface larger than consumed surface”: confirmed and concentrated in a few subsystems.**  
The main drift cluster is the split between `schema.py` and `train.py`: `training` is the runtime source for batch size / LR / print frequency / worker count / stratified sampling; `logging` only partially drives runtime (`verbose`, `histogram_every`, `enhanced_metrics`), while `log_dir`, `tensorboard`, `embedding_every`, and `logging.print_every` are not used in the searched runtime path; `data` is mostly decorative next to `DataAuditor.prepare_data(val_frac)`; `checkpoints` is largely metadata next to `log_dir / "checkpoints"`; `precision.dtype` is superseded by `set_determinism(..., use_float64=True)`.

**Item 5 — `anchor_checkpoint: null` crash: confirmed and structurally localized.**  
The failure is concentrated in `ModelAuditor.create_and_validate_model()`: normalized config can legally contain `anchor_checkpoint: None`, but the runtime path immediately assumes a mapping and calls `.get("path")`. This is a narrow fix with high payoff because it blocks `--validate-only` on the recommended preset before training even begins.

**Item 6 — compatibility blocks vs “removed dead config” comments: confirmed as policy drift.**  
The deep-dive reading shows a real split in maintenance policy: `schema.py` intentionally keeps several legacy fields for preset compatibility, while `v6.yaml` comments frame several related fields as removed dead config. The repo therefore mixes “compatibility preserved” and “removed/archived” narratives without one authoritative migration policy.

**Items 19 and 20 — stale StateNet docs: confirmed and connected to the same root problem.**  
`src/README.md` still teaches the older StateNet threshold and annealing model, while `statenet_config.py` has already moved to a leaner dataclass surface. These are not isolated stale lines; they indicate that the docs still describe an older control philosophy than the runtime dataclass now embodies.

**Appendix config surfaces A–O: all confirmed after direct re-read, with these notable implementation clarifications.**  
`device.num_workers` and `training.num_workers` are a real duplicate surface, but runtime consistently chooses the training block. `device.pin_memory` and `device.empty_cache_freq` are preset-visible but bypassed by hardcoded or different-section runtime logic. `memory.empty_cache` and `memory.gradient_checkpointing` are schema-visible but do not participate in the searched training path. `targets` is fully modeled and fully populated in maintained presets yet absent from `train.py`. `data.shuffle` does not steer the deterministic split or loader behavior. `logging.tensorboard` / `logging.log_dir` are especially important because they *look* like user controls but the runtime always constructs its own `RUNS_DIR / run_name` path and always instantiates `TensorBoardLogger` against it. `schema.py`’s module docstring overstates fidelity by claiming the schema models the live fields the training script actually consumes without silently dropping sections; the current runtime-ignored fields and the `anchor_checkpoint: None` crash contradict that claim.

### Brittle type-enforcement items — deeper context

**Item 7 — weak static type health: confirmed, but the strongest evidence is structural rather than numeric.**  
The high-value part of this finding is not the exact diagnostic count; it is that `train.py`, `visualization.py`, optional-import modules, and checkpoint/runtime glue all exhibit patterns that static analysis predictably dislikes: unresolved names, optional-member access, and loose `Dict[str, Any]` flow across the training script.

**Item 8 — TensorBoard abstraction bypass: strongly confirmed.**  
`TensorBoardLogger` exposes a safe wrapper style (`is_available`, `log_batch`, `log_histograms`, `flush`, `close`) but `train.py` still reaches inside to use `tb_logger.writer.add_*` directly in dozens of places. This deepens the original audit point: the abstraction is not just thinly used; it is systematically bypassed for most epoch-level logging and hparam writing.

**Item 9 — `TernaryVAEV6.forward()` return contract: confirmed.**  
The return type remains `Dict[str, torch.Tensor]`, but `logits_B` can be `None` when `decode_b=False` and the comments explicitly say `r_A` / `r_B` may be `None` outside factored mode. The practical fix is not conceptual; it is a concrete annotation mismatch between the published interface and the actual returned structure.

**Item 10 — optional-dependency brittleness: confirmed, with false-positive risk reduced by runtime guards.**  
The direct code read shows `visualization.py` is more disciplined than a pure grep view suggests: runtime methods check `_HAS_*` flags before many backend-specific operations, and `VisualizationPipeline.__init__()` reports missing packages. So the deeper judgment is: runtime fragility is lower than the initial wording might imply, but the typing/import structure is still brittle and noisy for analysis tools because symbols may be unavailable even though later code tries to guard them.

**Item 11 — broad mypy suppression: confirmed.**  
The suppressions are not hypothetical; they are explicitly targeted at the three files that dominate the audit’s type-brittleness story (`src.train`, `src.losses.padic_geodesic`, `src.utils.tensorboard_logger`). The effect is that the repo has type tooling configured, but the most operationally important modules are also the least strongly enforced.

**Items 12 and 12b — missing `get_model_state_dict` import: confirmed as both static and runtime breakage.**  
The direct code read shows the unresolved symbol is invoked in the anchor-checkpoint path, while the import section only brings in `load_checkpoint_compat`. The command-verified failures on `v6.yaml` and `5.12.4.yaml` prove this is not just a theoretical lint issue; it is a real executable failure in shipped preset flows.

### Repo-slop / process items — deeper context

**Item 13 — duplicated pytest config: confirmed, and the mismatch is three-way.**  
The repo does not just have `pyproject.toml` plus `pytest.ini`; it also mutates markers inside `tests/conftest.py`. The practical result is three distinct marker/config stories: pyproject says `slow` + `gpu`, pytest.ini says `slow` + `integration`, and runtime test setup adds `slow` + `gpu` again.

**Item 14 — dependency metadata drift: confirmed and materially relevant to packaging.**  
The deep dive on `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, and `docs/DEPENDENCIES.md` shows that editable/package installs and requirements installs still do not describe the same environment. This is not cosmetic: runtime schema validation depends on `pydantic`, training metrics depend on `scikit-learn`, and the visualization pipeline depends on packages absent from `[project.dependencies]`.

**Item 15 — CI gap: confirmed, and the missing gate is broader than just typing.**  
CI currently runs yamllint and pytest, but it also skips the exact runtime preset-validation path that exposed the most serious config/runtime breakages. So the process gap is both “no static-type gate” and “no executable preset gate.”

**Item 16 — schema-test safety illusion: confirmed.**  
`tests/test_config_schema.py` is a good schema-focused suite, but it concentrates on `validate_config()` / `normalize_config()` behavior and individual sub-schema defaults. It does not run the normalized config through `main()` or `ModelAuditor`, which is why runtime failures can survive alongside a green schema suite.

**Item 17 — optimistic repo health narrative: confirmed.**  
`docs/STATUS.md` still presents the training loop and config validation as fixed and validated, while direct command checks on maintained presets show runtime validation failures. This is a documentation drift problem, not a claim that every status note is false.

**Item 18 — “single source of truth” narrative: confirmed as aspirational wording.**  
The config package exports both the dataclass StateNet path and the Pydantic full-schema path, and the surrounding docs repeat “single source of truth” language in more than one place. The direct code read shows that the repo is instead operating with parallel config authorities.

### Newly discovered issues during the per-item deep dive

**Item 21 — `src/README.md` still teaches a stale LR-scale merge pattern that bypasses the validation now used in `train.py`.**  
The integration guide in `src/README.md` shows direct field mutation of `sn_config.lr_scales.encoder_a / encoder_b / projections`, while `train.py` now contains an explicit comment explaining why runtime goes through the `LRScales(...)` constructor instead: direct field assignment bypasses `LRScales.__post_init__` validation. This is a documentation-level false positive risk because readers are taught an outdated merge pattern.

**Evidence**

- `src/README.md:63-67`
- `src/train.py:930-939`
- `src/config/statenet_config.py:81-88`

**Item 22 — `src/README.md` still describes pure “Option C” optimizer-only control, but runtime now also toggles `requires_grad`.**  
The guide says LR multipliers are the single source of truth and presents that as the cleaner replacement for `requires_grad=False`, but the runtime training loop now explicitly synchronizes `requires_grad` with controller decisions after applying LR scales. This is not necessarily wrong in code, but it means the documentation’s control philosophy no longer matches the actual implementation.

**Evidence**

- `src/README.md:24-31`
- `src/train.py:1595-1618`

**Item 23 — `docs/DEPENDENCIES.md` overstates the TensorBoard failure mode.**  
The dependency doc says TensorBoard absence causes logging to be “silently skipped — no error, no warning,” but `train.py` does print a warning when `tb_logger.is_available` is false, and `TensorBoardLogger` can also emit a warning through its callback when TensorBoard was requested but not installed. The operational point that metrics are lost remains correct; the “no warning” phrasing is the stale part.

**Evidence**

- `docs/DEPENDENCIES.md:31-40`
- `src/utils/tensorboard_logger.py:65-73`
- `src/train.py:1056-1058`

### False-positive review summary

After the direct re-read pass, the meaningful corrections/softenings are:

- do **not** treat `lr_scales.decoders` as part of the StateNet schema/runtime mismatch,
- describe `--force` fallback as **post-warning raw-YAML continuation**, not silent entry,
- describe the static-analysis problem in qualitative structural terms unless an exact reproduced count is part of the evidence at hand,
- treat the optional-import issue as **type/maintainability brittleness with runtime guards**, not blanket runtime breakage.

---

## Pass 6 — Appendix A–O Individual Re-Verification

This pass revisits each appendix item individually so the lower-level drift surfaces are not only grouped under a single confirmation paragraph.

### Appendix A — duplicated worker config

`device.num_workers` is a real duplicate surface, not just an unused comment artifact. The actual DataLoader worker count is read from `train_cfg.get("num_workers", 4)` immediately before loader construction, while the early device-handling path only reads `cuda_device` and `use_amp` from `device_cfg`. That makes `device.num_workers` vestigial in the searched runtime path.

### Appendix B — `device.pin_memory`

This is confirmed as a true runtime bypass rather than an indirect alias. Both `train_loader` and `val_loader` hardcode `pin_memory=torch.cuda.is_available()` and do not thread through any config-sourced value, so changing `device.pin_memory` in YAML cannot affect loader construction in the searched path.

### Appendix C — `data.use_full_dataset` / `data.n_operations`

This remains a direct runtime drift. `DataAuditor.prepare_data()` constructs the data from `TERNARY.all_ternary()` and the `main()` path only passes `val_frac` into it, so the `data` block currently does not control dataset cardinality or full/subset selection in the searched training path.

### Appendix D — `logging.print_every`

This is not just duplicated naming; it creates a misleading knob. Epoch progress printing is governed by `training.print_every`, while `logging.print_every` is only part of the logging schema and is never threaded into the runtime printing branch or the TensorBoard logger callback.

### Appendix E — checkpoint block metadata drift

The deeper runtime context strengthens this finding: checkpoint filenames and destinations are hardwired around `ckpt_dir / "best_Q.pt"`, `ckpt_dir / f"epoch_{epoch}.pt"`, and `ckpt_dir / "final.pt"`, all under `log_dir / "checkpoints"`. That means `checkpoints.save_dir`, `best_metric`, and `checkpoint_name` are not merely unreferenced keys; they are displaced by explicit runtime naming logic.

### Appendix F — `grokking_detection` ignored keys

The runtime behavior is fully explicit here. `train.py` builds a small key-map for the constructor-supported parameters, skips `enabled`, and then silently drops everything else via the `# else: silently ignore unknown keys` branch. So `gradient_norm_track` and `representation_analysis` are accepted by schema but intentionally discarded by runtime.

### Appendix G — `logging.tensorboard` / `logging.log_dir`

This drift is stronger than a missing conditional. The run directory is always derived from `RUNS_DIR / run_name`, written to disk immediately, and then passed into `TensorBoardLogger`; `logging.tensorboard` does not gate logger creation, and `logging.log_dir` does not participate in the directory path chosen by `main()`.

### Appendix H — `device.empty_cache_freq`

This is a live placement mismatch, not just a stale field. The active presets place `empty_cache_freq` under `device`, but the runtime reads only `memory_cfg.get("empty_cache_freq", 25)` and then uses that value in the periodic CUDA cache cleanup branch.

### Appendix I — broader inert checkpoint fields

This remains confirmed after deeper read. `save_best` and `save_freq` in the checkpoint block are not wired into the actual save branches; periodic checkpoint cadence is controlled by `training.save_every`, and the best-checkpoint logic is hardcoded to Q improvement rather than a configurable `best_metric` or `save_best` switch.

### Appendix J — `schema.py` module contract overstatement

This remains a fair criticism after re-read. The module docstring advertises the schema as the live model of the fields the training script actually consumes and as preserving sections without silent drops, but the runtime still ignores entire accepted surfaces (`targets`, logging/checkpoint/device extras) and fails on the normalized `anchor_checkpoint: None` case.

### Appendix K — `device.name`

This is correctly classified as unused in the searched runtime path. Device selection is driven by CLI `--device`, optional `device.cuda_device`, and optional `device.use_amp`; no read of `device.name` participates in device construction.

### Appendix L — `memory.empty_cache` / `memory.gradient_checkpointing`

The deeper read confirms these are schema-visible but behaviorally inert in the searched training path. Runtime only reads `memory.empty_cache_freq` plus `memory.cudnn_benchmark`; the actual `torch.cuda.empty_cache()` call is scheduled solely from the frequency value, and no gradient-checkpointing branch is wired into training setup.

### Appendix M — `targets` block

This remains one of the cleanest unused-surface findings. The schema has a dedicated `TargetsConfig`, all maintained presets populate it, and the searched `train.py` path never reads it. There is no evidence that it participates indirectly through another helper in the main training entrypoint.

### Appendix N — `data.shuffle`

This finding also survives deeper review. Data splitting is deterministic from seeded permutation inside `DataAuditor.prepare_data()`, train loading is sampler-driven, and validation loading is explicitly `shuffle=False`; no config-level shuffle flag influences any of those decisions in the searched runtime path.

### Appendix O — `precision.dtype`

This remains a true orphaned surface in the main training path. The schema and presets expose it, but dtype behavior in `train.py` is governed by the `use_float64` function parameter and `torch.set_default_dtype(torch.float64)` when that flag is active, not by any read from the `precision` config block.

### Additional nuance discovered while revisiting repo-slop items

The packaging-drift conclusion in Item 14 still stands, but the strongest formulation is: these metadata files describe different install stories, not that every package absent from `pyproject.toml` is mandatory in every runtime path. Some of the visualization/analysis packages are optional or guarded at runtime, but their omission from `[project.dependencies]` still conflicts with `requirements.txt` and the project docs.
