# Silent Failure & Reliability Audit — 2026-05-02

**Scope:** Full adversarial review of `src/` for silent failures, exception swallowing,
None-dereference time bombs, metric corruption, and training loop correctness issues.  
**Method:** Manual code review + automated grep for exception patterns + runtime verification.  
**Files audited:** `train.py`, `losses/combined.py`, `losses/padic_geodesic.py`,
`models/vae.py`, `models/lr_controller.py`, `config/schema.py`, `config/statenet_config.py`,
`utils/checkpoint.py`

---

## Severity Legend

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Silent wrong results or crash that corrupts saved state |
| **HIGH** | Silent wrong metric or training behaviour across whole run |
| **MEDIUM** | Silent wrong value in specific conditions; low reproduction rate |
| **LOW** | Code smell / fragility; no observable wrong result yet |

---

## Summary Table

| ID | File | Line(s) | Category | Severity | Status |
|----|------|---------|----------|----------|--------|
| F-01 | train.py | 1238 | Falsy-zero resume | CRITICAL | **Fixed 2026-05-02** |
| F-02 | train.py | 566–568 | NaN→0 metric masking | HIGH | **Fixed 2026-05-03** |
| F-03 | train.py | 314–318 | Silent anchor fallback | HIGH | **Fixed 2026-05-03** |
| F-04 | losses/combined.py | 628–638 | One-time AC warning | MEDIUM | **Fixed 2026-05-03** |
| F-05 | train.py | 369–372 | Broad except in model audit | MEDIUM | **Fixed 2026-05-03** |
| F-06 | models/lr_controller.py | 448–451 | Name-mismatch silent skip | MEDIUM | **Fixed 2026-05-03** |
| F-07 | train.py | 309–313 | Broad except checkpoint load | MEDIUM | **Fixed 2026-05-03** |
| F-08 | losses/padic_geodesic.py | 227–229 | corrcoef on n=1 | MEDIUM | **Fixed 2026-05-03** |
| F-09 | train.py | 1429 | Division by n_batches=0 | MEDIUM | **Fixed 2026-05-03** |
| F-10 | config/statenet_config.py | 158–175 | Deep-key typo silent default | MEDIUM | **Fixed 2026-05-03** |
| F-11 | train.py | 586–588 | dist_corr NaN→0 masking | MEDIUM | **Fixed 2026-05-03** |
| F-12 | losses/combined.py | 302–308 | AC warn flag not reset | LOW | **Fixed 2026-05-03** |
| F-13 | train.py | 1054 | TensorBoard not closed on exception | LOW | **Fixed 2026-05-03** |

---

## Detailed Findings

---

### F-01 — CRITICAL: Falsy-zero corrupts resume best_Q

**File:** `src/train.py:1238`

```python
best_Q = float(ckpt.get("Q", ckpt.get("best_Q", -1.0)) or -1.0)
```

**What goes wrong:** The trailing `or -1.0` is applied to the result of `float(...)`, not
to the raw dict value. Python's `or` is falsy-based: `0.0 or -1.0` evaluates to `-1.0`.
If a checkpoint was saved at a point where Q=0.0 (common in early epochs or after a reset),
resuming from that checkpoint sets `best_Q = -1.0` instead of `0.0`. On the next epoch
where Q > -1.0 (i.e., always), the trainer immediately overwrites the `best_Q.pt` file,
breaking the invariant that `best_Q.pt` always holds the best model ever seen.

Same pattern on line 1239 for `best_hierarchy` and line 1240 for `best_coverage` — both
use the same `or 0.0` pattern, which is safe only because 0.0 hierarchy and 0.0 coverage
are the correct floor values. But best_Q with floor -1.0 requires the `or` chain to never
produce a falsy float from a valid checkpoint, which it can.

**Fix:**
```python
# Replace the or-shortcircuit pattern with explicit None-check
_raw_q = ckpt.get("Q") if ckpt.get("Q") is not None else ckpt.get("best_Q", -1.0)
best_Q = float(_raw_q) if _raw_q is not None else -1.0
```

Or more concisely:
```python
best_Q = float(ckpt["Q"]) if "Q" in ckpt else float(ckpt.get("best_Q", -1.0))
```

---

### F-02 — HIGH: Spearman NaN masked as 0 — controller misinterprets collapse

**File:** `src/train.py:566–568`

```python
hierarchy = -spearmanr(valuations, radii).correlation
if np.isnan(hierarchy):
    hierarchy = 0.0
```

**What goes wrong:** `spearmanr` returns `NaN` when all values in either array are
identical — e.g., when `tangent_scale` collapses and all points are projected to the same
Poincaré radius, or when all items in the batch happen to share the same valuation level.
Replacing NaN with 0.0 is wrong in two ways:

1. **Indistinguishable from zero hierarchy:** The LR controller receives `hierarchy=0.0`
   and interprets it as "no hierarchy structure formed yet", potentially triggering state
   transitions (e.g., unfreezing encoder_b early) when the real situation is that the
   metric is undefined because all radii collapsed to a constant.

2. **Q metric understated:** `Q = dist_corr + 1.5 * |hierarchy|`. With hierarchy=0 the
   Q is artificially low, preventing best_Q checkpoint saves even if the model's actual
   geometry is valid. The diagnostic problem: the trainer log shows `Q=0.78` but it may
   mean "radii collapsed" not "hierarchy absent".

**Fix:** Log NaN explicitly before substituting; add a separate `radii_collapsed` flag to
the metric dict so the controller can distinguish collapse from low-hierarchy:
```python
hierarchy_raw = -spearmanr(valuations, radii).correlation
hierarchy_collapsed = bool(np.isnan(hierarchy_raw))
hierarchy = 0.0 if hierarchy_collapsed else hierarchy_raw
# Pass hierarchy_collapsed to TrainingMetrics so controller can gate on it
```

---

### F-03 — HIGH: Missing anchor checkpoint silently trains from scratch

**File:** `src/train.py:314–318`

```python
else:
    print(f"  [WARN] Checkpoint not found: {ckpt_path}")
    self.audit_log["checkpoint_loaded"] = False
    if not force:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
```

**What goes wrong:** The `raise` is inside `if not force:`. When invoked with
`--force`, the model silently proceeds with random initialization even though an explicit
anchor path was configured. For a researcher continuing training from a specific checkpoint,
`--force` is a debugging flag that should override validation errors, not silently discard
the intent to load a pre-trained model.

Additionally, even without `--force`, the error propagates to the caller at line 2076–2080:
```python
except Exception as e:
    print(f"\n[ERROR] Model validation failed: {e}")
    if not args.force:
        sys.exit(1)
    model = None  # continues training with None model → crash later
```

So with `--force` and a bad checkpoint path: model is set to `None`, training reaches
`if model is None: sys.exit(1)` at line 2094. This is consistent, but the user sees
two separate prints and no clear indication of what failed.

**Fix:** Separate "allow force" (validation errors) from "allow missing checkpoint" (a
data-integrity issue). The checkpoint path failure should always be a hard stop when
explicitly configured.

---

### F-04 — MEDIUM: AC one-time warning; silently zero for all subsequent epochs

**File:** `src/losses/combined.py:628–638`

```python
elif self.angular_coherence is not None and r is None and not self._ac_warned_no_r:
    warnings.warn("AngularCoherenceLoss is enabled ... r=None ... ZERO gradient.", ...)
    self._ac_warned_no_r = True
```

**What goes wrong:** After the first forward pass, `_ac_warned_no_r=True` permanently.
All subsequent epochs produce zero gradient from the AC loss with no log entry. In a 1500-
epoch run, the researcher sees one warning at epoch 0 and then nothing. If they miss the
warning (e.g., redirected to log file), they may analyse TensorBoard and see `AC_loss=0`
for 1500 epochs and not know why.

The condition is triggered when `model.factored=False` but `angular_coherence.enabled=true`.
With the current `v7_large.yaml` this cannot occur (`factored=true`), but a future config
or preset could re-introduce it.

**Fix:** Log the zero-AC-gradient condition every `eval_every` epochs, or add it as a
metric to TensorBoard so it's visible in the training history.

---

### F-05 — MEDIUM: Broad `except Exception` in model health check allows broken model through

**File:** `src/train.py:369–372`

```python
except Exception as e:
    print(f"  [ERROR] Model health check failed: {e}")
    if not force:
        raise
```

**What goes wrong:** Any exception during the gradient-flow check (forward pass + backward)
is caught here. With `--force`, the health check failure is printed but execution continues.
The function then returns the model without zeroing gradients (line 367 only runs inside the
`try`), meaning the model's parameters carry stale gradients into the first real training
step. If the health-check failure was a CUDA error, the gradients may be invalid tensors.

The first real training step then calls `optimizer.zero_grad()` at line 1292, which does
clear them, so in practice this is low-risk. But the pattern of `return model` unconditionally
after a caught exception is fragile.

**Fix:** Zero gradients in `finally:` block, and re-raise all CUDA errors regardless of `force`.

---

### F-06 — MEDIUM: LR group name mismatch silently leaves groups at base_lr

**File:** `src/models/lr_controller.py:448–451`

```python
for group in optimizer.param_groups:
    name = group.get('name', '')
    if name in lr_scales:
        group['lr'] = base_lr * lr_scales[name]
```

**What goes wrong:** If a param group has no `name` key, or if the name in the group
(`encoder_a`) doesn't match the key in `lr_scales` (also `encoder_a`), the group's LR
is never updated by the controller. It stays at whatever the scheduler set last. This
is a silent failure: the controller appears to run (no error), but the differential LR
scheme is completely bypassed.

This is currently safe because `get_param_groups()` in vae.py explicitly sets names
that match the lr_scales dict keys. But it's one refactor away from breaking silently.

**Fix:** After the update loop, log any `lr_scales` keys that were not matched to any
group, and any optimizer groups that have a name not in `lr_scales`. Add an assertion
in tests that the names round-trip.

---

### F-07 — MEDIUM: Checkpoint load exception swallowed without re-raise on continue

**File:** `src/train.py:309–313`

```python
except Exception as e:
    print(f"  [WARN] Checkpoint load failed: {e}")
    self.audit_log["checkpoint_loaded"] = False
    if not force:
        raise
```

**What goes wrong:** Same pattern as F-05. With `--force`, a corrupted checkpoint
(disk read error, wrong format, version mismatch) is silently discarded and training
proceeds from random initialization. No mechanism distinguishes "checkpoint not found"
from "checkpoint found but corrupt" in the audit log — both set `checkpoint_loaded=False`.

**Fix:** Differentiate in `audit_log`: `"checkpoint_error": str(e)` so post-hoc inspection
can identify the cause.

---

### F-08 — MEDIUM: `torch.corrcoef` on 1-element tensors silently returns NaN metric

**File:** `src/losses/padic_geodesic.py:227–229`

```python
with torch.no_grad():
    corr = torch.corrcoef(torch.stack([d_actual, d_target]))[0, 1]
    if torch.isnan(corr):
        corr = torch.zeros(1, device=z_hyp.device, dtype=torch.float64)
```

**What goes wrong:** When fewer than 2 valid pairs remain after filtering (e.g., small
batch or highly uniform valuation distribution), `d_actual` is a 1-element tensor.
`torch.corrcoef` on shape `(2, 1)` input requires at least 2 observations per row and
returns a `(2, 2)` matrix of NaN. The code catches and replaces with 0, but:

1. The TensorBoard metric `Losses/geodesic_corr` shows 0 for these batches, which is
   misleading (0 means "no correlation" not "insufficient data").
2. If this happens every batch, the geodesic loss never logs a meaningful correlation,
   making it impossible to tell if the loss is working.

This is inside `torch.no_grad()` so the loss computation (outside this block) is
unaffected — only the metric is corrupted.

**Fix:** Guard with `if d_actual.numel() < 2: corr = float("nan"); metric_valid = False`.
Log `Losses/geodesic_corr_valid_batches` as a separate counter.

---

### F-09 — MEDIUM: Division by `n_batches=0` if DataLoader returns empty iterable

**File:** `src/train.py:1429–1430`

```python
avg_train_loss = train_loss_sum / n_batches
avg_train_acc = train_acc_sum / n_batches
```

**What goes wrong:** If the training DataLoader yields zero batches (e.g., dataset is
empty after a filter, or a custom sampler returns an empty iterator), `n_batches` stays
at 0 and this line raises `ZeroDivisionError`. The error is unhandled and kills training
at the end of the first epoch without saving a checkpoint.

Currently impossible with the ternary dataset (19,683 fixed items) but would silently
break any attempt to swap in a filtered or custom dataset.

**Fix:** Add `assert n_batches > 0, f"No training batches produced (epoch {epoch})"` after
the batch loop, before the division.

---

### F-10 — MEDIUM: Deep-key typo in `statenet_config.py` produces silent defaults

**File:** `src/config/statenet_config.py:158–175`

```python
if 'coverage' in d:
    cov = d['coverage']
    config.coverage = CoverageThresholds(
        fix_threshold=cov.get('fix_threshold', 0.995),
        ...
    )
```

**What goes wrong:** If a user misspells a *sub-key* within a known section (e.g.
`statenet.coverage.fix_threshhold:` instead of `fix_threshold:`), the outer key
`coverage` is present and passes the top-level unknown-key guard. The inner `cov.get()`
silently returns the default. The misspelled key is not detected. The user's intentional
override is lost.

The pydantic schema (`StateNetConfigSchema`) uses `StrictConfigModel` which has
`extra="forbid"` — but only for the top-level `statenet.*` keys. The `StateNetConfig.from_dict`
path bypasses pydantic entirely and uses plain dict `.get()`.

**Fix:** Replace the manual `from_dict` parsing with pydantic validation (which already
has `extra="forbid"` at all nesting levels). The schema is already fully specified in
`schema.py`; the manual `from_dict` is a duplication that diverges from it.

---

### F-11 — MEDIUM: `dist_corr` NaN→0 masking in pairwise distance correlation

**File:** `src/train.py:586–588`

```python
dist_corr = spearmanr(r_dists[triu_idx], v_dists[triu_idx]).correlation
if np.isnan(dist_corr):
    dist_corr = 0.0
```

**What goes wrong:** If all sampled radii are identical (tangent_scale collapse),
`r_dists` is an all-zero matrix, `r_dists[triu_idx]` is all zeros. `spearmanr` returns
NaN when one variable is constant. Replacing with 0.0 makes `Q = 0.0 + 1.5 * |hierarchy|`,
which loses the `dist_corr` contribution entirely, silently understating Q during collapse.

This is the pairwise counterpart to F-02. Both should be handled with the same
`collapsed` flag pattern.

---

### F-12 — LOW: AC warning flag never reset; stale after model swap

**File:** `src/losses/combined.py:302–308`

```python
if self.angular_coherence is None:
    self._ac_warned_no_r = True   # nothing to warn about
else:
    self._ac_warned_no_r = False  # emit the missing-r warning at most once
```

**What goes wrong:** `_ac_warned_no_r` is set at construction time. If the user creates
a `CombinedLoss` instance, passes `r=None` once (triggering the warning and setting
`_ac_warned_no_r=True`), then later in the same run changes the model to factored mode
and passes a valid `r` tensor — the warning flag is now correct (no spurious warning).
But if the opposite happens (was factored, flag stays False, model later becomes
non-factored), the warning fires again correctly. So this is symmetric and safe.

The actual risk is: the flag is per-instance. If two `CombinedLoss` instances are created
(train and validation, or loss_fn and loss_fn_b), each has its own flag. `loss_fn_b` gets
its own flag and would also emit one warning. This is expected behaviour but produces two
confusingly similar warnings.

**Fix:** Convert to a module-level warning (using `warnings.warn` with the same `stacklevel`
and a unique `message`, relying on Python's built-in de-duplication by source location).

---

### F-13 — LOW: TensorBoard writer not closed on unhandled exception

**File:** `src/train.py:1054`

```python
atexit.register(tb_logger.close)
```

**What goes wrong:** `atexit` callbacks run when the interpreter exits normally or after
an unhandled exception reaches the top level. However, if the training loop is killed via
`SIGKILL` (e.g., OOM killer on Linux) or if the process is hard-killed, `atexit` does
not run and the TensorBoard protobuf buffer is not flushed. The last few seconds of
metrics may be lost.

Additionally, if an exception occurs inside the `atexit` callback itself (e.g.,
`tb_logger.close()` fails because the writer is already closed), Python silently ignores
it and prints a message to stderr.

**Fix:** Wrap the main training call in a `try/finally` that explicitly calls
`tb_logger.close()`:
```python
try:
    results = train(...)
finally:
    tb_logger.close()
```

---

## False Positives (Investigated and Dismissed)

| Claim | Why it is NOT a bug |
|-------|---------------------|
| `logits_B=None` crashes loss_fn_b (Finding 2.1 from initial audit) | `combined.py:480-483` correctly sets `_call_logits=None` when `coverage_weight=0.0`, which is what `loss_fn_b` always has. The RichHierarchyLoss does not dereference None logits when coverage is 0. |
| K-means with n_clusters=1 inflates ARI | `train.py:1556` correctly guards `if km_k < 2: continue`. Finding was wrong — the guard exists. |
| `float(ckpt.get("Q", nested_get) or -1.0)` crashes on `None` | The inner default chain `ckpt.get("best_Q", -1.0)` always returns a float, so `float(...)` never receives `None`. However F-01 documents the separate falsy-zero bug. |
| `_ac_warned_no_r` flag stuck across model reload | Loss instances are re-created per run; the flag is per-instance. No cross-run contamination. |

---

## Recommendations (Priority Order)

1. **Fix F-01** immediately — corrupts resume semantics in any run where Q=0 was checkpointed.
2. **Fix F-02 + F-11** together — add `collapsed` flag to metric dict so controller and logs
   can distinguish radius-collapse from genuine low hierarchy.
3. **Fix F-06** — add name-matching assertion in `update_optimizer_lr_scales` and a test that
   verifies all group names appear in `lr_scales`.
4. **Migrate F-10** — retire the manual `StateNetConfig.from_dict` dict-walking in favour of
   the pydantic `StateNetConfigSchema` which already has `extra="forbid"` at all depths.
5. **Fix F-09** — one-line assert before the division is zero cost and catches a whole class
   of future dataset swapping bugs.
6. **Fix F-08** — make `corrcoef` metric validity explicit with a counter; misleading 0 is
   worse than logged NaN.
7. **Fix F-13** — wrap `train()` call in `try/finally` for `tb_logger.close()`.

---

## Run 10 Observation (from live training)

During this audit session, Run 10 (`v7_large.yaml`, 1500 epochs, factored latent) was
executed for the first time. At epoch 1055 the metrics are:

| Metric | Value |
|--------|-------|
| Coverage | 0.996 |
| Hier A/B | 0.839 / 0.839 |
| Q | 2.162 |

This is converging to the structural ceiling of Q≈2.163 — the same ceiling as V6 — consistent
with the NEXT-STEPS-ROADMAP finding that the ceiling is data-structural (66% ops at v=0 tied
Spearman ranks), not architectural. The factored-latent V7 representation did not break the
ceiling under this config. Next step: positional significance encoding (Step 4A in roadmap).
