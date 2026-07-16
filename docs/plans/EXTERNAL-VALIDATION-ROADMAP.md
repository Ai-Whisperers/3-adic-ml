# External Validation Roadmap

**Date:** 2026-07-15 (updated 2026-07-16)
**Status:** Partially attempted, unevenly — see "Prior art found in `scripts/`"
below. Not a committed plan; still requires a decision on which thread (if
any) to pursue deliberately.

## The question

Does mapping 3-adic valuation to hyperbolic radius actually buy anything
useful, or is it an elaborate self-consistent exercise where the geometric
"discoveries" are guaranteed by construction?

## Why this is a fair question

The mathematical premise is legitimate: ultrametric trees genuinely fit
hyperbolic geometry well (the same idea underlies Poincaré embeddings for
real-world taxonomies), and p-adic integers have a real tree structure
(Bruhat–Tits tree). That part isn't in doubt.

What's in doubt is what the current results actually demonstrate. `v_3(n)`
is a deterministic, trivially-computable function of the index `n` — it is
not inferred from data, it's handed directly to the loss functions as a
target. `RadialHierarchyLoss`, `MonotonicRadialLoss`, and `RichHierarchyLoss`
all explicitly push each point's radius toward a `target_radius(v_3(n))`
computed from a fixed formula. `AngularCoherenceLoss` similarly pulls
same-digit-prefix points together directionally. So results like "radius is
perfectly ordered by valuation" (Spearman ceiling documented at 0.8335,
structural per `docs/plans/NEXT-STEPS-ROADMAP.md`) or "ARI=1.0 between
learned direction clusters and `digit_prefix_class`" (V21.0, see CLAUDE.md)
are close to *proving the loss functions do what they were written to do*,
not evidence that hyperbolic geometry helped the model discover a hierarchy
it wasn't told about.

This doesn't mean the codebase is wrong or the work was wasted — the
architecture, loss composition, and numerical stability work (V6→V24) are
all sound engineering, and the project is a legitimate demonstration that a
VAE *can* be steered into a hyperbolic geometry matching a known p-adic
hierarchy. It means that demonstration and a *useful application* are two
different milestones, and only the first one has been reached so far.

## What would change the verdict

1. **Move off the synthetic ternary domain.** Apply the same architecture to
   a real dataset with a hierarchy that is known (for evaluation) but *not*
   injected directly into the loss as a per-sample target — i.e. something
   closer to weak/indirect supervision than `target_radius(v_3(n))`. The
   existing `SurrogatePropertyLoss` (net hydropathy / transition complexity
   targets in `src/losses/surrogate.py`) already gestures at a protein/codon
   sequence use case, but as of this writing it's trained only on the
   synthetic ternary operation set (`{-1,0,1}^9`), not real biological
   sequences. Wiring it to a real dataset (e.g. actual codon sequences with
   known secondary structure or taxonomic hierarchy) would be the natural
   first step.

2. **Benchmark against baselines on a downstream task.** Without a
   head-to-head comparison — same dataset, same task, (a) plain Euclidean
   VAE, (b) hyperbolic VAE without p-adic structure, (c) this p-adic +
   hyperbolic approach — there's no way to tell whether the p-adic/hyperbolic
   prior earns its complexity (better accuracy, fewer samples needed, more
   interpretable latents) versus a simpler baseline achieving the same
   downstream result. No such comparison exists yet.

## Explicitly out of scope for now

Per user direction (2026-07-15): this is being documented for future
consideration, not acted on. The active work at time of writing is a
module-by-module maintainability pass over the existing codebase
(`src/training/`, `src/losses/`, `src/utils/` done; `src/models/`,
`src/config/`, `src/core/`, `src/geometry/`, `src/analysis/`,
`src/symbolic/` pending).

## Prior art found in `scripts/` (2026-07-16)

A maintainability pass over `scripts/` (49 files) turned up real, already-written
attempts at both items above — this section exists so a future session doesn't
re-derive them from scratch, and doesn't overestimate their rigor either.

**Item 1 (real, non-synthetic data) — three different attempts, uneven quality:**

- `scripts/data/prep_human_tp53.py` + `prepare_rosetta_dataset.py`: build a
  mixed dataset (19,683 synthetic ops + human TP53 codon windows + curcumin/
  ginger peptide windows) into `data/rosetta_indices.pt`. `runs/` and `/data/`
  are gitignored, so none of the checkpoints these scripts reference
  (`v15.0`, `v16.0`, `v17.1`, `v19.0`, …) survive in a fresh clone — this
  thread is not currently reproducible, only the data-prep code is.
- `scripts/analysis/validate_humsavar_tp53.py`, `validate_clinical_benchmark.py`,
  `scan_human_proteome.py`, `probe_foreign_genome.py`, `qspr_bioactivity_scoring.py`:
  each tests 2–6 hardcoded sequences with no train/test split and no baseline
  comparison (roadmap item 2, below, was never done for any of these).
  `validate_humsavar_tp53.py`'s "pathogenic mutation" is a sequence typed by
  hand in a comment (`# Mapping simulation: Based on Humsavar p.Cys141Tyr`),
  not loaded from a real HumSavar record — treat any of these scripts'
  historical output as illustrative, not as evidence.
- `scripts/analysis/probe_anomaly_detection.py` had its own inline
  reimplementation of the codon→ternary-index conversion with the *same*
  reversed-digit-order bug documented in
  `docs/DATA-SEMANTICS.md`/[[project-codon-index-bug-fixed]] — it didn't
  import the already-fixed canonical `seq_to_ternary_index`. Fixed
  2026-07-16 (now imports the shared, correct function). Any anomaly-detection
  numbers this script produced before the fix should be treated as invalid —
  the model was scoring sequences with scrambled nucleotide positions.
- `scripts/analysis/filesystem_transfer.py` is the one thread that actually
  matches item 1's spirit (a domain with a hierarchy that's independently
  computable but never injected into the loss): it tests whether the
  trained geometry transfers to filesystem-path depth, an unrelated domain.
  It was fully broken (`ctypes.CDLL` pointed at `scripts/src/c/ternary_hash.so`
  instead of `src/c/ternary_hash.so` — an off-by-one in `parents[N]`) — fixed
  2026-07-16, confirmed working end-to-end after compiling
  `src/c/ternary_hash.c` locally (build artifact, gitignored, not committed).
  Nobody has run it against a real trained checkpoint yet, only against random
  init (expected null result).

**A relevant negative result already exists:** `scripts/validation/check_zero_count_semantics.py`
is a rigorous, self-skeptical check (explicit null hypothesis + random-permutation
baseline) asking whether `zero_count_valuation` — an alternative hierarchy signal
derived from the operation's truth table instead of the index `n` — correlates
with real algebraic structure (commutativity, associativity, identity/absorbing
elements, etc.). Its own verdict, run 2026-07-16: **"NO — same-level operations
are NOT more algebraically similar than random"** (lift 1.03x vs. 1.00x random
baseline). This is exactly the kind of self-critical validation item 1 calls
for, already in the repo, and it already returned a negative result for one
candidate hierarchy signal.

**Item 2 (baseline comparison) — still not done anywhere.** No script in
`scripts/` runs a Euclidean-VAE or non-p-adic-hyperbolic-VAE baseline
side-by-side with this architecture on any dataset, synthetic or real.

**`scripts/analysis/project_audit.py`** (1026 lines, the largest and most
carefully engineered file in `scripts/`) is worth knowing about independent of
the above: it's an explicitly self-skeptical audit tool ("treats `results.json`
as run-log evidence, not proof of full-domain behavior") with a real
retrieval-ablation suite, symbolic-orbit retrieval benchmark, and transparent
Monte Carlo scenario modeling. If this roadmap is picked up again, its
`representation_probe_suite`/`retrieval_ablation_suite` machinery is the most
reusable starting point for building the item-2 baseline comparison.
