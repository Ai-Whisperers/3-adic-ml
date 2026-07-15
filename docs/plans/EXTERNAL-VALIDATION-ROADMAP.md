# External Validation Roadmap

**Date:** 2026-07-15
**Status:** Not started — this document records an open question raised during a
maintainability review, not a committed plan. See discussion context in
conversation history around that date.

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
