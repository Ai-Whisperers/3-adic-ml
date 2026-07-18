# Plans

This directory contains design proposals, roadmaps, and feature plans. These documents describe intended future work and are not authoritative descriptions of the current codebase.

## Active / pending

| Document | Description |
|----------|-------------|
| [TESTS_CRITICAL_TARGETS.md](TESTS_CRITICAL_TARGETS.md) | Test coverage strategy and critical targets |
| [GRAPH-TOPOLOGY-VISUALIZATION-PLAN.md](GRAPH-TOPOLOGY-VISUALIZATION-PLAN.md) | Topology visualization pipeline plan — Parts 1–2 (scatter refactor, viz pipeline) done; Part 3 (PyTorch Geometric tree graph, `TreeMessagePassingLoss`) not started |
| [ALGEBRAIC-VISUALIZATION-ROADMAP.md](ALGEBRAIC-VISUALIZATION-ROADMAP.md) | Algebraic visualization roadmap — Phase 1 (probing) partially done; Phase 2 (native D3/Three.js Poincaré renderer) and Phase 3 (dashboard/paper) not started |
| [EXTERNAL-VALIDATION-ROADMAP.md](EXTERNAL-VALIDATION-ROADMAP.md) | Open question: does the p-adic/hyperbolic prior help on real (non-synthetic) data vs. baselines? Answered for the cytochrome c thread (2026-07-17, negative) — see PHYLOGENY-VALIDATION-PIPELINE.md and TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md for the narrower follow-up question left open |
| [PHYLOGENY-VALIDATION-PIPELINE.md](PHYLOGENY-VALIDATION-PIPELINE.md) | Cytochrome C phylogeny + 3-condition baseline comparison (Euclidean / generic-hyperbolic / p-adic). Fases 1-5 done, full runs completed 2026-07-17 — negative result, none of A/B/C beat a zero-model baseline |
| [TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md](TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md) | Follow-up ("Condition D"): does pointing the hyperbolic geodesic loss at real taxonomic distance (instead of `v_3(index)`) generalize to held-out species? Proposed 2026-07-17, not started |

## Archived (resolved)

| Document | Resolution |
|----------|-------------|
| [archive/NEXT-STEPS-ROADMAP.md](archive/NEXT-STEPS-ROADMAP.md) | Superseded by V8–V24 progress; both listed pending items (Pydantic validation, Run 10) are done |
| [archive/VALUATION_CONDITIONED_PRIOR.md](archive/VALUATION_CONDITIONED_PRIOR.md) | Phases 1, 2, 3B implemented (`src/losses/prior.py`, `src/losses/lagrangian.py`); 3A explicitly skipped |
| [archive/PYDANTIC_VALIDATION.md](archive/PYDANTIC_VALIDATION.md) | Implemented as `src/config/schema.py` (full schema, beyond the recommended minimal option) |
| [archive/PLAN-PHASE-17.md](archive/PLAN-PHASE-17.md) | Executed — `v17.0`/`v17.1_rosetta_manifold` runs completed |
