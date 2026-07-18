# Plan: Taxonomy-Conditioned Hyperbolic Embedding ("Condition D")

**Date:** 2026-07-17
**Status:** Proposed — not started, no code written yet.
**Prerequisite:** `docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md` (Fases 1-5, done)
and `docs/plans/EXTERNAL-VALIDATION-ROADMAP.md`'s 2026-07-17 negative result.

## Why this plan exists

The Fase 1-4 result (A/B/C vs. `raw_encoding_baseline`) answered "does the
p-adic index curriculum help on real data?" — no, and C actively hurt.
It did **not** answer a different, narrower question that
`EXTERNAL-VALIDATION-ROADMAP.md` explicitly left open: hyperbolic geometry
*is* mathematically well-suited to tree-like structure (that part "isn't in
doubt" per the roadmap's own framing) — but nothing trained so far ever
pointed a hyperbolic loss at the *real* tree (taxonomy). C pointed it at
`v_3(index)`, a self-referential target with no causal link to species.
B pointed it at nothing tree-shaped at all (plain KL, no geodesic term).
Neither condition tests "hyperbolic geometry shaped toward the actual
right answer." That's what Condition D is: same evaluation harness,
same dataset, but the geodesic target during training is real
`taxonomic_distance.npy` instead of `v_3(index)`.

`docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md`'s Fase 3 already named and
deliberately deferred this exact idea ("variante D … queda deliberadamente
fuera de alcance hasta tener resultados de las 3 primeras" — reapuntar
`target_radius` a la taxonomía real vía el `valuation_type` plegable). We
now have those results, so this plan picks it back up — and corrects one
detail from that earlier note: the pluggable `valuation_type` mechanism in
`src/losses/combined.py:120`/`src/core/ternary.py:get_valuation_fn` maps a
*single index to a scalar depth* (0-9), which is the wrong shape for
taxonomy — taxonomic distance is a *pairwise* relationship between species,
not a per-index depth. The right lever is `PAdicGeodesicLoss`
(`src/losses/geodesic.py`), which already targets pairwise Poincaré
distance directly — Condition D swaps its `d_target` source, not the
per-index valuation function.

## What would make this worth doing

A real answer to a cleaner question than Fase 1-4 asked: **given the same
architecture and the same tiny, coarse dataset, does explicitly supervising
hyperbolic distance with the real answer generalize better than trivial
sequence-identity, on species the model never saw?** If yes: the first real
positive evidence in this project that the hyperbolic/tree-shaped prior
earns its complexity. If no: the negative result gets *stronger* — it would
mean even directly telling the model the right answer doesn't help on this
data, which points at dataset scale/encoding coarseness as the limiting
factor, not the loss target.

## Fase 0 — cheap gut check before building anything (hours, not days)

Before touching the VAE at all: fit a classic Poincaré embedding (Nickel &
Kiela 2017 style — no encoder, no reconstruction, just N=39 free points on
the ball directly optimized against `taxonomic_distance.npy` via the
existing `poincare_distance`/`poincare_distance_matrix` in
`src/geometry/poincare.py`) and Mantel-test it the same way
`evaluate_phylogeny_recovery.py` does.

**Why first:** this isolates "can hyperbolic geometry fit *this specific*
39-species distance matrix well at all" from every VAE/encoder/dataset-size
confound. If even a maximally-flexible direct embedding (no bottleneck
through 429 coarse windows) can't clear `raw_encoding_baseline` (0.7228),
the ceiling is in the taxonomy data/Mantel-test noise at n=39, not in
anything Condition D could fix — and the rest of this plan should be
shelved, not built. This is ~50 lines, reuses existing geometry code, no
new loss class, no new script beyond a small standalone check.

**New file:** `scripts/analysis/probe_direct_poincare_embedding.py`

## Fase 1 — `TaxonomyGeodesicLoss`

Only proceed here if Fase 0 clears the baseline.

**New class in `src/losses/geodesic.py`** (same file as `PAdicGeodesicLoss`,
same `HierarchyLossBase` base, same `sample_random_pairs` utility — mirrors
`PAdicGeodesicLoss.forward()` structurally):

```python
class TaxonomyGeodesicLoss(HierarchyLossBase):
    """Like PAdicGeodesicLoss, but d_target comes from real taxonomic
    distance between the *species* each sample's window belongs to, not
    from v_3(index). Requires a per-sample species_id tensor (not just the
    ternary index) and the taxonomic_distance.npy matrix from Fase 1 of
    PHYLOGENY-VALIDATION-PIPELINE.md."""

    def __init__(self, taxonomic_distance: torch.Tensor, max_target_distance: float = 3.0,
                 n_pairs: int = 500, use_smooth_l1: bool = True):
        ...

    def forward(self, z_hyp, species_ids, **kwargs):
        i_idx, j_idx = sample_random_pairs(z_hyp.size(0), self.n_pairs, z_hyp.device)
        d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], cur_c)
        tax_dist = self.taxonomic_distance[species_ids[i_idx], species_ids[j_idx]]
        d_target = self.max_target * tax_dist / tax_dist.max()
        loss = F.smooth_l1_loss(d_actual, d_target) if self.use_smooth_l1 else F.mse_loss(d_actual, d_target)
        ...
```

Needs `species_ids` (int tensor, one entry per window, aligned to
`indices.pt`) which doesn't exist yet as a first-class artifact —
`window_map.json` has `species` as a string per window; add
`scripts/data/prepare_cytochrome_c_dataset.py` output
`data/cytochrome_c/species_ids.pt` (species name → integer index, same
`species_order.json` ordering `taxonomic_distance.npy` already uses) as a
small addition, not a rewrite.

## Fase 2 — species-level holdout (finally implemented, not just flagged)

Every prior condition (A/B/C) trained on windows from all 39 species — the
"no species-level holdout" caveat in `evaluate_phylogeny_recovery.py` has
been open since Fase 4. Condition D is the right place to close it, since
it needs new data-loading code anyway:

- Hold out ~8 species (~20%) chosen once, seeded, stratified across the
  taxonomic range already computed (not just random — e.g. one bacterium,
  one plant, one invertebrate, one close-to-human mammal, etc., so the
  holdout isn't accidentally all-one-clade).
- Train Condition D only on the remaining ~31 species' windows.
- At evaluation, report Mantel correlation separately for (a) held-in
  species pairs, (b) held-out-species-only pairs, (c) mixed pairs. (b) is
  the number that actually answers "did this generalize."
- Recompute `raw_encoding_baseline` restricted to the same held-out-species
  subset for a fair comparison — the existing 0.7228 number used all 39
  species and isn't directly comparable to a held-out-only correlation.

**Modified:** `scripts/data/prepare_cytochrome_c_dataset.py` (optional
`--exclude-species` flag, or a sibling script that filters before saving
`indices.pt`) — data-level split, not a loss-level trick, so the model
genuinely never sees held-out windows.

## Fase 3 — train + evaluate

**New file:** `scripts/applications/train_taxonomy_conditioned.py`, following
the Condition A precedent (`train_euclidean_baseline.py`): a standalone
loop, not routed through `src/train.py`/`CombinedLoss`, because Condition D
needs species-aware batching (`CombinedLoss` and the shared `DataAuditor`
split only know about ternary index, not species) and because forcing
StateNet/Lagrangian/algebraic-loss machinery designed for the 19,683-op
synthetic domain onto this would be more adaptation work than a direct loop.
Reuses `TernaryVAEV6Controllable` (single VAE head is enough — no need for
the dual-VAE split Condition B/C use, since there's no coverage-vs-hierarchy
role split to justify it here) with: reconstruction (coverage) +
`hyperbolic_kl` (same as Condition B) + `TaxonomyGeodesicLoss`.

**Modified:** `scripts/analysis/evaluate_phylogeny_recovery.py` — add
`embed_condition_d` (loads the standalone checkpoint, same aggregation
pattern as `embed_condition_a`) and the held-in/held-out split reporting
from Fase 2.

## Success criteria

| Check | Bar |
|---|---|
| Fase 0 (direct embedding, no VAE) | Must beat 0.7228 to justify Fase 1-3 at all |
| Condition D, held-in species | Informative but not the real test (same leakage shape as A/B/C) |
| Condition D, held-out species only | Must beat `raw_encoding_baseline` recomputed on the same held-out subset — this is the actual bar |

## What NOT to do

- Don't skip Fase 0 to save time — it's the cheapest possible falsifier and
  building Fases 1-3 first risks discovering the same n=39 ceiling after
  much more engineering.
- Don't reuse Condition C's full loss curriculum with the target swapped —
  that reintroduces every p-adic-specific loss (algebraic, angular
  coherence, monotonic radial) that has no justification once the geodesic
  target is taxonomy instead of `v_3(index)`. Condition D should stay as
  minimal as Condition B plus the one new geodesic term, so a positive or
  negative result is attributable to the taxonomy-conditioning change
  specifically.
- Don't declare a result from held-in-species correlation alone — Fase 2's
  entire point is that held-in numbers are not evidence of generalization.

## Effort estimate

Fase 0: hours. Fases 1-3: comparable to Fases 1-4 of the phylogeny pipeline
combined (new loss class, new data artifact, new standalone script, eval
script extension) — call it the same order of magnitude, days not weeks,
given the dataset is tiny and training itself takes minutes on the RTX 3050.

---

## Result (2026-07-17): Fase 0-3 executed, held-out generalization test negative

**Fase 0** (`scripts/analysis/probe_direct_poincare_embedding.py`, 39 free
points, no VAE): Spearman=0.9057 vs. `raw_encoding_baseline`=0.7228,
non-overlapping bootstrap CIs, no directional collapse (mean pairwise cosine
similarity ≈ -0.01), radii ordered sensibly (mammals near origin, bacteria
pushed to the boundary). **Gate passed** — proceeded to Fase 1-3.

**Fase 1** (`TaxonomyGeodesicLoss`, `src/losses/geodesic.py`): implemented,
manually verified (finite nonzero gradient, same-species pairs correctly
target distance 0, missing `species_ids` raises instead of silently
no-op-ing). Exported, full test suite unaffected.

**Fase 2** (`scripts/data/select_holdout_species.py`): 9/39 species (23%)
held out, stratified across all 5 kingdom groups present (Metazoa, Fungi,
Viridiplantae, Bacteria, unranked protists) — includes both a "near" case
(*Pan troglodytes*, one node from human) and a "far" case (*Pseudomonas
aeruginosa*, leaving only one other bacterium in training). Zero overlap
with the 30-species training set, verified directly.

**Fase 3** (`scripts/applications/train_taxonomy_conditioned.py`, 500
epochs, RTX 3050, ~5 min): trained successfully (reconstruction accuracy
~84-90% on the row-level monitoring split; `tax_dist_corr` on training
batches climbed from ~0 to 0.4-0.8 range, confirming the geodesic loss was
having a real effect). Extended `evaluate_phylogeny_recovery.py` with
`embed_condition_d` and `split_evaluate_condition_d` (held-in / held-out /
mixed reporting).

**The real test — full run, n_permutations=9999:**

| Split | Spearman | Mantel p | Bootstrap 95% CI | n pairs |
|---|---|---|---|---|
| D, held-in (30 species, seen in training) | 0.8404 | 0.0001 | — | 435 |
| D, held-out (9 species, never in training) | **0.5091** | 0.0028 | **[-0.124, 0.874]** | 36 |
| raw_encoding_baseline, same held-out subset | **0.7803** | 0.0023 | [-0.035, 0.944] | 36 |
| D, held-in vs. held-out (mixed, descriptive only) | 0.6870 | — (no permutation test built) | — | 270 |

**Verdict: negative on the test that matters, but underpowered.** Condition
D beats `raw_encoding_baseline` overall (0.7566 vs. 0.7228) and strongly on
held-in species (0.8404) — unsurprising, since held-in species were directly
supervised with real taxonomic distance during training. On the actual
generalization test — species the model never saw — D scores **lower** than
doing no training at all (0.509 vs. 0.780). Both point estimates come with
very wide, overlapping bootstrap CIs at n=9 held-out species (36 pairs is
little data for a permutation test), so this should be read as "D shows no
detectable held-out generalization advantage," not "D is definitively worse"
— the sample is too small to distinguish those confidently.

**Interpretation:** consistent with an encoder that fit the training
species' positions rather than learning a rule that transfers to new
sequences — plausible given only 30 species / 330 windows and the same
coarse 3-symbol hydropathy encoding that caused the 73.9% cross-species
index collision rate throughout this whole pipeline. Fase 0's positive
result (hyperbolic geometry *can* represent this tree, given unconstrained
free points) does not transfer once the embedding has to come from an
encoder generalizing over noisy, collision-heavy biological input — the
bottleneck was never the geometry, it was the encoder/dataset combination.

**What this does and doesn't settle:** it doesn't rule out that a taxonomy-
conditioned geodesic loss could work with more species, less collision-prone
encoding, or more training data — Fase 0 already showed the geometry itself
isn't the obstacle. It does mean this specific implementation, on this
specific dataset, provides no evidence that directly supervising with real
taxonomy generalizes better than trivial sequence identity. Combined with
the Condition A/B/C result (`EXTERNAL-VALIDATION-ROADMAP.md`), the pattern
across both experiments on this dataset is the same: nothing trained so far
has beaten the zero-model baseline on data it wasn't directly fit to.
