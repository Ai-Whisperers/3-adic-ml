# Data Semantics — Indexing-Derived vs Intrinsic Hierarchy

**Date:** 2026-03-23

This document clarifies a subtle but important distinction in how "hierarchy" is defined in
this codebase, explains the v=9 singleton convention, documents the intrinsic properties that
exist but are currently unused, and evaluates dataset expansion options.

---

## 1. The Fundamental Distinction: Indexing-Derived vs Intrinsic

### What the dataset is

The 19,683 "ternary operations" are all integers in [0, 19682], each represented as a
9-digit base-3 number with digits shifted to {-1, 0, 1}. Index k maps to digits:

```
digit[i] = ((k // 3^i) % 3) - 1   for i = 0..8
```

Index 0 → `(-1,-1,-1,-1,-1,-1,-1,-1,-1)` (all -1s)
Index 1 → `( 0,-1,-1,-1,-1,-1,-1,-1,-1)` (first digit 0, rest -1)
Index 6561 → `(-1,-1,-1,-1,-1,-1,-1,-1, 1)` (last digit +1, rest -1)

The 19,683 operations are the **complete space** of 3^9 — not a sample, not a subset.

### Indexing-derived hierarchy

The 3-adic valuation `v_3(k)` measures divisibility of the **index integer k** by powers of 3:

```
v_3(k) = max{ v ≥ 0 : 3^v divides k }
v_3(0) = MAX_VALUATION = 9  (convention; mathematical value is ∞)
```

**This is a property of k as an integer, not of the 9-digit operation it represents.**

- Index 3 → v=1, because 3 divides 3
- Index 9 → v=2, because 9=3^2 divides 9
- Index 7 → v=0, because 3 does not divide 7

The digits of the operation at index 3 are `(0, 0, -1, -1, ...)` — there is nothing
algebraically special about these digits that "deserves" v=1. The hierarchy comes from
the enumeration order, not from the content of the operation.

**All hierarchy losses in this codebase are indexing-derived:**

| Loss | Signal |
|------|--------|
| `PAdicGeodesicLoss` | `v_3(\|i-j\|)` — valuation of index difference |
| `RadialHierarchyLoss` | `v_3(k)` — target radius from level |
| `GlobalRankLoss` | `v_3(k)` — ordinal rank between pairs |
| `MonotonicRadialLoss` | `v_3(k)` — per-level mean radius ordering |
| `RichHierarchyLoss` | `v_3(k)` — per-level target radius |
| `WithinLevelContrastiveLoss` | `v_3(k)` — same-level cohort |

**Only `AngularCoherenceLoss` uses intrinsic digit content** via `digit_prefix_class(k)`,
which reads the actual digit values to form sub-clusters within each valuation level.
See Section 3.

### Why this matters

The model is learning to embed a hierarchical structure that is an artifact of how we
enumerate the space, not an intrinsic property of the algebra. This is a deliberate
research choice — the goal is to test whether p-adic ultrametric structure, imposed
via indexing, can be geometrically encoded in hyperbolic space and whether this encoding
captures meaningful algebraic relationships.

Whether the learned geometry corresponds to any intrinsic algebraic structure in the
operations (e.g., compositional similarity, algebraic closure properties) is an open
empirical question not currently measured by the codebase.

---

## 1b. Algebraic Structure of Each Valuation Level

This is the key structural fact that connects the indexing-derived hierarchy to actual
algebraic content of the operations:

**For all operations at level v=k, exactly the first k digit positions are fixed to -1,
position k is always in {0, +1} (never -1), and positions k+1..8 are completely free.**

Proof: v_3(idx) = k means idx is divisible by 3^k but not 3^(k+1). In the base-3
expansion, this means unshifted digits 0..k-1 are 0 (shifted: -1), unshifted digit k
is 1 or 2 (shifted: 0 or +1, never -1), and digits k+1..8 are unconstrained.

```
Level v=k operation layout:
  positions:  0    1   ...  k-1    k       k+1  ...  8
  values:    -1   -1   ...  -1    {0,+1}  free  ... free
              ←─── k fixed zeros ───→  ↑pivot    ←─ 9-k-1 free ─→
```

Empirically verified for v=0..4 — no operation at level v has digit[v] = -1.

### The class count formula

`digit_prefix_class(idx, depth)` reads the first `depth` digits. For operations at
level v, only positions v..depth-1 vary (0..v-1 are constant -1). The number of
distinct classes is:

```
n_classes(v, depth) = 2 × 3^(depth - v - 1)
```

The factor 2 comes from digit[v] ∈ {0, +1}; the factor 3^(depth-v-1) comes from the
`depth-v-1` free positions after the pivot.

Verified against the actual LUT:

| Level v | Prefix depth k | Formula | Actual | ops/class |
|---------|---------------|---------|--------|-----------|
| v=0 | 3 | 2×3^2=18 | 18 | 729 |
| v=1 | 4 | 2×3^2=18 | 18 | 243 |
| v=2 | 3 | 2×3^0=2 | 2 | 729 |
| v=3 | 4 | 2×3^0=2 | 2 | 243 |
| v=4 | 5 | 2×3^0=2 | 2 | 81 |
| v=5 | 6 | 2×3^0=2 | 2 | 27 |

**The formula determines the only useful prefix depths:**
- `depth = v+1` → 2 classes (binary split on the pivot digit alone)
- `depth = v+2` → 6 classes
- `depth = v+3` → 18 classes

18 classes is the maximum used (matches v=0 structure). Deeper splits are unstable
because ops/class drops below a reliable K-means minimum (~5 for validation).

### What the binary split means algebraically

For levels v≥2, we use the binary split (depth = v+1, 2 classes). This separates
operations by the value of their "pivot digit" — the first position where the operation
is not algebraically constrained to be -1:

- **Class A** (digit[v] = 0): the pivot digit has unshifted value 1 in {0,1,2}
- **Class B** (digit[v] = +1): the pivot digit has unshifted value 2 in {0,1,2}

Both classes always have exactly 50% of the level's operations (81 each for v=4,
243 each for v=3, etc.) and identical freedom in all remaining positions. The split
is the p-adic tree branching at depth v: the two non-zero branches at the first
active trit position.

### Choosing prefix depth for any level

Given level v and available ops N_v:
1. Compute ops/class for each depth option: `N_v / (2 × 3^(depth - v - 1))`
2. Require ops/class ≥ 80 for training stability (AC loss needs enough same-class pairs)
3. Require ops/class ≥ 5 for validation K-means stability
4. Prefer the coarsest depth that gives ≥ 2 stable classes

For the current dataset (3^9, 19683 ops):

| Level | N_ops | Viable depths | Recommended |
|-------|-------|--------------|-------------|
| v=0 | 13122 | v+1(2cls), v+2(6cls), v+3(18cls) | v+3=3 (18 cls) |
| v=1 | 4374 | v+1, v+2, v+3 | v+3=4 (18 cls) |
| v=2 | 1458 | v+1, v+2 | v+1=3 (2 cls — v+2 gives 81/cls, borderline) |
| v=3 | 486 | v+1 | v+1=4 (2 cls) |
| v=4 | 162 | v+1 | v+1=5 (2 cls) |
| v=5 | 54 | v+1 (27/cls, marginal) | v+1=6 (2 cls, borderline) |
| v=6 | 18 | none viable | skip |
| v=7 | 6 | none viable | skip |
| v=8 | 2 | none viable | skip |
| v=9 | 1 | none viable | skip |

---

## 2. The v=9 Singleton

### Why v=9 has exactly 1 operation

Level v=9 contains exactly one operation: index 0.

- `v_3(0) = MAX_VALUATION = 9` by convention (mathematical v_3(0) = ∞, capped at 9)
- `3^9 = 19683 > 19682` → no positive integer in [1, 19682] is divisible by 3^9
- Therefore index 0 is the sole inhabitant of v=9

The complete level population follows the exact formula:

| Level v | Count | Formula |
|---------|-------|---------|
| v=0 | 13122 | 2 × 3^8 |
| v=1 | 4374 | 2 × 3^7 |
| v=2 | 1458 | 2 × 3^6 |
| v=3 | 486 | 2 × 3^5 |
| v=4 | 162 | 2 × 3^4 |
| v=5 | 54 | 2 × 3^3 |
| v=6 | 18 | 2 × 3^2 |
| v=7 | 6 | 2 × 3^1 |
| v=8 | 2 | 2 × 3^0 |
| v=9 | 1 | convention (v_3(0) = ∞, capped) |
| **Total** | **19683** | **3^9** |

General formula for 3^n dataset: level v=k has `2 × 3^(n-1-k)` ops for k=0..n-1,
and 1 op for v=n (the index-0 convention).

### What index 0 represents

The operation at index 0 is `(-1,-1,-1,-1,-1,-1,-1,-1,-1)` — all digits are -1. There
is **no algebraic reason** for this operation to be at the apex of the p-adic hierarchy.
It is there because 3^9 divides 0 (by convention), which is a consequence of the
enumeration starting at 0.

The "deepest" operation in the p-adic tree is purely an enumeration artifact.

---

## 3. Intrinsic Properties — Available but Unused

`TernarySpace` precomputes six intrinsic properties that derive from the actual digit
values of the operation, not from the index:

| Property | Method | Range | Description |
|----------|--------|-------|-------------|
| `digit_count` | `TERNARY.digit_count(k)` | [0, 9] | Number of non-zero digits |
| `digit_sum` | `TERNARY.digit_sum(k)` | [-9, +9] | Sum of {-1,0,1} values |
| `first_nonzero` | `TERNARY.first_nonzero(k)` | [0, 9] | Position of first non-zero digit (9 if all zero) |
| `last_nonzero` | `TERNARY.last_nonzero(k)` | [-1, 8] | Position of last non-zero digit |
| `nonzero_pattern` | `TERNARY.nonzero_pattern(k)` | [0, 511] | 9-bit encoding of which positions are non-zero |
| `valuation_prefix_class` | `TERNARY.valuation_prefix_class(k)` | [0, 5] | Sign of first non-zero digit × value of next digit (6 sub-classes per level) |

**None of these are currently used by any loss function.** They are computed and cached
in the LUT but never referenced in `src/losses/`.

`digit_prefix_class(k, depth)` (used by `AngularCoherenceLoss`) is a semi-intrinsic
classifier: it reads the actual digit values at positions [0..depth-1] but interprets
them as a base-3 number, mixing content with positional significance.

### Potential uses of intrinsic properties

- **Content-based hierarchy**: use `digit_count` as an alternative valuation — operations
  with more non-zero digits could be considered "richer" or "denser"
- **Algebraic sub-clustering**: use `nonzero_pattern` to group operations by their
  zero-structure, independent of sign
- **Cross-level verification**: test whether the indexing-derived hierarchy correlates
  with intrinsic algebraic similarity (e.g., do same-v operations share similar `digit_count`?)

---

## 4. Dataset Expansion — Options and Reasoning

### Why expansion is non-trivial

The current dataset is the **complete space** of 3^9. There is no sampling — every
possible 9-digit ternary operation is present exactly once. Expansion is not about
adding more examples from the same distribution; it requires changing the domain.

### Option A: Increase digit count to 3^n (n > 9)

The most natural extension. Going from 9 to 10 digits:

| Property | n=9 (current) | n=10 | n=12 |
|----------|---------------|------|------|
| Total operations | 19,683 | 59,049 | 531,441 |
| Model input dims | 9 | 10 | 12 |
| MAX_VALUATION | 9 | 10 | 12 |
| v=9 count | 1 (singleton) | 2 | 2 |
| Deepest singleton | v=9 | v=10 | v=12 |
| Training time | baseline | ~3× | ~27× |

**Key effect**: going from n=9 to n=10 makes v=9 have 2 ops (indices 19683 and 39366,
both divisible by 3^9 but not 3^10) and shifts the singleton to v=10. Each increment
adds one level and triples the dataset.

**Constraint**: The RTX 3050 6GB can handle n=10 comfortably (model size, not dataset
size, is the bottleneck — the data itself is <5MB at n=10). n=12 likely requires
architecture changes to handle 531K operations.

**Code changes required**:
- `TernarySpace.N_DIGITS`: 9 → n
- `TernarySpace.N_OPERATIONS`: 19683 → 3^n
- Model input layer: 9 → n dimensions
- No loss changes required (all are valuation-based, dimension-agnostic)

**Domain implication**: A 10-digit operation is a different mathematical object — it
represents a ternary function of 10 variables (or a 10-position ternary word). The
research question must be restated for the new dimensionality. The p-adic structure
is preserved identically.

### Option B: Intrinsic hierarchy relabeling

Keep the dataset as-is but replace indexing-derived valuation with a content-derived
measure such as `digit_count` (number of non-zero digits, range 0–9):

- **Distribution**: binomial-like — C(9,d) × 2^d operations with d non-zero digits
- **Sparsest operations** (d=0): 1 op (index 0, all -1s) — same singleton as v=9
- **Densest operations** (d=9): 2^9 = 512 ops
- **Content-meaning**: "depth" reflects algebraic sparsity, not enumeration order

This is a research hypothesis rather than a dataset expansion. It changes what the
model is asked to learn and would require significant validation of whether content-based
hierarchy produces meaningful geometry.

### Option C: Multi-scale dataset (multiple n simultaneously)

Treat operations of different lengths as a unified dataset with variable-depth trees.
Use n=6, n=7, n=8, n=9 in the same training run with zero-padding.

- Preserves the complete space at each scale
- Naturally trains the model on multiple p-adic tree depths
- Requires architecture that handles variable input length
- Engineering-heavy; no evidence it helps Q

### Recommendation

**For continuity with current work**: Option A at n=10 is the cleanest extension.
It preserves all mathematical structure, triples the data, makes v=9 a proper pair
instead of a singleton, and requires only minor code changes.

**For research novelty**: Option B (intrinsic hierarchy) asks a fundamentally different
question — whether the model can encode algebraic content structure rather than an
enumeration-derived structure. This would validate (or refute) whether the learned
geometry has semantic meaning beyond the indexing artifact.

**Not recommended**: n≥12 until n=10 is validated; the 27× data increase is not
justified without evidence that the added levels improve the geometry.

---

## 5. Implications for Current Metrics

### Q ceiling at 2.163 is indexing-structure-dependent

The Spearman-based Q metric measures correlation between radial position and v_3(k).
Since v_3(k) is indexing-derived and 66% of operations are at v=0, the tied-rank
ceiling is provably at Q≈2.163 regardless of architecture. See
`docs/audits/22-03-2026-Q-CEILING-ANALYSIS.md`.

Breaking this ceiling without changing the data structure requires either:
- A different metric not based on Spearman (e.g., per-level ARI, which is already
  tracked and reflects the actual learned geometry rather than just Spearman rank)
- A different dataset (Option A/B above)

### ARI metrics are less affected by the indexing artifact

Per-level ARI (v=0: 0.970, v=1: 0.905, v=2: 1.000 in Run 5) measures whether the model
groups operations by `digit_prefix_class` within each valuation level. Since
`digit_prefix_class` reads actual digit values, these metrics partially escape the
indexing-derived tautology — they test whether the direction geometry encodes
content-adjacent structure, not just rank order.

---

## Summary

| Aspect | Indexing-Derived | Intrinsic |
|--------|-----------------|-----------|
| **Source** | v_3(k), d_3(i,j) | digit values |
| **Used by** | All hierarchy losses | AngularCoherenceLoss only |
| **v=9** | 1 op (convention) | — |
| **Expandable by** | Increasing n digits | Relabeling with content measures |
| **Q ceiling** | Structural (tied ranks) | Would change with intrinsic labels |
| **ARI metrics** | Partially intrinsic (digit_prefix_class reads digits) | — |
