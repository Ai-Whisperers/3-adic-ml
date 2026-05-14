#!/usr/bin/env python3
"""Semantic validity check for zero_count_valuation hierarchy (Option B / v9).

Each of the 19,683 "operations" is a binary operation f: {-1,0,1}² → {-1,0,1},
with its 9-digit ternary representation encoding the full lookup table:

    digit[k] = f(a, b)   where  a = (k//3) - 1,  b = (k%3) - 1

zero_count_valuation(op) = number of k where digit[k] = 0
                         = number of (a,b) pairs where f(a,b) = 0

Question: Do same-level operations share algebraic structure, or is the
zero-count distribution merely a convenient statistical artifact?

Output: per-level algebraic property table + within-level homogeneity test.

Usage:
    python scripts/validation/check_zero_count_semantics.py
"""

import os
import sys

import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from src.core.ternary import TERNARY

# ─── Input domain ────────────────────────────────────────────────────────────
# Canonical mapping: digit[k] = f(a_k, b_k)
#   a_k = (k // 3) - 1 ∈ {-1, 0, 1}
#   b_k = (k %  3) - 1 ∈ {-1, 0, 1}
# Row = a (first input), Col = b (second input), reading left-to-right top-to-bottom.

K_LOOKUP = np.array([
    # (a, b) -> k:  k = (a+1)*3 + (b+1)
    [0, 1, 2],   # a=-1: k=0,1,2
    [3, 4, 5],   # a= 0: k=3,4,5
    [6, 7, 8],   # a= 1: k=6,7,8
], dtype=np.int32)  # row=a_idx, col=b_idx

# Pairs that must match for commutativity (anti-diagonal off-diagonal pairs)
COMM_PAIRS = [(1, 3), (2, 6), (5, 7)]  # k(a,b) vs k(b,a)

# Diagonal positions: f(a, a) positions for a in {-1, 0, 1}
DIAG_K = [0, 4, 8]   # k(-1,-1), k(0,0), k(1,1)
DIAG_V = [-1, 0, 1]  # expected value for fixed-point


# ─── Build operation table ───────────────────────────────────────────────────

def build_op_table() -> np.ndarray:
    """Return (19683, 9) int8 array of all operation tables, values in {-1,0,1}."""
    all_idx = torch.arange(TERNARY.N_OPERATIONS)
    ops = TERNARY.to_ternary(all_idx).numpy().astype(np.int8)
    return ops


def compute_zero_count(ops: np.ndarray) -> np.ndarray:
    """Count zero digits per operation. Shape: (19683,)."""
    return (ops == 0).sum(axis=1).astype(np.int32)


# ─── Per-operation algebraic properties ──────────────────────────────────────

def compute_properties(ops: np.ndarray) -> dict:
    """Compute boolean algebraic properties for every operation.

    Returns a dict of property_name -> (19683,) bool ndarray.
    """
    N = ops.shape[0]

    props = {}

    # Commutativity: f(a,b) = f(b,a) for all (a,b)
    comm = np.ones(N, dtype=bool)
    for k1, k2 in COMM_PAIRS:
        comm &= (ops[:, k1] == ops[:, k2])
    props["commutative"] = comm

    # Idempotent: f(a,a) = a for all a in {-1,0,1}
    idempotent = (
        (ops[:, DIAG_K[0]] == DIAG_V[0]) &
        (ops[:, DIAG_K[1]] == DIAG_V[1]) &
        (ops[:, DIAG_K[2]] == DIAG_V[2])
    )
    props["idempotent"] = idempotent

    # Fixed-point count: number of a where f(a,a)=a
    fp = (
        (ops[:, 0] == -1).astype(np.int32) +
        (ops[:, 4] ==  0).astype(np.int32) +
        (ops[:, 8] ==  1).astype(np.int32)
    )
    props["fixed_points"] = fp  # integer 0..3

    # Identity element: ∃ e ∈ {-1,0,1}: f(e,a)=f(a,e)=a for all a
    # e=-1 (e_idx=0): row [0,1,2]=[-1,0,1]  col [0,3,6]=[-1,0,1]
    # e= 0 (e_idx=1): row [3,4,5]=[-1,0,1]  col [1,4,7]=[-1,0,1]
    # e= 1 (e_idx=2): row [6,7,8]=[-1,0,1]  col [2,5,8]=[-1,0,1]
    identity_checks = {
        -1: (
            (ops[:, 0] == -1) & (ops[:, 1] ==  0) & (ops[:, 2] ==  1) &
            (ops[:, 0] == -1) & (ops[:, 3] ==  0) & (ops[:, 6] ==  1)
        ),
        0: (
            (ops[:, 3] == -1) & (ops[:, 4] ==  0) & (ops[:, 5] ==  1) &
            (ops[:, 1] == -1) & (ops[:, 4] ==  0) & (ops[:, 7] ==  1)
        ),
        1: (
            (ops[:, 6] == -1) & (ops[:, 7] ==  0) & (ops[:, 8] ==  1) &
            (ops[:, 2] == -1) & (ops[:, 5] ==  0) & (ops[:, 8] ==  1)
        ),
    }
    has_identity = identity_checks[-1] | identity_checks[0] | identity_checks[1]
    props["has_identity"] = has_identity
    props["identity_is_0"] = identity_checks[0]  # e=0 is the additive identity

    # Absorbing element: ∃ z ∈ {-1,0,1}: f(z,a)=f(a,z)=z for all a
    # z=-1: row[0,1,2]=[-1,-1,-1]  col[0,3,6]=[-1,-1,-1]
    # z= 0: row[3,4,5]=[0,0,0]     col[1,4,7]=[0,0,0]
    # z= 1: row[6,7,8]=[1,1,1]     col[2,5,8]=[1,1,1]
    absorb_checks = {
        -1: (
            (ops[:, 0] == -1) & (ops[:, 1] == -1) & (ops[:, 2] == -1) &
            (ops[:, 0] == -1) & (ops[:, 3] == -1) & (ops[:, 6] == -1)
        ),
        0: (
            (ops[:, 3] == 0) & (ops[:, 4] == 0) & (ops[:, 5] == 0) &
            (ops[:, 1] == 0) & (ops[:, 4] == 0) & (ops[:, 7] == 0)
        ),
        1: (
            (ops[:, 6] == 1) & (ops[:, 7] == 1) & (ops[:, 8] == 1) &
            (ops[:, 2] == 1) & (ops[:, 5] == 1) & (ops[:, 8] == 1)
        ),
    }
    has_absorbing = absorb_checks[-1] | absorb_checks[0] | absorb_checks[1]
    props["has_absorbing"] = has_absorbing
    props["absorbing_is_0"] = absorb_checks[0]

    # Constant operation: all 9 outputs equal
    props["constant"] = (ops == ops[:, :1]).all(axis=1)

    # Left projection: f(a,b) = a — same output per row
    # Row a=-1: all digits in {0,1,2} = -1 → d[0]=d[1]=d[2]=-1
    # Row a= 0: d[3]=d[4]=d[5]=0
    # Row a= 1: d[6]=d[7]=d[8]=1
    left_proj = (
        (ops[:, 0] == -1) & (ops[:, 1] == -1) & (ops[:, 2] == -1) &
        (ops[:, 3] ==  0) & (ops[:, 4] ==  0) & (ops[:, 5] ==  0) &
        (ops[:, 6] ==  1) & (ops[:, 7] ==  1) & (ops[:, 8] ==  1)
    )
    props["left_projection"] = left_proj

    # Right projection: f(a,b) = b — same output per column
    # Col b=-1: d[0]=d[3]=d[6]=-1
    # Col b= 0: d[1]=d[4]=d[7]=0
    # Col b= 1: d[2]=d[5]=d[8]=1
    right_proj = (
        (ops[:, 0] == -1) & (ops[:, 3] == -1) & (ops[:, 6] == -1) &
        (ops[:, 1] ==  0) & (ops[:, 4] ==  0) & (ops[:, 7] ==  0) &
        (ops[:, 2] ==  1) & (ops[:, 5] ==  1) & (ops[:, 8] ==  1)
    )
    props["right_projection"] = right_proj

    # Output diversity: number of distinct values used (1, 2, or 3)
    uses_neg1 = (ops == -1).any(axis=1)
    uses_zero = (ops ==  0).any(axis=1)
    uses_pos1 = (ops ==  1).any(axis=1)
    props["output_diversity"] = uses_neg1.astype(np.int32) + uses_zero.astype(np.int32) + uses_pos1.astype(np.int32)
    props["uses_zero_output"] = uses_zero

    # Zero-output symmetry: when f(a,b)=0, is f(b,a)=0?
    # For all anti-symmetric pairs, check that both or neither are zero.
    zero_sym = np.ones(N, dtype=bool)
    for k1, k2 in COMM_PAIRS:
        z1 = (ops[:, k1] == 0)
        z2 = (ops[:, k2] == 0)
        zero_sym &= (z1 == z2)  # both zero or both non-zero
    props["zero_output_symmetric"] = zero_sym

    # Associativity: f(f(a,b),c) = f(a,f(b,c)) for all a,b,c ∈ {-1,0,1}
    # Build full table for vectorized check
    VALS = [-1, 0, 1]
    assoc_mask = np.ones(N, dtype=bool)
    for a in VALS:
        for b in VALS:
            for c in VALS:
                k_ab = (a + 1) * 3 + (b + 1)
                k_bc = (b + 1) * 3 + (c + 1)
                fab = ops[:, k_ab]            # f(a,b) for each op
                fbc = ops[:, k_bc]            # f(b,c) for each op
                # k(fab, c) and k(a, fbc) — values may be -1/0/1
                k_fab_c = (fab + 1) * 3 + (c + 1)
                k_a_fbc = (a + 1) * 3 + (fbc + 1)
                lhs = ops[np.arange(N), k_fab_c]
                rhs = ops[np.arange(N), k_a_fbc]
                assoc_mask &= (lhs == rhs)
    props["associative"] = assoc_mask

    return props


# ─── Level-stratified statistics ──────────────────────────────────────────────

def per_level_stats(zero_counts: np.ndarray, props: dict) -> list:
    """Compute property rates per zero_count level. Returns list of row dicts."""
    rows = []
    for level in range(10):
        mask = zero_counts == level
        n = mask.sum()
        if n == 0:
            continue

        row = {"level": level, "count": n, "pct": 100 * n / TERNARY.N_OPERATIONS}

        # Boolean property rates
        for key in ["commutative", "idempotent", "has_identity", "identity_is_0",
                    "has_absorbing", "absorbing_is_0", "constant",
                    "left_projection", "right_projection",
                    "uses_zero_output", "zero_output_symmetric", "associative"]:
            row[key] = props[key][mask].mean() * 100  # as %

        # Integer properties (mean)
        row["mean_fixed_points"] = props["fixed_points"][mask].mean()
        row["mean_output_diversity"] = props["output_diversity"][mask].mean()

        rows.append(row)
    return rows


# ─── Within-level homogeneity test ────────────────────────────────────────────

def build_property_vector(props: dict, N: int) -> np.ndarray:
    """Stack boolean+integer properties into an (N, D) float32 feature matrix."""
    columns = []
    for key in ["commutative", "idempotent", "has_identity", "has_absorbing",
                "constant", "zero_output_symmetric", "associative"]:
        columns.append(props[key].astype(np.float32))
    columns.append(props["fixed_points"].astype(np.float32) / 3.0)
    columns.append(props["output_diversity"].astype(np.float32) / 3.0)
    return np.stack(columns, axis=1)


def within_level_homogeneity(zero_counts: np.ndarray, feat: np.ndarray, n_sample: int = 5000) -> dict:
    """Compare within-level vs. between-level mean property agreement.

    Samples `n_sample` pairs; reports observed vs. random-permutation baseline.
    Higher ratio = same-level ops are more algebraically similar than random.
    """
    rng = np.random.default_rng(42)
    N = len(zero_counts)

    # Sample random pairs
    i_idx = rng.integers(0, N, n_sample)
    j_idx = rng.integers(0, N, n_sample)
    same_level = zero_counts[i_idx] == zero_counts[j_idx]

    # Cosine similarity between property vectors
    fi = feat[i_idx]
    fj = feat[j_idx]
    dot = (fi * fj).sum(axis=1)
    ni = np.linalg.norm(fi, axis=1) + 1e-9
    nj = np.linalg.norm(fj, axis=1) + 1e-9
    cos_sim = dot / (ni * nj)

    within_sim = cos_sim[same_level].mean()
    across_sim = cos_sim[~same_level].mean()

    # Random baseline: shuffle level assignments
    shuffled_counts = zero_counts.copy()
    rng.shuffle(shuffled_counts)
    same_shuffled = shuffled_counts[i_idx] == shuffled_counts[j_idx]
    within_sim_rand = cos_sim[same_shuffled].mean() if same_shuffled.any() else np.nan
    across_sim_rand = cos_sim[~same_shuffled].mean() if (~same_shuffled).any() else np.nan

    return {
        "within_sim": within_sim,
        "across_sim": across_sim,
        "lift": within_sim / (across_sim + 1e-9),
        "within_sim_rand": within_sim_rand,
        "lift_rand": within_sim_rand / (across_sim_rand + 1e-9),
        "n_within_pairs": same_level.sum(),
        "n_across_pairs": (~same_level).sum(),
    }


# ─── Printing helpers ─────────────────────────────────────────────────────────

def _fmt(v, digits=1):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def print_summary_table(rows: list):
    header_fields = [
        ("Lv", "level", 2),
        ("N", "count", 5),
        ("%tot", "pct", 5),
        ("Comm%", "commutative", 6),
        ("Idmp%", "idempotent", 6),
        ("Idnt%", "has_identity", 6),
        ("Absb%", "has_absorbing", 6),
        ("Asc%",  "associative", 5),
        ("Cnst%", "constant", 5),
        ("FP",   "mean_fixed_points", 4),
        ("OutD", "mean_output_diversity", 4),
        ("ZOSym%", "zero_output_symmetric", 7),
    ]

    header = "  ".join(f"{h[0]:>{h[2]}}" for h in header_fields)
    print(header)
    print("-" * len(header))

    for row in rows:
        parts = []
        for label, key, width in header_fields:
            v = row[key]
            if key in ("level", "count"):
                parts.append(f"{int(v):>{width}}")
            elif key in ("mean_fixed_points", "mean_output_diversity", "pct"):
                parts.append(f"{v:>{width}.2f}")
            else:
                parts.append(f"{v:>{width}.1f}")
        print("  ".join(parts))


def print_notable_ops(ops: np.ndarray, zero_counts: np.ndarray, props: dict):
    """Print algebraically notable operations at key levels."""
    VAL_NAMES = {-1: "−1", 0: " 0", 1: "+1"}

    def op_to_table_str(op):
        rows = []
        for a_idx in range(3):
            row_vals = [VAL_NAMES[op[a_idx * 3 + b_idx]] for b_idx in range(3)]
            rows.append(" ".join(row_vals))
        return "  |  ".join(rows)

    print("\nNotable operations by algebraic class:")
    print("  Format: [f(-1,·)] [f(0,·)] [f(1,·)]  (columns: input b = -1, 0, +1)")

    categories = [
        ("Commutative + Idempotent + Has Identity",
         props["commutative"] & props["idempotent"] & props["has_identity"]),
        ("Associative + Commutative + Has Identity (group-like)",
         props["associative"] & props["commutative"] & props["has_identity"]),
        ("Left projection", props["left_projection"]),
        ("Right projection", props["right_projection"]),
        ("Constant = 0 (fully absorbing)", props["constant"] & (ops[:, 0] == 0)),
        ("Absorbing element = 0", props["absorbing_is_0"]),
        ("Identity element = 0", props["identity_is_0"]),
    ]

    for label, mask in categories:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"\n  [{label}]  — none found")
            continue
        # Show up to 3 examples
        show = idxs[:3]
        print(f"\n  [{label}]  ({len(idxs)} total):")
        for idx in show:
            zc = zero_counts[idx]
            v3 = TERNARY.valuation(torch.tensor([idx])).item()
            print(f"    idx={idx:5d}  zero_count={zc}  v3={v3}  table: {op_to_table_str(ops[idx])}")


def print_level_composition(ops: np.ndarray, zero_counts: np.ndarray, props: dict):
    """Show what fraction of each level consists of key algebraic classes."""
    print("\nLevel composition by algebraic class (% of operations at that level):")

    prop_cols = [
        ("Commut.", "commutative"),
        ("Idmpt.",  "idempotent"),
        ("Assoc.",  "associative"),
        ("CommIdmptAssoc", None),  # special composite
        ("HasIdnt.", "has_identity"),
        ("HasAbsb.", "has_absorbing"),
    ]

    header = f"  {'Lv':>2}  {'N':>5}  " + "  ".join(f"{h:>12}" for h, _ in prop_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for level in range(10):
        mask = zero_counts == level
        n = mask.sum()
        if n == 0:
            continue
        parts = [f"  {level:>2}  {n:>5}"]
        for _label, key in prop_cols:
            if key is None:
                v = (props["commutative"] & props["idempotent"] & props["associative"])[mask].mean() * 100
            else:
                v = props[key][mask].mean() * 100
            parts.append(f"  {v:>12.1f}")
        print("".join(parts))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" Semantic Validity Check: zero_count_valuation Hierarchy (v9 / Option B)")
    print("=" * 72)
    print()
    print("Each operation is a binary function f: {-1,0,1}² → {-1,0,1} (9-cell table).")
    print("zero_count = number of cells where f(a,b) = 0.")
    print("Null hypothesis: same-level operations are NOT more algebraically similar.")
    print()

    print("Loading TERNARY LUT and building operation tables...")
    ops = build_op_table()
    zero_counts = compute_zero_count(ops)

    print(f"  Total operations: {len(ops):,}")
    print(f"  Zero-count distribution: {dict(zip(*np.unique(zero_counts, return_counts=True)))}")
    print()

    print("Computing algebraic properties for all 19,683 operations...")
    props = compute_properties(ops)
    print()

    # ── Per-level statistics table
    print("─" * 72)
    print("PER-LEVEL ALGEBRAIC PROPERTY RATES")
    print("─" * 72)
    rows = per_level_stats(zero_counts, props)
    print_summary_table(rows)
    print()
    print("  Lv=level, Comm=commutative, Idmp=idempotent, Idnt=identity element,")
    print("  Absb=absorbing element, Asc=associative, Cnst=constant op, FP=fixed points,")
    print("  OutD=output diversity, ZOSym=zero-output symmetric")

    # ── Level composition
    print()
    print("─" * 72)
    print_level_composition(ops, zero_counts, props)

    # ── Notable operations
    print()
    print("─" * 72)
    print_notable_ops(ops, zero_counts, props)

    # ── Global counts
    print()
    print("─" * 72)
    print("GLOBAL ALGEBRAIC CLASS COUNTS (all 19,683 operations)")
    print("─" * 72)
    global_props = [
        ("Commutative", props["commutative"]),
        ("Idempotent", props["idempotent"]),
        ("Associative", props["associative"]),
        ("Comm + Idmpt + Assoc (band-like)", props["commutative"] & props["idempotent"] & props["associative"]),
        ("Has identity element", props["has_identity"]),
        ("Has absorbing element", props["has_absorbing"]),
        ("Left projection (f(a,b)=a)", props["left_projection"]),
        ("Right projection (f(a,b)=b)", props["right_projection"]),
        ("Constant function", props["constant"]),
        ("Zero-output symmetric", props["zero_output_symmetric"]),
    ]
    for label, mask in global_props:
        n = mask.sum()
        pct = 100 * n / len(ops)
        print(f"  {label:<40s}  {n:>6,}  ({pct:.2f}%)")

    # ── Within-level homogeneity test
    print()
    print("─" * 72)
    print("WITHIN-LEVEL HOMOGENEITY (cosine similarity of property vectors)")
    print("─" * 72)
    feat = build_property_vector(props, len(ops))
    hom = within_level_homogeneity(zero_counts, feat, n_sample=20000)
    print(f"  Within-level mean cosine similarity:   {hom['within_sim']:.4f}")
    print(f"  Between-level mean cosine similarity:  {hom['across_sim']:.4f}")
    print(f"  Lift (within / between):               {hom['lift']:.4f}x")
    print(f"  Random baseline lift:                  {hom['lift_rand']:.4f}x")
    print(f"  Within/between pair counts:            {hom['n_within_pairs']:,} / {hom['n_across_pairs']:,}")
    print()
    lift = hom["lift"]
    rand = hom["lift_rand"]
    if lift > rand * 1.05:
        verdict = "YES — same-level operations are algebraically more similar than random."
    else:
        verdict = "NO  — same-level operations are NOT more similar than random."
    print(f"  Verdict: {verdict}")

    # ── Index-derived v_3 vs zero_count comparison
    print()
    print("─" * 72)
    print("INDEX-DERIVED v_3(n) vs ZERO_COUNT: DISTRIBUTION COMPARISON")
    print("─" * 72)
    v3_all = TERNARY.all_valuations().numpy()
    print(f"  {'Level':>5}  {'v3(n) count':>12}  {'v3(n) %':>8}  {'zero_count':>12}  {'zero_ct %':>9}")
    for level in range(10):
        n_v3 = (v3_all == level).sum()
        n_zc = (zero_counts == level).sum()
        pct_v3 = 100 * n_v3 / len(ops)
        pct_zc = 100 * n_zc / len(ops)
        print(f"  {level:>5}  {n_v3:>12,}  {pct_v3:>8.2f}%  {n_zc:>12,}  {pct_zc:>9.2f}%")

    print()
    v3_max = (v3_all == 0).mean() * 100
    print(f"  Index-derived: dominant level (v=0) holds {v3_max:.1f}% of ops -> tied-rank ceiling.")
    print(f"  Zero-count:    peak level holds ≤{max(100*np.bincount(zero_counts)/len(ops)):.1f}% of ops → no dominant tier.")
    print()
    print("=" * 72)
    print(" Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
