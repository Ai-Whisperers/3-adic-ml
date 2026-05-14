"""Step 2 — z_θ Direction Geometry Diagnostic (identity geometry audit).

Loads best V7 checkpoint, extracts direction vectors for all 19683 operations,
and answers four questions:

  Q1: Do same-valuation operations cluster angularly?
      → intra-level cosine similarity vs random baseline, per level

  Q2: Do operations sharing a digit pattern at position i cluster angularly?
      → for each of the 9 digit positions and 3 digit values {-1,0,1},
        measure mean cosine similarity within the group vs random

  Q3: 2D UMAP of all 19683 direction vectors, colored by valuation level
      → saved to runs/v7_*/direction_umap.png

  Q4: kNN@5 in direction space — do nearest neighbors share more digit
      patterns than random chance?

No code changes to src/ required. Direction vectors recovered as:
    dir = z_A_hyp / ||z_A_hyp||   (exact since z_hyp = r * dir, ||dir||=1)
"""

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset

from src.core.ternary import TERNARY
from src.models.vae import TernaryVAEV6Controllable

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT = "runs/v7_large_20260323_064646/checkpoints/best_Q.pt"
CONFIG     = "src/presets/v7_large.yaml"
OUT_DIR    = Path("runs/v7_large_20260323_064646")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  Z_θ DIRECTION GEOMETRY DIAGNOSTIC")
print("=" * 70)

# ── Load model ─────────────────────────────────────────────────────────────────
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

model = TernaryVAEV6Controllable(
    latent_dim=mc["latent_dim"],
    hidden_dim=mc["hidden_dim"],
    max_radius=mc["max_radius"],
    factored=mc["factored"],
    radial_dims=mc["radial_dims"],
    n_projection_layers=mc["projection_layers"],
    projection_dropout=mc["projection_dropout"],
    init_identity=mc["init_identity"],
    tangent_scale_init=mc["tangent_scale"],
    learnable_curvature=mc["learnable_curvature"],
    encoder_a_trainable=True,
    encoder_b_trainable=True,
    projections_trainable=True,
).to(DEVICE)

ck = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"  Loaded checkpoint: epoch {ck['epoch']} | Q={ck['Q']:.4f}")

# ── Extract directions for all 19683 operations ────────────────────────────────
all_ops  = TERNARY.all_ternary().to(DEVICE)   # (19683, 9)
all_idx  = torch.arange(len(all_ops), device=DEVICE)
all_vals = TERNARY.valuation(all_idx).float() # (19683,)

dl = DataLoader(TensorDataset(all_ops, all_idx), batch_size=512, shuffle=False)

z_hyp_list, r_list = [], []
with torch.no_grad():
    for x_batch, _ in dl:
        out = model(x_batch.float())
        z_hyp_list.append(out["z_A_hyp"].cpu())
        r_list.append(out["r_A"].cpu())

z_hyp = torch.cat(z_hyp_list, dim=0)  # (19683, 28)
r     = torch.cat(r_list,     dim=0)  # (19683,)

# Recover direction: dir = z_hyp / ||z_hyp|| (== z_hyp / r since ||dir||=1)
norms = z_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
dirs  = z_hyp / norms                  # (19683, 28) unit-normalized

vals_np = all_vals.cpu().numpy()
dirs_np = dirs.numpy()
r_np    = r.numpy()

norms_check = np.linalg.norm(dirs_np, axis=1)
print(f"  Directions extracted: {dirs_np.shape}, norm range "
      f"[{norms_check.min():.6f}, {norms_check.max():.6f}]  "
      f"(should be ≈1.0)")

# ── Q1: Intra-level angular clustering ────────────────────────────────────────
print()
print("─" * 70)
print("Q1: Intra-level cosine similarity vs random baseline")
print("─" * 70)
print(f"  {'Level':>6} | {'N':>5} | {'intra_sim':>9} | {'random_sim':>10} | {'delta':>7} | {'ratio':>6}")
print(f"  {'──────':>6}-+-{'─────':>5}-+-{'─────────':>9}-+-{'──────────':>10}-+-{'───────':>7}-+-{'──────':>6}")

rng = np.random.default_rng(42)
overall_intra, overall_random = [], []

for v in range(10):
    mask = vals_np == v
    n = mask.sum()
    if n < 2:
        print(f"  {v:>6} | {n:>5} | {'(skip)':>9} | {'':>10} | {'':>7} | {'':>6}")
        continue

    d_v = dirs_np[mask]  # (n, 28)

    # Intra-level: mean pairwise cosine sim (sample up to 2000 pairs)
    n_pairs = min(2000, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, n_pairs)
    j_idx = rng.integers(0, n, n_pairs)
    same = i_idx == j_idx
    i_idx, j_idx = i_idx[~same], j_idx[~same]
    intra = (d_v[i_idx] * d_v[j_idx]).sum(axis=1).mean()

    # Random baseline: sample same number of random cross-level pairs
    r_idx = rng.integers(0, len(dirs_np), len(i_idx))
    s_idx = rng.integers(0, len(dirs_np), len(i_idx))
    random_sim = (dirs_np[r_idx] * dirs_np[s_idx]).sum(axis=1).mean()

    delta = intra - random_sim
    ratio = intra / (abs(random_sim) + 1e-8)
    flag = "✓ cluster" if delta > 0.05 else ("~ weak" if delta > 0.01 else "✗ flat")
    print(f"  {v:>6} | {n:>5} | {intra:>+9.4f} | {random_sim:>+10.4f} | {delta:>+7.4f} | {ratio:>6.3f}  {flag}")
    overall_intra.append(intra)
    overall_random.append(random_sim)

print(f"\n  Overall mean intra-level sim: {np.mean(overall_intra):+.4f}  |  "
      f"Random baseline: {np.mean(overall_random):+.4f}  |  "
      f"Delta: {np.mean(overall_intra) - np.mean(overall_random):+.4f}")

# ── Q2: Digit-position grouping ────────────────────────────────────────────────
print()
print("─" * 70)
print("Q2: Digit-position grouping — do ops sharing digit value at position i cluster?")
print("─" * 70)

all_ternary_np = all_ops.cpu().numpy()  # (19683, 9)  values in {-1, 0, 1}

print(f"  {'Pos':>4} | {'Val':>4} | {'N':>5} | {'intra_sim':>9} | {'random_sim':>10} | {'delta':>7}")
print(f"  {'────':>4}-+-{'────':>4}-+-{'─────':>5}-+-{'─────────':>9}-+-{'──────────':>10}-+-{'───────':>7}")

pos_deltas = np.zeros((9, 3))
for pos in range(9):
    for vi, dval in enumerate([-1, 0, 1]):
        mask = all_ternary_np[:, pos] == dval
        n = mask.sum()
        d_g = dirs_np[mask]

        n_pairs = min(1000, n * (n-1) // 2)
        i_idx = rng.integers(0, n, n_pairs)
        j_idx = rng.integers(0, n, n_pairs)
        same = i_idx == j_idx
        i_idx, j_idx = i_idx[~same], j_idx[~same]
        intra = (d_g[i_idx] * d_g[j_idx]).sum(axis=1).mean()

        r_idx = rng.integers(0, len(dirs_np), len(i_idx))
        s_idx = rng.integers(0, len(dirs_np), len(i_idx))
        random_sim = (dirs_np[r_idx] * dirs_np[s_idx]).sum(axis=1).mean()

        delta = intra - random_sim
        pos_deltas[pos, vi] = delta
        flag = "✓" if delta > 0.05 else ("~" if delta > 0.01 else "✗")
        print(f"  {pos:>4} | {dval:>+4} | {n:>5} | {intra:>+9.4f} | {random_sim:>+10.4f} | {delta:>+7.4f}  {flag}")

print(f"\n  Mean delta across all (pos, val) groups: {pos_deltas.mean():+.4f}")
print(f"  Best position (highest mean delta): pos={pos_deltas.mean(axis=1).argmax()}, "
      f"delta={pos_deltas.mean(axis=1).max():+.4f}")

# ── Q3: UMAP 2D of directions ──────────────────────────────────────────────────
print()
print("─" * 70)
print("Q3: UMAP 2D - fitting on direction vectors (19683 x 28)...")
print("─" * 70)

try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
                        metric="cosine", verbose=False)
    embedding = reducer.fit_transform(dirs_np)   # (19683, 2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by valuation
    cmap = plt.cm.get_cmap("plasma", 10)
    for v in range(10):
        mask = vals_np == v
        if mask.sum() == 0:
            continue
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[cmap(v)], s=1.0, alpha=0.5, label=f"v={v} (n={mask.sum()})")
    axes[0].set_title("Direction UMAP — colored by valuation level", fontsize=12)
    axes[0].legend(markerscale=5, fontsize=7, loc="upper right")
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")

    # Right: colored by digit-0 value
    colors_d0 = {-1: "royalblue", 0: "gray", 1: "tomato"}
    labels_d0 = {-1: "digit0=−1", 0: "digit0=0", 1: "digit0=+1"}
    for dval in [-1, 0, 1]:
        mask = all_ternary_np[:, 0] == dval
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=colors_d0[dval], s=0.8, alpha=0.4, label=labels_d0[dval])
    axes[1].set_title("Direction UMAP — colored by digit position 0 value", fontsize=12)
    axes[1].legend(markerscale=5, fontsize=9)
    axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")

    plt.tight_layout()
    out_path = OUT_DIR / "direction_umap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  UMAP saved → {out_path}")
except Exception as e:
    print(f"  UMAP failed: {e}")
    embedding = None

# ── Q4: kNN@5 recall in direction space ───────────────────────────────────────
print()
print("─" * 70)
print("Q4: kNN@5 in direction space — digit pattern overlap vs random")
print("─" * 70)

# Sample 500 query operations, find 5 nearest neighbors in direction space
# Measure: mean fraction of shared digits (out of 9) between query and each neighbor
# Compare to random baseline

N_QUERIES = 500
query_idx = rng.choice(len(dirs_np), N_QUERIES, replace=False)

# Build cosine similarity matrix for queries vs all (batch)
dirs_t = torch.from_numpy(dirs_np).float()  # (19683, 28)
query_dirs = dirs_t[query_idx]               # (500, 28)

# Cosine sim: query_dirs @ dirs_t.T (already unit-norm)
cos_sim_full = (query_dirs @ dirs_t.T).numpy()  # (500, 19683)

# kNN@5: exclude self
cos_sim_full[np.arange(N_QUERIES), query_idx] = -2.0
nn5_idx = np.argsort(cos_sim_full, axis=1)[:, -5:]  # (500, 5)

# Digit overlap: fraction of 9 digits matching
def digit_overlap(ops_a, ops_b):
    """Mean fraction of matching digits between row sets (B, 9) each."""
    return (ops_a == ops_b).mean()

nn_overlaps, rand_overlaps = [], []
for qi, q in enumerate(query_idx):
    q_digits = all_ternary_np[q]  # (9,)
    nn_digits = all_ternary_np[nn5_idx[qi]]  # (5, 9)
    nn_overlaps.append((nn_digits == q_digits).mean())

    rand5 = rng.choice(len(dirs_np), 5, replace=False)
    rand_digits = all_ternary_np[rand5]
    rand_overlaps.append((rand_digits == q_digits).mean())

nn_mean   = np.mean(nn_overlaps)
rand_mean = np.mean(rand_overlaps)
expected_random = 1/3  # three equally likely digit values → P(match) = 1/3

print(f"  kNN@5 mean digit overlap:    {nn_mean:.4f}")
print(f"  Random@5 mean digit overlap: {rand_mean:.4f}")
print(f"  Expected random baseline:    {expected_random:.4f}")
print(f"  Delta (kNN vs random):       {nn_mean - rand_mean:+.4f}")

if nn_mean - rand_mean > 0.05:
    print("  → ✓ Direction space encodes digit similarity: kNN neighbors share significantly more digits")
elif nn_mean - rand_mean > 0.01:
    print("  → ~ Weak signal: slight digit overlap enrichment in kNN")
else:
    print("  → ✗ Direction space does not encode digit patterns above random")

# Also check: are kNN@5 neighbors more likely to share the same valuation?
nn_val_match   = [float((vals_np[nn5_idx[qi]] == vals_np[query_idx[qi]]).mean()) for qi in range(N_QUERIES)]
rand_val_match = [float((vals_np[rng.choice(len(vals_np), 5)] == vals_np[query_idx[qi]]).mean()) for qi in range(N_QUERIES)]
print(f"\n  kNN@5 same-valuation fraction:    {np.mean(nn_val_match):.4f}")
print(f"  Random@5 same-valuation fraction: {np.mean(rand_val_match):.4f}")
print(f"  Delta:                            {np.mean(nn_val_match) - np.mean(rand_val_match):+.4f}")

# ── Step 5: v=0 Sub-island Labeling ───────────────────────────────────────────
print()
print("─" * 70)
print("Step 5: v=0 K-means sub-island labeling (k=15)")
print("─" * 70)

v0_mask    = vals_np == 0
v0_dirs    = dirs_np[v0_mask]                         # (13122, 28)
v0_indices = all_idx.cpu().numpy()[v0_mask]           # operation indices

km = KMeans(n_clusters=15, random_state=42, n_init=10, max_iter=500)
km_labels  = km.fit_predict(v0_dirs)                  # (13122,)

# Compare K-means clusters against digit-pattern classifiers
v0_idx_t   = torch.from_numpy(v0_indices.astype('int64'))
prefix2    = TERNARY.digit_prefix_class(v0_idx_t, k=2).numpy()   # 9 classes
prefix3    = TERNARY.digit_prefix_class(v0_idx_t, k=3).numpy()   # 27 classes
nzpat      = TERNARY.nonzero_pattern(v0_idx_t).numpy()           # 512 classes
valprefix  = TERNARY.valuation_prefix_class(v0_idx_t).numpy()    # 6 classes

ari_p2     = adjusted_rand_score(prefix2,   km_labels)
ari_p3     = adjusted_rand_score(prefix3,   km_labels)
ari_nz     = adjusted_rand_score(nzpat,     km_labels)
ari_vp     = adjusted_rand_score(valprefix, km_labels)

print("  Adjusted Rand Index — K-means(15) vs:")
print(f"    digit_prefix_class(k=2):      ARI={ari_p2:.4f}  {'★ prefix explains islands' if ari_p2 > 0.5 else ('~ partial' if ari_p2 > 0.2 else '✗ not explained')}")
print(f"    digit_prefix_class(k=3):      ARI={ari_p3:.4f}  {'★ prefix explains islands' if ari_p3 > 0.5 else ('~ partial' if ari_p3 > 0.2 else '✗ not explained')}")
print(f"    nonzero_pattern (9-bit):      ARI={ari_nz:.4f}  {'★ zero-structure explains' if ari_nz > 0.5 else ('~ partial' if ari_nz > 0.2 else '✗ not explained')}")
print(f"    valuation_prefix_class:       ARI={ari_vp:.4f}  {'★ val-prefix explains' if ari_vp > 0.5 else ('~ partial' if ari_vp > 0.2 else '✗ not explained')}")

print()
print("  Per K-means cluster digit statistics (d0-d2 means, zero-frac):")
print(f"  {'Cls':>4} | {'N':>5} | {'d0':>5} | {'d1':>5} | {'d2':>5} | {'zeros':>6} | {'prefix2':>8} | {'nz_mode':>8}")
print(f"  {'────':>4}-+-{'─────':>5}-+-{'─────':>5}-+-{'─────':>5}-+-{'─────':>5}-+-{'──────':>6}-+-{'────────':>8}-+-{'────────':>8}")
v0_ternary = all_ops.cpu().numpy()[v0_mask]           # (13122, 9)

for c in range(15):
    mc = km_labels == c
    ops_c = v0_ternary[mc]                            # (n_c, 9)
    # Most common prefix2 class in this cluster
    from collections import Counter
    p2_mode  = Counter(prefix2[mc]).most_common(1)[0][0]
    nz_mode  = Counter(nzpat[mc]).most_common(1)[0][0]
    print(f"  {c:>4} | {mc.sum():>5} | {ops_c[:,0].mean():>+5.2f} | {ops_c[:,1].mean():>+5.2f} | "
          f"{ops_c[:,2].mean():>+5.2f} | {(ops_c==0).mean():>6.3f} | {p2_mode:>8} | {nz_mode:>8}")

# UMAP colored by K-means cluster (append to existing UMAP figure)
if embedding is not None:
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    cmap15 = plt.cm.get_cmap("tab20", 15)
    full_colors = np.full(len(dirs_np), -1, dtype=int)
    full_colors[v0_mask] = km_labels
    for c in range(15):
        mask_c = full_colors == c
        ax2.scatter(embedding[mask_c, 0], embedding[mask_c, 1],
                    c=[cmap15(c)], s=1.5, alpha=0.7, label=f"C{c} (n={mask_c.sum()})")
    not_v0 = ~v0_mask
    ax2.scatter(embedding[not_v0, 0], embedding[not_v0, 1],
                c="lightgray", s=0.5, alpha=0.3, label="v≥1")
    ax2.set_title("v=0 K-means sub-islands (15 clusters) on direction UMAP", fontsize=11)
    ax2.legend(markerscale=5, fontsize=6, ncol=3, loc="upper right")
    ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
    plt.tight_layout()
    out2 = OUT_DIR / "direction_umap_v0_clusters.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"\n  Cluster UMAP saved → {out2}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Step5 — K-means ARI:          prefix2={ari_p2:.4f} | prefix3={ari_p3:.4f} | nzpat={ari_nz:.4f} | valprefix={ari_vp:.4f}")
print(f"  Q1 — Intra-level clustering:   delta={np.mean(overall_intra) - np.mean(overall_random):+.4f}")
print(f"  Q2 — Best digit-pos grouping:  delta={pos_deltas.mean(axis=1).max():+.4f} (pos={pos_deltas.mean(axis=1).argmax()})")
print(f"  Q3 — UMAP:                     {'saved → ' + str(OUT_DIR / 'direction_umap.png') if embedding is not None else 'failed'}")
print(f"  Q4 — kNN digit overlap delta:  {nn_mean - rand_mean:+.4f}")
print(f"  Q4 — kNN valuation match delta:{np.mean(nn_val_match) - np.mean(rand_val_match):+.4f}")
print()
