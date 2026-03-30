# Graph + Topology + Visualization Integration Plan
**Date**: 2026-03-24
**Status**: Design
**Scope**: PyTorch Geometric · pytorch_scatter · Persistent Homology · UMAP/PaCMAP/TriMAP · 3D TensorBoard

---

## Executive Summary

This plan integrates three classes of tools into the 3-adic VAE codebase:

1. **pytorch_scatter** — replace Python loops over valuation levels (v=0–9) with vectorized scatter operations throughout the loss system and training loop
2. **PyTorch Geometric (PyG)** — represent the 3-adic ultrametric tree as a proper graph; add a tree-aware message-passing loss that exploits parent-child structure; build the graph once from `TernarySpace.PROP_PARENT`
3. **Topology + Dimensionality Reduction Visualization** — compute persistent homology and multi-algorithm 2D/3D projections using the *hyperbolic* distance matrix (not Euclidean raw coordinates), logged to TensorBoard and saved as interactive HTML

The integrations are grounded in the mathematics of the codebase. None are added as decoration.

---

## Part 1 — pytorch_scatter: Vectorized Valuation Aggregations

### 1.1 Why It's Needed

Currently there are **Python loops over valuation levels (v=0..9)** in at least 4 places:

| File | Location | Operation |
|------|----------|-----------|
| `src/losses/padic_geodesic.py` | `MonotonicRadialLoss.forward()` line ~741 | Per-level mean radii |
| `src/losses/padic_geodesic.py` | `PAdicGeodesicLoss.forward()` line ~561 | Per-level scatter tensors |
| `src/losses/padic_geodesic.py` | `RichHierarchyLoss.forward()` line ~928 | Per-level MSE targets |
| `src/train.py` | `compute_hierarchy_metrics()` line ~524, 811 | Per-level radius stats |

Each loop masks and indexes into full-batch tensors per level — this is O(B × 10) memory reads instead of O(B). With B=4096 and float64, this matters on an RTX 3050.

### 1.2 What scatter Enables

```python
# BEFORE (Python loop, 10 iterations):
level_means = []
for v in range(10):
    mask = valuations == v
    if mask.any():
        level_means.append(radii[mask].mean())

# AFTER (single CUDA kernel):
from torch_scatter import scatter_mean
level_means = scatter_mean(radii, valuations.long(), dim=0, dim_size=10)
# → (10,) tensor, differentiable, single GPU call
```

Available operations from `torch_scatter`:
- `scatter_mean` — per-level mean radii, per-level within-class cosine similarity
- `scatter_std` — per-level radius variance (currently untracked)
- `scatter_max` — per-level maximum radius (for margin constraint checking)
- `scatter_min` — complement for margin gaps

### 1.3 Files to Modify

**`src/losses/padic_geodesic.py`**:
- `MonotonicRadialLoss.forward()`: replace loop at ~line 741 with `scatter_mean(radii, valuations, dim=0, dim_size=11)`
- `PAdicGeodesicLoss.forward()`: replace per-level scatter loop at ~line 558 with vectorized batch
- `RichHierarchyLoss.forward()`: replace `for v in present_levels` loop at ~line 928

**`src/train.py`**:
- `compute_hierarchy_metrics()`: replace Python loops at ~line 524 and 811 with scatter ops
- This also enables computing **per-level radius std** without extra cost (log as `Radius/std_v{v}_A`)

**`src/losses/combined.py`**:
- No structural change needed, but the faster per-level ops allow logging all per-level metrics cheaply

### 1.4 New Metrics Unlocked

Once scatter_std is free, add to TensorBoard logging in `train.py`:
- `Radius/std_v{v}_A` and `_B` — per-level radius spread (currently only mean is logged; std reveals collapse vs. spread within a level)
- `Geometry/level_gap_v{v}` — `r_v{v} - r_v{v+1}` as an explicit margin signal (currently only violation count logged)

### 1.5 Implementation Order

1. Add `torch-scatter` to `requirements.txt` with CUDA-compatible install note
2. Add helper `src/utils/scatter_utils.py` with fallback (pure-torch loop) if torch_scatter not installed
3. Refactor `MonotonicRadialLoss` first (most isolated)
4. Refactor `PAdicGeodesicLoss` per-level scatter
5. Refactor `train.py` hierarchy metrics
6. Update tests in `test_losses.py` to verify scatter outputs match loop outputs on known data

---

## Part 2 — PyTorch Geometric: 3-Adic Tree as a Graph

### 2.1 The Core Insight

The 3-adic valuation structure **is** a tree graph. `TernarySpace` already stores `PROP_PARENT` in its properties lookup table — every index `n` has a canonical parent index `m` where `v_3(m) = v_3(n) + 1`. This defines 19,682 directed edges (root has no parent).

This tree is not currently exploited as a *graph* — losses treat it as index arithmetic. PyG lets us make the tree structure a first-class citizen.

### 2.2 Build the 3-Adic Tree Graph (Once, at Startup)

**New file: `src/graph/ternary_tree.py`**

```python
from torch_geometric.data import Data
import torch
from src.core.ternary import TERNARY

def build_ternary_tree_graph(device) -> Data:
    """
    Build the full 19,683-node 3-adic tree as a PyG Data object.

    Nodes: indices 0..19682
    Node features: valuation (1-dim), ternary digits (9-dim), target_radius (1-dim)
    Edges: parent→child (directed; also add reverse for undirected message passing)
    Edge attributes: |v_child - v_parent| = 1 (always, by construction)

    Built once and cached on the target device.
    Returns PyG Data with:
        x: (19683, 11) node features
        edge_index: (2, ~39364) — bidirectional parent-child edges
        edge_attr: (edges, 1) — valuation depth of child
    """
    ...
```

**Why build it**:
- MessagePassing operates on this graph to propagate hyperbolic embeddings up/down the tree
- Visualization tools use the edge structure for 3D graph layout
- The graph is immutable (fixed by math) — build once, reuse across epochs

### 2.3 Tree-Coherence Message Passing Loss

**Current limitation**: `TreeCoherence` metrics are computed offline in `diagnose_direction_geometry.py` and not part of the loss. The existing `RichHierarchyLoss` only looks at aggregate statistics, not local parent-child relationships.

**New loss: `TreeMessagePassingLoss`** in `src/losses/graph_losses.py`

**Idea**: A parent node's embedding in hyperbolic space should be geometrically "between" its children. In the Poincaré ball, this means the parent's embedding should be closer to the origin than any of its children *and* roughly on the geodesic from origin toward the child cluster centroid.

```python
class TreeMessagePassingLoss(nn.Module):
    """
    For each (parent, child) edge in the 3-adic tree,
    enforce: poincare_dist(z_parent, centroid(z_children)) < margin

    Uses PyG MessagePassing to aggregate children → parent messages:
    - Message: z_child (hyperbolic embedding)
    - Aggregate: Fréchet mean on Poincaré ball (approximate: Euclidean mean + project back to ball)
    - Update: compute distance(z_parent, agg_children) — should be small

    Loss = mean over all edges of max(0, dist(parent, children_mean) - margin)
    """
```

**Why this is non-trivial**: Existing losses treat valuation levels globally (mean radius per level). Tree message passing is *local* — it checks that each individual parent is geometrically consistent with its own specific children, not just that average radii are ordered. A model could satisfy all radial constraints globally while having chaotic local structure.

**Where to wire it**: Add as optional loss component in `CombinedLoss` with `tree_message_passing: enabled: true/false` in YAML. Keep disabled by default (adds ~300ms per epoch for full 19,683-node graph).

### 2.4 Graph-Based Stratified Pair Sampling

**Current limitation**: `PAdicGeodesicLoss` uses random within-batch pair sampling, which can be dominated by v=0 pairs (most common valuation). The comment in CLAUDE.md notes this is intentional but acknowledged as a limitation.

**PyG improvement**: Use the tree graph's edge structure for structured pair sampling:
- Sample pairs along edges at each tree depth
- Guarantees coverage of all (parent_valuation, child_valuation) combinations
- Uses PyG's `RandomLinkSplit` or custom walk sampling

**New file function: `src/graph/pair_sampler.py`**

```python
def sample_tree_pairs(
    tree_graph: Data,
    batch_indices: Tensor,
    n_pairs: int,
    strategy: str = "random_walk",  # or "edge_uniform", "level_balanced"
) -> Tuple[Tensor, Tensor]:
    """
    Sample (i, j) index pairs from within-batch indices guided by tree structure.

    strategy="level_balanced": equal pairs per valuation-difference bucket
    strategy="edge_uniform": pairs that are tree-neighbors (v_diff=1 exactly)
    strategy="random_walk": pairs reachable by k-hop walk on tree
    """
```

**This is genuinely valuable**: Level-balanced sampling ensures the geodesic loss gets gradient signal from all 9 hierarchy levels, not just the statistically dominant v=0 level.

### 2.5 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/graph/__init__.py` | Create | Package init |
| `src/graph/ternary_tree.py` | Create | Build/cache PyG Data object from TernarySpace |
| `src/graph/pair_sampler.py` | Create | Structured pair sampling from tree topology |
| `src/losses/graph_losses.py` | Create | TreeMessagePassingLoss |
| `src/losses/combined.py` | Modify | Add graph_losses instantiation + tree_graph injection |
| `src/train.py` | Modify | Build tree_graph at startup, pass to CombinedLoss |
| `requirements.txt` | Modify | Add torch_geometric, torch_scatter, torch_sparse |

---

## Part 3 — Topology + Visualization Pipeline

### 3.1 Why Hyperbolic Distance Matrix Is Critical

The key insight that separates non-trivial from trivial visualization:

> Current TensorBoard embeddings use raw Euclidean coordinates of z_A_hyp ∈ ℝ¹⁶. This is wrong — the meaningful distances are *hyperbolic*, not Euclidean. UMAP/PaCMAP/TriMAP must receive the *Poincaré distance matrix* as input, not Euclidean coordinates.

```python
# WRONG (currently used by TensorBoard add_embedding):
euclidean_projector = PCA(n_components=3).fit_transform(z_A_hyp.cpu().numpy())

# RIGHT (using hyperbolic distances):
from src.geometry.poincare import poincare_distance
D = poincare_distance_matrix(z_A_hyp)  # (B, B) symmetric, uses manifold metric
umap_3d = UMAP(n_components=3, metric="precomputed").fit_transform(D)
```

The 3-adic ultrametric structure lives in the hyperbolic metric, not in the embedding coordinates. All three dimensionality reduction algorithms should receive `D` (hyperbolic pairwise distances) as their input.

### 3.2 Visualization Suite — What to Compute and Why

#### 3.2.1 Poincaré Ball 3D Projection

**What**: Project z_A_hyp from 16-dim → 3-dim via PCA on the *tangent space* (using logmap0), then visualize as 3D scatter on the Poincaré ball surface.

**Why non-trivial**: Using logmap0 to project to tangent space *preserves the origin-distance relationship* (radius is invariant to tangent projection direction). A naive PCA on raw coordinates loses the ball structure. The 3D ball visualization should show:
- v=9 (identity elements): cluster near origin
- v=0 (primitive elements): spread near boundary
- Concentric shells by valuation level

**Output**:
- Plotly 3D scatter saved as `visualizations/epoch_{N}/poincare_3d.html` (interactive)
- TensorBoard `add_figure` with matplotlib 3D scatter (per-checkpoint)
- TensorBoard `add_mesh` for the ball surface wireframe

**New function**: `src/utils/visualization.py → plot_poincare_ball_3d(z_hyp, valuations, epoch, writer)`

#### 3.2.2 Persistent Homology

**What**: Compute Vietoris-Rips persistence on the hyperbolic distance matrix D.

**Why genuinely useful**: The 3-adic ultrametric has a known topological signature:
- **H0** (connected components): A perfect ultrametric tree gives a *dendrogram* — exactly 1 component that splits at each distance threshold in a binary/ternary pattern. If the model learns perfect 3-adic structure, the persistence diagram for H0 should have a specific shape: 3 components that merge at v=1 threshold, 9 that merge at v=2, etc.
- **H1** (loops): Trees have **zero** 1-cycles. If H1 is large, the model is learning non-tree structure (loops = failure to learn ultrametric).
- Persistence = long-lived topological features = structurally meaningful clusters

**This is a mathematical verification**, not just visualization. If homology confirms the expected signature, the architecture is learning the right geometry.

**Library**: `gudhi` (or `ripser` for speed) — add to requirements

**Output**:
- Persistence diagram: TensorBoard `add_figure`
- Betti numbers (β0, β1) over training: TensorBoard scalar `Topology/betti_0`, `Topology/betti_1`
- Persistence entropy: scalar `Topology/persistence_entropy` — measures how "clean" the hierarchy is

**New function**: `src/utils/topology.py → compute_persistence(D_hyp, max_dimension=1) → (diagram, betti_numbers)`

**Frequency**: Every 50 epochs (expensive on full dataset — subsample to 2000 points per call)

#### 3.2.3 UMAP

**What**: UMAP(metric="precomputed") on hyperbolic distance matrix → 2D and 3D projections.

**Why UMAP specifically**: UMAP preserves global structure (hierarchy levels should stay separated). Best for seeing whether the 10 valuation levels form concentric shells.

**Config**: `n_neighbors=15, min_dist=0.1, n_components=3, metric="precomputed"`

**Output**: TensorBoard `add_embedding` (already existing) — but replace current Euclidean input with UMAP 3D coordinates computed from hyperbolic distances. Keep metadata (valuation, prefix_class) as labels.

**Key**: Replaces the existing `Embedding/VAE_A_Poincare` embedding log with a geometrically correct version.

#### 3.2.4 PaCMAP

**What**: PaCMAP on hyperbolic distance matrix → 2D projection.

**Why PaCMAP specifically**: Preserves *mid-range* distances better than UMAP. In the 3-adic tree, mid-range distances (v=4,5,6) are where hierarchy is least clearly captured. PaCMAP will show whether the middle levels are properly interpolated between origin and boundary.

**Library**: `pacmap` — add to requirements

**Output**: TensorBoard `add_figure` matplotlib 2D scatter, colored by valuation level. Saved as `visualizations/epoch_{N}/pacmap_2d.html` (interactive Plotly).

#### 3.2.5 TriMAP

**What**: TriMAP (triplet-based) on hyperbolic distance matrix.

**Why TriMAP specifically**: TriMAP uses triplet constraints `(anchor, positive, negative)` — directly analogous to the ranking constraints in `GlobalRankLoss`. It preserves the *ordering* of distances, making it the most faithful to ultrametric structure.

**Key advantage over UMAP/PaCMAP**: TriMAP will expose whether the distance ordering is correct (v=0 farther from origin than v=1, etc.) not just cluster membership.

**Library**: `trimap` — add to requirements

**Output**: TensorBoard `add_figure`, Plotly HTML.

### 3.3 Architecture of the Visualization System

**New file: `src/utils/visualization.py`**

```
VisualizationPipeline
├── __init__(config, writer: SummaryWriter)
├── run(epoch, z_A_hyp, z_B_hyp, valuations, prefix_classes)
│   ├── [every eval_every]: compute_hyperbolic_distance_matrix()  ← shared, compute once
│   ├── [every eval_every]: plot_poincare_3d(D_hyp, z_A_hyp)
│   ├── [every eval_every]: run_umap_3d(D_hyp) → replace add_embedding
│   ├── [every eval_every]: run_pacmap_2d(D_hyp) → add_figure
│   ├── [every eval_every]: run_trimap_2d(D_hyp) → add_figure
│   └── [every 50 epochs]: run_persistence(D_hyp) → Betti numbers + diagram
└── save_html(epoch, figures)  → visualizations/epoch_N/
```

**Critical design decision**: All algorithms share a single `D_hyp` computation per eval. Computing it once and passing to all algorithms is important — it's O(N²) and the bottleneck.

**Subsampling strategy**: For N=19,683, full pairwise distance is 19683² floats = 3GB. Use stratified subsampling:
- 500 points per valuation level (levels 0-8)
- All 1 point at v=9 (the singleton — critical landmark)
- Total: ~4,501 points
- D_hyp: ~160MB float64 — manageable
- Set `max_vis_points` config parameter

### 3.4 TensorBoard Integration Points

**Existing `tensorboard_logger.py` additions needed**:

```python
# Add to TensorBoardLogger:

def log_embedding_hyperbolic(self, tag, z_hyp, umap_3d, metadata, metadata_header, step):
    """Replace Euclidean embedding with hyperbolic-UMAP 3D coordinates."""
    self.writer.add_embedding(umap_3d, metadata=metadata,
                              metadata_header=metadata_header,
                              global_step=step, tag=tag)

def log_persistence_diagram(self, tag, diagram, step):
    """Log persistence diagram as matplotlib figure."""
    fig = plot_persistence_diagram(diagram)
    self.writer.add_figure(tag, fig, global_step=step)

def log_betti_numbers(self, betti_0, betti_1, step):
    self.writer.add_scalar("Topology/betti_0", betti_0, step)
    self.writer.add_scalar("Topology/betti_1", betti_1, step)
    self.writer.add_scalar("Topology/betti_ratio", betti_1 / (betti_0 + 1e-6), step)

def log_projection_figure(self, tag, coords_2d, valuations, step):
    """PaCMAP/TriMAP 2D scatter colored by valuation."""
    fig = scatter_colored_by_valuation(coords_2d, valuations)
    self.writer.add_figure(tag, fig, global_step=step)

def log_poincare_3d(self, tag, coords_3d, valuations, step):
    """Poincaré ball 3D scatter as matplotlib figure."""
    fig = plot_poincare_ball_3d(coords_3d, valuations)
    self.writer.add_figure(tag, fig, global_step=step)
```

**New TensorBoard layout panels** (add to `add_custom_scalars_layout` in `train.py`):
```python
"Topology": {
    "Betti Numbers": ["Multiline", ["Topology/betti_0", "Topology/betti_1"]],
    "Persistence Entropy": ["Margin", ["Topology/persistence_entropy"]],
},
"Geometry": {
    "Level Gaps": ["Multiline", [f"Geometry/level_gap_v{v}" for v in range(9)]],
    "Level Std": ["Multiline", [f"Radius/std_v{v}_A" for v in range(10)]],
}
```

### 3.5 Interactive HTML Output (not in TensorBoard)

TensorBoard's embedding projector is limited. For true interactive exploration, save:

```
visualizations/
├── epoch_100/
│   ├── poincare_3d.html     ← Plotly 3D scatter, hover shows index/valuation
│   ├── umap_3d.html         ← UMAP 3D with hyperbolic metric
│   ├── pacmap_2d.html       ← PaCMAP, colored by valuation level
│   ├── trimap_2d.html       ← TriMAP, colored by prefix class
│   └── persistence.html     ← Persistence diagram (Plotly barcode)
├── epoch_200/
│   └── ...
└── latest -> epoch_200/     ← symlink to most recent
```

**Library**: `plotly` — add to requirements

---

## Part 4 — Implementation Sequence

### Phase 1: pytorch_scatter refactor (1-2 days)
**Risk**: Low. Pure refactor, tests must pass before/after.

1. `requirements.txt`: add torch-scatter with CUDA wheel note
2. `src/utils/scatter_utils.py`: create with fallback
3. `src/losses/padic_geodesic.py`: refactor MonotonicRadialLoss, PAdicGeodesicLoss, RichHierarchyLoss
4. `src/train.py`: refactor compute_hierarchy_metrics
5. Add `Radius/std_v{v}` logging
6. Run `pytest tests/ -v` — all 280 must pass

### Phase 2: Visualization pipeline (3-4 days)
**Risk**: Medium (new deps, output format choices).

1. `requirements.txt`: add umap-learn (already there), pacmap, trimap, gudhi, plotly
2. `src/utils/topology.py`: persistent homology with gudhi
3. `src/utils/visualization.py`: VisualizationPipeline class
4. Integrate into `train.py` eval block (after existing ARI computation)
5. Add to TensorBoardLogger
6. Test with a checkpoint: run 10 epochs, confirm all figures appear in TensorBoard

### Phase 3: PyTorch Geometric tree graph (5-7 days)
**Risk**: Higher (new architecture component, must not break training).

1. `requirements.txt`: add torch-geometric, torch-sparse
2. `src/graph/ternary_tree.py`: build PyG Data from TernarySpace
3. Test graph construction: 19683 nodes, correct edge count, correct PROP_PARENT wiring
4. `src/graph/pair_sampler.py`: level-balanced pair sampler
5. Wire level-balanced sampler as optional strategy in PAdicGeodesicLoss config
6. `src/losses/graph_losses.py`: TreeMessagePassingLoss
7. Wire into CombinedLoss as optional component
8. `tests/test_graph.py`: unit tests for tree construction, pair sampler, message passing loss
9. Add to v7_large.yaml as optional (disabled by default until evaluated)

---

## Part 5 — Requirements Changes

```
# requirements.txt additions:

# Graph Neural Networks (Phase 3)
torch-geometric>=2.4.0    # PyG core
torch-scatter>=2.1.2      # Vectorized scatter ops (also used standalone in Phase 1)
torch-sparse>=0.6.18      # Sparse tensor ops (required by PyG)
# NOTE: torch-scatter/sparse require CUDA-specific wheels:
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.X.X+cuXXX.html

# Topology (Phase 2)
gudhi>=3.8.0              # Persistent homology (Vietoris-Rips)
# Alternative: ripser>=0.6.0 (faster for large point clouds)

# Dimensionality Reduction (Phase 2)
pacmap>=0.7.0             # PaCMAP mid-range distance preserving
trimap>=1.1.0             # Triplet-based, best for ordering constraints
umap-learn>=0.5.0         # Already in requirements — confirm metric="precomputed" works

# Interactive Visualization (Phase 2)
plotly>=5.18.0            # Interactive HTML figures
```

---

## Part 6 — What NOT to Do

**Avoid these pitfalls**:

1. **Do not** use PyG's `GCNConv` on the ternary tree for embedding — this would replace the VAE architecture, which is the research object. PyG is for auxiliary losses and sampling, not the primary encoder.

2. **Do not** use UMAP/PaCMAP/TriMAP on raw Euclidean coordinates `z_A_hyp.cpu().numpy()`. The hyperbolic distance matrix is mandatory. Using Euclidean distances defeats the purpose.

3. **Do not** compute persistence on every epoch. Vietoris-Rips on 4500 points takes ~2-5 seconds. Every 50 epochs or checkpoint-only.

4. **Do not** log all 19,683 embeddings to TensorBoard's projector — it becomes unresponsive. Use the existing 5000-point subsample.

5. **Do not** make TreeMessagePassingLoss enabled by default until its effect on Q metric is evaluated against a baseline. Add it to YAML as `enabled: false` initially.

---

## Part 7 — Success Criteria

| Component | Success Metric |
|-----------|---------------|
| pytorch_scatter refactor | All 280 tests pass; MonotonicRadialLoss runtime < 50% of current |
| Per-level std logging | `Radius/std_v{v}_A` appears in TensorBoard, non-zero spread |
| Hyperbolic UMAP embedding | `Embedding/VAE_A_Poincare_Hyp` shows valuation-ordered shells (v=9 inner, v=0 outer) |
| Persistent homology | β0 ≈ 1 (single connected component); β1 ≈ 0 (no loops in embedding) |
| PaCMAP/TriMAP | Figures show cleaner level separation than Euclidean PCA at same epoch |
| PyG tree graph | 19,683 nodes, edge_index has exactly 2×(N-1) = 39,364 entries (bidirectional tree) |
| Level-balanced sampler | PAdicGeodesicLoss gradient magnitude is more uniform across valuation levels |
| TreeMessagePassingLoss | Reduces offline TreeCoherence metric vs. baseline without degrading Q |

---

---

## Appendix A: Phase 1 Implementation — Exact Code Details

### A.1 Differentiable vs. Metrics-Only Loops

Critical distinction: two loops live in the differentiable forward pass (gradients must flow through the scatter result), two are metrics-only (no grad needed).

| Loop Site | File | In Grad Graph? | Safe to Use numpy/fallback? |
|-----------|------|---------------|----------------------------|
| `MonotonicRadialLoss.forward()` line ~741 | `padic_geodesic.py` | **YES** — `level_means` feeds into hinge loss | No: must use differentiable scatter |
| `RichHierarchyLoss.forward()` line ~928 | `padic_geodesic.py` | **YES** — `mean_r` and `level_radii.var()` are loss terms | No: must use differentiable scatter |
| `compute_level_stratified_hierarchy()` line ~524 | `train.py` | No — called under `torch.no_grad()` | Yes: pure-torch loop or scatter |
| `compute_hierarchy_metrics()` line ~539 | `train.py` | No — called under `torch.no_grad()` | Yes: already uses numpy |

**Consequence**: `src/utils/scatter_utils.py` must provide a differentiable path (used in loss forward) AND a non-differentiable fallback path (used in metrics). Use the same API for both; the differentiable variant just calls `torch_scatter.scatter_mean` while the fallback uses a vectorized pure-torch implementation.

### A.2 `src/utils/scatter_utils.py` — Fallback Design

```python
"""Scatter utilities with optional torch_scatter acceleration.

All functions are differentiable (autograd-compatible) even in fallback mode.
The fallback uses torch.zeros + scatter_add_ which is fully differentiable.
"""
try:
    from torch_scatter import scatter_mean as _scatter_mean
    from torch_scatter import scatter_std as _scatter_std
    _HAS_TORCH_SCATTER = True
except ImportError:
    _HAS_TORCH_SCATTER = False

def level_scatter_mean(
    src: torch.Tensor,      # (N,) float values (e.g. radii)
    index: torch.LongTensor,  # (N,) level index per element (0..dim_size-1)
    dim_size: int = 10,     # number of output levels
) -> torch.Tensor:          # (dim_size,) — differentiable
    """Per-level mean. Differentiable. Uses torch_scatter if available."""
    if _HAS_TORCH_SCATTER:
        return _scatter_mean(src, index, dim=0, dim_size=dim_size)
    # Fallback: differentiable pure-torch implementation
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    sums = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(src))
    sums.scatter_add_(0, index, src)
    safe_counts = counts.clamp(min=1.0)
    return sums / safe_counts  # (dim_size,) — 0 where count=0

def level_scatter_std(
    src: torch.Tensor,
    index: torch.LongTensor,
    dim_size: int = 10,
) -> torch.Tensor:          # (dim_size,) — differentiable (via mean)
    """Per-level std. Differentiable."""
    if _HAS_TORCH_SCATTER:
        return _scatter_std(src, index, dim=0, dim_size=dim_size)
    means = level_scatter_mean(src, index, dim_size)
    deviations = src - means[index]
    variance = level_scatter_mean(deviations ** 2, index, dim_size)
    return variance.clamp(min=0.0).sqrt()

def level_has_data(index: torch.LongTensor, dim_size: int = 10) -> torch.BoolTensor:
    """(dim_size,) bool: which levels have at least 1 sample."""
    counts = torch.zeros(dim_size, dtype=torch.long, device=index.device)
    counts.scatter_add_(0, index, torch.ones_like(index))
    return counts > 0
```

**Installation note for `requirements.txt`**:
```
# torch-scatter requires a CUDA-version-specific wheel:
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.X.X+cuXXX.html
# Fallback (pure-torch) is used automatically if not installed.
torch-scatter>=2.1.2  # optional; fallback active if absent
```

### A.3 MonotonicRadialLoss.forward() — Before/After

**Before** (lines ~741–746, differentiable loop):
```python
level_means = []
level_counts = []
levels_present = []

for v in range(self.max_valuation + 1):
    mask = valuations == v
    if mask.any():
        level_means.append(radii[mask].mean())
        level_counts.append(mask.sum().item())
        levels_present.append(v)

# ...
level_means = torch.stack(level_means)
```

**After** (single scatter call):
```python
from src.utils.scatter_utils import level_scatter_mean, level_has_data

dim_size = self.max_valuation + 1
# level_means_all: (dim_size,) — 0.0 for absent levels (differentiable)
level_means_all = level_scatter_mean(radii, valuations.long(), dim_size=dim_size)
present_mask = level_has_data(valuations.long(), dim_size=dim_size)
levels_present = present_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
level_means = level_means_all[present_mask]  # (n_present,) — differentiable slice
level_counts = [int((valuations == v).sum().item()) for v in levels_present]
```

**Gradient check**: `level_means` is a sliced view of `level_means_all`, which is differentiable w.r.t. `radii`. The gradient flows back through `scatter_add_` in the fallback (PyTorch autograd supports `scatter_add_` as of 1.8) or through `torch_scatter.scatter_mean`.

### A.4 RichHierarchyLoss.forward() — Before/After

**Before** (line ~928):
```python
for v in present_levels:
    mask = valuations == v
    level_radii = radii[mask]
    if level_radii.numel() > 0:
        mean_r = level_radii.mean()
        target_r = target_radii[int(v.item())]
        hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
        if level_radii.numel() > 1:
            variance_loss = variance_loss + level_radii.var()
```

**After**:
```python
from src.utils.scatter_utils import level_scatter_mean, level_scatter_std, level_has_data

dim_size = 10
vals_long = valuations.long()
present_mask = level_has_data(vals_long, dim_size=dim_size)  # (10,) bool

means_all = level_scatter_mean(radii, vals_long, dim_size=dim_size)  # (10,)
stds_all = level_scatter_std(radii, vals_long, dim_size=dim_size)    # (10,) — NEW: free!

# Hierarchy loss: MSE to target radii for present levels
means_present = means_all[present_mask]           # differentiable
targets_present = target_radii[present_mask]      # (n_present,)
hierarchy_loss = ((means_present - targets_present) ** 2).sum()

# Variance loss: sum of per-level std for present levels (was .var(), now .std())
variance_loss = stds_all[present_mask].sum()

# NEW: these can now be logged for free (pass out via metrics dict)
# per_level_std = stds_all.detach()  → log as Radius/std_v{v}_A
```

**New metric unlocked**: `stds_all` is computed as a byproduct with zero extra cost. Pass it out through the loss metrics dict:
```python
# In RichHierarchyLoss.forward() return metrics:
for v in range(dim_size):
    if present_mask[v]:
        metrics[f"r_mean_v{v}"] = means_all[v].item()
        metrics[f"r_std_v{v}"] = stds_all[v].item()  # NEW
```

### A.5 train.py — New TB Logging After Scatter Refactor

After the `RichHierarchyLoss` and `MonotonicRadialLoss` scatter refactor, add these TB tags in the eval block (after the existing `Radius/r_v{v}_A` logging):

```python
# Per-level radius std (from RichHierarchyLoss metrics, accumulated per epoch)
if "rich_hierarchy_metrics" in losses:
    rh_m = losses["rich_hierarchy_metrics"]
    for v in range(10):
        std_key = f"r_std_v{v}"
        if std_key in rh_m:
            tb_logger.writer.add_scalar(f"Radius/std_v{v}_A", rh_m[std_key], epoch)

# Per-level gaps (level_means_all is now (10,) from MonotonicRadialLoss)
if "monotonic_metrics" in losses:
    mono_m = losses["monotonic_metrics"]
    r_means = [mono_m.get(f"r_v{v}") for v in range(10)]
    for v in range(9):
        if r_means[v] is not None and r_means[v+1] is not None:
            gap = r_means[v] - r_means[v+1]
            tb_logger.writer.add_scalar(f"Geometry/level_gap_v{v}", gap, epoch)
```

**TensorBoard custom scalars layout additions** (add to the `add_custom_scalars` call in train.py startup):
```python
"Radius Spread": {
    "Per-Level Std A": ["Multiline", [f"Radius/std_v{v}_A" for v in range(10)]],
},
"Geometry": {
    "Level Gaps": ["Multiline", [f"Geometry/level_gap_v{v}" for v in range(9)]],
},
```

### A.6 compute_level_stratified_hierarchy() — Metrics-Only Simplification

This function is under `torch.no_grad()` and its result is never backpropagated. The loop can be replaced with the fallback scatter for clarity, or left as-is. If replaced:

```python
# Current (line ~524):
for level in range(TERNARY.MAX_VALUATION + 1):
    mask = valuations == level
    count = mask.sum().item()
    if count < 2: correlations[level] = float("nan"); continue
    level_radii = radii[mask]
    radius_std = level_radii.std().item()
    correlations[level] = -1.0 / (1.0 + radius_std)

# After scatter (if desired):
from src.utils.scatter_utils import level_scatter_std, level_has_data
dim_size = TERNARY.MAX_VALUATION + 1
stds = level_scatter_std(radii, valuations.long(), dim_size=dim_size)
# Note: radii here is already a tensor (poincare_distance output)
correlations = {}
for level in range(dim_size):
    has = (valuations == level).sum().item()
    correlations[level] = float("nan") if has < 2 else -1.0 / (1.0 + stds[level].item())
```

This is a minor cleanup (loop still present but body is simpler). The payoff here is minor vs. the differentiable loops above — prioritize A.3 and A.4 first.

### A.7 Expected Speedup and Risk Assessment

| Loop | Current cost (B=4096) | After scatter | Risk |
|------|-----------------------|---------------|------|
| `MonotonicRadialLoss` | 10× masking + 10× `.mean()` | 1 scatter kernel | Low — test shows same output |
| `RichHierarchyLoss` | N× masking + N× `.mean()` + N× `.var()` | 2 scatter kernels | Low — new `r_std_v{v}` is bonus metric |
| `compute_level_stratified_hierarchy` | 10× masking + 10× `.std()` | 1 scatter kernel | Negligible — metrics-only |

**Test requirement**: Before and after each refactor, verify on fixed random seed that:
```python
# In tests/test_losses.py (new test):
def test_scatter_matches_loop():
    radii = torch.rand(500, dtype=torch.float64)
    valuations = torch.randint(0, 10, (500,))
    loop_means = [radii[valuations == v].mean() for v in range(10) if (valuations == v).any()]
    scatter_means = level_scatter_mean(radii, valuations, dim_size=10)
    for v in range(10):
        if (valuations == v).any():
            assert abs(scatter_means[v].item() - radii[valuations == v].mean().item()) < 1e-10
```

---

## Appendix B: scatter_std NaN Gradient — Deep Analysis (2026-03-24)

### B.1 Root Cause

`torch_scatter.scatter_std` computes per-group standard deviation via:

```
std(group) = sqrt( E[(x - E[x])²] )
```

Backward: `d/dx sqrt(v)|_{v=0} = 1/(2·sqrt(0)) = ∞ → NaN`

This NaN occurs when a group has 0 or 1 samples (variance=0, sqrt gradient undefined).

**Critical additional finding**: The NaN is not contained to the singleton group. It contaminates the entire backward kernel. Backpropagating through ANY group in the same `scatter_std` call will produce NaN gradients for elements in singleton groups, even if those elements are not part of the group being backpropagated. Confirmed empirically:

```python
# Group v=0: singleton [0.5]
# Group v=2: normal [0.3, 0.7, 0.4, 0.6]
x = torch.tensor([0.5, 0.3, 0.7, 0.4, 0.6], requires_grad=True)
idx = torch.tensor([0, 2, 2, 2, 2])
out = scatter_std(x, idx, dim=0, dim_size=3)
out[2].backward()  # backprop through GROUP 2 (normal)
# x.grad = [NaN, -0.365, 0.365, -0.183, 0.183]
# ↑ element 0 (group 0, singleton) is NaN even though we never touched group 0
```

**This means**: even if we only indexed into `stds_all` for levels with ≥2 samples, the backward through `level_scatter_std(radii, ...)` itself would NaN any `radii` element belonging to a singleton level.

### B.2 Per-Level Risk Table (batch_size=4096, dataset=19683)

| v | Total ops | E[per batch] | P(count=0) | P(count=1) | NaN risk |
|---|-----------|-------------|-----------|-----------|----------|
| 0 | 13,122 | 2,730.7 | ~0 | ~0 | None |
| 1 | 4,374  | 910.2   | ~0 | ~0 | None |
| 2 | 1,458  | 303.4   | 1e-137 | 4e-135 | None |
| 3 | 486    | 101.1   | 3e-45  | 3e-43  | None |
| 4 | 162    | 33.7    | 2e-15  | 7e-14  | None |
| 5 | 54     | 11.2    | 1e-5   | 1e-4   | Rare |
| 6 | 18     | 3.75    | 2.4%   | 8.8%   | **Frequent** |
| 7 | 6      | 1.25    | 28.7%  | 35.8%  | **Most batches** |
| 8 | 2      | 0.42    | 66.0%  | 27.5%  | **Most batches** |
| 9 | 1      | 0.21    | 81.2%  | 16.9%  | **Most batches** |

Without the fix, training would produce NaN gradients in approximately:
- Every batch containing exactly 1 sample from v=6 (~9% of batches)
- Most batches for v=7/8/9 due to frequent singletons

### B.3 Fix Applied

Replaced `scatter_std` in the differentiable loss path with variance computed via `scatter_add_`:

```python
# BEFORE (NaN gradient at v=6,7,8,9):
variance_loss = stds_all[present_mask].mean()  # used scatter_std output

# AFTER (safe for all levels):
deviations = radii - means_all[vals_long]       # (B,) zero-mean per level
variance_all = zeros(10).scatter_add_(0, vals_long, deviations**2) / counts_all.clamp(min=1)
variance_loss = variance_all[present_mask].mean()
```

`scatter_add_` gradient is `1` (linear accumulation) — no sqrt, no divide-by-zero in backward.

For metrics logging only, `stds_all = variance_all.clamp(min=0).sqrt()` is computed under `torch.no_grad()`. Since no `.backward()` is ever called on `stds_all`, the sqrt gradient issue is irrelevant.

### B.4 Why `variance_all` Gradient is Safe

The gradient of `variance_all[v] = sum((r_i - mean_v)²) / n_v` w.r.t. `r_i` is:

```
∂ variance_all[v] / ∂ r_i = 2 * (r_i - mean_v) / n_v
```

At variance=0 (all samples identical): `r_i = mean_v` → gradient = 0. This is correct and defined. No NaN.

### B.5 Summary of Final Code

| Code path | Uses | Safe? |
|-----------|------|-------|
| `hierarchy_loss` | `means_all` via `scatter_add_` | ✓ |
| `variance_loss` | `variance_all` via `scatter_add_` | ✓ |
| `stds_all` metric | `variance_all.sqrt()` under `no_grad` | ✓ |
| NaN guard in metrics | `if std_val != std_val else 0.0` | ✓ |

`level_scatter_std` is no longer imported in `padic_geodesic.py`. It remains in `scatter_utils.py` for external use with the `# noqa: NaN-grad-at-std-0` warning.

---

## Appendix: Mathematical Justification for Hyperbolic Distance in Viz

The Poincaré ball with curvature c=1 has distance:
```
d_P(x, y) = (2/√c) arctanh(√c · ||(-x) ⊕ y||)
```
where ⊕ is the Möbius addition. For our 16-dim case, this is implemented in `src/geometry/poincare.py → poincare_distance()`.

The Euclidean distance `||x - y||₂` does NOT reflect the learned hierarchy — two points at the same Euclidean distance from the origin can have very different hyperbolic distances depending on their radii. Points near the boundary (v=0) have exponentially larger hyperbolic distances than their Euclidean proximity suggests.

UMAP with `metric="precomputed"` accepting `D_hyp` will correctly reflect this geometry. The resulting 2D/3D projection will show v=0 points spread widely even if their raw 16-dim coordinates are only slightly farther from the origin than v=1 points — because hyperbolic distance correctly amplifies that difference.

This is the difference between a visualization that confirms the architecture works versus one that shows nothing meaningful.
