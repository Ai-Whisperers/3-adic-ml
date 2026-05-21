# Visualizing the 3-Adic Latent Space 🎨

One of the most powerful aspects of this project is the ability to **see** the mathematics. Because we use hyperbolic geometry, our "maps" of the data look and behave differently than standard Euclidean maps.

## 🛰️ The Visualization Pipeline

During training, the `VisualizationPipeline` automatically generates several types of maps. These are saved as interactive HTML files in `runs/visualizations/<run_name>/`.

### 1. Native Poincaré Disk
This is our "Gold Standard" visualization. 
- **What it shows**: The latent space projected directly onto a 2D disk.
- **Radial Preservation**: The distance from the center represents the **3-adic valuation**. Points at the center are the most "fundamental" (divisible by many powers of 3), while points at the boundary are the most "specific."
- **Direction**: The angle represents the **prefix structure** (the first few digits of the ternary operation).
- **Recent Renders**:
    - **[V7.2 Large Poincaré Disk](v7_large_poincare.html)**: The recommended large-scale baseline.
    - **[V14.1 Ring Completion Poincaré Disk](v14_ring_completion_poincare.html)**: Advanced model exploring ring completion and high algebraic consistency.
- **Features**: 
    - **Tree Edges**: Thin lines connecting "parents" to "children" in the 3-adic tree.
    - **Prefix Shading**: Colored regions showing where different digit-prefix classes live.

### 2. Hyperbolic UMAP & PaCMAP
Traditional algorithms like UMAP usually use Euclidean distance. We have modified them to use the **Hyperbolic Distance Matrix**.
- **UMAP (Uniform Manifold Approximation and Projection)**: Great for seeing the overall "skeleton" of the data.
- **PaCMAP**: Better at balancing local clusters with the global tree structure.
- **Why it matters**: By using hyperbolic distances, we ensure that the "curved" nature of the space is preserved even when we flatten it to 2D or 3D.

### 3. Persistent Homology
We use topology to check if the AI is learning "holes" or "connected components" in the data.
- **Betti Numbers**: We track $H_0$ (connected components). Ideally, as the AI learns the tree structure, the number of connected components should align with the 3-adic branching.

## 🛠️ How to Generate Visualizations

### During Training
Visualizations are enabled by default in the YAML configs:
```yaml
visualization:
  max_per_level: 500     # Subsample size per valuation level
  persist_every: 50      # Generate new files every 50 epochs
  save_html: true
```

### Manual Generation
You can run the analysis scripts to generate specific visualizations:
- **`scripts/analysis/create_evolution_animation.py`**: Creates a video showing how the latent space "unfolds" during training.
- **`scripts/analysis/visualize_algebraic_trajectories.py`**: Shows how the latent space moves when you perform algebraic operations like $n \to n+1$ or $n \to 3n$.

## 🕵️ How to "Read" a Poincaré Map

When you open an interactive HTML visualization:

1.  **The Center is the Root**: The single point at the very center (usually $v=9$) is the "0" index, the root of the 3-adic tree.
2.  **Rings are Valuations**: You will see concentric rings of points. Each ring corresponds to a different 3-adic valuation.
3.  **Wedges are Prefixes**: Points in the same "pie slice" usually share the same starting digits.
4.  **Zooming**: In the Poincaré ball, the "action" happens near the boundary. Zoom in on the edges to see the fine-grained branching of the $v=0$ and $v=1$ levels.

---

*"A picture is worth a thousand valuations."*
