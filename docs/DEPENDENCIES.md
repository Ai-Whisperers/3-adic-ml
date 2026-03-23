# Dependencies Reference

All dependencies are listed in `requirements.txt` at the project root.
Install with `pip install -r requirements.txt`.

## Required Packages

| Package | Min Version | Used By | Purpose |
|---------|-------------|---------|---------|
| `torch` | 2.0.0 | Entire codebase | Neural network framework |
| `numpy` | 1.24.0 | Entire codebase | Array operations |
| `scipy` | 1.10.0 | `train.py` | Spearman correlation (`scipy.stats.spearmanr`) |
| `geoopt` | 0.5.0 | `geometry/poincare.py`, `hyperbolic_projection.py` | Riemannian optimization, Poincare ball manifold |
| `pyyaml` | 6.0 | `train.py` | YAML config loading |
| `scikit-learn` | 1.3.0 | `train.py`, `diagnose_direction_geometry.py` | K-means clustering, ARI metric |
| `tensorboard` | 2.13.0 | `train.py`, `utils/tensorboard_logger.py` | Training metrics visualization |
| `tqdm` | 4.65.0 | `train.py` | Progress bars |
| `psutil` | 5.9.0 | `utils/hardware_monitor.py` | RAM/GPU memory monitoring |
| `matplotlib` | 3.7.0 | `diagnose_direction_geometry.py` | UMAP plots, cluster visualizations |
| `umap-learn` | 0.5.0 | `diagnose_direction_geometry.py` | 2D direction space projections |
| `pillow` | 10.0.0 | matplotlib backend | Image rendering |
| `pandas` | 2.0.0 | Analysis scripts | Data manipulation |

## Testing

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `pytest` | 7.3.0 | Test runner |
| `pytest-cov` | 4.1.0 | Coverage reporting |

## Critical: TensorBoard

TensorBoard is **required**, not optional. Without it:
- `tb_logger.is_available` evaluates to `False`
- All TensorBoard logging is **silently skipped** — no error, no warning
- Live metrics (Q, ARI, hierarchy, coverage, AQ, LR scales) are not recorded
- Training still runs but you lose all visibility into what the model is doing

If you see no TensorBoard event files in your run directory, this is almost
certainly because tensorboard is not installed. Fix:

```bash
pip install tensorboard>=2.13.0
```

Verify:

```bash
python -c "import tensorboard; print(tensorboard.__version__)"
```

## Checking Your Environment

```bash
# Verify all required packages
python -c "
import torch, numpy, scipy, geoopt, yaml, sklearn, tensorboard, tqdm, psutil
print(f'torch {torch.__version__}')
print(f'numpy {numpy.__version__}')
print(f'scipy {scipy.__version__}')
print(f'geoopt {geoopt.__version__}')
print(f'sklearn {sklearn.__version__}')
print(f'tensorboard {tensorboard.__version__}')
print('All OK')
"
```
