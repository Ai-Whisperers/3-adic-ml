# External Sources & References

## TensorBoard

| Resource | URL |
|----------|-----|
| PyTorch TensorBoard Tutorial | https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html |
| `torch.utils.tensorboard` API | https://pytorch.org/docs/stable/tensorboard.html |
| TensorBoard Get Started (TF) | https://www.tensorflow.org/tensorboard/get_started |
| SummaryWriter API Reference | https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter |

### Key Rules Applied in This Codebase

1. **`add_scalar` only, never `add_scalars`** — `add_scalars` creates phantom sub-run
   directories that clutter the log folder and confuse the TensorBoard UI. Use hierarchical
   tag names instead (e.g., `"Hierarchy/Q_VAE_A"`, `"Hierarchy/Q_VAE_B"`).

2. **`flush()` at end of each eval cycle** — prevents data loss on crash.

3. **`atexit.register(tb_logger.close)`** — guarantees writer is closed even on unclean exit.

4. **Explicit warning when tensorboard is missing** — `tb_logger.is_available` is checked
   at startup and a `[WARN]` message is printed. Previously this was silent.

5. **Log every `eval_every` epochs, not every batch** — scalars logged at epoch granularity
   (default every 5 epochs). Histograms logged less frequently.

6. **Single `SummaryWriter` instance** — created once in `train()`, reused throughout.

7. **Always `.item()` on tensors before logging** — all metrics are Python floats when
   they reach `add_scalar`. No computation graph leaks.

## Hyperbolic Geometry

| Resource | URL |
|----------|-----|
| geoopt (Riemannian optimization) | https://github.com/geoopt/geoopt |
| Poincare Ball Model | https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model |

## P-Adic Mathematics

| Resource | URL |
|----------|-----|
| p-adic numbers (Wikipedia) | https://en.wikipedia.org/wiki/P-adic_number |
| Ultrametric space | https://en.wikipedia.org/wiki/Ultrametric_space |

## Machine Learning

| Resource | URL |
|----------|-----|
| Kendall et al. 2018 (Multi-task loss weighting) | https://arxiv.org/abs/1705.07115 |
| Adjusted Rand Index | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html |
