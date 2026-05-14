# Copyright 2024-2025 AI Whisperers

"""Tests for visualization utilities and pipeline control flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils import visualization as viz
from src.utils.visualization import (
    VisualizationPipeline,
    VisualizationRuntimeConfig,
    _poincare_tangent_pca,
    _stratified_subsample,
    compute_hyperbolic_distance_matrix,
)


class DummyWriter:
    """Minimal TensorBoard-like writer for pipeline tests."""

    def __init__(self) -> None:
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


def test_stratified_subsample_caps_per_level_and_is_reproducible() -> None:
    z_hyp = torch.arange(90, dtype=torch.float64).reshape(30, 3)
    valuations = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    indices = torch.arange(30)

    z_1, val_1, _ = _stratified_subsample(z_hyp, valuations, indices, max_per_level=4, seed=7)
    z_2, val_2, _ = _stratified_subsample(z_hyp, valuations, indices, max_per_level=4, seed=7)

    assert z_1.shape == (12, 3)
    assert np.array_equal(z_1, z_2)
    assert np.array_equal(val_1, val_2)

    counts = np.bincount(val_1, minlength=3)
    assert counts.tolist() == [4, 4, 4]


def test_hyperbolic_distance_matrix_is_symmetric_and_zero_diagonal() -> None:
    z_np = np.array(
        [
            [0.05, 0.00, 0.00],
            [0.00, 0.08, 0.00],
            [0.03, 0.02, 0.01],
        ],
        dtype=np.float32,
    )

    dist = compute_hyperbolic_distance_matrix(z_np, c=1.0)

    assert dist.shape == (3, 3)
    assert np.allclose(dist, dist.T, atol=1e-6)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-7)
    assert np.all(dist >= 0.0)


def test_poincare_tangent_pca_returns_expected_shape() -> None:
    z_np = np.array(
        [
            [0.05, 0.00, 0.00, 0.01],
            [0.00, 0.08, 0.00, 0.02],
            [0.03, 0.02, 0.01, 0.01],
            [0.02, 0.01, 0.04, 0.00],
        ],
        dtype=np.float32,
    )

    coords = _poincare_tangent_pca(z_np, n_components=3, c=1.0)

    assert coords is not None
    assert coords.shape == (4, 3)
    assert coords.dtype == np.float32


def test_visualization_runtime_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_per_level"):
        VisualizationRuntimeConfig.from_mapping({"max_per_level": 0})

    with pytest.raises(ValueError, match="persist_every"):
        VisualizationRuntimeConfig.from_mapping({"persist_every": 0})

    with pytest.raises(ValueError, match="umap_neighbors"):
        VisualizationRuntimeConfig.from_mapping({"umap_neighbors": 1})

    with pytest.raises(ValueError, match="curvature"):
        VisualizationRuntimeConfig.from_mapping({"curvature": 0.0})


def test_visualization_pipeline_validates_tensor_shapes(tmp_path) -> None:
    pipeline = VisualizationPipeline(
        config={"save_html": False, "html_dir": str(tmp_path)},
        writer=DummyWriter(),
    )

    with pytest.raises(ValueError, match="same first dimension"):
        pipeline.run(
            epoch=1,
            z_hyp=torch.randn(4, 3, dtype=torch.float64),
            valuations=torch.tensor([0, 1, 2]),
            indices=torch.arange(3),
        )


def test_visualization_pipeline_runs_expected_steps(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    writer = DummyWriter()
    pipeline = VisualizationPipeline(
        config={"save_html": False, "persist_every": 3, "html_dir": str(tmp_path)},
        writer=writer,
    )

    def fake_distance_matrix(z_np: np.ndarray, c: float = 1.0) -> np.ndarray:
        assert c == 1.0
        return np.zeros((len(z_np), len(z_np)), dtype=np.float32)

    monkeypatch.setattr(viz, "compute_hyperbolic_distance_matrix", fake_distance_matrix)
    monkeypatch.setattr(
        pipeline,
        "_run_umap_step",
        lambda epoch, D, z_np, val_np, html_dir: calls.append("umap"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_pacmap_step",
        lambda epoch, D, z_np, val_np, html_dir: calls.append("pacmap"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_trimap_step",
        lambda epoch, D, val_np, html_dir: calls.append("trimap"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_poincare3d_step",
        lambda epoch, z_np, val_np, html_dir: calls.append("poincare"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_persistence_step",
        lambda epoch, D, html_dir: calls.append("persist"),
    )

    z_hyp = torch.randn(20, 4, dtype=torch.float64) * 0.05
    valuations = torch.arange(20) % 10
    indices = torch.arange(20)

    pipeline.run(epoch=1, z_hyp=z_hyp, valuations=valuations, indices=indices)
    assert calls == ["umap", "pacmap", "trimap", "poincare", "persist"]
    assert writer.flush_calls == 1

    calls.clear()
    pipeline.run(epoch=2, z_hyp=z_hyp, valuations=valuations, indices=indices)
    assert calls == ["umap", "pacmap", "trimap", "poincare"]
    assert writer.flush_calls == 2


# ---------------------------------------------------------------------------
# compute_persistence
# ---------------------------------------------------------------------------

def _small_dist_matrix(n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = rng.random((n, n)).astype(np.float32)
    D = (raw + raw.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def test_compute_persistence_returns_dict() -> None:
    from src.utils.visualization import compute_persistence
    D = _small_dist_matrix()
    result = compute_persistence(D)
    assert isinstance(result, dict)


def test_compute_persistence_empty_without_ripser(monkeypatch) -> None:
    import src.utils.visualization as vis_module
    from src.utils.visualization import compute_persistence
    monkeypatch.setattr(vis_module, "_HAS_RIPSER", False)
    D = _small_dist_matrix()
    result = compute_persistence(D)
    assert result == {}, "Expected empty dict when ripser absent"


def test_compute_persistence_keys_when_ripser_available() -> None:
    try:
        import ripser  # noqa: F401
    except ImportError:
        pytest.skip("ripser not installed")
    from src.utils.visualization import compute_persistence
    D = _small_dist_matrix()
    result = compute_persistence(D)
    assert "betti_0" in result
    assert "betti_1" in result
    assert "dgms" in result


def test_compute_persistence_betti_0_ge_1() -> None:
    """A non-empty distance matrix must have at least one connected component."""
    try:
        import ripser  # noqa: F401
    except ImportError:
        pytest.skip("ripser not installed")
    from src.utils.visualization import compute_persistence
    D = _small_dist_matrix(8)
    result = compute_persistence(D)
    assert result["betti_0"] >= 1


def test_compute_persistence_entropy_non_negative() -> None:
    try:
        import ripser  # noqa: F401
    except ImportError:
        pytest.skip("ripser not installed")
    from src.utils.visualization import compute_persistence
    D = _small_dist_matrix()
    result = compute_persistence(D)
    if "persistence_entropy" in result:
        assert result["persistence_entropy"] >= 0.0


# ---------------------------------------------------------------------------
# CombinedLossOutput TypedDict
# ---------------------------------------------------------------------------

def test_combined_loss_output_importable() -> None:
    from src.losses import CombinedLossOutput
    assert CombinedLossOutput is not None


def test_combined_loss_output_total_key_pattern() -> None:
    """Verify the recommended access pattern works on a real forward pass."""
    import torch

    from src.losses import CombinedLoss, CombinedLossOutput

    config = {
        "rich_hierarchy": {"enabled": True, "hierarchy_weight": 1.0,
                           "coverage_weight": 1.0, "separation_weight": 0.5},
        "geodesic": {"enabled": False},
        "radial": {"enabled": False},
        "monotonic": {"enabled": False},
        "rank": {"enabled": False},
        "hyperbolic_kl": {"enabled": False},
        "angular_coherence": {"enabled": False},
        "within_level_contrastive": {"enabled": False},
        "valuation_prior": {"enabled": False},
        "lagrangian": {"enabled": False},
    }
    loss_fn = CombinedLoss(config, curvature=1.0)

    B = 32
    torch.manual_seed(0)
    z = torch.randn(B, 16, dtype=torch.float64)
    z = z / (z.norm(dim=-1, keepdim=True) + 0.1) * 0.4
    indices = torch.arange(B) % 100
    logits = torch.randn(B, 9, 3)
    # CombinedLoss expects targets in {-1, 0, 1} (shifted to {0,1,2} internally)
    targets = (torch.arange(B * 9) % 3 - 1).reshape(B, 9).long()

    losses: CombinedLossOutput = loss_fn(z, indices, logits, targets, epoch=0)

    assert "total" in losses
    assert isinstance(losses["total"], torch.Tensor)
    assert losses["total"].ndim == 0  # scalar

    # Optional key access via .get — must not raise
    rich_metrics = losses.get("rich_metrics", {})  # type: ignore[attr-defined]
    assert isinstance(rich_metrics, dict)


# ---------------------------------------------------------------------------
# Distance matrix additional properties
# ---------------------------------------------------------------------------

def test_distance_matrix_off_diagonal_positive() -> None:
    """Distinct Poincaré points must have strictly positive pairwise distance."""
    torch.manual_seed(3)
    z = torch.randn(8, 4, dtype=torch.float64)
    z = z / (z.norm(dim=-1, keepdim=True) + 0.2) * 0.3
    D = compute_hyperbolic_distance_matrix(z.float().numpy())
    off_diag = D[~np.eye(8, dtype=bool)]
    assert np.all(off_diag > 0.0), f"Non-positive off-diagonal entry: {off_diag.min()}"


def test_stratified_subsample_empty_level_excluded() -> None:
    """Levels absent from the batch must not appear in output valuations."""
    z = torch.randn(20, 4, dtype=torch.float64) * 0.3
    vals = torch.zeros(20, dtype=torch.long)
    vals[10:] = 2  # levels 0 and 2 only — level 1 absent
    indices = torch.arange(20)
    _, v_out, _ = _stratified_subsample(z, vals, indices, max_per_level=100)
    assert 1 not in v_out.tolist()


def test_stratified_subsample_dtype_contract() -> None:
    z = torch.randn(30, 8, dtype=torch.float64)
    vals = torch.arange(30) % 5
    indices = torch.arange(30)
    z_out, v_out, i_out = _stratified_subsample(z, vals, indices)
    assert z_out.dtype == np.float32, "z output must be float32"
    assert v_out.dtype == np.int32, "val output must be int32"
    assert i_out.dtype == np.int32, "idx output must be int32"
