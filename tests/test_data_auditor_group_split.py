# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Tests for DataAuditor's group-aware train/val split (group_map_path).

The phylogeny pipeline (docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md) documented
a real leakage: the split was by window row, so every species appeared in both
train and val. The group split holds out whole groups instead. These tests
pin the properties that make that fix meaningful: no group straddles the
split, the split is deterministic per seed, and misaligned group maps fail
loudly instead of silently mispairing rows.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.training.bootstrap import DataAuditor


def _write_dataset(tmp_path, groups):
    """Write indices.pt + row-aligned group map for the given group labels."""
    indices = torch.arange(len(groups), dtype=torch.long)
    indices_path = tmp_path / "indices.pt"
    torch.save(indices, indices_path)
    map_path = tmp_path / "group_map.json"
    map_path.write_text(json.dumps([{"species": g} for g in groups]))
    return str(indices_path), str(map_path)


@pytest.fixture
def ten_groups_dataset(tmp_path):
    # 10 groups x 6 rows each = 60 rows
    groups = [f"sp_{i}" for i in range(10) for _ in range(6)]
    return _write_dataset(tmp_path, groups)


class TestGroupSplit:
    def test_no_group_appears_in_both_splits(self, ten_groups_dataset):
        indices_path, map_path = ten_groups_dataset
        auditor = DataAuditor(seed=42)
        train_ds, val_ds, _ = auditor.prepare_data(
            val_frac=0.2, custom_indices_path=indices_path, group_map_path=map_path,
        )
        # Row index == group index // 6 by construction
        train_groups = {int(idx) // 6 for _, idx in train_ds}
        val_groups = {int(idx) // 6 for _, idx in val_ds}
        assert train_groups.isdisjoint(val_groups)
        assert len(train_ds) + len(val_ds) == 60

    def test_val_holds_out_expected_group_count(self, ten_groups_dataset):
        indices_path, map_path = ten_groups_dataset
        auditor = DataAuditor(seed=42)
        _, val_ds, _ = auditor.prepare_data(
            val_frac=0.2, custom_indices_path=indices_path, group_map_path=map_path,
        )
        # 2 of 10 groups -> 12 of 60 rows
        assert len(val_ds) == 12
        assert len(auditor.audit_log["val_groups"]) == 2

    def test_same_seed_same_split(self, ten_groups_dataset):
        indices_path, map_path = ten_groups_dataset
        runs = []
        for _ in range(2):
            auditor = DataAuditor(seed=7)
            auditor.prepare_data(
                val_frac=0.2, custom_indices_path=indices_path, group_map_path=map_path,
            )
            runs.append(auditor.audit_log["val_groups"])
        assert runs[0] == runs[1]

    def test_different_seed_can_change_split(self, ten_groups_dataset):
        indices_path, map_path = ten_groups_dataset
        val_groups = []
        for seed in range(6):
            auditor = DataAuditor(seed=seed)
            auditor.prepare_data(
                val_frac=0.2, custom_indices_path=indices_path, group_map_path=map_path,
            )
            val_groups.append(tuple(auditor.audit_log["val_groups"]))
        assert len(set(val_groups)) > 1

    def test_row_misaligned_map_raises(self, tmp_path):
        indices_path, _ = _write_dataset(tmp_path, ["a"] * 10)
        bad_map = tmp_path / "bad_map.json"
        bad_map.write_text(json.dumps([{"species": "a"}] * 9))
        with pytest.raises(ValueError, match="row-aligned"):
            DataAuditor(seed=42).prepare_data(
                custom_indices_path=indices_path, group_map_path=str(bad_map),
            )

    def test_val_frac_swallowing_all_groups_raises(self, tmp_path):
        indices_path, map_path = _write_dataset(tmp_path, ["a", "a", "b", "b"])
        with pytest.raises(ValueError, match="nothing left to train"):
            DataAuditor(seed=42).prepare_data(
                val_frac=1.0, custom_indices_path=indices_path, group_map_path=map_path,
            )

    def test_no_group_map_keeps_row_split(self, ten_groups_dataset):
        indices_path, _ = ten_groups_dataset
        auditor = DataAuditor(seed=42)
        train_ds, val_ds, _ = auditor.prepare_data(
            val_frac=0.2, custom_indices_path=indices_path,
        )
        assert "val_groups" not in auditor.audit_log
        assert len(val_ds) == 12  # round(60 * 0.2)
