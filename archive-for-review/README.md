# Archive for Review

This directory contains code and configuration that was removed from the active
codebase during the 2026-03-10 audit. Items here are preserved for potential
future reintegration, not deleted.

## Motive: `archive-for-review`

Each item was archived because it meets one or more of these criteria:
- **Dead code**: Defined but never called from production training path
- **Config drift**: YAML keys that have no corresponding implementation
- **Experimental**: Prototypes not yet ready for production

## Contents

### `dead_code/`

| File | Original Location | Reason |
|------|-------------------|--------|
| `learnable_lr_controller.py` | `src/models/lr_controller.py` | Experimental MLP-based controller; never used in training |
| `schedule_based_lr.py` | `src/models/lr_controller.py` | Predetermined schedule controller; superseded by MetricBasedLR |
| `combined_geodesic_loss.py` | `src/losses/padic_geodesic.py` | Curriculum wrapper; superseded by CombinedLoss factory |
| `checkpoint_validator.py` | `src/utils/checkpoint_validator.py` | Config validator; superseded by ModelAuditor in train.py |

### `config_drift/`

| File | Reason |
|------|--------|
| `v6_archived_keys.yaml` | YAML config keys removed from v6.yaml (no implementation exists) |

## Reintegration Guide

To reintroduce any archived item:
1. Copy the file back to its original location (see table above)
2. Re-add the import to the relevant `__init__.py`
3. Wire the feature into `train.py` or the appropriate consumer
4. Add tests covering the reintroduced functionality
5. Remove from this archive
