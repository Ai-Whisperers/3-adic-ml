# Project Structure Documentation

## Directory Layout

```
vaes_ai/
├── configs/                    # Root-level configs (extended/production)
│   └── v5_12_extended.yaml
├── models/                     # Trained model artifacts
│   ├── INDEX.md               # Checkpoint documentation
│   └── checkpoints/
│       ├── v5_5/              # V5.5 Euclidean bridge models
│       ├── v5_11/             # V5.11 StateNet models
│       │   ├── v5_11_11_production/  # RECOMMENDED production checkpoint
│       │   └── ...
│       └── v5_12/             # V5.12 experimental
├── runs/                       # Training outputs (gitignored)
│   └── checkpoints/           # New checkpoints saved here
├── src/                        # Source code
│   ├── config/                # Configuration constants and paths
│   │   ├── __init__.py
│   │   ├── constants.py       # STATENET_* constants
│   │   └── paths.py           # PROJECT_ROOT, RUNS_DIR, etc.
│   ├── configs/               # Training configuration YAML files
│   ├── core/                  # Core ternary math
│   │   └── ternary.py
│   ├── data/                  # Dataset generation
│   │   └── generation.py
│   ├── geometry/              # Hyperbolic/Poincare geometry
│   │   └── poincare.py
│   ├── losses/                # Loss functions
│   │   └── padic_geodesic.py
│   ├── models/                # Model definitions
│   │   ├── statenet.py        # StateNet controller
│   │   ├── vae.py             # TernaryVAE models
│   │   └── hyperbolic_projection.py
│   ├── utils/                 # Utilities
│   │   ├── checkpoint.py
│   │   ├── checkpoint_validator.py
│   │   ├── coverage_evaluator.py
│   │   └── tensorboard_logger.py
│   ├── launch_statenet_training.py
│   ├── train_v5_12_7_scientific_rigor.py
│   └── train_validated_unbiased.py
└── venv/                       # Virtual environment (gitignored)
```

## Path Conventions

### Input Checkpoints (Pre-trained models)
All pre-trained checkpoints are stored in `models/checkpoints/`:
- `models/checkpoints/v5_5/latest.pt` - V5.5 Euclidean bridge
- `models/checkpoints/v5_11/v5_11_11_production/best.pt` - Production V5.11
- `models/checkpoints/v5_12/v5_12_4/best_Q.pt` - V5.12 experimental

### Output Checkpoints (Training outputs)
New checkpoints are saved to `runs/checkpoints/<experiment_name>/`:
- `runs/checkpoints/v5_12/` - V5.12 training runs
- `runs/checkpoints/v5_12_5_statenet_diff/` - Differential StateNet experiments

### Configuration Files
- `configs/` - Legacy/deprecated configurations
- `src/presets/` - Training preset configurations

## Module Imports

```python
# Configuration
from src.config import PROJECT_ROOT, RUNS_DIR, CHECKPOINTS_DIR
from src.config.constants import STATENET_*, N_TERNARY_OPERATIONS

# Core modules
from src.core import TERNARY, TernarySpace, valuation, distance
from src.data import generate_all_ternary_operations
from src.geometry import poincare_distance, get_manifold, ManifoldParameter

# Models
from src.models import StateNet, compute_Q, TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze

# Losses
from src.losses import PAdicGeodesicLoss, RichHierarchyLoss, RadialHierarchyLoss

# Utilities
from src.utils import (
    CheckpointValidator,
    validate_training_config,
    evaluate_coverage,
    TensorBoardLogger
)
```

## Refactoring History

### 2026-01-23: Path Standardization
- All `sandbox-training/checkpoints/` references updated to:
  - Input: `models/checkpoints/`
  - Output: `runs/checkpoints/`
- All `checkpoints/` (root-relative) updated to `models/checkpoints/`

### 2026-01-21: Homeostasis → StateNet Rename
- `homeostasis.py` renamed to `statenet.py`
- All internal imports updated to use `statenet` module
- Class name `StateNet` preserved (was already correct)
- Constants renamed from `HOMEOSTASIS_*` to `STATENET_*`
