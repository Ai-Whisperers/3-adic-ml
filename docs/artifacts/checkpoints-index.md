# INDEX for the AI .pt checkpoints & models

The current state is the following:

## Directory Structure
```
models/checkpoints/
├── v5_5/                    # V5.5 models (Euclidean bridge)
├── v5_11/                   # V5.11 StateNet models
│   ├── v5_11_11_production/ # Production-ready checkpoint (recommended)
│   ├── v5_11_homeostasis/   # Legacy naming, contains mature models
│   └── ...other variants
├── v5_12/                   # V5.12 experiments
└── non_production_ready_checkpoints/
```

## Model Descriptions

* **v5_5/** - V5.5 full 19683 ternary operations 100% coverage BUT it enforces non-euclidean behaviour through euclidean algorithms, so the mathematical purity is lost. Can be used as a "bridge" system between non-euclidean and euclidean, as well as "continuum and discrete bridge".

* **v5_11/** - StateNet-enabled models with proper hyperbolic geometry:
  - `v5_11_11_production/` - **Recommended** production checkpoint
  - `v5_11_homeostasis/` - Legacy naming, contains mature trained models (note: internal code now uses "StateNet" terminology, folder name preserved for checkpoint compatibility)