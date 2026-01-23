from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"  # Canonical checkpoint location
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SRC_CONFIGS_DIR = PROJECT_ROOT / "src" / "configs"
