#!/usr/bin/env python3
# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Canonical Training Entry Point for p-adic VAE.

Modularized version (V6.2):
- src/training/bootstrap.py: Setup and auditors
- src/training/setup.py: Component initialization
- src/training/engine.py: Training loops
- src/training/reporting.py: Checkpointing and logging
- src/training/contracts.py: Typed contracts
"""

import argparse
import atexit
from pathlib import Path
import sys

import torch
import yaml

from src.config.schema import normalize_config
from src.training.bootstrap import DataAuditor, ModelAuditor, get_timestamp, set_determinism
from src.training.engine import train_model
from src.training.metrics import GrokkingDetector
from src.training.reporting import ReportingManager
from src.training.setup import (
    setup_controller,
    setup_dataloaders,
    setup_lagrangian,
    setup_losses,
    setup_optimizer,
    setup_scheduler,
)
from src.utils import TensorBoardLogger, VisualizationPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Unified P-Adic VAE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit config and data then exit without training",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Fail if CUDA is not available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override config seed",
    )

    args = parser.parse_args()

    # 1. Load and Validate Configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    try:
        config = normalize_config(raw_config)
        print(f"[OK] Config loaded and validated: {config_path.name}\n")
    except Exception as e:
        print(f"[ERROR] Config validation failed:\n{e}")
        sys.exit(1)

    # 2. Hardware & Determinism
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_gpu and device.type != "cuda":
        print("[ERROR] --force-gpu specified but CUDA not available.")
        sys.exit(1)

    use_amp = config.get("device", {}).get("use_amp", False)
    if use_amp and device.type == "cuda":
        # Model uses float64 globally; CUDA autocast only downcasts float32→float16/bf16
        # and leaves float64 untouched, so AMP provides no benefit here.
        print("[WARN] use_amp=true has no effect: the model uses float64 throughout. "
              "Disable use_amp to remove the GradScaler overhead.")

    seed = args.seed if args.seed is not None else config.get("training", {}).get("seed", 42)
    set_determinism(seed)

    # 3. Data Auditing
    data_auditor = DataAuditor(seed=seed)
    train_ds, val_ds, _ = data_auditor.prepare_data(
        val_frac=config.get("training", {}).get("val_frac", 0.1),
        device=device,
        custom_indices_path=config.get("data", {}).get("indices_path"),
    )

    # 4. Component Initialization
    train_loader, val_loader = setup_dataloaders(train_ds, val_ds, config, seed)

    # Extract loss params (e.g. learnable weights) if any
    loss_fn, loss_fn_b = setup_losses(config, device)
    loss_params = []
    if hasattr(loss_fn, "parameters"):
        loss_params.extend(list(loss_fn.parameters()))
    if hasattr(loss_fn_b, "parameters"):
        loss_params.extend(list(loss_fn_b.parameters()))

    model_auditor = ModelAuditor(config, device)
    model = model_auditor.create_and_validate_model()
    optimizer = setup_optimizer(model, config, loss_params)
    scheduler = setup_scheduler(optimizer, config)
    lr_controller = setup_controller(config)
    dual_state = setup_lagrangian(config)

    # 5. Validation-only mode
    if args.validate_only:
        print("\n[OK] Validation complete. Exiting (--validate-only)")
        sys.exit(0)

    # 6. Run Directory Setup
    timestamp = get_timestamp()
    run_name = f"{config_path.stem}_{timestamp}"
    log_dir = PROJECT_ROOT / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run directory: {log_dir}")

    # 7. Logger Setup
    tb_logger = TensorBoardLogger(log_dir)
    reporting = ReportingManager(log_dir, config, tb_logger)

    vis_cfg = config.get("visualization", {})
    if "html_dir" not in vis_cfg:
        vis_cfg["html_dir"] = log_dir / "visualizations"
    vis_pipeline = VisualizationPipeline(vis_cfg, tb_logger)

    # Save a copy of the validated config
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # 8. Late Registration (Ensure cleanup on crash)
    atexit.register(tb_logger.close)

    # 9. Optional grokking detection
    grokking_cfg = config.get("training", {}).get("grokking_detection", {})
    grokking_detector = None
    if grokking_cfg.get("enabled", False):
        grokking_detector = GrokkingDetector(
            window=grokking_cfg.get("monitor_window", 20),
            slope_eps=grokking_cfg.get("plateau_threshold", 1e-4),
            val_lift_min=grokking_cfg.get("accuracy_jump_threshold", 0.02),
        )
        print(f"  [OK] Grokking detector enabled (window={grokking_detector.window})")

    # 10. Training Execution
    print("\n[OK] Starting training engine...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        loss_fn_b=loss_fn_b,
        device=device,
        reporting=reporting,
        config=config,
        lr_controller=lr_controller,
        dual_state=dual_state,
        grokking_detector=grokking_detector,
        vis_pipeline=vis_pipeline,
        use_amp=use_amp,
    )

    # 11. Summary
    reporting.save_results(results)
    reporting.print_summary(results)


if __name__ == "__main__":
    main()
