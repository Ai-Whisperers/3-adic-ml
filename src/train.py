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
import os
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import normalize_config
from src.training.bootstrap import DataAuditor, ModelAuditor, set_determinism, get_timestamp
from src.training.setup import (
    setup_dataloaders,
    setup_losses,
    setup_optimizer,
    setup_scheduler,
    setup_controller,
    setup_lagrangian,
)
from src.training.engine import train_model
from src.training.reporting import ReportingManager, _build_checkpoint_payload
from src.utils import TensorBoardLogger


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Unified P-Adic VAE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Target device (cuda or cpu)"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Run audit and exit"
    )
    parser.add_argument(
        "--force", action="store_true", help="Continue even if audit fails"
    )
    args = parser.parse_args()

    # 1. Load and Normalize Config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    try:
        config = normalize_config(raw_config)
        print(f"\n[OK] Config loaded and validated: {config_path.name}")
    except Exception as e:
        print(f"[ERROR] Config validation failed: {e}")
        if not args.force:
            sys.exit(1)
        config = raw_config

    # 2. Setup Determinism
    seed = args.seed
    set_determinism(seed, deterministic=True, use_float64=True)

    # 3. Setup Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 4. Data Audit & Preparation
    data_auditor = DataAuditor(seed)
    train_ds, val_ds, _ = data_auditor.prepare_data(
        val_frac=config.get("training", {}).get("val_frac", 0.1),
        device=device
    )

    # 5. Model Audit & Creation
    model_auditor = ModelAuditor(config, device)
    model = model_auditor.create_and_validate_model(force=args.force)

    if args.validate_only:
        print("\n[OK] Validation complete. Exiting (--validate-only)")
        sys.exit(0)

    # 6. Run Directory Setup
    timestamp = get_timestamp()
    run_name = f"{config_path.stem}_{timestamp}"
    log_dir = PROJECT_ROOT / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run directory: {log_dir}")

    # 7. Component Setup
    train_loader, val_loader = setup_dataloaders(train_ds, val_ds, config, seed)
    loss_fn, loss_fn_b = setup_losses(config, device)
    
    # Extract loss parameters if learnable weights enabled
    loss_cfg = config.get("loss", {})
    loss_params = (
        list(loss_fn.parameters()) + list(loss_fn_b.parameters())
        if loss_cfg.get("learnable_weights", False) else []
    )
    
    optimizer = setup_optimizer(model, config, loss_params)
    scheduler = setup_scheduler(optimizer, config)
    lr_controller = setup_controller(config)
    dual_state = setup_lagrangian(config)

    # 8. Reporting Setup
    tb_logger = TensorBoardLogger(
        tensorboard_dir=str(log_dir),
        experiment_name=run_name,
        log_callback=lambda msg: print(f"  {msg}") if config.get("logging", {}).get("verbose", False) else None
    )
    atexit.register(tb_logger.close)
    
    reporting = ReportingManager(log_dir, config, tb_logger)

    # 9. Train Model
    print(f"\n[OK] Starting training engine...")
    results = train_model(
        config=config,
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        loss_fn_b=loss_fn_b,
        reporting=reporting,
        lr_controller=lr_controller,
        dual_state=dual_state,
        use_amp=config.get("device", {}).get("use_amp", False),
    )

    # 10. Summary
    reporting.save_results(results)
    reporting.print_summary(results)


if __name__ == "__main__":
    main()
