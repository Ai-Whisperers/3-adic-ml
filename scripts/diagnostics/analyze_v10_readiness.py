#!/usr/bin/env python3
from pathlib import Path
import sys

import torch


def analyze_readiness(checkpoint_path):
    print(f"Analyzing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Check for StateNet controller state
    if "lr_controller_state" in ckpt:
        print("\n[OK] MetricBasedLR state persisted:")
        statenet_state = ckpt["lr_controller_state"]
        print(f"  Best Q: {statenet_state.get('best_q', 'N/A'):.4f}")
        print(f"  Active states: {statenet_state.get('active', 'N/A')}")
        print(f"  Last epoch: {statenet_state.get('last_epoch', 'N/A')}")
    else:
        print("\n[WARN] No MetricBasedLR state found in checkpoint.")

    # Check for Lagrangian Dual state
    if "lagrangian_state" in ckpt:
        print("\n[OK] LagrangianDualState persisted:")
        lag_state = ckpt["lagrangian_state"]
        print(f"  Epoch: {lag_state.get('epoch', 'N/A')}")
        print(f"  Max lambda_prior: {max(lag_state.get('lambda_prior', [0])):.4f}")
        print(f"  Max lambda_margin: {max(lag_state.get('lambda_margin', [0])):.4f}")
    else:
        print("\n[WARN] No LagrangianDualState found in checkpoint.")

    # Model parameters
    state_dict = ckpt['model_state_dict']
    print(f"\nKeys in state_dict (sample): {list(state_dict.keys())[:10]}")

    print("\nModel Parameter Analysis:")

    # 1. Curvature
    c_keys = [k for k in state_dict.keys() if "manifold.c" in k]
    for k in c_keys:
        print(f"  Curvature parameter ({k}): {state_dict[k].item():.6f}")

    # 2. Tangent Scales
    ts_keys = [k for k in state_dict.keys() if "tangent_scale" in k]
    for k in ts_keys:
        print(f"  Tangent Scale ({k}): {state_dict[k].item():.6f}")

    # 3. Component Norms
    print("\nWeight Norms (Stability Check):")
    for name, param in state_dict.items():
        if "weight" in name and "norm" not in name:
            norm = torch.norm(param).item()
            if norm > 100:
                print(f"  [CRITICAL] High norm in {name}: {norm:.2f}")
            elif norm < 1e-6:
                print(f"  [WARN] Zero norm in {name}: {norm:.2e}")

    # 4. Algebraic Support
    if any("algebraic" in k for k in state_dict.keys()):
        print("\n[INFO] Model contains algebraic-specific parameters (not expected in base VAE, checking CombinedLoss instead).")

    # Check CombinedLoss weights if learnable
    if "loss_state_dict" in ckpt:
        loss_sd = ckpt["loss_state_dict"]
        if any("weight" in k for k in loss_sd.keys()):
            print("\nLearnable Loss Weights:")
            for k, v in loss_sd.items():
                if "weight" in k:
                    print(f"  {k}: {v.item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        # Auto-find latest run
        runs = sorted(Path("runs").glob("v10_algebraic_*"))
        if not runs:
            print("No v10 runs found.")
            sys.exit(1)
        checkpoint_path = runs[-1] / "checkpoints" / "final.pt"

    analyze_readiness(checkpoint_path)
