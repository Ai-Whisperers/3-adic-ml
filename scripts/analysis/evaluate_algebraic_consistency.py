#!/usr/bin/env python3
"""Evaluate Zero-Shot Algebraic Consistency in the latent space.

Hypothesis:
- Addition: z(a) + z(b) ≈ z(a ⊕ b)
- Multiplication: z(a) * z(b) ≈ z(a ⊗ b) (element-wise)
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from src.core import TERNARY
from src.utils.checkpoint import load_checkpoint_compat
from src.models.vae import TernaryVAEV6Controllable
from src.training.metrics import compute_accuracy

def run_consistency_probe(checkpoint_path, n_samples=1000):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config if available
    run_dir = Path(checkpoint_path).parents[1]
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
    else:
        print("[WARN] Config not found, using default V10/V11 settings")
        model_cfg = {
            "latent_dim": 64,
            "hidden_dim": 128,
            "max_radius": 0.99,
            "curvature": 1.0,
            "factored": True,
            "radial_dims": 4,
            "positional_encoding": True
        }

    # Auto-detect positional_encoding from state_dict
    pos_enc = False
    if "head_A.backbone.0.weight" in ckpt['model_state_dict']:
        in_dim = ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1]
        if in_dim == 18:
            pos_enc = True

    # Reconstruct model
    model = TernaryVAEV6Controllable(
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        max_radius=model_cfg.get("max_radius", 0.99),
        curvature=model_cfg.get("curvature", 1.0),
        factored=model_cfg.get("factored", True),
        radial_dims=model_cfg.get("radial_dims", 4),
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        projection_dropout=model_cfg.get("projection_dropout", 0.0)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sample pairs
    torch.manual_seed(42)
    idx_a = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    idx_b = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))

    # 1. Evaluate ADDITION
    idx_sum_gt = TERNARY.ternary_add(idx_a, idx_b)
    ternary_sum_gt = TERNARY.to_ternary(idx_sum_gt).to(device)

    print(f"\n--- Addition Consistency (n={n_samples}) ---")
    with torch.no_grad():
        mu_a = model.get_mu_representations(idx_a, device)
        mu_b = model.get_mu_representations(idx_b, device)
        mu_sum_pred = mu_a + mu_b
        logits_add = model.decoder_A(mu_sum_pred)
        acc_add = compute_accuracy(logits_add, ternary_sum_gt)
        mu_sum_gt = model.get_mu_representations(idx_sum_gt, device)
        cos_sim_add = torch.nn.functional.cosine_similarity(mu_sum_pred, mu_sum_gt).mean().item()

    print(f"  Digit Accuracy (decoded): {acc_add:.4%}")
    print(f"  Mu-Space Cosine Sim:     {cos_sim_add:.6f}")

    # 2. Evaluate MULTIPLICATION
    idx_prod_gt = TERNARY.ternary_mul(idx_a, idx_b)
    ternary_prod_gt = TERNARY.to_ternary(idx_prod_gt).to(device)

    print(f"\n--- Multiplicative Consistency (n={n_samples}) ---")
    with torch.no_grad():
        mu_prod_pred = mu_a * mu_b
        logits_mul = model.decoder_A(mu_prod_pred)
        acc_mul = compute_accuracy(logits_mul, ternary_prod_gt)
        mu_prod_gt = model.get_mu_representations(idx_prod_gt, device)
        cos_sim_mul = torch.nn.functional.cosine_similarity(mu_prod_pred, mu_prod_gt).mean().item()

    print(f"  Digit Accuracy (decoded): {acc_mul:.4%}")
    print(f"  Mu-Space Cosine Sim:     {cos_sim_mul:.6f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Prioritize V11 then V10
        runs = sorted(Path("runs").glob("v11*"))
        if not runs:
            runs = sorted(Path("runs").glob("v10*"))
            
        if not runs:
            print("No v10/v11 runs found.")
            sys.exit(1)
            
        path = runs[-1] / "checkpoints" / "final.pt"
        if not path.exists():
            # Try Best Q
            path = runs[-1] / "checkpoints" / "best_Q.pt"
            if not path.exists():
                # Try any checkpoint
                pts = sorted((runs[-1] / "checkpoints").glob("*.pt"))
                if pts:
                    path = pts[-1]
            
    run_consistency_probe(path)
