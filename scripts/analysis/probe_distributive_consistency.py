#!/usr/bin/env python3
"""Distributive Property Probe: Evaluate ring completeness in latent space.

Hypothesis: If the latent space is a true ring homomorphism, it should respect:
    z(a ⊗ (b ⊕ c)) ≈ z(a) ⊙ (z(b) + z(c))
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

# Set project root for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.training.metrics import compute_accuracy

def probe_distributive_consistency(checkpoint_path, n_samples=1000):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    run_dir = Path(checkpoint_path).parents[1]
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    model = TernaryVAEV6Controllable(
        **{k: v for k, v in model_cfg.items() if k not in ["name", "projection_layers", "tangent_scale", "positional_encoding"]},
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        tangent_scale_init=model_cfg.get("tangent_scale", 0.1)
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Sample triplets (a, b, c)
    torch.manual_seed(42)
    idx_a = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)
    idx_b = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)
    idx_c = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)

    # Ground truth: z(a ⊗ (b ⊕ c))
    idx_sum_bc = TERNARY.ternary_add(idx_b, idx_c)
    idx_dist_gt = TERNARY.ternary_mul(idx_a, idx_sum_bc)
    ternary_gt = TERNARY.to_ternary(idx_dist_gt).to(device)

    print(f"Probing distributive consistency (n={n_samples})...")

    with torch.no_grad():
        # Get representations
        mu_a = model.get_mu_representations(idx_a, device)
        mu_b = model.get_mu_representations(idx_b, device)
        mu_c = model.get_mu_representations(idx_c, device)
        
        # Predicted representation: z(a) * (z(b) + z(c))
        mu_pred = mu_a * (mu_b + mu_c)
        
        # Expanded representation: (z(a)*z(b)) + (z(a)*z(c))
        # Mathematically identical in R^D but useful to check for drift
        mu_expanded = (mu_a * mu_b) + (mu_a * mu_c)
        
        # Target representation
        mu_gt = model.get_mu_representations(idx_dist_gt, device)
        
        # Metrics
        mse = torch.nn.functional.mse_loss(mu_pred, mu_gt).item()
        cos_sim = torch.nn.functional.cosine_similarity(mu_pred, mu_gt).mean().item()
        
        # Decoding check
        logits = model.decoder_A(mu_pred)
        preds = torch.argmax(logits.view(-1, 9, 3), dim=-1) - 1
        acc = (preds == ternary_gt.long()).float().mean().item()

    print("\nResults:")
    print(f"  Mu-Space MSE (pred vs gt):   {mse:.6f}")
    print(f"  Mu-Space Cosine Sim:        {cos_sim:.6f}")
    print(f"  Final Digit Accuracy:       {acc:.4%}")

if __name__ == "__main__":
    runs = sorted(Path("runs").glob("v13*"))
    if not runs:
        runs = sorted(Path("runs").glob("v11*"))
    if not runs:
        print("No valid runs found.")
        sys.exit(1)
    
    ckpt_path = runs[-1] / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = runs[-1] / "checkpoints" / "best_Q.pt"
        
    probe_distributive_consistency(ckpt_path)
