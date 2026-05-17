#!/usr/bin/env python3
"""Evaluate Zero-Shot Symbolic Addition in the latent space.

Hypothesis: If z(a) + z(b) ≈ z(a ⊕ b) is enforced in mu space,
then decoding (z(a) + z(b)) should yield the correct ternary result (a ⊕ b).
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

def run_addition_probe(checkpoint_path, n_samples=1000):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config if available (usually in the run dir)
    run_dir = Path(checkpoint_path).parents[1]
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
    else:
        # Fallback to standard V10 settings
        print("[WARN] Config not found, using default V10 settings")
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

    # 1. Sample pairs
    torch.manual_seed(42)
    idx_a = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    idx_b = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    
    # Ground truth sum in index space
    idx_sum_gt = TERNARY.ternary_add(idx_a, idx_b)
    ternary_sum_gt = TERNARY.to_ternary(idx_sum_gt).to(device)

    print(f"\nEvaluating addition accuracy (n={n_samples})...")

    with torch.no_grad():
        # 2. Get representations
        mu_a = model.get_mu_representations(idx_a, device)
        mu_b = model.get_mu_representations(idx_b, device)
        
        # 3. Perform addition in mu space
        mu_sum_pred = mu_a + mu_b
        
        # 4. Decode the result
        # Note: decoder_A expects tangent space input
        logits = model.decoder_A(mu_sum_pred)
        
        # 5. Compute metrics
        acc = compute_accuracy(logits, ternary_sum_gt)
        
        # Coverage check (perfect decoding)
        preds = torch.argmax(logits.view(-1, 9, 3), dim=-1) - 1
        correct_per_sample = (preds == ternary_sum_gt.long()).all(dim=1)
        cov = correct_per_sample.float().mean().item()

        # Per-digit position accuracy
        digit_accs = (preds == ternary_sum_gt.long()).float().mean(dim=0).cpu().numpy()

        # Baseline: get direct mu_sum from model
        mu_sum_gt = model.get_mu_representations(idx_sum_gt, device)
        mu_mse = torch.nn.functional.mse_loss(mu_sum_pred, mu_sum_gt).item()
        cos_sim = torch.nn.functional.cosine_similarity(mu_sum_pred, mu_sum_gt).mean().item()

    print(f"Results:")
    print(f"  Digit Accuracy (overall):      {acc:.4%}")
    print(f"  Perfect Coverage:              {cov:.4%}")
    print(f"  Mu-Space MSE (pred vs gt):     {mu_mse:.6f}")
    print(f"  Mu-Space Cosine Sim:          {cos_sim:.6f}")

    print("\nAccuracy by digit position (0=LSB, 8=MSB):")
    for i, d_acc in enumerate(digit_accs):
        print(f"  Pos {i}: {d_acc:.4%}")

    # Check breakdown by valuation level of the sum
    v_sum = TERNARY.valuation(idx_sum_gt).cpu().numpy()
    print("\nAccuracy by sum valuation level:")
    for v in range(TERNARY.MAX_VALUATION + 1):
        mask = (v_sum == v)
        if mask.any():
            v_acc = compute_accuracy(logits[mask], ternary_sum_gt[mask])
            print(f"  v={v}: {v_acc:.4%} (n={mask.sum()})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Find latest run
        runs = sorted(Path("runs").glob("v10_algebraic_*"))
        if not runs:
            print("No v10 runs found.")
            sys.exit(1)
        path = runs[-1] / "checkpoints" / "final.pt"
        if not path.exists():
            path = runs[-1] / "checkpoints" / "periodic_epoch_5.pt" # diagnostic run
            
    run_addition_probe(path)
