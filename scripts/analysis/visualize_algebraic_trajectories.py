#!/usr/bin/env python3
"""Algebraic Latent Calculator - CSV Trajectory Exporter."""

import sys
from pathlib import Path
import torch
import yaml
import csv

# Set project root for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.utils.checkpoint import load_checkpoint_compat

def visualize_trajectories(checkpoint_path):
    print(f"Exporting algebraic trajectories for: {checkpoint_path}")
    run_dir = Path(checkpoint_path).parents[1]
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"].copy()
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    # Remove conflicting keys
    for k in ["name", "projection_layers", "tangent_scale", "positional_encoding"]:
        if k in model_cfg:
            del model_cfg[k]
    
    model = TernaryVAEV6Controllable(
        **model_cfg,
        positional_encoding=pos_enc,
        n_projection_layers=config["model"].get("projection_layers", 1),
        tangent_scale_init=config["model"].get("tangent_scale", 0.1)
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 1. Sample pairs
    torch.manual_seed(42)
    n_samples = 100
    idx_a = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    idx_b = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    
    # Addition targets
    idx_sum = TERNARY.ternary_add(idx_a, idx_b)
    
    with torch.no_grad():
        mu_a = model.get_mu_representations(idx_a, torch.device('cpu'))
        mu_b = model.get_mu_representations(idx_b, torch.device('cpu'))
        mu_sum = mu_a + mu_b
        
        # Project to ball
        z_sum, _ = model.projections.proj_A(mu_sum)
        z_sum = z_sum.numpy()
        
        z_a_plus_b, _ = model.projections.proj_A(model.get_mu_representations(idx_sum, torch.device('cpu')))
        z_a_plus_b = z_a_plus_b.numpy()

    # 2. Export to CSV (using first 2 dims as projection)
    csv_path = run_dir / "algebraic_trajectories.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_sum', 'y_sum', 'x_gt', 'y_gt'])
        for i in range(n_samples):
            writer.writerow([z_sum[i,0], z_sum[i,1], z_a_plus_b[i,0], z_a_plus_b[i,1]])
    print(f"Trajectories exported to {csv_path}")

if __name__ == "__main__":
    runs = sorted(Path("runs").glob("v11_multiplicative_*"))
    if not runs:
        print("No V11 runs found.")
        sys.exit(1)
    
    ckpt_path = runs[-1] / "checkpoints" / "final.pt"
    visualize_trajectories(ckpt_path)
