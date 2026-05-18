#!/usr/bin/env python3
"""LSB Accuracy Probe: Analyze reconstruction fidelity by digit position."""

import sys
from pathlib import Path
import torch
import numpy as np
import yaml

# Set project root for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.utils.checkpoint import load_checkpoint_compat

def probe_lsb_accuracy(checkpoint_path):
    print(f"Analyzing LSB accuracy for: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    run_dir = Path(checkpoint_path).parents[1]
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
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
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_indices = torch.arange(TERNARY.N_OPERATIONS, device=device)
    all_ternary = TERNARY.to_ternary(all_indices) 

    with torch.no_grad():
        out = model(all_ternary)
        logits = out["logits_A"]
        logits_3 = logits.view(-1, 9, 3)
        preds = torch.argmax(logits_3, dim=-1) - 1
        
        correct = (preds == all_ternary.long())
        pos_acc = correct.float().mean(dim=0).cpu().numpy()

    print("Accuracy by digit position (0=LSB, 8=MSB):")
    for i, acc in enumerate(pos_acc):
        print("  Pos " + str(i) + ": " + "{:.4%}".format(acc))

    valuations = TERNARY.valuation(all_indices).cpu().numpy()
    
    print("LSB (Pos 0) Accuracy vs Valuation:")
    lsb_correct = correct[:, 0].float().cpu().numpy()
    for v in range(TERNARY.MAX_VALUATION + 1):
        mask = (valuations == v)
        if mask.any():
            val_acc = lsb_correct[mask].mean()
            print("  v=" + str(v) + ": " + "{:.4%}".format(val_acc) + " (n=" + str(mask.sum()) + ")")

if __name__ == "__main__":
    runs = sorted(Path("runs").glob("v11_multiplicative_*"))
    if not runs:
        print("No V11 runs found.")
        sys.exit(1)
    
    ckpt_path = runs[-1] / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    probe_lsb_accuracy(ckpt_path)
