#!/usr/bin/env python3
"""LSB Resolution Probe.

Analyzes reconstruction accuracy for each ternary digit position (0=LSB, 8=MSB).
Helps isolate whether the LSB resolution gap is a systematic bottleneck
in the VAE encoder/decoder backbone.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.utils.checkpoint import load_checkpoint_compat

def probe_lsb_accuracy(checkpoint_path):
    print(f"Probing LSB accuracy: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Load config
    run_dir = Path(checkpoint_path).parents[1]
    import yaml
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    # 2. Reconstruct model (auto-detect pos_enc)
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    model = TernaryVAEV6Controllable(
        **{k: v for k, v in model_cfg.items() if k not in ["name", "projection_layers", "tangent_scale"]},
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        tangent_scale_init=model_cfg.get("tangent_scale", 0.1)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(torch.float64)
    
    # 3. Full Domain Probe
    all_indices = torch.arange(TERNARY.N_OPERATIONS)
    all_ternary = TERNARY.to_ternary(all_indices)
    
    with torch.no_grad():
        out = model(all_ternary)
        logits = out["logits_A"]
        # Reshape to (N, 9, 3)
        logits_3 = logits.view(-1, 9, 3)
        preds = torch.argmax(logits_3, dim=-1) - 1
        
        # Compare to targets (-1, 0, 1)
        targets = all_ternary.long()
        correct = (preds == targets) # (N, 9)

        # Accuracy per position (0 = LSB)
        pos_acc = correct.float().mean(dim=0).cpu().numpy()
        
    print("
Accuracy by digit position (0=LSB, 8=MSB):")
    for i, acc in enumerate(pos_acc):
        print(f"  Pos {i}: {acc:.4%}")

    # Correlate LSB accuracy with valuation level
    valuations = TERNARY.valuation(all_indices).cpu().numpy()
    lsb_correct = correct[:, 0].cpu().numpy()
    
    print("
LSB (Pos 0) Accuracy by valuation level:")
    for v in range(TERNARY.MAX_VALUATION + 1):
        mask = (valuations == v)
        if mask.any():
            print(f"  v={v}: {lsb_correct[mask].mean():.4%}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(Path("runs").glob("v11*"))[-1] / "checkpoints" / "final.pt"
    probe_lsb_accuracy(path)
