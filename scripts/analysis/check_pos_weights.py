#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def check_pos_weights(checkpoint_path):
    print(f"Checking positional weights in: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model_state_dict']
    
    if "pos_weights" in sd:
        weights = sd["pos_weights"].squeeze()
        print(f"Positional Weights (9 digits):")
        for i, w in enumerate(weights):
            print(f"  Digit {i}: {w.item():.6f}")
    else:
        print("No pos_weights found in state_dict.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/v10.1_algebraic_index_20260517_124236/checkpoints/epoch_50.pt"
    check_pos_weights(path)
