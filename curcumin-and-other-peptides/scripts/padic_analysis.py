import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from src.core import TERNARY

# 1. Amino Acid Mapping (Hydropathy-based)
# -1: Hydrophilic (D, E, N, Q, K, R)
#  0: Neutral (G, S, T, Y, P, H)
#  1: Hydrophobic (V, L, I, M, F, W, C, A)
AA_MAP = {
    'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1,
    'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0,
    'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1
}

KNOWN_PEPTIDES = {
    "Antioxidant (T)": "WTLTPLTPA",
    "Cur-1 (T)": "KLHLLILI",
    "ACE-Inh (T)": "CACGGV",
    "P2 Peptide (G)": "RALGWSCL",
    "Antioxidant (G)": "SVAGRAQGM",
    "Antihypertensive (G)": "VTYM"
}

def encode_peptide(seq, length=9):
    # Pad or truncate to length
    encoded = []
    for i in range(length):
        if i < len(seq):
            aa = seq[i].upper()
            encoded.append(AA_MAP.get(aa, 0))
        else:
            encoded.append(0) # Padding
    return torch.tensor(encoded, dtype=torch.float64).unsqueeze(0)

def load_model(ckpt_path, config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    mapping = {
        'projection_layers': 'n_projection_layers',
        'tangent_scale': 'tangent_scale_init',
        'projection_dropout': 'projection_dropout'
    }
    init_cfg = {mapping.get(k, k): v for k, v in model_cfg.items() if k not in ['name']}
    model = TernaryVAEV6Controllable(**init_cfg)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def find_latest_run_paths(run_prefix="v19.1_peptide_retrain"):
    import glob
    base_dir = Path(__file__).resolve().parents[2]
    runs_dir = base_dir / "runs"
    matching_runs = sorted(glob.glob(str(runs_dir / f"{run_prefix}_*")))
    if matching_runs:
        latest_run = Path(matching_runs[-1])
        ckpt = latest_run / "checkpoints" / "best_Q.pt"
        cfg = latest_run / "config.yaml"
        if ckpt.exists() and cfg.exists():
            return str(ckpt), str(cfg)
    
    # Fallback to the archived Phase 11 model if no v19.1 run is found
    ckpt = base_dir / "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/checkpoints/best_Q.pt"
    cfg = base_dir / "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/config.yaml"
    return str(ckpt), str(cfg)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P-adic Peptide Latent Analysis")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--cfg", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    if args.ckpt and args.cfg:
        ckpt, cfg = args.ckpt, args.cfg
    else:
        ckpt, cfg = find_latest_run_paths()
    
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found at {ckpt}")
        return

    print(f"Loading model from {ckpt}...")
    model = load_model(ckpt, cfg)
    
    embeddings = {}
    with torch.no_grad():
        for name, seq in KNOWN_PEPTIDES.items():
            x = encode_peptide(seq)
            output = model(x)
            # Use z_A_hyp (Hyperbolic embedding in Poincaré ball)
            z_hyp = output['z_A_hyp'].numpy()[0]
            embeddings[name] = z_hyp
            
            # Compute radius
            radius = np.linalg.norm(z_hyp)
            print(f"{name:20} | Seq: {seq:10} | Radius: {radius:.4f}")

    # Analysis: Geodesic distances
    print("\n--- Hyperbolic Geodesic Distances ---")
    names = list(embeddings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u = embeddings[names[i]]
            v = embeddings[names[j]]
            
            # Poincaré distance formula: d(u,v) = arcosh(1 + 2*||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
            dist_sq = np.sum((u - v)**2)
            denom = (1 - np.sum(u**2)) * (1 - np.sum(v**2))
            dist = np.arccosh(1 + 2 * dist_sq / denom)
            
            if "Antioxidant" in names[i] and "Antioxidant" in names[j]:
                print(f"Antioxidant Match: {names[i]} <-> {names[j]} | Dist: {dist:.4f} (UNLOCKED!)")
            elif "Antihypertensive" in names[i] or "ACE-Inh" in names[i]:
                 if "Antihypertensive" in names[j] or "ACE-Inh" in names[j]:
                    print(f"Hypertension Match: {names[i]} <-> {names[j]} | Dist: {dist:.4f} (UNLOCKED!)")
            else:
                # print(f"{names[i]} <-> {names[j]} | Dist: {dist:.4f}")
                pass

if __name__ == "__main__":
    main()
