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
    model = TernaryVAEV6Controllable(
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        max_radius=model_cfg['max_radius'],
        curvature=model_cfg.get('curvature', 1.0),
        encoder_type=model_cfg['encoder_type'],
        decoder_type=model_cfg['decoder_type'],
        n_projection_layers=model_cfg['projection_layers'],
        projection_dropout=model_cfg['projection_dropout'],
        learnable_curvature=model_cfg['learnable_curvature'],
        init_identity=model_cfg['init_identity'],
        factored=model_cfg['factored'],
        radial_dims=model_cfg['radial_dims'],
        positional_encoding=model_cfg['positional_encoding']
    )
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    ckpt = "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/checkpoints/best_Q.pt"
    cfg = "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/config.yaml"
    
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
