import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from src.core import TERNARY
from scripts.data.peptide_encoding import AA_MAP

def encode_window(seq):
    digits = [AA_MAP.get(aa.upper(), 0) for aa in seq]
    return torch.tensor(digits, dtype=torch.float64).unsqueeze(0)

def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/checkpoints/final.pt"
    cfg = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/config.yaml"
    
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config['model']
    mapping = {'projection_layers': 'n_projection_layers', 'tangent_scale': 'tangent_scale_init', 'projection_dropout': 'projection_dropout'}
    init_cfg = {mapping.get(k, k): v for k, v in model_cfg.items() if k not in ['name']}
    model = TernaryVAEV6Controllable(**init_cfg).to(device)
    
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define Bioactive Centroid (Average of Antiox/Antihyper peptides from previous phase)
    # Using the same mapping and centroid logic as hotspot_scanner.py
    def get_centroid(peptides):
        embs = []
        with torch.no_grad():
            for p in peptides:
                seq_padded = p[:9].ljust(9, 'G')
                x = encode_window(seq_padded).to(device)
                # Augmented input
                pos_weights = torch.tensor([1.0 / (3.0 ** k) for k in range(9)], dtype=torch.float64).to(device)
                x_aug = torch.cat([x, x * pos_weights], dim=-1)
                embs.append(model(x_aug)['z_A_hyp'].squeeze().cpu().numpy())
        return np.mean(embs, axis=0)

    bioactive_peps = ["WTLTPLTPA", "SVAGRAQGM", "CACGGV", "VTYM"]
    centroid = get_centroid(bioactive_peps)

    # Scan Human TP53 (CCTGCC...)
    fasta_path = base_dir / "data/human/tp53_ref.fasta"
    with open(fasta_path, 'r') as f:
        seq = f.read().strip()
        
    print(f"Scanning Human TP53 (Len: {len(seq)})...")
    
    hits = []
    with torch.no_grad():
        for i in range(len(seq) - 8):
            window = seq[i:i+9]
            x = encode_window(window).to(device)
            pos_weights = torch.tensor([1.0 / (3.0 ** k) for k in range(9)], dtype=torch.float64).to(device)
            x_aug = torch.cat([x, x * pos_weights], dim=-1)
            
            emb = model(x_aug)['z_A_hyp'].squeeze().cpu().numpy()
            
            # Poincaré distance
            dist_sq = np.sum((emb - centroid)**2)
            denom = (1 - np.sum(emb**2)) * (1 - np.sum(centroid**2))
            dist = np.arccosh(1 + 2 * dist_sq / max(denom, 1e-12))
            
            if dist < 0.1:
                hits.append((window, dist))
                
    hits.sort(key=lambda x: x[1])
    print(f"\n--- Top {min(len(hits), 10)} Novel TP53 Bioactive Candidates ---")
    for w, d in hits[:10]:
        print(f"Window: {w} | Dist: {d:.4f}")

if __name__ == "__main__":
    main()
