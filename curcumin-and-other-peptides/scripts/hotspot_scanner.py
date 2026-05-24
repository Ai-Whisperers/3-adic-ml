import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable

# AA Mapping (Hydropathy-based)
AA_MAP = {
    'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1,
    'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0,
    'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1
}

def encode_peptide(seq, length=9):
    encoded = []
    for i in range(length):
        if i < len(seq):
            aa = seq[i].upper()
            encoded.append(AA_MAP.get(aa, 0))
        else:
            encoded.append(0)
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

def hyperbolic_dist(u, v):
    dist_sq = np.sum((u - v)**2)
    denom = (1 - np.sum(u**2)) * (1 - np.sum(v**2))
    # Clip for stability
    val = 1 + 2 * dist_sq / max(denom, 1e-12)
    return np.arccosh(val)

def read_fasta(file_path):
    sequences = {}
    current_id = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line
    return sequences

def main():
    ckpt = "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/checkpoints/best_Q.pt"
    cfg = "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/config.yaml"
    model = load_model(ckpt, cfg)
    
    # Define Targets (Centroids of known bioactive pairs)
    # Antioxidant Centroid
    antiox_t = encode_peptide("WTLTPLTPA")
    antiox_g = encode_peptide("SVAGRAQGM")
    with torch.no_grad():
        emb_antiox_t = model(antiox_t)['z_A_hyp'].numpy()[0]
        emb_antiox_g = model(antiox_g)['z_A_hyp'].numpy()[0]
    antiox_centroid = (emb_antiox_t + emb_antiox_g) / 2
    
    # Antihypertensive Centroid
    ace_t = encode_peptide("CACGGV")
    ace_g = encode_peptide("VTYM")
    with torch.no_grad():
        emb_ace_t = model(ace_t)['z_A_hyp'].numpy()[0]
        emb_ace_g = model(ace_g)['z_A_hyp'].numpy()[0]
    ace_centroid = (emb_ace_t + emb_ace_g) / 2

    fasta_path = "curcumin-and-other-peptides/data/sequences.fasta"
    all_seqs = read_fasta(fasta_path)
    
    hits = []
    print(f"Scanning {len(all_seqs)} proteins for hotspots...")
    
    for protein_id, full_seq in all_seqs.items():
        # Sliding window of 9
        for i in range(len(full_seq) - 8):
            window = full_seq[i:i+9]
            x = encode_peptide(window)
            with torch.no_grad():
                emb = model(x)['z_A_hyp'].numpy()[0]
            
            d_antiox = hyperbolic_dist(emb, antiox_centroid)
            d_ace = hyperbolic_dist(emb, ace_centroid)
            
            if d_antiox < 0.25:
                hits.append(("Antioxidant", protein_id, window, d_antiox))
            if d_ace < 0.25:
                hits.append(("Antihypertensive", protein_id, window, d_ace))

    # Sort and filter
    hits.sort(key=lambda x: x[3])
    
    print("\n--- NOVEL BIOACTIVE CANDIDATES FOUND ---")
    seen_windows = set()
    count = 0
    for category, pid, window, dist in hits:
        if window in seen_windows: continue
        # Filter out the original known ones
        if window in ["WTLTPLTPA", "SVAGRAQGM", "CACGGV", "VTYM"]: continue
        
        print(f"[{category}] {window} | Dist: {dist:.4f} | Source: {pid[:40]}...")
        seen_windows.add(window)
        count += 1
        if count >= 10: break

if __name__ == "__main__":
    main()
