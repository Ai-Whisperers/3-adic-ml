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
    parser = argparse.ArgumentParser(description="P-adic Peptide Hotspot Proteomic Scanner")
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
