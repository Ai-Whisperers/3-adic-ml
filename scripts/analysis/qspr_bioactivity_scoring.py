import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from scripts.data.peptide_encoding import encode_peptide_window

def encode_window(seq, device):
    digits = encode_peptide_window(seq)
    x = torch.tensor(digits, dtype=torch.float64).unsqueeze(0).to(device)
    # Augment: x is 1x9. pos_weights is 1x9. 
    # x * pos_weights produces 1x9.
    # cat them to get 1x18.
    pos_weights = torch.tensor([1.0 / (3.0 ** k) for k in range(9)], dtype=torch.float64).unsqueeze(0).to(device)
    return torch.cat([x, x * pos_weights], dim=-1)

def hyperbolic_dist(u, v):
    dist_sq = np.sum((u - v)**2)
    denom = (1 - np.sum(u**2)) * (1 - np.sum(v**2))
    val = 1 + 2 * dist_sq / max(denom, 1e-12)
    return np.arccosh(val)

def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt = base_dir / "runs/v19.0_hybrid_curriculum_20260525_203246/checkpoints/final.pt"
    cfg = base_dir / "runs/v19.0_hybrid_curriculum_20260525_203246/config.yaml"
    
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

    # Centroids
    bioactive_peps = {
        "Antioxidant": ["WTLTPLTPA", "SVAGRAQGM", "AVLGSSEGV", "ISLSEQQLV"],
        "Antihypertensive": ["CACGGV", "VTYM", "ALVVGSDPV", "VAVLGSSEG"]
    }
    
    centroids = {}
    with torch.no_grad():
        for cat, peps in bioactive_peps.items():
            embs = []
            for p in peps:
                x_aug = encode_window(p, device)
                embs.append(model(x_aug)['z_A_hyp'].squeeze().cpu().numpy())
            centroids[cat] = np.mean(embs, axis=0)
            
    # Test Candidates
    candidates = ["TGTGTAACA", "TGTCTCCTT", "TCTGTCTCC", "TTCCTCTCT", "CCTCAGGGT"]
    
    print(f"{'Candidate':<15} | {'Antiox Dist':<12} | {'Antihyper Dist':<15}")
    print("-" * 50)
    
    for c in candidates:
        x_aug = encode_window(c, device)
        with torch.no_grad():
            emb = model(x_aug)['z_A_hyp'].squeeze().cpu().numpy()
            
        d_antiox = hyperbolic_dist(emb, centroids["Antioxidant"])
        d_ace = hyperbolic_dist(emb, centroids["Antihypertensive"])
        print(f"{c:<15} | {d_antiox:<12.4f} | {d_ace:<15.4f}")

if __name__ == "__main__":
    main()
