import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable

def get_hotspot_embeddings(model, device):
    peptides = ["WTLTPLTPA", "SVAGRAQGM", "CACGGV", "VTYM", "AVLGSSEGV", "ISLSEQQLV"]
    names = ["Antiox(T)", "Antiox(G)", "ACE(T)", "Antihyper(G)", "Antiox(G-SOD)", "Antiox(G-Zin)"]
    
    AA_MAP = {'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1, 'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0, 'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1}
    
    embs = []
    with torch.no_grad():
        for seq in peptides:
            seq_padded = seq[:9].ljust(9, 'G')
            digits = [AA_MAP.get(aa.upper(), 0) for aa in seq_padded]
            x = torch.tensor(digits, dtype=torch.float64).unsqueeze(0).to(device)
            # Manually replicate positional encoding
            pos_weights = torch.tensor([1.0 / (3.0 ** k) for k in range(9)], dtype=torch.float64).to(device)
            x_aug = torch.cat([x, x * pos_weights], dim=-1)
            
            out = model(x_aug)
            embs.append(out['z_A_hyp'].squeeze().cpu().numpy())
    return np.array(embs), names

def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/checkpoints/final.pt"
    cfg = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/config.yaml"
    
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model_cfg = config['model']
    mapping = {'projection_layers': 'n_projection_layers', 'tangent_scale': 'tangent_scale_init', 'projection_dropout': 'projection_dropout'}
    init_cfg = {mapping.get(k, k): v for k, v in model_cfg.items() if k not in ['name']}
    model = TernaryVAEV6Controllable(**init_cfg).to(device)
    
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Extracting embeddings for visualization...")
    embs, names = get_hotspot_embeddings(model, device)
    
    # Project 3D subset (first 3 dims of 64) for visual inspection in Poincaré ball
    # Note: Full visualization requires TSNE/UMAP projection, 
    # but we will use the first 3 dims to visualize the hyperbolic radial embedding.
    embs_3d = embs[:, :3]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=embs_3d[:,0], y=embs_3d[:,1], z=embs_3d[:,2],
        mode='markers+text',
        text=names,
        marker=dict(size=8, color=np.linalg.norm(embs_3d, axis=1), colorscale='Viridis', opacity=0.8)
    ))
    
    # Add Poincaré sphere boundary
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, showscale=False))
    
    fig.update_layout(title="Poincaré Manifold Hotspot Projection (Factored Subset)", margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html("hotspot_projection.html")
    print("Visualization saved to hotspot_projection.html")

if __name__ == "__main__":
    main()
