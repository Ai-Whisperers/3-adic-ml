import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import yaml
import glob

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from scripts.data.peptide_encoding import encode_peptide_window

def find_latest_run_paths(run_prefix="v19.1_peptide_retrain"):
    base_dir = Path(__file__).resolve().parents[2]
    runs_dir = base_dir / "runs"
    matching_runs = sorted(glob.glob(str(runs_dir / f"{run_prefix}_*")))
    if matching_runs:
        latest_run = Path(matching_runs[-1])
        ckpt = latest_run / "checkpoints" / "best_Q.pt"
        cfg = latest_run / "config.yaml"
        if ckpt.exists() and cfg.exists():
            return str(ckpt), str(cfg)
    
    # Fallback
    ckpt = base_dir / "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/checkpoints/best_Q.pt"
    cfg = base_dir / "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/config.yaml"
    return str(ckpt), str(cfg)

def get_hotspot_embeddings(model, device):
    peptides = [
        "WTLTPLTPA", "SVAGRAQGM", "CACGGV", "VTYM", 
        "AVLGSSEGV", "FGWTFYFLN", "AHPGNWGIM", 
        "WAIIDAIEA", "VLAVCSEVT", "ALVVGSDPV"
    ]
    names = [
        "Antiox(T-Known)", "Antiox(G-Known)", "ACE(T-Known)", "Antihyper(G-Known)",
        "Antiox(G-SOD-Hit)", "Antihyper(T-CURS1-Hit)", "Antihyper(T-CURS3-Hit)",
        "Antihyper(T-CURS1-Hit2)", "Antiox(T-DCS-Hit)", "Antiox(T-DCS-Hit2)"
    ]
    
    embs = []
    with torch.no_grad():
        for seq in peptides:
            digits = encode_peptide_window(seq)
            x = torch.tensor(digits, dtype=torch.float64).unsqueeze(0).to(device)
            # The model's forward pass handles positional encoding internally
            out = model(x)
            embs.append(out['z_A_hyp'].squeeze().cpu().numpy())
    return np.array(embs), names, peptides

def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt_str, cfg_str = find_latest_run_paths()
    
    with open(cfg_str, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model_cfg = config['model']
    mapping = {'projection_layers': 'n_projection_layers', 'tangent_scale': 'tangent_scale_init', 'projection_dropout': 'projection_dropout'}
    init_cfg = {mapping.get(k, k): v for k, v in model_cfg.items() if k not in ['name']}
    model = TernaryVAEV6Controllable(**init_cfg).to(device)
    
    checkpoint = torch.load(ckpt_str, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Extracting embeddings for visualization...")
    embs, names, peptides = get_hotspot_embeddings(model, device)
    
    # Project 3D subset (first 3 dims of 128) for visual inspection in Poincaré ball
    embs_3d = embs[:, :3]
    
    fig = go.Figure()
    
    # Add Poincaré sphere boundary
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, 
        opacity=0.08, 
        showscale=False, 
        hoverinfo='skip',
        colorscale=[[0, 'rgba(200,200,250,0.2)'], [1, 'rgba(200,200,250,0.2)']]
    ))
    
    # Annotations for hovered points
    text_labels = [f"<b>{names[i]}</b><br>Seq: {peptides[i]}<br>R: {np.linalg.norm(embs[i]):.4f}" for i in range(len(names))]
    
    fig.add_trace(go.Scatter3d(
        x=embs_3d[:,0], y=embs_3d[:,1], z=embs_3d[:,2],
        mode='markers+text',
        text=[names[i].split('(')[0] for i in range(len(names))],
        hoverinfo='text',
        hovertext=text_labels,
        textposition="top center",
        marker=dict(
            size=10, 
            color=np.linalg.norm(embs, axis=1), 
            colorscale='Viridis', 
            opacity=0.9,
            colorbar=dict(title="True Hyp Radius"),
            line=dict(color='white', width=2)
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="Poincaré Manifold Hotspot Projection (v19.1 Hybrid Curriculum)",
            font=dict(size=18, color="#1a73e8"),
            x=0.5, y=0.95
        ),
        scene=dict(
            xaxis=dict(title="Z_0", range=[-1.05, 1.05]),
            yaxis=dict(title="Z_1", range=[-1.05, 1.05]),
            zaxis=dict(title="Z_2", range=[-1.05, 1.05])
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    output_path = base_dir / "curcumin-and-other-peptides/visualizations/hotspots_poincare.html"
    fig.write_html(str(output_path))
    fig.write_html("hotspot_projection.html") # also save to root as fallback
    print(f"Visualization saved to {output_path}")
    print("Visualization also saved to hotspot_projection.html")

if __name__ == "__main__":
    main()
