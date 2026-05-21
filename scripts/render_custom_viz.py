import sys
from pathlib import Path
import torch
import yaml
import numpy as np

# Set project root for imports
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.utils.poincare_renderer import save_poincare_disk

def render_visualization(checkpoint_path, output_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    run_dir = Path(checkpoint_path).parents[1]
    
    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config not found at {config_path}, skipping.")
        return
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"].copy()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Check for positional encoding
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    # Clean up model config for constructor
    for k in ["name", "projection_layers", "tangent_scale", "positional_encoding"]:
        if k in model_cfg:
            del model_cfg[k]
            
    # Initialize model
    model = TernaryVAEV6Controllable(
        **model_cfg,
        positional_encoding=pos_enc,
        n_projection_layers=config["model"].get("projection_layers", 1),
        tangent_scale_init=config["model"].get("tangent_scale", 0.1)
    ).to(torch.float64)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Subsample data for visualization (e.g., 2000 points)
    n_samples = 2000
    indices = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,))
    valuations = TERNARY.valuation(indices).numpy()
    
    with torch.no_grad():
        # Get mu representations (tangent space)
        mu_A = model.get_mu_representations(indices, torch.device('cpu'))
        # Project mu to Poincaré ball
        z_hyp, _ = model.projections.proj_A(mu_A)
        z_hyp = z_hyp.numpy()
        
    # Render and save
    print(f"Rendering visualization to: {output_path}")
    save_poincare_disk(
        z_hyp=z_hyp,
        valuations=valuations,
        output_path=output_path,
        indices=indices.numpy(),
        title=f"Native Poincaré Disk - {run_dir.name}",
        show_tree=True
    )
    print("Done.")

if __name__ == "__main__":
    # Target a recent high-quality run
    target_run = "runs/v7_large_20260502_153618"
    ckpt_path = f"{target_run}/checkpoints/best_Q.pt"

    if Path(ckpt_path).exists():
        output_html = "docs/v7_large_poincare.html"
        render_visualization(ckpt_path, output_html)
    else:
        print(f"Target checkpoint {ckpt_path} not found.")

