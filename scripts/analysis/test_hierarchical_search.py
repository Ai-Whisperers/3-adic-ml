import torch
import numpy as np
from pathlib import Path
import yaml
from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.geometry import poincare_distance

def setup_model(checkpoint_path):
    print(f"Loading model: {checkpoint_path}")
    run_dir = Path(checkpoint_path).parents[1]
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model_cfg = config["model"]
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    model = TernaryVAEV6Controllable(
        **{k: v for k, v in model_cfg.items() if k not in ["name", "projection_layers", "tangent_scale", "positional_encoding"]},
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        tangent_scale_init=model_cfg.get("tangent_scale", 0.1)
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def test_hierarchical_search(model, target_indices):
    device = torch.device("cpu")
    with torch.no_grad():
        # Get latent embeddings (Poincaré Ball)
        mu = model.get_mu_representations(torch.tensor(target_indices, device=device), device)
        z_hyp, _ = model.projections.proj_A(mu)
        z_hyp = z_hyp.cpu()

    # Calculate distances from a query node to others
    # Node 0 is the root (highest valuation)
    root_idx = 0
    root_emb = z_hyp[0].unsqueeze(0)
    
    print("\nQuerying Hierarchical Distances from Root (Index 0):")
    for i, idx in enumerate(target_indices):
        if i == 0: continue
        dist = poincare_distance(root_emb, z_hyp[i].unsqueeze(0), c=1.0)
        val = TERNARY.valuation(torch.tensor([idx])).item()
        print(f"Node {idx:04d} (v={val}): Dist={dist.item():.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    model = setup_model(ckpt_path)
    
    # Test a small subset representing different levels of the hierarchy
    # 0 is root (v=9), others are lower valuations
    test_nodes = [0, 1, 3, 9, 27] 
    test_hierarchical_search(model, test_nodes)
