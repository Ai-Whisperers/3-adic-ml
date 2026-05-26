import torch
import sys
import os
import yaml
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from src.core import TERNARY
from src.training.engine import validate_epoch
from src.losses.combined import CombinedLoss
from torch.utils.data import DataLoader, TensorDataset
from src.core.ternary import get_valuation_fn

def load_model(ckpt_path, config_path, device):
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
    ).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # Models
    configs = [
        ("V11", "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/checkpoints/best_Q.pt", "archive-for-review/phase_11_multiplicative/v11_multiplicative_20260517_130238/config.yaml"),
        ("V17", "runs/v17.1_rosetta_manifold_resume_20260525_060109/checkpoints/final.pt", "runs/v17.1_rosetta_manifold_resume_20260525_060109/config.yaml")
    ]

    all_ops = TERNARY.all_ternary()
    all_indices = torch.arange(len(all_ops), dtype=torch.long)
    ds = TensorDataset(all_ops, all_indices)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    results = {}
    for name, ckpt, cfg in configs:
        print(f"Evaluating {name}...")
        model, config = load_model(ckpt, cfg, device)
        loss_fn = CombinedLoss(config['loss'], curvature=config['model']['curvature'], device=device)
        v_fn = get_valuation_fn(config.get("data", {}).get("valuation_type", "index"))
        res = validate_epoch(0, model, loader, device, v_fn, config, loss_fn=loss_fn)
        results[name] = res

    print(f"\n{'Metric':<20} | {'V11':<15} | {'V17':<15}")
    print("-" * 55)
    
    for metric in ['Coverage', 'Hierarchy', 'Q-Metric']:
        v11_val = "N/A"
        v17_val = "N/A"
        
        # Extract V11
        if metric == 'Coverage': v11_val = f"{results['V11'].get('avg_val_coverage', 0.0):.4%}"
        if metric == 'Hierarchy': v11_val = f"{results['V11']['hier_metrics_A'].get('hierarchy', 0.0):.4f}"
        if metric == 'Q-Metric': v11_val = f"{results['V11']['hier_metrics_A'].get('Q', 0.0):.4f}"
        
        # Extract V17
        if metric == 'Coverage': v17_val = f"{results['V17'].get('avg_val_coverage', 0.0):.4%}"
        if metric == 'Hierarchy': v17_val = f"{results['V17']['hier_metrics_A'].get('hierarchy', 0.0):.4f}"
        if metric == 'Q-Metric': v17_val = f"{results['V17']['hier_metrics_A'].get('Q', 0.0):.4f}"
        
        print(f"{metric:<20} | {v11_val:<15} | {v17_val:<15}")

if __name__ == "__main__":
    main()
