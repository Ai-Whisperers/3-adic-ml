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
import json

def load_model(ckpt_path, config_path, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    mapping = {'projection_layers': 'n_projection_layers', 'tangent_scale': 'tangent_scale_init', 'projection_dropout': 'projection_dropout'}
    init_cfg = {mapping.get(k, k): v for k, v in model_cfg.items() if k not in ['name']}
    
    model = TernaryVAEV6Controllable(**init_cfg).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    base_dir = Path(__file__).resolve().parents[2]
    
    # Path to the current run
    run_dir = base_dir / "runs/v19.0_hybrid_curriculum_20260525_203246"
    ckpt = run_dir / "checkpoints/epoch_1999.pt" # Target epoch
    cfg = run_dir / "config.yaml"
    
    if not ckpt.exists():
        print(f"Checkpoint {ckpt} not reached yet.")
        return

    model, config = load_model(ckpt, cfg, device)
    
    # Synthetic Data Benchmark for Algebraic Audit
    all_ops = TERNARY.all_ternary()
    all_indices = torch.arange(len(all_ops), dtype=torch.long)
    ds = TensorDataset(all_ops, all_indices)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    loss_fn = CombinedLoss(config['loss'], curvature=config['model']['curvature'], device=device)
    v_fn = get_valuation_fn(config.get("data", {}).get("valuation_type", "index"))
    
    results = validate_epoch(1999, model, loader, device, v_fn, config, loss_fn=loss_fn)
    
    # Filter for Algebraic Losses
    algebraic_metrics = {k: v for k, v in results.items() if 'algebraic' in k}
    
    with open(run_dir / "pre_transition_audit_1999.json", "w") as f:
        json.dump(algebraic_metrics, f, indent=4)
        
    print(f"Audit complete. Results saved to {run_dir / 'pre_transition_audit_1999.json'}")
    for k, v in algebraic_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
