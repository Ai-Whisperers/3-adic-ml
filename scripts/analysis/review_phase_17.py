import torch
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from src.core import TERNARY
from src.training.engine import validate_epoch
from src.losses.combined import CombinedLoss
from torch.utils.data import DataLoader, TensorDataset

def review_checkpoint(ckpt_path, config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # 1. Load Model
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

    # 2. Setup Data (Full Synthetic for base benchmark)
    all_ops = TERNARY.all_ternary()
    all_indices = torch.arange(len(all_ops), dtype=torch.long)
    ds = TensorDataset(all_ops, all_indices)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    # 3. Setup Loss for validation
    loss_fn = CombinedLoss(config['loss'], curvature=model_cfg['curvature'], device=device)

    # 4. Validate
    print(f"Reviewing Checkpoint: {ckpt_path}")
    from src.core.ternary import get_valuation_fn
    v_fn = get_valuation_fn(config.get("data", {}).get("valuation_type", "index"))
    results = validate_epoch(0, model, loader, device, v_fn, config, loss_fn=loss_fn)
    
    print("\n--- Rosetta Manifold Review (Epoch 1000) ---")
    print(f"Loss: N/A (Algebraic review mode)")
    print(f"Coverage: {results['avg_val_coverage']:.2%} (Target: >95%)")
    hier_A = results["hier_metrics_A"]
    print(f"ARI (v=2): {hier_A.get('hierarchy', 0.0):.4f} (Hierarchy Clarity)")
    print(f"Q-Metric: {hier_A.get('Q', 0.0):.4f}")
    
    # Check for specifically requested Phase 17 goals
    if results['avg_val_coverage'] > 0.99:
        print("STATUS: RECONSTRUCTION PERFECTED")
    elif results['avg_val_coverage'] > 0.95:
        print("STATUS: HIGH FIDELITY REACHED")
    else:
        print("STATUS: HIERARCHY STILL FORMING")

if __name__ == "__main__":
    ckpt = "runs/v17.0_rosetta_manifold_20260524_215350/checkpoints/epoch_1000.pt"
    cfg = "runs/v17.0_rosetta_manifold_20260524_215350/config.yaml"
    if os.path.exists(ckpt):
        review_checkpoint(ckpt, cfg)
    else:
        print(f"Checkpoint {ckpt} not found yet.")
