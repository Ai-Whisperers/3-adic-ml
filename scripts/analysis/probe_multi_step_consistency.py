#!/usr/bin/env python3
"""Multi-Step Consistency Probe: Evaluate compositional algebraic homomorphisms."""

import sys
from pathlib import Path
import torch
import yaml

# Set project root for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable

def probe_multi_step_consistency(checkpoint_path, n_samples=1000):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    run_dir = Path(checkpoint_path).parents[1]
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    # Auto-detect pos_enc
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    
    # Robust model reconstruction matching checkpoint
    model = TernaryVAEV6Controllable(
        **{k: v for k, v in model_cfg.items() if k not in ["name", "projection_layers", "tangent_scale", "positional_encoding"]},
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        tangent_scale_init=model_cfg.get("tangent_scale", 0.1)
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Sample triplets (a, b, c)
    torch.manual_seed(42)
    idx_a = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)
    idx_b = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)
    idx_c = torch.randint(0, TERNARY.N_OPERATIONS, (n_samples,), device=device)

    # Ground truth: z((a ⊕ b) ⊗ c)
    idx_sum = TERNARY.ternary_add(idx_a, idx_b)
    idx_prod_gt = TERNARY.ternary_mul(idx_sum, idx_c)

    print(f"Probing multi-step consistency (n={n_samples})...")

    with torch.no_grad():
        # Get representations
        mu_a = model.get_mu_representations(idx_a, device)
        mu_b = model.get_mu_representations(idx_b, device)
        mu_c = model.get_mu_representations(idx_c, device)
        
        # Predicted representation: (z(a) + z(b)) * z(c)
        mu_pred = (mu_a + mu_b) * mu_c
        
        # Ground truth representation: z((a + b) * c)
        mu_gt = model.get_mu_representations(idx_prod_gt, device)
        
        # Metrics
        mse = torch.nn.functional.mse_loss(mu_pred, mu_gt).item()
        cos_sim = torch.nn.functional.cosine_similarity(mu_pred, mu_gt).mean().item()
        
        # Decoding check
        logits = model.decoder_A(mu_pred)
        ternary_gt = TERNARY.to_ternary(idx_prod_gt).to(device)
        preds = torch.argmax(logits.view(-1, 9, 3), dim=-1) - 1
        acc = (preds == ternary_gt.long()).float().mean().item()

    print("")
    print("Results:")
    print(f"  Mu-Space MSE (pred vs gt):   {mse:.6f}")
    print(f"  Mu-Space Cosine Sim:        {cos_sim:.6f}")
    print(f"  Final Digit Accuracy:       {acc:.4%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    probe_multi_step_consistency(ckpt_path)
