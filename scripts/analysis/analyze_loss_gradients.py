#!/usr/bin/env python3
"""Analyze gradient norms of different loss components.

Helps determine if loss weights are balanced for Phase 10 research.
"""

import torch
import torch.nn as nn
from src.models.vae import TernaryVAEV6Controllable
from src.losses.combined import CombinedLoss
from src.core import TERNARY

def analyze_gradients(config_path="src/presets/v10_algebraic.yaml"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    model = TernaryVAEV6Controllable(
        latent_dim=config['model']['latent_dim'],
        hidden_dim=config['model']['hidden_dim'],
        max_radius=config['model']['max_radius'],
        curvature=config['model']['curvature'],
        factored=config['model']['factored'],
        radial_dims=config['model']['radial_dims'],
        positional_encoding=config['model']['positional_encoding']
    ).to(device).to(torch.float64)
    
    # 2. Setup Loss
    loss_fn = CombinedLoss(config['loss'], curvature=1.0, device=device)
    
    # 3. Dummy Batch
    batch_size = 128
    indices = torch.randint(0, TERNARY.N_OPERATIONS, (batch_size,), device=device)
    batch_ternary = TERNARY.to_ternary(indices).to(device)
    
    # 4. Compute Gradients for each component separately
    print(f"{'Loss Component':<25} | {'Loss Value':<10} | {'Grad Norm':<10}")
    print("-" * 55)
    
    components = [
        'rich_hierarchy', 'radial', 'geodesic', 'rank', 'monotonic', 
        'algebraic_coherence', 'algebraic_addition'
    ]
    
    # Forward pass once to get latents
    out = model(batch_ternary)
    
    for comp in components:
        # Clear grads
        model.zero_grad()
        
        # Override config to enable only this component
        temp_cfg = {k: {'enabled': False} for k in components}
        temp_cfg[comp] = config['loss'].get(comp, {'enabled': True})
        temp_cfg[comp]['enabled'] = True
        
        # We need a fresh loss_fn or careful override
        comp_loss_fn = CombinedLoss(temp_cfg, curvature=1.0, device=device)
        
        # Forward pass components
        res = comp_loss_fn(
            out["z_A_hyp"], indices, out.get("logits_A"), batch_ternary,
            epoch=100, # ensure phase-gated losses are active
            mu=out.get("mu_A"), logvar=out.get("logvar_A"),
            curvature=1.0, r=out.get("r_A"), model=model
        )
        
        loss_val = res['total']
        if loss_val.item() == 0:
            print(f"{comp:<25} | {loss_val.item():<10.4f} | {'N/A':<10}")
            continue
            
        loss_val.backward(retain_graph=True)
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"{comp:<25} | {loss_val.item():<10.4f} | {total_norm:<10.4f}")

if __name__ == "__main__":
    analyze_gradients()
