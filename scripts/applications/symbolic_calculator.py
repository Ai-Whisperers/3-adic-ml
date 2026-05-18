#!/usr/bin/env python3
"""Algebraic Latent Calculator."""
import argparse
import sys
from pathlib import Path
import torch
import yaml
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.core import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.utils.checkpoint import load_checkpoint_compat

def parse_ternary(s):
    vals = [int(x) for x in s.split()]
    return torch.tensor(vals, dtype=torch.float64)

def calculate(ternary_a, ternary_b, op):
    runs = sorted(Path("runs").glob("v11_multiplicative_*"))
    if not runs: return
    ckpt_path = runs[-1] / "checkpoints" / "final.pt"
    run_dir = runs[-1]
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    pos_enc = (ckpt['model_state_dict']["head_A.backbone.0.weight"].shape[1] == 18)
    model = TernaryVAEV6Controllable(
        **{k: v for k, v in model_cfg.items() if k not in ["name", "projection_layers", "tangent_scale", "positional_encoding"]},
        positional_encoding=pos_enc,
        n_projection_layers=model_cfg.get("projection_layers", 1),
        tangent_scale_init=model_cfg.get("tangent_scale", 0.1)
    ).to(torch.float64)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    t_a = parse_ternary(ternary_a).unsqueeze(0)
    t_b = parse_ternary(ternary_b).unsqueeze(0)
    idx_a = TERNARY.from_ternary(t_a)
    idx_b = TERNARY.from_ternary(t_b)
    if op == 'add':
        idx_gt = TERNARY.ternary_add(idx_a, idx_b)
        op_fn = torch.add
    else:
        idx_gt = TERNARY.ternary_mul(idx_a, idx_b)
        op_fn = torch.mul
    ternary_gt = TERNARY.to_ternary(idx_gt)
    with torch.no_grad():
        mu_a = model.get_mu_representations(idx_a, torch.device('cpu'))
        mu_b = model.get_mu_representations(idx_b, torch.device('cpu'))
        mu_res = op_fn(mu_a, mu_b)
        logits = model.decoder_A(mu_res)
        pred = torch.argmax(logits.view(-1, 9, 3), dim=-1) - 1
    print(f"\nOperation: {ternary_a} {op} {ternary_b}")
    print(f"Ground Truth: {ternary_gt.squeeze().tolist()}")
    print(f"Latent Pred:  {pred.squeeze().tolist()}")
    match = torch.equal(pred, ternary_gt.long())
    print(f"\nResult: {'MATCH' if match else 'MISMATCH'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("op", choices=['add', 'mul'])
    parser.add_argument("b")
    args = parser.parse_args()
    calculate(args.a, args.b, args.op)
