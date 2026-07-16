#!/usr/bin/env python3
"""Diagnose algebraic structure in a trained checkpoint.

Loads a model checkpoint, runs all 19,683 operations through the encoder,
then measures whether the hyperbolic embeddings cluster by algebraic
signature (4-bit: comm=8, assoc=4, has_id=2, has_abs=1).

Usage:
    python scripts/diagnostics/diagnose_algebraic_structure.py \\
        --checkpoint runs/<run>/checkpoints/best_Q.pt \\
        --config src/presets/v23.0_algebraic.yaml

Output: Markdown table to stdout + optional JSON to --output.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.core.ternary import algebraic_signature


def _build_model(config: dict, device: torch.device):
    from src.models.vae import TernaryVAEV6Controllable
    mc = config["model"]
    return TernaryVAEV6Controllable(
        latent_dim=mc["latent_dim"],
        hidden_dim=mc["hidden_dim"],
        max_radius=mc["max_radius"],
        factored=mc["factored"],
        radial_dims=mc["radial_dims"],
        n_projection_layers=mc["projection_layers"],
        projection_dropout=mc.get("projection_dropout", 0.0),
        learnable_curvature=mc.get("learnable_curvature", False),
        positional_encoding=mc.get("positional_encoding", False),
        encoder_type=mc.get("encoder_type", "standard"),
        decoder_type=mc.get("decoder_type", "standard"),
        tangent_scale_init=mc.get("tangent_scale", 0.1),
        detach_radial=mc.get("detach_radial", False),
        init_identity=mc.get("init_identity", True),
    ).to(device)


def _load_checkpoint(path: Path, model, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    epoch = ckpt.get("epoch", "?")
    q = ckpt.get("Q", ckpt.get("best_Q", "?"))
    print(f"Loaded checkpoint: epoch={epoch}, Q={q}")


@torch.no_grad()
def _embed_all(model, device: torch.device, batch_size: int = 1024):
    """Run all 19,683 ops through the encoder; return z_hyp (Poincaré) and radius."""
    all_idx = torch.arange(TERNARY.N_OPERATIONS)
    z_all, r_all = [], []
    for start in range(0, TERNARY.N_OPERATIONS, batch_size):
        batch_idx = all_idx[start:start + batch_size].to(device)
        ops = TERNARY.to_ternary(batch_idx).float().to(device)
        out = model(ops)
        z_all.append(out["z_A_hyp"].cpu())
        r_all.append(out["r_A"].cpu())
    return torch.cat(z_all), torch.cat(r_all)


def _knn_purity(z: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """Fraction of k-nearest neighbors sharing the same label (cosine distance)."""
    z_norm = torch.nn.functional.normalize(z.float(), dim=-1)
    sim = z_norm @ z_norm.T  # (N, N)
    sim.fill_diagonal_(-1.0)
    topk = sim.topk(k, dim=-1).indices  # (N, k)
    neighbor_labels = labels[topk]  # (N, k)
    same = (neighbor_labels == labels.unsqueeze(1))
    return same.float().mean().item()


def _within_class_cosine_sim(z: torch.Tensor, labels: torch.Tensor, sig_val: int) -> float:
    """Mean cosine similarity between all pairs within a given algebraic class."""
    mask = labels == sig_val
    z_cls = z[mask]
    if z_cls.shape[0] < 2:
        return float("nan")
    z_norm = torch.nn.functional.normalize(z_cls.float(), dim=-1)
    sims = (z_norm @ z_norm.T).triu(diagonal=1)
    n_pairs = z_cls.shape[0] * (z_cls.shape[0] - 1) / 2
    return (sims.sum() / n_pairs).item()


SIG_NAMES = {
    0:  "none",
    1:  "abs",
    2:  "id",
    3:  "id+abs",
    4:  "assoc",
    5:  "assoc+abs",
    6:  "assoc+id",
    7:  "assoc+id+abs",
    8:  "comm",
    9:  "comm+abs",
    10: "comm+id",
    11: "comm+id+abs",
    12: "comm+assoc",
    13: "comm+assoc+abs",
    14: "comm+assoc+id",
    15: "comm+assoc+id+abs",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config used for training")
    parser.add_argument("--k", type=int, default=10, help="kNN k for purity metric")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", help="Write JSON results to this path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = _build_model(config, device)
    _load_checkpoint(Path(args.checkpoint), model, device)
    model.eval()

    print("Embedding all 19,683 operations...")
    z_all, r_all = _embed_all(model, device, batch_size=args.batch_size)

    # Ground-truth algebraic signatures for all ops
    all_idx = torch.arange(TERNARY.N_OPERATIONS)
    sigs = algebraic_signature(all_idx)      # [0..15]
    vals = TERNARY.valuation(all_idx)        # [0..9] for hierarchy reference

    print(f"Embeddings: {z_all.shape}  (radius range [{r_all.min():.3f}, {r_all.max():.3f}])\n")

    # Per-class stats
    rows = []
    for sig_val in range(16):
        mask = (sigs == sig_val)
        n = mask.sum().item()
        if n < 2:
            continue

        z_cls = z_all[mask]
        r_cls = r_all[mask]
        mean_r = r_cls.mean().item()
        within_sim = _within_class_cosine_sim(z_all, sigs, sig_val)

        rows.append({
            "sig": sig_val,
            "name": SIG_NAMES[sig_val],
            "n_ops": n,
            "mean_radius": round(mean_r, 4),
            "within_cos_sim": round(within_sim, 4),
        })

    # Global kNN purity — only over the 12 non-trivial classes (sig != 0)
    nontrivial_mask = sigs != 0
    z_nt = z_all[nontrivial_mask]
    sigs_nt = sigs[nontrivial_mask]
    global_purity = _knn_purity(z_nt, sigs_nt, k=args.k)

    # Baseline: random purity (1/n_classes weighted by class size)
    class_sizes = torch.bincount(sigs_nt, minlength=16).float()
    total_nt = nontrivial_mask.sum().float()
    baseline = (class_sizes / total_nt).pow(2).sum().item()

    # Per-class kNN purity
    for row in rows:
        if row["sig"] == 0:
            continue
        cls_mask = sigs == row["sig"]
        z_cls_only = z_all[cls_mask]
        # Since all same class, purity within class is always 1; measure purity of
        # the class's kNN among the full population instead (lift over baseline).
        # knn_purity_global: for ops in this class, what fraction of their k-NNs share the same class?
        if z_cls_only.shape[0] >= 2:
            z_norm_all = torch.nn.functional.normalize(z_all.float(), dim=-1)
            z_norm_cls = torch.nn.functional.normalize(z_cls_only.float(), dim=-1)
            sim_cls = z_norm_cls @ z_norm_all.T  # (n_cls, N)
            k = min(args.k, z_cls_only.shape[0] - 1)
            topk_idx = sim_cls.topk(k + 1, dim=-1).indices[:, 1:]  # exclude self
            neighbor_sigs = sigs[topk_idx]
            class_purity = (neighbor_sigs == row["sig"]).float().mean().item()
            row["knn_purity"] = round(class_purity, 4)
        else:
            row["knn_purity"] = float("nan")

    # Print markdown table
    print("## Algebraic Structure Analysis\n")
    print(f"kNN purity (k={args.k}) over non-trivial classes: **{global_purity:.4f}**  "
          f"(random baseline: {baseline:.4f}, lift: {global_purity/baseline:.2f}×)\n")
    print(f"{'sig':>4} | {'name':<20} | {'n_ops':>6} | {'mean_r':>7} | {'within_sim':>10} | {'knn_purity':>10}")
    print("-" * 72)
    for row in rows:
        knn_purity = row.get("knn_purity", float("nan"))
        knn = "   nan" if math.isnan(knn_purity) else f"{knn_purity:.4f}"
        print(f"{row['sig']:>4} | {row['name']:<20} | {row['n_ops']:>6} | {row['mean_radius']:>7.4f} | {row['within_cos_sim']:>10.4f} | {knn:>10}")

    results = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "k": args.k,
        "global_knn_purity_nontrivial": round(global_purity, 6),
        "random_baseline": round(baseline, 6),
        "lift": round(global_purity / baseline, 4),
        "classes": rows,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return results


if __name__ == "__main__":
    main()
