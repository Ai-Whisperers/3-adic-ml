"""Train Condition D (taxonomy-conditioned hyperbolic VAE) for the phylogeny
validation pipeline. Fase 3 of docs/plans/TAXONOMY-CONDITIONED-EMBEDDING-PLAN.md.

Same architecture class as Conditions B/C (TernaryVAEV6Controllable, same
factored=false/positional_encoding=false choice as Condition B), but the
geodesic loss targets real taxonomic distance (TaxonomyGeodesicLoss) instead
of v_3(index) (PAdicGeodesicLoss, Condition C) or nothing (Condition B).

Trains only on window_map_train.json/indices_train.pt (Fase 2's species-level
holdout -- 9/39 species never seen here). Standalone loop, not routed through
src/train.py/CombinedLoss, for the same reason Condition A's script is
standalone: CombinedLoss and the shared DataAuditor split only know about
ternary index, not species, and this needs species-aware batching for
TaxonomyGeodesicLoss. decode_b=False throughout -- VAE-B is never trained or
used, since there's no coverage-vs-hierarchy role split to justify the
dual-VAE machinery for this condition.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core import TERNARY
from src.losses.geodesic import TaxonomyGeodesicLoss
from src.losses.hyperbolic_kl import HyperbolicKLDivergence
from src.losses.utils import compute_coverage_loss
from src.models.vae import TernaryVAEV6Controllable
from src.training.bootstrap import set_determinism


def load_training_data(window_map_path: Path, indices_path: Path, taxonomy_dir: Path):
    window_map = json.loads(window_map_path.read_text())
    indices = torch.load(indices_path, weights_only=True).long()
    if len(window_map) != len(indices):
        raise ValueError(f"window_map ({len(window_map)}) and indices ({len(indices)}) length mismatch")

    species_order_all = json.loads((taxonomy_dir / "species_order.json").read_text())
    tax_dist_all = torch.tensor(
        np.load(taxonomy_dir / "taxonomic_distance.npy"), dtype=torch.float64,
    )
    species_order = sorted({w["species"] for w in window_map})
    keep_idx = [species_order_all.index(s) for s in species_order]
    tax_dist = tax_dist_all[keep_idx][:, keep_idx]

    species_to_id = {s: i for i, s in enumerate(species_order)}
    species_ids = torch.tensor([species_to_id[w["species"]] for w in window_map], dtype=torch.long)
    x = TERNARY.to_ternary(indices)

    return x, indices, species_ids, species_order, tax_dist


def evaluate(model, loader, kl_loss, tax_loss, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, idx, sp in loader:
            x, idx, sp = x.to(device), idx.to(device), sp.to(device)
            out = model(x, decode_b=False)
            c = model.projections.get_curvature()
            recon = compute_coverage_loss(out["logits_A"], x)
            kl = kl_loss(out["mu_A"], out["logvar_A"], out["z_A_hyp"], curvature=c)
            tax, _ = tax_loss(out["z_A_hyp"], idx, species_ids=sp, curvature=c)
            total_loss += (recon + kl + tax).item() * x.size(0)
            preds = out["logits_A"].view(-1, 9, 3).argmax(-1) - 1
            total_correct += (preds == x).all(dim=-1).sum().item()
            total_n += x.size(0)
    model.train()
    return total_loss / total_n, total_correct / total_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-map-train", default="data/cytochrome_c/window_map_train.json")
    parser.add_argument("--indices-train", default="data/cytochrome_c/indices_train.pt")
    parser.add_argument("--taxonomy-dir", default="data/cytochrome_c")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--kl-free-bits", type=float, default=0.5)
    parser.add_argument("--taxonomy-weight", type=float, default=5.0)
    parser.add_argument("--taxonomy-n-pairs", type=int, default=500)
    parser.add_argument("--taxonomy-max-target-distance", type=float, default=4.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-out", default="runs/cytochrome_c_D_taxonomy/best.pt")
    args = parser.parse_args()

    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, indices, species_ids, species_order, tax_dist = load_training_data(
        Path(args.window_map_train), Path(args.indices_train), Path(args.taxonomy_dir),
    )
    print(f"[data] {len(species_order)} training species, {len(x)} windows, "
          f"tax_dist range=[{tax_dist.min():.0f}, {tax_dist.max():.0f}]")

    n = len(x)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, round(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(x[train_idx], indices[train_idx], species_ids[train_idx])
    val_ds = TensorDataset(x[val_idx], indices[val_idx], species_ids[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print("[NOTE] This train/val split is row-level, for training-time monitoring only -- "
          "it is NOT the species-level holdout test (that's Fase 2's excluded 9 species, "
          "scored separately by evaluate_phylogeny_recovery.py after training).")

    model = TernaryVAEV6Controllable(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        max_radius=0.99, curvature=1.0, learnable_curvature=True,
        factored=False, positional_encoding=False,
        n_projection_layers=3, projection_dropout=0.1,
        encoder_type="improved", decoder_type="improved", init_identity=True,
        tangent_scale_init=0.1,
        encoder_a_trainable=True, encoder_b_trainable=True, projections_trainable=True,
        encoder_a_lr_scale=1.0, encoder_b_lr_scale=1.0, projections_lr_scale=1.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    kl_loss = HyperbolicKLDivergence(
        curvature=1.0, beta=args.kl_beta, free_bits=args.kl_free_bits, variance_only=False,
    )
    tax_loss = TaxonomyGeodesicLoss(
        tax_dist, max_target_distance=args.taxonomy_max_target_distance,
        n_pairs=args.taxonomy_n_pairs,
    ).to(device)

    best_val_loss = float("inf")
    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = None
        for x_b, idx_b, sp_b in train_loader:
            x_b, idx_b, sp_b = x_b.to(device), idx_b.to(device), sp_b.to(device)
            optimizer.zero_grad()
            out = model(x_b, decode_b=False)
            c = model.projections.get_curvature()
            recon = compute_coverage_loss(out["logits_A"], x_b)
            kl = kl_loss(out["mu_A"], out["logvar_A"], out["z_A_hyp"], curvature=c)
            tax, tax_metrics = tax_loss(out["z_A_hyp"], idx_b, species_ids=sp_b, curvature=c)
            train_loss = recon + args.kl_weight * kl + args.taxonomy_weight * tax
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, kl_loss, tax_loss, device)
            print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss.item():.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"tax_dist_corr={tax_metrics['distance_correlation']:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                    "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                    "factored": False, "positional_encoding": False,
                    "curvature": 1.0, "max_radius": 0.99,
                    "n_projection_layers": 3, "projection_dropout": 0.1,
                    "encoder_type": "improved", "decoder_type": "improved",
                    "init_identity": True, "tangent_scale_init": 0.1,
                    "train_species_order": species_order,
                }, out_path)

    print(f"\n[OK] Best val_loss={best_val_loss:.4f}. Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
