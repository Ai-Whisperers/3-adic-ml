"""Train Condition A (flat Euclidean VAE baseline) for the phylogeny
validation pipeline. Fase 3 of docs/plans/PHYLOGENY-VALIDATION-PIPELINE.md.

Simple loop, no curricula/StateNet/Lagrangian dual ascent: reconstruction
cross-entropy + standard Gaussian KL, Adam, over data/cytochrome_c/indices.pt.
Forcing the p-adic curriculum engine (src/train.py) onto a plain VAE would be
more code than writing this directly -- see src/models/vae_baseline.py.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.losses.utils import compute_coverage_loss
from src.models.vae_baseline import TernaryVAEEuclideanBaseline
from src.training.bootstrap import DataAuditor, set_determinism


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return torch.mean(torch.sum(kl_per_dim, dim=-1))


def evaluate(
    model: TernaryVAEEuclideanBaseline, loader: DataLoader, device: torch.device, free_bits: float,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            recon = compute_coverage_loss(out["logits"], x)
            kl = kl_divergence(out["mu"], out["logvar"], free_bits)
            total_loss += (recon + kl).item() * x.size(0)
            preds = out["logits"].view(-1, 9, 3).argmax(-1) - 1
            total_correct += (preds == x).all(dim=-1).sum().item()
            total_n += x.size(0)
    model.train()
    return total_loss / total_n, total_correct / total_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices-path", default="data/cytochrome_c/indices.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128, help="matches Conditions B/C model.latent_dim")
    parser.add_argument("--hidden-dim", type=int, default=256, help="matches Conditions B/C model.hidden_dim")
    parser.add_argument(
        "--kl-weight", type=float, default=0.05,
        help="At 1.0 with free_bits=0, the 128-dim posterior collapses to the "
             "prior and reconstruction never improves. Not equivalent to B/C's "
             "effective multiplier (weight=0.05 * beta=0.1 = 0.005) -- see also --free-bits.",
    )
    parser.add_argument(
        "--free-bits", type=float, default=0.5,
        help="Per-dimension KL floor, same mechanism/default as B/C's hyperbolic_kl.free_bits.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-out", default="runs/cytochrome_c_A_euclidean/best.pt")
    args = parser.parse_args()

    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, _ = DataAuditor(seed=args.seed).prepare_data(
        val_frac=args.val_frac, device=device, custom_indices_path=args.indices_path,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TernaryVAEEuclideanBaseline(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = None
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            recon = compute_coverage_loss(out["logits"], x)
            kl = kl_divergence(out["mu"], out["logvar"], args.free_bits)
            train_loss = recon + args.kl_weight * kl
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, device, args.free_bits)
            print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss.item():.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch,
                     "val_loss": val_loss, "val_acc": val_acc,
                     "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                     "kl_weight": args.kl_weight, "free_bits": args.free_bits, "lr": args.lr},
                    out_path,
                )

    print(f"\n[OK] Best val_loss={best_val_loss:.4f}. Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
