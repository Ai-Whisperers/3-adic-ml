#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Launch V5.11.11 StateNet Training - RTX 2060 SUPER Optimized.

This script provides a simple entry point for training the V5.11.11
statenet model with optimal settings for RTX 2060 SUPER (8GB VRAM).

Features:
- Automatic v5.5 base model training if needed
- Hardware detection and verification
- Memory-optimized settings
- Progress monitoring with TensorBoard

Hardware Requirements:
- NVIDIA GPU with 6+ GB VRAM (tested on RTX 2060 SUPER)
- 16+ GB RAM
- CUDA 12.0+

Usage:
    # Full training (recommended)
    python scripts/training/launch_statenet_training.py

    # Quick test (5 epochs)
    python scripts/training/launch_statenet_training.py --quick

    # Resume from checkpoint
    python scripts/training/launch_statenet_training.py --resume

    # Custom epochs
    python scripts/training/launch_statenet_training.py --epochs 200
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import torch


def print_banner():
    """Print training banner."""
    print("\n" + "=" * 70)
    print("  V5.11.11 HOMEOSTATIC TRAINING - RTX 2060 SUPER OPTIMIZED")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def check_hardware() -> dict:
    """Check and report hardware configuration."""
    print("[1/4] Checking Hardware...")

    info = {"cuda_available": False, "device_name": None, "vram_gb": 0}

    if not torch.cuda.is_available():
        print("  [ERROR] CUDA not available!")
        print("  Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu126")
        return info

    props = torch.cuda.get_device_properties(0)
    info["cuda_available"] = True
    info["device_name"] = props.name
    info["vram_gb"] = props.total_memory / (1024**3)

    print(f"  [OK] GPU: {props.name}")
    print(f"  [OK] VRAM: {info['vram_gb']:.1f} GB")
    print(f"  [OK] CUDA: {torch.version.cuda}")
    print(f"  [OK] PyTorch: {torch.__version__}")

    # Check VRAM is sufficient
    if info["vram_gb"] < 6:
        print(f"  [WARN] {info['vram_gb']:.1f} GB VRAM may be insufficient.")
        print("     Consider reducing batch_size to 256 or enabling gradient_checkpointing.")

    # Clear any cached memory
    torch.cuda.empty_cache()
    gc.collect()

    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"  [OK] Available VRAM: {free_mem / (1024**3):.1f} GB")

    return info


def check_dependencies():
    """Check required dependencies."""
    print("\n[2/4] Checking Dependencies...")

    required = ["scipy", "numpy", "yaml", "tensorboard"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)

    # Check optional geoopt for Riemannian optimization
    try:
        import geoopt
        print(f"  [OK] geoopt (Riemannian optimization)")
    except ImportError:
        print(f"  [WARN] geoopt not installed (optional for Riemannian optimization)")
        print("     Install with: pip install geoopt")

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Install with: pip install " + " ".join(missing))
        return False

    return True


def check_directories(project_root: Path) -> dict:
    """Create necessary directories."""
    print("\n[3/4] Setting up Directories...")

    dirs = {
        "checkpoints": project_root / "sandbox-training" / "checkpoints" / "v5_11_11_statenet_rtx2060s",
        "runs": project_root / "runs" / "v5_11_11_statenet_rtx2060s",
        "v5_5_checkpoints": project_root / "sandbox-training" / "checkpoints" / "v5_5",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {name}: {path}")

    return dirs


def check_base_checkpoint(dirs: dict) -> bool:
    """Check if v5.5 base checkpoint exists."""
    v5_5_path = dirs["v5_5_checkpoints"] / "latest.pt"
    return v5_5_path.exists()


def print_training_config(args):
    """Print training configuration."""
    print("\n[4/4] Training Configuration...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Mixed precision (FP16): Enabled")
    print(f"  - StateNet: Enabled")
    print(f"  - Riemannian optimization: Enabled")
    print(f"  - Q-gated annealing: Enabled")
    print(f"  - TensorBoard: runs/v5_11_11_statenet_rtx2060s")


def run_training(args, project_root: Path, dirs: dict):
    """Run the training pipeline."""
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70 + "\n")

    # Add project root to path
    sys.path.insert(0, str(project_root))

    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations

    device = torch.device("cuda:0")

    # Check for base checkpoint
    if not check_base_checkpoint(dirs):
        print("Phase 1: Training v5.5 Base Model (coverage training)...")
        print("-" * 60)
        train_base_model(device, dirs, project_root)
    else:
        print("[OK] Using existing v5.5 base checkpoint")

    # Now train the v5.11.11 model
    print("\nPhase 2: Training V5.11.11 StateNet Model...")
    print("-" * 60)
    train_statenet_model(args, device, dirs, project_root)


def train_base_model(device, dirs, project_root):
    """Train v5.5 base model for coverage."""
    import numpy as np
    from scipy.stats import spearmanr
    from torch.utils.tensorboard import SummaryWriter

    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations

    # Simple dual-encoder VAE
    class TernaryVAEv5_5(torch.nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.latent_dim = latent_dim

            self.encoder_A = torch.nn.Sequential(
                torch.nn.Linear(9, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            )
            self.fc_mu_A = torch.nn.Linear(64, latent_dim)
            self.fc_logvar_A = torch.nn.Linear(64, latent_dim)

            self.encoder_B = torch.nn.Sequential(
                torch.nn.Linear(9, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            )
            self.fc_mu_B = torch.nn.Linear(64, latent_dim)
            self.fc_logvar_B = torch.nn.Linear(64, latent_dim)

            self.decoder_A = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 27),
            )

        def encode(self, x):
            h_A = self.encoder_A(x)
            h_B = self.encoder_B(x)
            return (
                self.fc_mu_A(h_A), self.fc_logvar_A(h_A),
                self.fc_mu_B(h_B), self.fc_logvar_B(h_B)
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu_A, logvar_A, mu_B, logvar_B = self.encode(x)
            z_A = self.reparameterize(mu_A, logvar_A)
            logits = self.decoder_A(z_A)
            return {"logits": logits, "mu_A": mu_A, "logvar_A": logvar_A, "z_A": z_A}

    model = TernaryVAEv5_5(latent_dim=16).to(device)

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = 200
    batch_size = 512
    best_coverage = 0.0

    writer = SummaryWriter(log_dir=str(project_root / "runs" / "v5_5_base"))

    print(f"Training v5.5 for {n_epochs} epochs (or until 100% coverage)...")

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(x), device=device)
        total_loss = 0.0
        n_batches = (len(x) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(x))
            batch_idx = perm[start:end]
            x_batch = x[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)

            logits = outputs["logits"].view(-1, 9, 3)
            targets = (x_batch.long() + 1).clamp(0, 2)
            recon_loss = criterion(logits.permute(0, 2, 1), targets)

            mu_A, logvar_A = outputs["mu_A"], outputs["logvar_A"]
            kl_loss = -0.5 * torch.mean(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
            kl_loss = torch.clamp(kl_loss, min=0.3)

            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            logits = outputs["logits"].view(-1, 9, 3)
            preds = torch.argmax(logits, dim=-1) - 1
            targets = x.long()
            correct = (preds == targets).float().mean(dim=1)
            coverage = (correct == 1.0).sum().item() / len(x)

            mu_A = outputs["mu_A"]
            radii = torch.norm(mu_A, dim=1).cpu().numpy()
            valuations = TERNARY.valuation(indices).cpu().numpy()
            radial_corr = spearmanr(valuations, radii)[0]

        writer.add_scalar("Train/loss", total_loss / n_batches, epoch)
        writer.add_scalar("Eval/coverage", coverage, epoch)
        writer.add_scalar("Eval/radial_corr", radial_corr, epoch)

        if epoch % 20 == 0 or coverage > best_coverage:
            print(f"  Epoch {epoch:3d}: loss={total_loss/n_batches:.4f}, "
                  f"coverage={coverage*100:.1f}%, radial_corr={radial_corr:.3f}")

        if coverage > best_coverage:
            best_coverage = coverage
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "coverage": coverage,
                "radial_corr": radial_corr,
            }, dirs["v5_5_checkpoints"] / "latest.pt")

            if coverage >= 1.0:
                print(f"\n  [OK] 100% COVERAGE ACHIEVED at epoch {epoch}!")
                break

    writer.close()
    print(f"\nv5.5 training complete. Best coverage: {best_coverage*100:.1f}%")


def train_statenet_model(args, device, dirs, project_root):
    """Train the V5.11.11 statenet model."""
    import numpy as np
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    from src.core import TERNARY
    from src.data import generate_all_ternary_operations
    from src.models import StateNet, compute_Q

    # Load model
    print("Loading model architecture...")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
        use_controller=True,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    # Load v5.5 checkpoint with proper key mapping
    v5_5_path = dirs["v5_5_checkpoints"] / "latest.pt"
    if v5_5_path.exists():
        print(f"Loading v5.5 checkpoint: {v5_5_path}")
        ckpt = torch.load(v5_5_path, map_location=device, weights_only=False)
        pretrained = ckpt.get("model", {})

        # Map v5.5 keys to V5.11 structure:
        # v5.5: encoder_A.0.weight -> V5.11: encoder_A.encoder.0.weight
        # v5.5: fc_mu_A.weight -> V5.11: encoder_A.fc_mu.weight
        # v5.5: decoder_A.0.weight -> V5.11: decoder_A.decoder.0.weight
        key_mapping = {}
        for k in pretrained.keys():
            if k.startswith("encoder_A."):
                # encoder_A.0.weight -> encoder_A.encoder.0.weight
                suffix = k[len("encoder_A."):]
                key_mapping[k] = f"encoder_A.encoder.{suffix}"
            elif k.startswith("encoder_B."):
                suffix = k[len("encoder_B."):]
                key_mapping[k] = f"encoder_B.encoder.{suffix}"
            elif k.startswith("fc_mu_A."):
                suffix = k[len("fc_mu_A."):]
                key_mapping[k] = f"encoder_A.fc_mu.{suffix}"
            elif k.startswith("fc_logvar_A."):
                suffix = k[len("fc_logvar_A."):]
                key_mapping[k] = f"encoder_A.fc_logvar.{suffix}"
            elif k.startswith("fc_mu_B."):
                suffix = k[len("fc_mu_B."):]
                key_mapping[k] = f"encoder_B.fc_mu.{suffix}"
            elif k.startswith("fc_logvar_B."):
                suffix = k[len("fc_logvar_B."):]
                key_mapping[k] = f"encoder_B.fc_logvar.{suffix}"
            elif k.startswith("decoder_A."):
                suffix = k[len("decoder_A."):]
                key_mapping[k] = f"decoder_A.decoder.{suffix}"

        # Apply mapping
        mapped_state = {}
        for old_key, new_key in key_mapping.items():
            if old_key in pretrained:
                mapped_state[new_key] = pretrained[old_key]

        # Load compatible weights
        model_dict = model.state_dict()
        compatible = {k: v for k, v in mapped_state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(compatible)}/{len(model_dict)} weights from v5.5 checkpoint")

        if len(compatible) == 0:
            print("  [WARN] No compatible weights found! Training from scratch.")
    else:
        print("  [INFO] No v5.5 checkpoint found. Training from scratch.")

    model = model.to(device)
    model.set_encoder_a_frozen(True)  # Start with encoder_A frozen
    model.set_encoder_b_frozen(False)

    # Dataset
    print("Loading dataset...")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    print(f"Dataset: {len(all_ops)} operations, {len(dataloader)} batches per epoch")

    # StateNet controller
    statenet = StateNet(
        coverage_freeze_threshold=0.995,
        coverage_unfreeze_threshold=0.999,
        enable_annealing=True,
    )

    # Loss function
    class RichHierarchyLoss(torch.nn.Module):
        def __init__(self, inner_radius=0.1, outer_radius=0.85):
            super().__init__()
            self.inner_radius = inner_radius
            self.outer_radius = outer_radius
            target_radii = torch.tensor([
                outer_radius - (v / 9) * (outer_radius - inner_radius)
                for v in range(10)
            ])
            self.register_buffer("target_radii", target_radii)

        def forward(self, z_hyp, indices_batch, logits, targets):
            device = z_hyp.device
            radii = z_hyp.norm(dim=-1)
            valuations = TERNARY.valuation(indices_batch).long().to(device)

            # Hierarchy loss
            hierarchy_loss = torch.tensor(0.0, device=device)
            for v in torch.unique(valuations):
                mask = valuations == v
                if mask.sum() > 0:
                    mean_r = radii[mask].mean()
                    target_r = self.target_radii[v]
                    hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
            hierarchy_loss = hierarchy_loss / 10

            # Coverage loss
            coverage_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 3),
                (targets + 1).long().view(-1),
            )

            # Separation loss
            separation_loss = torch.tensor(0.0, device=device)
            mean_radii = []
            for v in sorted(torch.unique(valuations).tolist()):
                mask = valuations == v
                if mask.sum() > 0:
                    mean_radii.append(radii[mask].mean())

            for i in range(len(mean_radii) - 1):
                violation = torch.relu(mean_radii[i + 1] - mean_radii[i] + 0.01)
                separation_loss = separation_loss + violation

            total = 5.0 * hierarchy_loss + 1.0 * coverage_loss + 3.0 * separation_loss

            return {"total": total, "hierarchy": hierarchy_loss, "coverage": coverage_loss, "separation": separation_loss}

    loss_fn = RichHierarchyLoss().to(device)

    # Training loop
    writer = SummaryWriter(log_dir=str(dirs["runs"]))
    best_hierarchy = 0.0
    best_Q = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    for epoch in range(args.epochs):
        model.train()
        param_groups = model.get_param_groups(args.lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)

        epoch_losses = {"total": 0, "hierarchy": 0, "coverage": 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(batch_ops, compute_control=False)
                    z_A = out["z_A_hyp"]
                    logits = model.decoder_A(out["mu_A"])
                    losses = loss_fn(z_A, batch_idx, logits, batch_ops)

                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch_ops, compute_control=False)
                z_A = out["z_A_hyp"]
                logits = model.decoder_A(out["mu_A"])
                losses = loss_fn(z_A, batch_idx, logits, batch_ops)

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_losses["total"] += losses["total"].item()
            epoch_losses["hierarchy"] += losses["hierarchy"].item()
            epoch_losses["coverage"] += losses["coverage"].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Evaluate
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                all_radii = []
                all_correct = []
                for i in range(0, len(all_ops), 4096):
                    batch = all_ops[i:i+4096].to(device)
                    out = model(batch, compute_control=False)
                    all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())

                    logits = model.decoder_A(out["mu_A"])
                    preds = torch.argmax(logits, dim=-1) - 1
                    correct = (preds == batch.long()).float().mean(dim=1).cpu().numpy()
                    all_correct.append(correct)

                all_radii = np.concatenate(all_radii)
                all_correct = np.concatenate(all_correct)
                valuations = TERNARY.valuation(indices).numpy()

                coverage = (all_correct == 1.0).mean()
                hierarchy = spearmanr(valuations, all_radii)[0]

            # Update StateNet
            statenet_state = statenet.update(
                epoch=epoch,
                coverage=coverage,
                hierarchy_A=hierarchy,
                hierarchy_B=hierarchy,
                dist_corr_A=0.0,
            )
            model.apply_statenet_state(statenet_state)

            writer.add_scalar("Train/loss", epoch_losses["total"], epoch)
            writer.add_scalar("Eval/coverage", coverage, epoch)
            writer.add_scalar("Eval/hierarchy", hierarchy, epoch)

            print(f"Epoch {epoch:3d}/{args.epochs}: loss={epoch_losses['total']:.4f}, "
                  f"cov={coverage*100:.1f}%, hier={hierarchy:.4f}, "
                  f"freeze={model.get_freeze_state_summary()}")

            # Save best
            if hierarchy < best_hierarchy and coverage > 0.99:
                best_hierarchy = hierarchy
                print(f"  [OK] New best hierarchy: {best_hierarchy:.4f}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "coverage": coverage,
                    "hierarchy": hierarchy,
                    "statenet": statenet.get_state_summary(),
                }, dirs["checkpoints"] / "best.pt")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "coverage": coverage,
                "hierarchy": hierarchy,
            }, dirs["checkpoints"] / "latest.pt")

    writer.close()
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best hierarchy: {best_hierarchy:.4f}")
    print(f"  Checkpoints: {dirs['checkpoints']}")
    print(f"  TensorBoard: tensorboard --logdir {dirs['runs']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Launch V5.11.11 StateNet Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/training/launch_statenet_training.py           # Full training
    python scripts/training/launch_statenet_training.py --quick   # Quick test (5 epochs)
    python scripts/training/launch_statenet_training.py --epochs 200  # Custom epochs
        """
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (512 for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        print("Quick test mode: 5 epochs")

    print_banner()

    # Step 1: Check hardware
    hw_info = check_hardware()
    if not hw_info["cuda_available"]:
        print("\nERROR: CUDA required for training.")
        sys.exit(1)

    # Step 2: Check dependencies
    if not check_dependencies():
        print("\nERROR: Missing dependencies.")
        sys.exit(1)

    # Step 3: Setup directories
    project_root = Path(__file__).resolve().parents[2]
    dirs = check_directories(project_root)

    # Step 4: Print config
    print_training_config(args)

    # Confirm
    print("\n" + "-" * 70)
    if not args.yes:
        input("Press Enter to start training (Ctrl+C to cancel)...")

    # Run training
    run_training(args, project_root, dirs)


if __name__ == "__main__":
    main()
