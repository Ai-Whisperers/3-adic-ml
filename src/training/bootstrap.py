# Copyright (c) 2024-2026 AI Whisperers
#
# Licensed under the MIT License.
# See LICENSE file in the repository root for full license text.

"""Bootstrapping and hardware setup for p-adic VAE training."""

from datetime import datetime
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config.paths import PROJECT_ROOT
from ..core import TERNARY
from ..models import TernaryVAEV6Controllable
from ..utils.checkpoint import get_model_state_dict, load_checkpoint_compat


def set_determinism(
    seed: int, deterministic: bool = True, use_float64: bool = True
) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but exact)
        use_float64: If True, use float64 as default dtype (required for geoopt stability)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set default dtype to float64 for geoopt compatibility
    if use_float64:
        torch.set_default_dtype(torch.float64)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUDA >= 10.2 requires this for cuBLAS determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            # Use warn_only=True because some ops (like nll_loss) lack deterministic implementations
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't have warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def get_timestamp() -> str:
    """Get formatted timestamp for run naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class DataAuditor:
    """Validates data integrity before training."""

    def __init__(self, seed: int):
        self.seed = seed
        self.audit_log: Dict[str, Any] = {}

    def prepare_data(
        self,
        val_frac: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.Tensor]:
        """Generate data, split, and validate.

        Args:
            val_frac: Fraction of data for validation
            device: Device for data tensors

        Returns:
            Tuple of (train_dataset, val_dataset, all_indices)
        """
        if device is None:
            device = torch.device("cpu")
        print("\n[AUDIT] Data Integrity Check...")

        # Generate all operations (uses cached LUT from TERNARY singleton)
        all_ops = TERNARY.all_ternary()
        n = len(all_ops)
        all_indices = torch.arange(n, dtype=torch.long)

        # Deterministic split
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, round(n * val_frac))

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train = all_ops[train_idx]
        X_val = all_ops[val_idx]
        idx_train = all_indices[train_idx]
        idx_val = all_indices[val_idx]

        # Leakage check
        train_set = {tuple(x.tolist()) for x in X_train}
        val_set = {tuple(x.tolist()) for x in X_val}
        overlap = train_set.intersection(val_set)

        self.audit_log["n_train"] = len(X_train)
        self.audit_log["n_val"] = len(X_val)
        self.audit_log["data_leakage_count"] = len(overlap)

        if len(overlap) > 0:
            print(
                f"  [WARN] Data leakage detected: {len(overlap)} samples in both sets"
            )
        else:
            print(f"  [OK] No data leakage (train={len(X_train)}, val={len(X_val)})")

        # Value distribution check
        vals, counts = torch.unique(X_train, return_counts=True)
        dist = counts.float() / counts.sum()
        self.audit_log["value_distribution"] = dict(zip(vals.tolist(), dist.tolist(), strict=False))
        print(
            f"  [OK] Value distribution: {dict(zip(vals.tolist(), [f'{d:.2%}' for d in dist.tolist()], strict=False))}"
        )

        # Create datasets
        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(X_train, idx_train)
        val_ds = TensorDataset(X_val, idx_val)

        return train_ds, val_ds, all_indices


class ModelAuditor:
    """Validates model health before training."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.audit_log: Dict[str, Any] = {}

    def create_and_validate_model(self, force: bool = False) -> nn.Module:
        """Create model, load checkpoint, validate gradients.

        Args:
            force: If True, continue even if validation fails

        Returns:
            Validated model ready for training
        """
        print("\n[AUDIT] Model Health Check...")

        model_cfg = self.config.get("model", {})
        option_c_cfg = self.config.get("option_c", {})
        model_name = model_cfg.get("name", "TernaryVAEV6Controllable")
        encoder_type = model_cfg.get("encoder_type", "improved")
        decoder_type = model_cfg.get("decoder_type", "improved")

        # LR scales from option_c config (differential learning rates per component)
        encoder_a_lr_scale = option_c_cfg.get("encoder_a_lr_scale", 0.05)
        encoder_b_lr_scale = option_c_cfg.get("encoder_b_lr_scale", 0.1)
        projections_lr_scale = option_c_cfg.get("projections_lr_scale", 1.0)

        # Initial trainability from statenet config (nested structure)
        statenet_cfg = self.config.get("statenet", {})
        initial_cfg = statenet_cfg.get("initial", {})
        encoder_a_trainable = initial_cfg.get("encoder_a_trainable", False)
        encoder_b_trainable = initial_cfg.get("encoder_b_trainable", True)
        projections_trainable = initial_cfg.get("projections_trainable", True)

        # Instantiate model
        model = TernaryVAEV6Controllable(
            latent_dim=model_cfg.get("latent_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            max_radius=model_cfg.get("max_radius", 0.95),
            curvature=model_cfg.get("curvature", 1.0),
            n_projection_layers=model_cfg.get("projection_layers", 2),
            projection_dropout=model_cfg.get("projection_dropout", 0.1),
            learnable_curvature=model_cfg.get("learnable_curvature", False),
            init_identity=model_cfg.get("init_identity", True),
            tangent_scale_init=model_cfg.get("tangent_scale", 0.1),
            factored=model_cfg.get("factored", False),
            radial_dims=model_cfg.get("radial_dims", 4),
            detach_radial=model_cfg.get("detach_radial", False),
            positional_encoding=model_cfg.get("positional_encoding", False),
            encoder_a_trainable=encoder_a_trainable,
            encoder_b_trainable=encoder_b_trainable,
            projections_trainable=projections_trainable,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            encoder_a_lr_scale=encoder_a_lr_scale,
            encoder_b_lr_scale=encoder_b_lr_scale,
            projections_lr_scale=projections_lr_scale,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] Model created: {model_name}")
        print(f"       Parameters: {n_params:,} total, {n_trainable:,} trainable")

        # Load anchor checkpoint if specified
        anchor_cfg = self.config.get("anchor_checkpoint") or {}
        ckpt_path_str = anchor_cfg.get("path")

        if ckpt_path_str and ckpt_path_str != "null":
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            if ckpt_path.exists():
                try:
                    ckpt = load_checkpoint_compat(ckpt_path, map_location=self.device)
                    state_dict = get_model_state_dict(ckpt)

                    # Load with strict=False (projections may not match)
                    missing, unexpected = model.load_state_dict(
                        state_dict, strict=False
                    )
                    print(f"  [OK] Loaded checkpoint: {ckpt_path.name}")
                    print(
                        f"       Missing keys: {len(missing)}, Unexpected: {len(unexpected)}"
                    )

                    self.audit_log["checkpoint_loaded"] = True
                    self.audit_log["checkpoint_path"] = str(ckpt_path)

                except Exception as e:
                    print(f"  [WARN] Checkpoint load failed: {e}")
                    self.audit_log["checkpoint_loaded"] = False
                    self.audit_log["checkpoint_error"] = str(e)
                    if not force:
                        raise
                    print("  [WARN] --force active: proceeding with RANDOM INIT (checkpoint ignored)")
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt_path}")
                self.audit_log["checkpoint_loaded"] = False
                self.audit_log["checkpoint_error"] = f"not found: {ckpt_path}"
                if not force:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                print("  [WARN] --force active: proceeding with RANDOM INIT (checkpoint not found)")
        else:
            print("  [INFO] No anchor checkpoint specified (training from scratch)")
            self.audit_log["checkpoint_loaded"] = False

        # Gradient flow check
        model.train()
        dummy_input = torch.randint(-1, 2, (32, 9)).double().to(self.device)

        try:
            out = model(dummy_input)
            logits = out.get("logits_A", out.get("logits"))

            # Simple forward check
            if logits is None:
                raise ValueError("Model output missing logits")

            # Compute dummy loss and backward
            loss = logits.mean()
            loss.backward()

            # Check gradient norm
            total_norm = 0.0
            dead_params = 0
            total_params = 0

            for p in model.parameters():
                if p.requires_grad:
                    total_params += 1
                    if p.grad is not None:
                        total_norm += p.grad.norm(2).item()
                        if p.grad.norm(2).item() == 0:
                            dead_params += 1
                    else:
                        dead_params += 1

            self.audit_log["initial_grad_norm"] = total_norm
            self.audit_log["dead_params"] = dead_params

            if total_norm == 0:
                print("  [ERROR] Vanishing gradients - model is dead")
                if not force:
                    raise RuntimeError("Dead model detected")
            elif total_norm > 1000:
                print("  [WARN] Large gradient norm - may explode")
            else:
                print(f"  [OK] Gradient flow active (norm={total_norm:.4f})")

        except Exception as e:
            print(f"  [ERROR] Model health check failed: {e}")
            if not force:
                raise
        finally:
            model.zero_grad()

        return model
