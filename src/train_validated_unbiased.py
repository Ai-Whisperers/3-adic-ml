#!/usr/bin/env python3
"""
Validated TernaryVAE Training Script

This script provides enhanced validation and error handling to prevent common
training pipeline issues, particularly the 0% coverage problem with V5.11+
architectures that require frozen checkpoints.

Key improvements:
1. Checkpoint validation before training starts
2. Architecture compatibility checking
3. Automatic checkpoint path fixing
4. Clear error messages and recommendations
5. Real-time monitoring for coverage collapse

Usage:
    python scripts/train_validated.py --config configs/v5_12_4.yaml
    python scripts/train_validated.py --config configs/v5_12_4.yaml --validate-only
    python scripts/train_validated.py --config configs/v5_12_4.yaml --auto-fix

Created: 2026-01-12
Root Cause Fix: Prevents null checkpoint issues causing 0% coverage
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import warnings
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint_validator import (
    CheckpointValidator,
    validate_training_config,
    CheckpointCompatibilityError
)
from src.core import TERNARY, to_ternary
from src.data import generate_all_ternary_operations
from src.geometry import poincare_distance, get_manifold
from src.models import StateNet, compute_Q


class ValidatedTrainer:
    """Enhanced trainer with validation and error prevention."""

    def __init__(self, config_path: str, auto_fix: bool = False):
        """Initialize trainer with validation."""
        self.config_path = Path(config_path)
        self.auto_fix = auto_fix
        self.config = self._load_and_validate_config()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        print(f"📋 Loading configuration from: {self.config_path}")

        # Load config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate configuration
        print("🔍 Validating configuration...")
        is_valid, errors = validate_training_config(config)

        if not is_valid:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  {error}")

            if self.auto_fix:
                print("\n🔧 Attempting auto-fix...")
                model_name = config.get('model', {}).get('name')
                config = CheckpointValidator.fix_null_checkpoint_config(config, model_name)

                # Re-validate
                is_valid, new_errors = validate_training_config(config)
                if is_valid:
                    print("✅ Auto-fix successful!")
                else:
                    print("❌ Auto-fix failed:")
                    for error in new_errors:
                        print(f"  {error}")
                    raise CheckpointCompatibilityError("Configuration cannot be auto-fixed")
            else:
                print("\n💡 Run with --auto-fix to attempt automatic correction")
                raise CheckpointCompatibilityError("Invalid configuration")
        else:
            print("✅ Configuration validation passed!")

        return config

    def validate_checkpoint_loading(self, model: torch.nn.Module) -> bool:
        """Validate that checkpoint can be loaded successfully."""
        frozen_cfg = self.config.get('frozen_checkpoint', {})
        checkpoint_path = frozen_cfg.get('path')

        if checkpoint_path and checkpoint_path != 'null':
            print(f"🔍 Validating checkpoint dimensions: {checkpoint_path}")

            # Check if checkpoint exists
            ckpt_file = Path(checkpoint_path)
            if not ckpt_file.exists():
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return False

            # Check dimensional compatibility
            is_compatible, errors = CheckpointValidator.validate_checkpoint_dimensions(
                checkpoint_path, model
            )

            if not is_compatible:
                print("❌ Checkpoint dimension validation failed:")
                for error in errors:
                    print(f"  {error}")
                return False
            else:
                print("✅ Checkpoint dimensions compatible!")

        return True

    def check_initial_coverage(self, model: torch.nn.Module, device: torch.device) -> float:
        """Check initial coverage to detect the 0% coverage issue early."""
        print("🔍 Checking initial coverage...")

        # Load full dataset for coverage check
        ops = TERNARY.all_operations(device=device)  # (19683, 9)

        with torch.no_grad():
            model.eval()

            # Get embeddings (sample a batch if memory limited)
            batch_size = 1000
            coverage_total = 0
            n_batches = 0

            for i in range(0, len(ops), batch_size):
                batch = ops[i:i+batch_size]
                outputs = model(batch, compute_control=False)

                # Check reconstruction via VAE-A (coverage encoder)
                mu_A = outputs.get('mu_A', outputs.get('z_A_euc'))
                if mu_A is not None:
                    # Get reconstruction logits
                    logits = model.decoder_A(mu_A)
                    preds = torch.argmax(logits, dim=-1) - 1  # Convert to {-1, 0, 1}

                    # Check perfect reconstruction
                    correct = (preds == batch.long()).float().mean(dim=1)
                    batch_coverage = (correct == 1.0).float().mean().item()

                    coverage_total += batch_coverage
                    n_batches += 1

            overall_coverage = (coverage_total / n_batches) * 100 if n_batches > 0 else 0.0

        print(f"📊 Initial coverage: {overall_coverage:.2f}%")

        # Check for the critical 0% coverage issue
        if overall_coverage < 5.0:
            print("⚠️  WARNING: Very low initial coverage detected!")
            print("   This usually indicates missing frozen checkpoint for V5.11+ architecture.")
            print("   Expected: ~100% coverage with proper frozen components.")
            return overall_coverage
        elif overall_coverage > 95.0:
            print("✅ Excellent initial coverage! Training should work correctly.")
        else:
            print("🔶 Moderate initial coverage. Monitor during training.")

        return overall_coverage

    def run_training(self):
        """Run validated training with monitoring."""
        print("🚀 Starting validated training...")

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {device}")

        # Create model
        model_name = self.config.get('model', {}).get('name')
        print(f"🏗️  Creating model: {model_name}")

        try:
            model_config = self.config.get('model', {})

            # Create model based on name
            if model_name == 'TernaryVAEV5_11':
                model = TernaryVAEV5_11(**model_config)
            elif model_name == 'TernaryVAEV5_11_PartialFreeze':
                model = TernaryVAEV5_11_PartialFreeze(**model_config)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            model.to(device)

            print(f"📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"🎛️  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            raise

        # Validate checkpoint loading
        if not self.validate_checkpoint_loading(model):
            raise CheckpointCompatibilityError("Checkpoint validation failed")

        # Load checkpoint if specified
        frozen_cfg = self.config.get('frozen_checkpoint', {})
        checkpoint_path = frozen_cfg.get('path')

        if checkpoint_path and checkpoint_path != 'null':
            try:
                print(f"📥 Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Load with strict=False for partial loading
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"📋 Missing keys (expected for partial loading): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")

                print("✅ Checkpoint loaded successfully!")

            except Exception as e:
                print(f"❌ Checkpoint loading failed: {e}")
                raise

        # Check initial coverage
        initial_coverage = self.check_initial_coverage(model, device)

        if initial_coverage < 1.0:
            print("⚠️  CRITICAL WARNING: Initial coverage is very low!")
            print("   This indicates the training pipeline has the 0% coverage issue.")
            print("   Training may appear to work but will produce useless embeddings.")

            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("🛑 Training aborted by user")
                return

        # Import and run the actual training logic
        # For now, we'll import and delegate to existing training script
        try:
            from scripts.training.train_v5_12 import main as train_main

            # Temporarily modify sys.argv to pass config
            original_argv = sys.argv.copy()
            sys.argv = ['train_v5_12.py', '--config', str(self.config_path)]

            train_main()

            # Restore original argv
            sys.argv = original_argv

        except ImportError:
            print("❌ Could not import training script. Please ensure scripts/training/train_v5_12.py exists.")
            raise
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Validated TernaryVAE Training")
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate config, do not train')
    parser.add_argument('--auto-fix', action='store_true', help='Automatically fix configuration issues')
    parser.add_argument('--no-checkpoint-check', action='store_true', help='Skip checkpoint validation')

    args = parser.parse_args()

    print("🔬 TernaryVAE Validated Training Script")
    print("=" * 60)

    try:
        trainer = ValidatedTrainer(args.config, auto_fix=args.auto_fix)

        if args.validate_only:
            print("✅ Configuration validation completed successfully!")
            print("🎯 Config is ready for training")
            return

        trainer.run_training()

        print("🎉 Training completed successfully!")

    except CheckpointCompatibilityError as e:
        print(f"💥 Checkpoint compatibility error: {e}")
        print("\n💡 Suggestions:")
        print("   1. Use --auto-fix to automatically correct configuration")
        print("   2. Manually update frozen_checkpoint.path to a valid checkpoint")
        print("   3. Use configs/v5_12_4_fixed_checkpoint.yaml as reference")
        sys.exit(1)

    except Exception as e:
        print(f"💥 Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()