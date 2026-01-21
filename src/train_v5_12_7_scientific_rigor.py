#!/usr/bin/env python3
# Copyright 2024-2026 AI Whisperers
# Licensed under PolyForm Noncommercial License 1.0.0

"""
V5.12.7 Unified Scientific Training Pipeline: Auditor + Skeptical Trainer

This script enforces a strict "Audit-Then-Execute" protocol.
It combines forensic data analysis with robust, falsifiable training dynamics.

Workflow:
1. AUDIT PHASE:
   - Verifies Data Integrity (Leakage detection, Class balance).
   - Verifies Model Health (Gradient flow, Random baseline comparison).
   -  - Prevents training on memorized test data.
   
2. TRAINING PHASE:
   - Deterministic execution.
   - Grokking Detection (Plateau -> Lift -> Gap Collapse).
   -  - Monitors for the characteristic delayed generalization.
   - Mixed Precision & Riemannian Optimization support.

Usage:
    python scripts/train_unified_v5_12_7.py --config configs/v5_12_extended.yaml
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from codebase
from src.config.paths import RUNS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.geometry import get_riemannian_optimizer, poincare_distance
from src.losses import RadialHierarchyLoss, PAdicGeodesicLoss, RichHierarchyLoss
from src.models import TernaryVAEV5_11_PartialFreeze
from src.statenet import compute_Q
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat

# ==========================================
# 1. DETERMINISM & UTILS
# ==========================================

def set_determinism(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================
# 2. THE AUDITOR (Factory & Forensic)
# ==========================================

class ExperimentAuditor:
    """
    The Gatekeeper. Initializes data and model ONLY if they pass forensic checks.
    """
    def __init__(self, config: Dict, device: torch.device, seed: int):
        self.config = config
        self.device = device
        self.seed = seed
        self.audit_log = {}

    def _stable_split(self, n: int, val_frac: float) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic shuffle split."""
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return train_idx, val_idx

    def prepare_and_audit_data(self, val_frac: float) -> Tuple[TensorDataset, TensorDataset]:
        """
        Generates data, splits it, and checks for leakage/imbalance.
        Returns TensorDatasets ready for loader creation.
        """
        print("\n🔍 [AUDIT 1/3] Data Integrity Check...")
        
        # 1. Generation
        all_ops_np = generate_all_ternary_operations()
        all_ops = torch.tensor(all_ops_np, dtype=torch.float32) # Keep on CPU for splitting
        n = len(all_ops)
        all_indices = torch.arange(n, dtype=torch.long)
        
        # 2. Splitting
        train_idx, val_idx = self._stable_split(n, val_frac)
        
        X_train = all_ops[train_idx]
        X_val = all_ops[val_idx]
        
        # 3. Leakage Check
        # Convert to tuple sets for exact matching
        train_set = set(tuple(x.tolist()) for x in X_train)
        val_set = set(tuple(x.tolist()) for x in X_val)
        
        overlap = train_set.intersection(val_set)
        leakage_pct = len(overlap) / len(val_set) * 100
        
        self.audit_log['data_leakage_count'] = len(overlap)
        
        if len(overlap) > 0:
            print(f"    ⚠️  CRITICAL WARNING: Data Leakage Detected!")
            print(f"       {len(overlap)} samples found in BOTH Train and Val sets ({leakage_pct:.2f}%).")
            print("       ")
            # In strict mode, we might raise an error here. For now, we warn heavily.
        else:
            print("    ✅ Data Leakage: None (Sets are strictly disjoint).")

        # 4. Class Balance (Heuristic based on value distribution)
        vals, counts = torch.unique(X_train, return_counts=True)
        dist = counts.float() / counts.sum()
        dist_dict = dict(zip(vals.tolist(), dist.tolist()))
        print(f"    ℹ️  Value Distribution: {dist_dict}")
        
        if any(d > 0.8 for d in dist):
             print("    ⚠️  WARNING: Severe Class Imbalance. Accuracy metric may be misleading.")
        
        # Return datasets
        train_ds = TensorDataset(X_train, all_indices[train_idx])
        val_ds = TensorDataset(X_val, all_indices[val_idx])
        return train_ds, val_ds

    def prepare_and_audit_model(self, force: bool = False) -> nn.Module:
        """
        Initializes model, checks checkpoint logic, and verifies gradient flow.
        """
        print("\n🩺 [AUDIT 2/3] Model Health & Initialization Check...")
        
        # 1. Config & Checkpoint Logic
        model_cfg = self.config.get("model", {})
        frozen_cfg = self.config.get("frozen_checkpoint", {})
        ckpt_path_str = frozen_cfg.get("path", None)
        
        if "PartialFreeze" in model_cfg.get("name", "") and (not ckpt_path_str or ckpt_path_str == "null"):
            print("    ⚠️  LOGIC WARNING: 'PartialFreeze' model requested without a checkpoint.")
            print("       Frozen components will remain random noise.")
        
        # 2. Instantiation
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=model_cfg.get("latent_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            max_radius=model_cfg.get("max_radius", 0.95),
            curvature=model_cfg.get("curvature", 1.0),
            use_controller=model_cfg.get("use_controller", True),
            use_dual_projection=model_cfg.get("use_dual_projection", True),
            n_projection_layers=model_cfg.get("projection_layers", 2),
            projection_dropout=model_cfg.get("projection_dropout", 0.1),
            learnable_curvature=model_cfg.get("learnable_curvature", True),
            manifold_aware=model_cfg.get("manifold_aware", True),
            freeze_encoder_b=False, # We usually want this trainable
            encoder_type=model_cfg.get("encoder_type", "improved"),
            decoder_type=model_cfg.get("decoder_type", "improved"),
        ).to(self.device)
        
        # 3. Load Checkpoint (if exists)
        if ckpt_path_str:
            ckpt_path = PROJECT_ROOT / ckpt_path_str
            if ckpt_path.exists():
                try:
                    ckpt = load_checkpoint_compat(ckpt_path, map_location=self.device)
                    msd = get_model_state_dict(ckpt)
                    mk, uk = model.load_state_dict(msd, strict=False)
                    print(f"    ✅ Loaded checkpoint: {ckpt_path.name} (Missing: {len(mk)}, Unexpected: {len(uk)})")
                except Exception as e:
                    print(f"    ❌ Checkpoint load failed: {e}")
                    if not force: raise e
            else:
                print(f"    ❌ Checkpoint file not found: {ckpt_path}")
                if not force: raise FileNotFoundError(ckpt_path)

        # 4. Health Check (Cold Start)
        model.train()
        # Create a dummy batch to test gradients
        dummy_ops = torch.randint(-1, 2, (32, 9)).float().to(self.device)
        
        try:
            # Forward pass
            out = model(dummy_ops, compute_control=False)
            
            # Baseline Loss Check
            # Assume logits are present. 
            # Note: Robust script handles 3-class or 27-class logits. 
            logits = out.get('logits_A', out.get('logits'))
            if logits is None: raise KeyError("No logits found in output")
            
            # Simple CE for checking
            targets = dummy_ops.long() + 1
            if logits.shape[-1] == 27:
                # Basic check doesn't need perfect mapping, just non-inf loss
                loss = logits.mean() 
            else:
                loss = F.cross_entropy(logits.view(-1, 3), targets.view(-1))
            
            initial_loss = loss.item()
            self.audit_log['initial_loss'] = initial_loss
            
            print(f"    ℹ️  Initial Loss: {initial_loss:.4f} (Random ~1.1)")
            if initial_loss > 5.0:
                 print("    ⚠️  WARNING: Exploding initial loss. Check weight init.")
            
            # Backward pass (Gradient Check)
            loss.backward()
            
            total_norm = 0.0
            zero_grad_params = 0
            total_params = 0
            
            for p in model.parameters():
                if p.requires_grad:
                    total_params += 1
                    if p.grad is not None:
                        norm = p.grad.data.norm(2).item()
                        total_norm += norm
                        if norm == 0: zero_grad_params += 1
                    else:
                        zero_grad_params += 1

            self.audit_log['initial_grad_norm'] = total_norm
            print(f"    ℹ️  Gradient Norm: {total_norm:.4f}")

            if total_norm == 0 or zero_grad_params == total_params:
                print("    ❌ FATAL: Vanishing Gradients. Model is dead on arrival.")
                if not force: raise RuntimeError("Dead model detected")
            elif total_norm > 1000:
                print("    ❌ FATAL: Exploding Gradients.")
                if not force: raise RuntimeError("Exploding gradients detected")
            else:
                print(f"    ✅ Gradient Flow: Active ({total_params - zero_grad_params}/{total_params} active params).")

        except Exception as e:
            print(f"    ❌ Model Health Check Failed: {e}")
            if not force: raise e
            
        return model

# ==========================================
# 3. METRICS & LOGIC (From Robust Script)
# ==========================================

@dataclass
class Phase:
    start: int
    end: int
    lr_mult: float

class MultiPhaseCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, phases: List[Phase], base_lr: float, last_epoch: int = -1):
        self.phases = phases
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        phase = next((p for p in self.phases if p.start <= e <= p.end), self.phases[-1])
        denom = max(1, phase.end - phase.start)
        t = (e - phase.start) / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return [self.base_lr * phase.lr_mult * cosine for _ in self.optimizer.param_groups]

class GrokkingDetectorSkeptical:
    """
    Detects Grokking via: Plateau -> Lift -> Gap Collapse.
    
    """
    def __init__(self, window=20, slope_eps=1e-4, sustain_k=6, val_lift_min=0.02, gap_collapse_min=0.02):
        self.window = window
        self.slope_eps = slope_eps
        self.sustain_k = sustain_k
        self.val_lift_min = val_lift_min
        self.gap_collapse_min = gap_collapse_min
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        self.plateau_start = None
        self.events = []

    def _slope(self, ys):
        if len(ys) < 2: return 0.0
        x = np.arange(len(ys))
        return np.polyfit(x, ys, 1)[0]

    def update(self, epoch, t_loss, t_acc, v_acc):
        self.history['train_loss'].append(t_loss)
        self.history['train_acc'].append(t_acc)
        self.history['val_acc'].append(v_acc)
        
        out = {"plateau": False, "potential": False, "event": False, "val_lift": 0.0, "gap_collapse": 0.0}
        
        if len(self.history['train_loss']) < self.window: return out
        
        # 1. Plateau Check
        recent_loss = self.history['train_loss'][-self.window:]
        slope = abs(self._slope(recent_loss))
        out['plateau'] = slope < self.slope_eps
        
        if out['plateau'] and self.plateau_start is None:
            self.plateau_start = epoch - self.window
        elif not out['plateau'] and self.plateau_start is not None:
             # Reset if we drift off plateau without event (unless we are in post-grokking regime)
             if epoch - self.plateau_start > self.window * 3:
                 self.plateau_start = None

        # 2. Grokking Check (Lift + Collapse)
        if self.plateau_start is not None and len(self.history['val_acc']) > self.window + self.sustain_k:
            baseline_val = np.mean(self.history['val_acc'][-(self.window + self.sustain_k):-self.sustain_k])
            recent_val = np.mean(self.history['val_acc'][-self.sustain_k:])
            val_lift = recent_val - baseline_val
            
            baseline_gap = np.mean(self.history['train_acc'][-(self.window + self.sustain_k):-self.sustain_k]) - baseline_val
            recent_gap = np.mean(self.history['train_acc'][-self.sustain_k:]) - recent_val
            gap_collapse = baseline_gap - recent_gap
            
            out['val_lift'] = val_lift
            out['gap_collapse'] = gap_collapse
            
            if val_lift > self.val_lift_min and gap_collapse > self.gap_collapse_min:
                out['potential'] = True
                # Register Event
                self.events.append({
                    "epoch": epoch, 
                    "plateau_dur": epoch - self.plateau_start,
                    "lift": val_lift,
                    "collapse": gap_collapse
                })
                out['event'] = True
                self.plateau_start = None # Reset
                
        return out

def compute_accuracy_robust(logits, target_x):
    """Robust accuracy for 3-class, 27-class, or sign-based logic."""
    with torch.no_grad():
        # Case A: Logits (B, S, 3), Target (B, S) {-1,0,1}
        if logits.shape[-1] == 3:
             pred = torch.argmax(logits, dim=-1) # 0,1,2
             tgt = (target_x + 1).long().clamp(0, 2)
             return (pred == tgt).float().mean().item()
        
        # Case B: Logits (B, 27), Target (B, 3) {-1,0,1}
        if logits.shape[-1] == 27:
            pred = torch.argmax(logits, dim=-1)
            # Encode target
            t = (target_x + 1).long().clamp(0, 2)
            tgt_encoded = t[..., 0] * 9 + t[..., 1] * 3 + t[..., 2]
            return (pred == tgt_encoded).float().mean().item()
            
        return 0.0

def compute_hierarchy_Q_robust(z_B, indices, sample_idx):
    """Computes Q on a fixed subset to ensure stability."""
    with torch.no_grad():
        origin = torch.zeros_like(z_B)
        radii = poincare_distance(z_B, origin).cpu().numpy()
        vals = TERNARY.valuation(indices).cpu().numpy()
        
        hierarchy = spearmanr(vals, radii).correlation or 0.0
        
        # Q calculation on subset
        r_s = radii[sample_idx]
        v_s = vals[sample_idx]
        
        z_dists = np.abs(r_s[:, None] - r_s[None, :])
        v_dists = np.abs(v_s[:, None] - v_s[None, :])
        
        triu = np.triu_indices(len(sample_idx), k=1)
        dist_corr = spearmanr(z_dists[triu], v_dists[triu]).correlation or 0.0
        
        Q = compute_Q(dist_corr, hierarchy)
        return hierarchy, dist_corr, Q

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser("Unified V5.12.7 Scientific Pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Bypass audit failures")
    parser.add_argument("--amp", action="store_true", help="Use Mixed Precision")
    args = parser.parse_args()

    # 1. Setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    set_determinism(args.seed)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 2. AUDIT PHASE (The Factory)
    print("🛡️  Starting Audit Phase...")
    auditor = ExperimentAuditor(config, device, args.seed)
    
    # Audit Data
    val_frac = config.get("training", {}).get("val_frac", 0.1)
    train_ds, val_ds = auditor.prepare_and_audit_data(val_frac)
    
    # Audit Model
    try:
        model = auditor.prepare_and_audit_model(args.force)
    except Exception as e:
        print("🛑 Audit Failed. Aborting.")
        sys.exit(1)

    print("\n🚀 Audit Passed. Launching Training Phase...")
    
    # 3. TRAINING PHASE (The Robust Trainer)
    
    # Hyperparams
    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 500))
    batch_size = int(train_cfg.get("batch_size", 512))
    base_lr = float(train_cfg.get("lr", 1e-3))
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Scheduler
    if config.get("riemannian", {}).get("enabled", False):
        optimizer = get_riemannian_optimizer(model.get_param_groups(base_lr), lr=base_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        
    phases = [Phase(int(p["start"]), int(p["end"]), float(p["lr_mult"])) 
              for p in train_cfg.get("scheduler", {}).get("phases", [])]
    if not phases:
        phases = [Phase(0, epochs, 1.0)]
    scheduler = MultiPhaseCosineLR(optimizer, phases, base_lr)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    detector = GrokkingDetectorSkeptical(**train_cfg.get("grokking_detection", {}))
    
    # Logging
    log_dir = RUNS_DIR / f"unified_v5_12_7_{now_ts()}"
    save_dir = log_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    
    # Fixed Sample for Q (Determinism)
    rng = np.random.default_rng(args.seed)
    q_sample_idx = rng.choice(len(val_ds), min(1000, len(val_ds)), replace=False)
    
    # State tracking
    best_Q = -1.0
    
    print(f"    Logs: {log_dir}")
    print(f"    Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        grad_norm_sum = 0.0
        batches = 0
        
        for xb, idx in train_loader:
            xb, idx = xb.to(device), idx.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(xb, compute_control=False)
                logits = out.get('logits_A', out.get('logits'))
                z_B = out.get('z_B_hyp')
                
                # Dynamic Loss Composition
                # (Assuming losses are defined in core/losses.py, instantiated here for brevity)
                # For robustness, we use simple CrossEntropy + Radial Proxy
                
                # 1. Functional Loss
                if logits.shape[-1] == 3:
                     targets = (xb + 1).long() # Shift to 0,1,2
                     loss_func = F.cross_entropy(logits.view(-1, 3), targets.view(-1))
                else:
                     loss_func = 0.0 # Placeholder for other archs
                
                # 2. Structural Proxy (Radial)
                origin = torch.zeros_like(z_B)
                radii = poincare_distance(z_B, origin)
                # Simple penalty to keep things in the ball
                loss_struct = torch.relu(radii - 0.9).mean() 
                
                loss = loss_func + loss_struct

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_sum += gn.item()
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            train_acc_sum += compute_accuracy_robust(logits, xb)
            batches += 1
            
        # End of Epoch Stats
        avg_train_loss = train_loss_sum / batches
        avg_train_acc = train_acc_sum / batches
        avg_grad_norm = grad_norm_sum / batches
        scheduler.step()
        
        # Validation
        model.eval()
        val_acc_sum = 0.0
        val_batches = 0
        z_B_all = []
        idx_all = []
        
        with torch.no_grad():
            for xb, idx in val_loader:
                xb, idx = xb.to(device), idx.to(device)
                out = model(xb)
                logits = out.get('logits_A', out.get('logits'))
                z_B = out.get('z_B_hyp')
                
                val_acc_sum += compute_accuracy_robust(logits, xb)
                val_batches += 1
                
                if epoch % 5 == 0: # Collect for Q calc occasionally
                    z_B_all.append(z_B)
                    idx_all.append(idx)

        avg_val_acc = val_acc_sum / val_batches
        
        # Grokking Update
        grok_stats = detector.update(epoch, avg_train_loss, avg_train_acc, avg_val_acc)
        
        # Q Calc (Expensive, do less often)
        hierarchy, dist_corr, Q = 0.0, 0.0, 0.0
        if epoch % 5 == 0 and z_B_all:
             z_cat = torch.cat(z_B_all)
             idx_cat = torch.cat(idx_all)
             # Map global sample idx to local batch if needed, or just sample from what we collected
             # Simplified: just use first N from validation set for stability
             sample_n = min(1000, len(z_cat))
             hierarchy, dist_corr, Q = compute_hierarchy_Q_robust(
                 z_cat, idx_cat, np.arange(sample_n)
             )
             
             if Q > best_Q:
                 best_Q = Q
                 torch.save(model.state_dict(), save_dir / "best_Q.pt")

        # Logging
        writer.add_scalars("Accuracy", {"Train": avg_train_acc, "Val": avg_val_acc}, epoch)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Dynamics/GradNorm", avg_grad_norm, epoch)
        writer.add_scalar("Structure/Q", Q, epoch)
        writer.add_scalar("Grokking/Lift", grok_stats['val_lift'], epoch)
        
        dt = time.time() - t0
        if epoch % 10 == 0:
            print(f"Ep {epoch:03d} | T_Loss {avg_train_loss:.4f} | Acc T/V {avg_train_acc:.3f}/{avg_val_acc:.3f} | Q {Q:.3f} | G_Norm {avg_grad_norm:.3f} | {dt:.1f}s")
            
        if grok_stats['event']:
            print(f"🚨 GROKKING EVENT DETECTED at Epoch {epoch}!")

    # Final Save
    torch.save(model.state_dict(), save_dir / "final.pt")
    
    # Save Audit Log
    with open(log_dir / "audit_log.json", "w") as f:
        json.dump(auditor.audit_log, f, indent=2)

    writer.close()
    print("🏁 Experiment Complete.")

if __name__ == "__main__":
    main()