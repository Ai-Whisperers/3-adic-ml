#!/usr/bin/env python3

"""V7 Pre-Training Validation — All Six Items (A–F + original Concerns 1–3).

Items covered:
  A  Re-run Concerns 1–3 with variance_only:false fix (new KL config)
  B  Actual Spearman hierarchy, dist_corr, Q, reconstruction accuracy
  C  Full-dataset hierarchy (includes rare v=8/v=9, not just random val split)
  D  Decoder reliance on z_r vs z_θ (gradient ratio per epoch)
  E  StateNet plateau detection simulation (would encoder_b freeze too early?)
  F  Within-level r scatter vs V6 baseline (scatter_weight=0.8 calibration)

Usage:
    python scripts/validation/validate_v7_concerns.py
"""

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.core.ternary import TERNARY
from src.losses.combined import CombinedLoss
from src.models.vae import TernaryVAEV6Controllable

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
EVAL_EVERY = 5
RADIAL_DIMS = 4

# plateau detection params (from statenet config)
PLATEAU_THRESHOLD = 0.0005
PLATEAU_PATIENCE = 10

# V6 reference scatter (v=0 radial std after WLC, from audit)
V6_SCATTER_V0_REF = 0.063

# ── Load config ────────────────────────────────────────────────────────────────
with open("src/presets/v7.yaml") as f:
    cfg = yaml.safe_load(f)

print(f"KL variance_only = {cfg['loss']['hyperbolic_kl']['variance_only']}  "
      f"(should be False after fix)")

mc = cfg["model"]
model = TernaryVAEV6Controllable(
    latent_dim=mc["latent_dim"],
    hidden_dim=mc["hidden_dim"],
    max_radius=mc["max_radius"],
    factored=mc["factored"],
    radial_dims=mc["radial_dims"],
    n_projection_layers=mc["projection_layers"],
    projection_dropout=mc["projection_dropout"],
    init_identity=mc["init_identity"],
    tangent_scale_init=mc["tangent_scale"],
    learnable_curvature=mc["learnable_curvature"],
    encoder_a_trainable=True,
    encoder_b_trainable=True,
    projections_trainable=True,
).to(DEVICE)

loss_fn = CombinedLoss(cfg["loss"]).to(DEVICE)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=cfg["training"]["lr"],
    weight_decay=cfg["training"]["weight_decay"],
)

# ── Build dataset ──────────────────────────────────────────────────────────────
all_ops  = TERNARY.all_ternary().to(DEVICE)        # (19683, 9)
all_idx  = torch.arange(len(all_ops), device=DEVICE)
all_vals = TERNARY.valuation(all_idx).float()      # (19683,)

n = len(all_ops)
rng = np.random.default_rng(42)
perm = rng.permutation(n)
n_val = max(1, round(n * 0.1))
val_mask   = torch.from_numpy(perm[:n_val]).to(DEVICE)
train_mask = torch.from_numpy(perm[n_val:]).to(DEVICE)

train_ds = TensorDataset(all_ops[train_mask], all_idx[train_mask])
val_ds   = TensorDataset(all_ops[val_mask],   all_idx[val_mask])
full_ds  = TensorDataset(all_ops, all_idx)           # Item C: all 19683 samples
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=512, shuffle=False)
full_dl  = DataLoader(full_ds,  batch_size=512, shuffle=False)

# Report val set valuation coverage (Item C)
val_vals_np = all_vals[val_mask].cpu().numpy()
print(f"\nVal set size: {len(val_ds)} | "
      f"v=8 count: {(val_vals_np == 8).sum()} | "
      f"v=9 count: {(val_vals_np == 9).sum()}")
print(f"{'─'*110}")

# Header
print(f"{'Ep':>4} | {'Loss':>6} | "
      f"{'A:Spear4':>9} | {'A:rsep':>6} | {'A:mu4n':>6} | "
      f"{'B:hier':>6} | {'B:dco':>6} | {'B:Q':>5} | {'B:acc':>5} | "
      f"{'C:hier_full':>11} | "
      f"{'D:zr/zt':>7} | "
      f"{'E:plat':>6} | "
      f"{'F:sc_v0':>7} | "
      f"{'C2:tsc':>6} | {'C3:KL':>6} | {'C3:mu+n':>7}")
print(f"{'─'*110}")

# ── Plateau detection state (Item E) ──────────────────────────────────────────
hier_history = []
plateau_fired_at = None

def check_plateau(hier_history, threshold, patience):
    """Return epoch index where plateau would fire, else None."""
    if len(hier_history) < patience:
        return None
    window = hier_history[-patience:]
    improvements = [abs(window[i+1] - window[i]) for i in range(len(window)-1)]
    if all(imp < threshold for imp in improvements):
        return True
    return False

# ── Training + Diagnostic Loop ─────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for x_batch, idx_batch in train_dl:
        out = model(x_batch.float())
        la = loss_fn(out["z_A_hyp"], idx_batch, out["logits_A"], x_batch,
                     epoch=epoch, mu=out["mu_A"], logvar=out["logvar_A"])
        lb = loss_fn(out["z_B_hyp"], idx_batch, out["logits_B"], x_batch,
                     epoch=epoch, mu=out["mu_B"], logvar=out["logvar_B"])
        total = la["total"] + lb["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg["training"]["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += total.item()

    if epoch % EVAL_EVERY != 0:
        continue

    model.eval()

    # ── Eval pass 1: val set ──────────────────────────────────────────────────
    mu_list, r_list, logvar_list, z_hyp_list = [], [], [], []
    logits_list, targets_list, val_idx_list = [], [], []

    with torch.no_grad():
        for x_batch, idx_batch in val_dl:
            out = model(x_batch.float())
            mu_list.append(out["mu_A"])
            r_list.append(out["r_A"])
            logvar_list.append(out["logvar_A"])
            z_hyp_list.append(out["z_A_hyp"])
            logits_list.append(out["logits_A"])
            targets_list.append(x_batch)
            val_idx_list.append(idx_batch)

    mu_A    = torch.cat(mu_list,    dim=0)
    r_A     = torch.cat(r_list,     dim=0)
    z_hyp_v = torch.cat(z_hyp_list, dim=0)
    logits_v = torch.cat(logits_list, dim=0)
    targets_v = torch.cat(targets_list, dim=0)
    val_idx_v = torch.cat(val_idx_list, dim=0)
    val_v_v   = all_vals[val_idx_v]

    r_np  = r_A.cpu().numpy()
    v_np  = val_v_v.cpu().numpy()
    mu_np = mu_A.cpu().numpy()

    # ── Item A: Concern 1 re-check (with variance_only:false) ────────────────
    mu4_norm = np.linalg.norm(mu_np[:, :RADIAL_DIMS], axis=1)
    a_spear  = spearmanr(mu4_norm, v_np).statistic       # negative = good
    r_by_v   = [r_np[v_np == v].mean() if (v_np == v).sum() > 0 else np.nan
                for v in range(10)]
    valid_r  = [r for r in r_by_v if not np.isnan(r)]
    a_rsep   = "✓" if all(valid_r[i] > valid_r[i+1]
                          for i in range(len(valid_r)-1)) else "✗"
    a_mu4n   = np.linalg.norm(mu_np[:, :RADIAL_DIMS], axis=1).mean()

    # ── Item B: True hierarchy, dist_corr, Q, accuracy ───────────────────────
    b_hier = float(-spearmanr(v_np, r_np).statistic)

    # dist_corr: Spearman(|r_i - r_j|, |v_i - v_j|) on 1000 random pairs
    N = len(r_np)
    rng2 = np.random.default_rng(0)
    i_idx = rng2.integers(0, N, 1000)
    j_idx = rng2.integers(0, N, 1000)
    dr = np.abs(r_np[i_idx] - r_np[j_idx])
    dv = np.abs(v_np[i_idx] - v_np[j_idx])
    same = dr == 0
    if same.all():
        b_dco = 0.0
    else:
        mask = dv > 0
        b_dco = float(spearmanr(dr[mask], dv[mask]).statistic) if mask.sum() > 10 else 0.0

    b_Q = b_dco + 1.5 * b_hier

    # Accuracy
    targs_shifted = (targets_v + 1).long().clamp(0, 2)
    preds = logits_v.view(-1, 9, 3).argmax(dim=-1)
    b_acc = (preds == targs_shifted).float().mean().item()

    # ── Item C: Full-dataset hierarchy (includes v=8/v=9) ────────────────────
    r_full_list, v_full_list = [], []
    with torch.no_grad():
        for x_batch, idx_batch in full_dl:
            out = model(x_batch.float())
            r_full_list.append(out["r_A"].cpu())
            v_full_list.append(all_vals[idx_batch].cpu())
    r_full = torch.cat(r_full_list).numpy()
    v_full = torch.cat(v_full_list).numpy()
    c_hier_full = float(-spearmanr(v_full, r_full).statistic)

    # ── Item D: Decoder reliance on z_r vs z_θ ───────────────────────────────
    # Use a fresh batch; compute recon grad w.r.t. z_tangent parts
    x_probe = all_ops[val_mask[:64]]
    z_t = model.head_A(x_probe.double())[0].detach().requires_grad_(True)
    logits_probe = model.decoder_A(z_t)
    tgt_shifted = (x_probe + 1).long().clamp(0, 2)
    recon = F.cross_entropy(logits_probe.view(-1, 9, 3).permute(0, 2, 1),
                            tgt_shifted)
    recon.backward()
    g = z_t.grad                                         # (64, 32)
    d_zr   = g[:, :RADIAL_DIMS].norm(dim=-1).mean().item()
    d_zth  = g[:, RADIAL_DIMS:].norm(dim=-1).mean().item()
    d_ratio = d_zr / (d_zth + 1e-10)                    # > 1.0 = decoder leaning on z_r

    # ── Item E: Plateau detection simulation ─────────────────────────────────
    hier_history.append(b_hier)
    if plateau_fired_at is None and check_plateau(hier_history,
                                                   PLATEAU_THRESHOLD,
                                                   PLATEAU_PATIENCE):
        plateau_fired_at = epoch
    e_flag = f"ep{plateau_fired_at}" if plateau_fired_at else "ok"

    # ── Item F: Within-level scatter of r (vs V6 baseline) ───────────────────
    v0_mask = v_np == 0
    f_sc_v0 = r_np[v0_mask].std() if v0_mask.sum() > 1 else float("nan")
    f_flag  = "ok" if f_sc_v0 < 0.02 else ("hi" if f_sc_v0 < V6_SCATTER_V0_REF else "V6")

    # ── Concern 2: tangent_scale (effective = exp(log_tangent_scale)) ────────
    c2_tsc = model.projections.proj_A.log_tangent_scale.exp().item()

    # ── Concern 3: KL + mu[:,4:] norm ────────────────────────────────────────
    with torch.no_grad():
        c3_kl = loss_fn.kl_loss(mu_A, torch.cat(logvar_list, dim=0), z_hyp_v).item() if loss_fn.kl_loss is not None else 0.0
    c3_mun = np.linalg.norm(mu_np[:, RADIAL_DIMS:], axis=1).mean()

    # ── Print row ─────────────────────────────────────────────────────────────
    print(f"{epoch:>4} | {epoch_loss/len(train_dl):>6.3f} | "
          f"{a_spear:>+9.3f} | {a_rsep:>6} | {a_mu4n:>6.3f} | "
          f"{b_hier:>6.3f} | {b_dco:>6.3f} | {b_Q:>5.3f} | {b_acc:>5.3f} | "
          f"{c_hier_full:>11.3f} | "
          f"{d_ratio:>7.3f} | "
          f"{e_flag:>6} | "
          f"{f_sc_v0:>6.4f}{f_flag:1} | "
          f"{c2_tsc:>6.4f} | {c3_kl:>6.2f} | {c3_mun:>7.4f}")

print(f"{'─'*110}")
print()
print("LEGEND:")
print("  A:Spear4  Spearman(||mu[:,:4]||, valuation) — negative=good (concern 1 + variance_only fix)")
print("  A:rsep    ✓ = r[v0]>…>r[v7] monotonic  |  A:mu4n = mean ||mu[:,:4]|| norm")
print("  B:hier    -Spearman(valuation, r)  |  B:dco = dist_corr  |  B:Q = dco + 1.5*hier")
print("  B:acc     per-digit reconstruction accuracy on val set")
print("  C:hier_full  hierarchy on ALL 19683 samples (includes v=8/v=9)")
print("  D:zr/zt   reconstruction grad norm ratio z_r/z_θ — >1.0 = decoder leaning on z_r")
print("  E:plat    'ok' = plateau not triggered | 'epN' = would freeze encoder_b at epoch N")
print("  F:sc_v0   within-v=0 r std | 'ok'<0.02 | 'hi'=0.02–0.063 | 'V6'≥0.063")
print("  C2:tsc    tangent_scale value  |  C3:KL = KL loss  |  C3:mu+n = ||mu[:,4:]|| norm")
print()

# ── Final per-level r summary ─────────────────────────────────────────────────
print("Per-level r.mean() at final epoch (val set + full dataset):")
for v in range(10):
    rv_val  = r_by_v[v]
    rv_full = r_full[v_full == v].mean() if (v_full == v).sum() > 0 else float("nan")
    bar = "█" * int(rv_val * 40) if not np.isnan(rv_val) else "(no val samples)"
    print(f"  v={v}: val={rv_val:.4f}  full={rv_full:.4f}  {bar}")

print()
if plateau_fired_at:
    print(f"  ⚠ Item E: StateNet would freeze encoder_b at epoch {plateau_fired_at} "
          f"— consider relaxing plateau_threshold or plateau_patience in v7.yaml")
else:
    print(f"  ✓ Item E: No early plateau firing in {EPOCHS} epochs")

print(f"\n  Item D: final decoder grad ratio z_r/z_θ = {d_ratio:.4f} "
      f"({'⚠ decoder leaning on z_r' if d_ratio > 0.5 else '✓ decoder using z_θ as expected'})")
print(f"  Item F: v=0 scatter = {f_sc_v0:.4f} "
      f"({'✓ near-zero, scatter_weight=0.8 may be excessive' if f_sc_v0 < 0.02 else 'still elevated'})")
print(f"  Item C: full-dataset hier = {c_hier_full:.4f} vs val-set hier = {b_hier:.4f} "
      f"(delta = {c_hier_full - b_hier:+.4f})")
