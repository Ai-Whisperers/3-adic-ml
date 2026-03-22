"""V7 Pre-Training Validation — Three Architectural Concerns.

Runs 40 training epochs on real data and checks:
  1. fc_mu[:, :4] pollution: does Spearman(||mu[:,:4]||, valuation) rise?
     Per-valuation r.mean() should be monotonically separated.
  2. tangent_scale shift: does it stay > 0.01? Direction diversity check.
  3. KL mean gap: mu[:, 4:] norm drift, KL loss value, conf_factor sanity.

Usage:
    python validate_v7_concerns.py
"""

import torch
import yaml
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from src.core.ternary import TERNARY
from src.models.vae import TernaryVAEV6Controllable
from src.losses.combined import CombinedLoss

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
EVAL_EVERY = 5

# ── Load config ────────────────────────────────────────────────────────────────
with open("src/presets/v7.yaml") as f:
    cfg = yaml.safe_load(f)

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
all_ops = TERNARY.all_ternary().to(DEVICE)          # (19683, 9)
all_idx = torch.arange(len(all_ops), device=DEVICE)
all_vals = TERNARY.valuation(all_idx).float()       # (19683,)

n = len(all_ops)
rng = np.random.default_rng(42)
perm = rng.permutation(n)
n_val = max(1, round(n * 0.1))
val_idx = torch.from_numpy(perm[:n_val]).to(DEVICE)
train_idx = torch.from_numpy(perm[n_val:]).to(DEVICE)

train_ds = TensorDataset(all_ops[train_idx], all_idx[train_idx])
val_ds   = TensorDataset(all_ops[val_idx],   all_idx[val_idx])
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=512, shuffle=False)

print(f"Device: {DEVICE} | Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"{'─'*80}")
print(f"{'Ep':>4} | {'Loss':>7} | {'C1 Spear':>9} | {'C1 r_sep':>8} | "
      f"{'C2 tscale':>9} | {'C2 cosim':>8} | {'C3 KL':>7} | {'C3 mu4+norm':>11}")
print(f"{'─'*80}")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += total.item()

    if epoch % EVAL_EVERY != 0:
        continue

    # ── Eval pass: collect mu, r, z_hyp, z_theta for full val set ────────────
    model.eval()
    all_mu_A, all_r_A, all_z_theta, all_logvar_A, all_z_hyp = [], [], [], [], []
    all_kl_vals = []

    with torch.no_grad():
        for x_batch, idx_batch in val_dl:
            out = model(x_batch.float())
            all_mu_A.append(out["mu_A"])
            all_r_A.append(out["r_A"])
            all_z_theta.append(out["mu_A"][:, mc["radial_dims"]:])  # mu z_theta part
            all_logvar_A.append(out["logvar_A"])
            all_z_hyp.append(out["z_A_hyp"])

            # KL computation for this batch
            kl_raw = loss_fn.kl_loss(out["mu_A"], out["logvar_A"], out["z_A_hyp"])
            all_kl_vals.append(kl_raw.item())

    mu_A     = torch.cat(all_mu_A,    dim=0)   # (N_val, 32)
    r_A      = torch.cat(all_r_A,     dim=0)   # (N_val,)
    z_theta  = torch.cat(all_z_theta, dim=0)   # (N_val, 28)
    logvar_A = torch.cat(all_logvar_A, dim=0)
    z_hyp    = torch.cat(all_z_hyp,   dim=0)   # (N_val, 28)

    val_vals = all_vals[val_idx]  # (N_val,)

    # ── Concern 1: fc_mu[:,:k] encodes valuation ──────────────────────────────
    mu_r_norm = mu_A[:, :mc["radial_dims"]].norm(dim=-1).cpu().numpy()
    val_np    = val_vals.cpu().numpy()
    c1_spear  = spearmanr(mu_r_norm, val_np).statistic  # should be negative (larger v → smaller r)

    # Per-level r.mean() — check monotonic separation
    r_np = r_A.cpu().numpy()
    r_by_level = []
    for v in range(10):
        mask = (val_vals == v)
        if mask.sum() > 0:
            r_by_level.append(r_np[mask.cpu().numpy()].mean())
        else:
            r_by_level.append(float("nan"))
    # "separated" = strictly decreasing r from v=0 to v=9
    valid = [r for r in r_by_level if not np.isnan(r)]
    c1_sep = all(valid[i] > valid[i+1] for i in range(len(valid)-1))

    # ── Concern 2: tangent_scale value + direction diversity ──────────────────
    c2_tscale = model.projections.proj_A.tangent_scale.item()

    # Within-level cosine similarity for v=0 (most samples)
    mask_v0 = (val_vals == 0)
    if mask_v0.sum() > 1:
        dirs_v0 = torch.nn.functional.normalize(z_hyp[mask_v0], dim=-1)
        # Sample 200 pairs
        n_v0 = min(200, dirs_v0.shape[0])
        idx_sample = torch.randperm(dirs_v0.shape[0])[:n_v0]
        d = dirs_v0[idx_sample]
        # Mean pairwise cosine sim (excluding diagonal)
        sim_matrix = d @ d.T
        off_diag = sim_matrix[~torch.eye(n_v0, dtype=bool, device=DEVICE)]
        c2_cosim = off_diag.mean().item()
    else:
        c2_cosim = float("nan")

    # ── Concern 3: KL value + mu[:, 4:] norm drift ────────────────────────────
    c3_kl      = np.mean(all_kl_vals)
    c3_mu4norm = z_theta.norm(dim=-1).mean().item()

    # Sanity: conf_factor on a batch
    from src.geometry import lambda_x
    conf = lambda_x(z_hyp[:8], c=1.0, keepdim=False)
    c3_cf_range = f"[{conf.min():.2f},{conf.max():.2f}]"

    # ── Print row ─────────────────────────────────────────────────────────────
    sep_mark = "✓" if c1_sep else "✗"
    ts_mark  = "✓" if c2_tscale > 0.01 else "✗"
    kl_mark  = "✓" if 0.01 < c3_kl < 200 else "✗"
    print(f"{epoch:>4} | {epoch_loss/len(train_dl):>7.3f} | "
          f"{c1_spear:>+9.3f} | {sep_mark:>8} | "
          f"{c2_tscale:>9.4f}{ts_mark} | {c2_cosim:>+8.3f} | "
          f"{c3_kl:>7.3f}{kl_mark} | {c3_mu4norm:>11.4f}  cf={c3_cf_range}")

print(f"{'─'*80}")
print()
print("LEGEND:")
print("  C1 Spear : Spearman(||mu[:,:4]||, valuation) — negative=good (high v → small r_norm)")
print("  C1 r_sep : ✓ = r[v0]>r[v1]>...>r[v9] strictly monotonic")
print("  C2 tscale: tangent_scale value; ✓ = > 0.01")
print("  C2 cosim : mean pairwise cosine sim within v=0 directions (lower=more diverse)")
print("  C3 KL    : KL loss value; ✓ = finite and > 0.01")
print("  C3 mu4+  : mean ||mu[:, radial_dims:]|| norm — watch for unbounded growth")
print("  cf       : conformal factor λ(z_hyp) range on a mini-batch")
print()

# ── Final per-level r summary ─────────────────────────────────────────────────
print("Per-level r.mean() at final epoch:")
for v in range(10):
    bar = "█" * int(r_by_level[v] * 40) if not np.isnan(r_by_level[v]) else ""
    print(f"  v={v}: {r_by_level[v]:.4f}  {bar}")
