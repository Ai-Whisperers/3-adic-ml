"""Filesystem Transfer Experiment.

Tests whether the p-adic geometry learned from ternary operations transfers
to filesystem path hierarchy — a structurally identical but superficially
unrelated domain.

Hypothesis:
    Paths sharing ancestors at depth divisible by 3 should cluster in the
    Poincaré ball at similar radii, mirroring the 3-adic valuation of depth.

Usage:
    # Synthetic tree (controlled ground truth):
    python experiments/filesystem_transfer.py --mode synthetic

    # Real filesystem:
    python experiments/filesystem_transfer.py --mode real --root /path/to/dir

    # With trained checkpoint:
    python experiments/filesystem_transfer.py --checkpoint checkpoints/best_Q.pt
"""

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np
import torch

from src.models.vae import TernaryVAEV6

# ---------------------------------------------------------------------------
# C ternary hash bridge
# ---------------------------------------------------------------------------

_LIB_PATH = Path(__file__).resolve().parents[2] / "src" / "c" / "ternary_hash.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.hash_string_to_ternary.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int8)]
_lib.hash_string_to_ternary.restype = None

_lib.ternary_index.argtypes = [ctypes.POINTER(ctypes.c_int8)]
_lib.ternary_index.restype = ctypes.c_uint32


def _hash_component(s: str) -> int:
    """Hash a single path component to a trit {-1, 0, 1} via C."""
    out = (ctypes.c_int8 * 9)()
    _lib.hash_string_to_ternary(s.encode(), out)
    # Use first trit — uniform in {-1,0,1}
    return int(out[0])


def path_to_ternary(path: str) -> list[int]:
    """Encode filesystem path hierarchy as 9-trit ternary sequence.

    Each of the first 9 path components contributes one trit via C hash.
    This preserves the hierarchical structure: component at depth d → trit[d].
    Deep paths share a common prefix in trit space, mirroring the ultrametric.
    Shallow paths (< 9 components) are zero-padded (valuation = high = near origin).
    """
    parts = Path(path).parts
    digits = [_hash_component(p) for p in parts[:9]]
    while len(digits) < 9:
        digits.append(0)  # padding = zero = high valuation
    return digits


def path_depth_valuation(path: str) -> int:
    """3-adic valuation of path depth.

    Depth d -> valuation = v_3(d+1) where v_3 counts powers of 3.
    Root (depth 0) -> highest valuation (closest to origin).
    """
    depth = len(Path(path).parts) - 1
    if depth == 0:
        return 9
    n = depth
    v = 0
    while n % 3 == 0:
        v += 1
        n //= 3
    return v


# ---------------------------------------------------------------------------
# Synthetic filesystem generator
# ---------------------------------------------------------------------------

def generate_synthetic_tree(branching: int = 3, max_depth: int = 9) -> list[str]:
    """Generate a synthetic filesystem tree with controlled 3-adic depth structure.

    Branching factor of 3 ensures depths divisible by 3 appear at natural
    intervals, creating clear valuation groupings.
    """
    paths = ["/"]
    queue = [("/", 0)]
    while queue:
        parent, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for i in range(branching):
            child = os.path.join(parent, f"d{depth}_{i}")
            paths.append(child)
            queue.append((child, depth + 1))
    return paths


# ---------------------------------------------------------------------------
# Embedding and analysis
# ---------------------------------------------------------------------------

def embed_paths(paths: list[str], model: TernaryVAEV6) -> tuple[torch.Tensor, list[int]]:
    """Run paths through model encoder, return hyperbolic embeddings and valuations."""
    model.eval()
    encodings = []
    valuations = []

    with torch.no_grad():
        for path in paths:
            seq = path_to_ternary(path)
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            out = model(x)
            z_hyp = out["z_A_hyp"]
            encodings.append(z_hyp.squeeze(0))
            valuations.append(path_depth_valuation(path))

    return torch.stack(encodings), valuations


def compute_radii(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings.norm(dim=-1)


def analyze_radii_by_valuation(radii: torch.Tensor, valuations: list[int]) -> dict:
    """Group radii by 3-adic valuation level and compute statistics."""
    results = {}
    vals = torch.tensor(valuations)
    for v in sorted(set(valuations)):
        mask = vals == v
        r = radii[mask]
        results[v] = {
            "count": mask.sum().item(),
            "mean_radius": r.mean().item(),
            "std_radius": r.std().item() if len(r) > 1 else 0.0,
        }
    return results


def compute_spearman_correlation(radii: torch.Tensor, valuations: list[int]) -> float:
    """Spearman correlation between radius and valuation.

    Expected: negative correlation — higher valuation → smaller radius (closer to origin).
    """
    from scipy.stats import spearmanr
    r = radii.numpy()
    v = np.array(valuations)
    corr, pval = spearmanr(r, v)
    return float(corr), float(pval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filesystem transfer experiment")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--root", type=str, default="/", help="Root for real filesystem scan")
    parser.add_argument("--max-paths", type=int, default=500, help="Max paths to sample")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--branching", type=int, default=3, help="Branching factor for synthetic tree")
    args = parser.parse_args()

    print("=" * 60)
    print("P-ADIC FILESYSTEM TRANSFER EXPERIMENT")
    print("=" * 60)

    # Load model
    model = TernaryVAEV6()
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    else:
        print("WARNING: No checkpoint — using random weights. Results show structural capacity only.")

    # Generate paths
    if args.mode == "synthetic":
        print(f"\nGenerating synthetic tree (branching={args.branching}, max_depth=9)...")
        paths = generate_synthetic_tree(branching=args.branching)
        paths = paths[:args.max_paths]
        print(f"Generated {len(paths)} paths.")
    else:
        print(f"\nScanning real filesystem: {args.root}")
        paths = []
        for root, _dirs, _files in os.walk(args.root):
            paths.append(root)
            if len(paths) >= args.max_paths:
                break
        print(f"Collected {len(paths)} paths.")

    # Embed
    print("\nEmbedding paths...")
    embeddings, valuations = embed_paths(paths, model)
    radii = compute_radii(embeddings)

    # Analyze
    print("\nRadii by 3-adic valuation level:")
    print(f"{'Valuation':>10} {'Count':>8} {'Mean Radius':>12} {'Std':>8}")
    print("-" * 42)
    stats = analyze_radii_by_valuation(radii, valuations)
    for v in sorted(stats.keys(), reverse=True):
        s = stats[v]
        print(f"{v:>10} {s['count']:>8} {s['mean_radius']:>12.4f} {s['std_radius']:>8.4f}")

    # Correlation
    print("\nSpearman correlation (radius vs valuation):")
    try:
        corr, pval = compute_spearman_correlation(radii, valuations)
        print(f"  corr = {corr:.4f}  (expected: negative)")
        print(f"  pval = {pval:.4e}")
        if corr < -0.3 and pval < 0.05:
            print("  RESULT: TRANSFER CONFIRMED — geometry preserves 3-adic hierarchy")
        elif corr < 0:
            print("  RESULT: WEAK SIGNAL — correct direction but not significant")
        else:
            print("  RESULT: NO TRANSFER — geometry does not preserve hierarchy")
    except ImportError:
        print("  scipy not available — install with: pip install scipy")
        print("  Raw: mean radius per level printed above.")

    print("\nDone.")


if __name__ == "__main__":
    main()
