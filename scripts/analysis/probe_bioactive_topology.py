import torch
import numpy as np
import gudhi
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vae import TernaryVAEV6Controllable
from src.core import TERNARY
import yaml

def get_hotspot_embeddings(model, device):
    # Load known bioactive peptides
    peptides = ["WTLTPLTPA", "SVAGRAQGM", "CACGGV", "VTYM", "AVLGSSEGV", "ISLSEQQLV"]
    
    # Simple hydropathy encoding
    AA_MAP = {'D': -1, 'E': -1, 'N': -1, 'Q': -1, 'K': -1, 'R': -1, 'G': 0, 'S': 0, 'T': 0, 'Y': 0, 'P': 0, 'H': 0, 'V': 1, 'L': 1, 'I': 1, 'M': 1, 'F': 1, 'W': 1, 'C': 1, 'A': 1}
    
    # Initialize positional weights if model has them
    # Note: Confirmed model does not use pos_weights in current checkpoint
    
    embs = []
    with torch.no_grad():
        for seq in peptides:
            # Pad or truncate to 9
            seq_padded = seq[:9].ljust(9, 'G') 
            digits = [AA_MAP.get(aa.upper(), 0) for aa in seq_padded]
            x = torch.tensor(digits, dtype=torch.float64).unsqueeze(0).to(device)
            # Manually replicate the positional encoding logic that the model expects
            # based on its checkpoint structure (which expects 18 dims)
            pos_weights = torch.tensor([1.0 / (3.0 ** k) for k in range(9)], dtype=torch.float64).to(device)
            x_aug = torch.cat([x, x * pos_weights], dim=-1)
            
            out = model(x_aug) # Passing augmented input
            embs.append(out['z_A_hyp'].squeeze().cpu().numpy())
    return np.array(embs)

def compute_persistent_homology(points):
    # Compute Vietoris-Rips complex
    skeleton = gudhi.RipsComplex(points=points)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    return persistence

def main():
    base_dir = Path(__file__).resolve().parents[2]
    ckpt = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/checkpoints/final.pt"
    cfg = base_dir / "runs/v17.1_rosetta_manifold_resume_20260525_060109/config.yaml"

    with open(cfg, 'r') as f:

        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Model
    model_cfg = config['model']
    # Filter only arguments accepted by TernaryVAEV6Controllable
    # Note: TernaryVAEV6Controllable passes kwargs to TernaryVAEV6, 
    # which passes to projections. The projections need specific keys.
    # The config 'model' has keys that correspond to both TernaryVAEV6 and HyperbolicProjection.

    # TernaryVAEV6Controllable.__init__ accepts:
    # encoder_a_lr_scale, encoder_b_lr_scale, projections_lr_scale,
    # encoder_a_trainable, encoder_b_trainable, projections_trainable, **kwargs

    # **kwargs goes to TernaryVAEV6.__init__, which accepts:
    # latent_dim, hidden_dim, max_radius, curvature, encoder_type, decoder_type,
    # n_projection_layers, projection_dropout, learnable_curvature,
    # init_identity, tangent_scale_init, factored, radial_dims, detach_radial,
    # positional_encoding, pos_weight_base

    # We need to map 'projection_layers' to 'n_layers' (this was the error!)
    # And 'tangent_scale' to 'tangent_scale_init'.

    # Create the mapping
    mapping = {
        'projection_layers': 'n_projection_layers',
        'tangent_scale': 'tangent_scale_init',
        # projection_dropout already matches
    }

    init_cfg = {}
    for k, v in model_cfg.items():
        if k in mapping:
            init_cfg[mapping[k]] = v
        else:
            init_cfg[k] = v

    # Remove 'name' which isn't part of any init
    if 'name' in init_cfg: del init_cfg['name']

    model = TernaryVAEV6Controllable(**init_cfg).to(device)

    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Extracting Hotspot Embeddings...")
    embs = get_hotspot_embeddings(model, device)
    
    print("Computing Persistent Homology...")
    persistence = compute_persistent_homology(embs)
    
    print("\n--- Persistence Diagram ---")
    print(persistence)
    
    # Simple interpretation of Betti numbers
    b0 = sum(1 for p in persistence if p[0] == 0) # Components
    b1 = sum(1 for p in persistence if p[0] == 1) # Loops
    
    print(f"\nEmergent Topological Features:")
    print(f"  Connected Components (B0): {b0}")
    print(f"  Topological Holes/Loops (B1): {b1}")

if __name__ == "__main__":
    main()
