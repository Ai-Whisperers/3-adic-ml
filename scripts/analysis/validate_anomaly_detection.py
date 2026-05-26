import torch
import numpy as np
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.test_hierarchical_search import setup_model
from src.geometry import poincare_distance
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))
from prepare_codon_data import seq_to_ternary_index

def mutate_sequence(seq: str, n: int = 2) -> str:
    """Introduces n point mutations into a sequence."""
    nucs = list(seq)
    for _ in range(n):
        idx = random.randint(0, len(nucs)-1)
        nucs[idx] = random.choice([n for n in 'ACGT' if n != nucs[idx]])
    return "".join(nucs)

def validate_anomaly_detection(checkpoint_path, normal_indices_path):
    model = setup_model(checkpoint_path)
    device = torch.device("cpu")
    
    # 1. Load normal embeddings
    normal_indices = torch.load(normal_indices_path, weights_only=True)
    # Use a subset for faster validation
    subset_size = min(len(normal_indices), 500)
    normal_indices = normal_indices[:subset_size]
    
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, device)
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    # 2. Compute Threshold
    nn_dists = []
    for i in range(len(z_norm)):
        dists = [poincare_distance(z_norm[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item() for j in range(len(z_norm)) if i != j]
        nn_dists.append(min(dists))
    threshold = np.mean(nn_dists) + 2 * np.std(nn_dists)
    
    # 3. Create test sets
    # Ground truth: Normal=0, Anomalous=1
    test_data = []
    labels = []
    
    # Normal (use validation E. coli data)
    test_data.extend(normal_indices.tolist())
    labels.extend([0] * len(normal_indices))
    
    # Anomalies
    anom_seqs = [mutate_sequence("ACGTACGTA", n=2) for _ in range(200)] # Subtle mutations
    anom_indices = [seq_to_ternary_index(s) for s in anom_seqs]
    test_data.extend(anom_indices)
    labels.extend([1] * len(anom_indices))
    
    # 4. Run detection
    print(f"Validating on {len(test_data)} samples...")
    predictions = []
    with torch.no_grad():
        test_indices = torch.tensor(test_data, device=device)
        mu_test = model.get_mu_representations(test_indices, device)
        z_test, _ = model.projections.proj_A(mu_test)
        
        for i in range(len(z_test)):
            dists = [poincare_distance(z_test[i].unsqueeze(0), z_norm[j].unsqueeze(0), c=1.0).item() for j in range(len(z_norm))]
            min_dist = min(dists)
            predictions.append(1 if min_dist > threshold else 0)
            
    # 5. Metrics
    tp = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(predictions, labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))
    
    print(f"TPR: {tp/(tp+fn):.4f}")
    print(f"FPR: {fp/(fp+tn):.4f}")
    print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    validate_anomaly_detection(ckpt_path, "data/ecoli_indices.pt")
