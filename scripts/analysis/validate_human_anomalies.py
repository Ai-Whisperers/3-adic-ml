import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.analysis.test_hierarchical_search import setup_model
from src.analysis.anomaly_detector import AnomalyDetector
from scripts.data.prepare_codon_data import seq_to_ternary_index

def validate_human_anomaly(checkpoint_path, human_data_path):
    model = setup_model(checkpoint_path)
    
    # 1. Fit detector on Normal Human TP53 data
    normal_indices = torch.load(human_data_path, weights_only=True)
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, torch.device("cpu"))
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    detector = AnomalyDetector(model)
    detector.fit(z_norm)
    
    # 2. Test sequences
    test_seqs = ["CCTGCCCTC", "AAAAAAAAA"] # TP53 segment vs Random
    test_indices = torch.tensor([seq_to_ternary_index(s) for s in test_seqs])
    
    with torch.no_grad():
        mu_test = model.get_mu_representations(test_indices, torch.device("cpu"))
        z_test, _ = model.projections.proj_A(mu_test)
    
    # 3. Detect
    results = detector.detect(z_test)
    
    print("\nAnomaly Detection Results:")
    for i, seq in enumerate(test_seqs):
        label = "ANOMALY" if results["is_anomaly"][i] else "NORMAL"
        print(f"[{label}] {seq} | Dist: {results['min_dist'][i]:.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v15.0_long_term_stability_20260521_120221/checkpoints/final.pt"
    # Actually use the fine-tuned checkpoint
    ckpt_path = "runs/v16.0_human_fine_tuning_20260523_094539/checkpoints/final.pt"
    validate_human_anomaly(ckpt_path, "data/human_tp53_indices.pt")
