import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.analysis.test_hierarchical_search import setup_model
from src.analysis.anomaly_detector import AnomalyDetector
from scripts.data.prepare_codon_data import seq_to_ternary_index

def validate_humsavar_tp53(checkpoint_path, human_data_path):
    model = setup_model(checkpoint_path)
    
    # 1. Fit detector on Human TP53 normal baseline
    normal_indices = torch.load(human_data_path, weights_only=True)
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, torch.device("cpu"))
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    detector = AnomalyDetector(model)
    detector.fit(z_norm, k=5, sigma_factor=2.0)
    
    # 2. Define TP53 segments (Real Pathogenic Mutations)
    # Mapping simulation: Based on Humsavar p.Cys141Tyr (Cys=TGC, Tyr=TAC -> T G C -> T A C)
    wt_seq = "TGCGGGCAG" 
    patho_seq = "TACGGGCAG" # Cys -> Tyr mutation
    
    test_seqs = [wt_seq, patho_seq]
    test_indices = torch.tensor([seq_to_ternary_index(s) for s in test_seqs])
    
    with torch.no_grad():
        mu_test = model.get_mu_representations(test_indices, torch.device("cpu"))
        z_test, _ = model.projections.proj_A(mu_test)
    
    # 3. Detect
    results = detector.detect(z_test)
    
    print("\nHumsavar TP53 Benchmark Results:")
    labels = ["NORMAL (WT)", "ANOMALY (Pathogenic)"]
    for i, seq in enumerate(test_seqs):
        status = "ANOMALY" if results["is_anomaly"][i] else "NORMAL"
        print(f"[{status}] {labels[i]} | Sequence: {seq} | Dist: {results['min_dist'][i]:.4f}")

if __name__ == "__main__":
    # V16.0 Human Fine-tuned
    ckpt_path = "runs/v16.0_human_fine_tuning_20260523_094539/checkpoints/final.pt"
    # Re-use the previously prepared human TP53 data as baseline
    validate_humsavar_tp53(ckpt_path, "data/human_tp53_indices.pt")
