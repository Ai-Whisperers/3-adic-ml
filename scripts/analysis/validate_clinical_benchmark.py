import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from scripts.analysis.test_hierarchical_search import setup_model
from src.analysis.anomaly_detector import AnomalyDetector
from prepare_codon_data import seq_to_ternary_index

def validate_tp53_benchmark(checkpoint_path, ecoli_data_path):
    model = setup_model(checkpoint_path)
    
    # 1. Fit detector on Normal E. coli data (Baseline)
    normal_indices = torch.load(ecoli_data_path, weights_only=True)
    with torch.no_grad():
        mu_norm = model.get_mu_representations(normal_indices, torch.device("cpu"))
        z_norm, _ = model.projections.proj_A(mu_norm)
        
    detector = AnomalyDetector(model)
    detector.fit(z_norm)
    
    # 2. Define TP53 segments (Normal vs Pathogenic)
    # Wild-type segment of TP53
    wt_seq = "CCTGCCCTC" # 9nt window
    
    # Pathogenic mutation (R175H example simulation: mutation in codon)
    # R175H in DNA often involves CGG -> CAG or similar. Let's simulate a mutation
    patho_seq = "CCTGCCATC" # Mutation
    
    test_seqs = [wt_seq, patho_seq]
    test_indices = torch.tensor([seq_to_ternary_index(s) for s in test_seqs])
    
    with torch.no_grad():
        mu_test = model.get_mu_representations(test_indices, torch.device("cpu"))
        z_test, _ = model.projections.proj_A(mu_test)
    
    # 3. Detect
    results = detector.detect(z_test)
    
    print("\nTP53 Benchmark Results:")
    for i, seq in enumerate(test_seqs):
        label = "ANOMALY" if results["is_anomaly"][i] else "NORMAL"
        print(f"[{label}] {seq} | Dist: {results['min_dist'][i]:.4f}")

if __name__ == "__main__":
    ckpt_path = "runs/v16.0_human_fine_tuning_20260523_094539/checkpoints/final.pt"
    # Use the human reference as normal baseline
    validate_tp53_benchmark(ckpt_path, "data/human_tp53_indices.pt")
