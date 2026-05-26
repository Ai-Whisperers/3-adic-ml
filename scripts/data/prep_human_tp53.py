import torch
from scripts.data.prepare_codon_data import prepare_codon_data

# Read TP53 reference
with open("data/human/tp53_ref.fasta", "r") as f:
    seq = "".join([line.strip() for line in f if not line.startswith(">")])

window_size = 9
sequences = [seq[i:i+window_size] for i in range(len(seq) - window_size + 1)]

indices = prepare_codon_data(sequences)
torch.save(torch.tensor(indices), "data/human_tp53_indices.pt")
print(f"Processed {len(indices)} sequences into data/human_tp53_indices.pt")
