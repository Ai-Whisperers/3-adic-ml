import re
from collections import Counter

def read_fasta(file_path):
    sequences = {}
    current_id = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:].split('|')[1] # Get Uniprot ID
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line
    return sequences

def find_repeats(sequence, min_len=3, max_len=6):
    repeats = []
    for length in range(min_len, max_len + 1):
        for i in range(len(sequence) - length + 1):
            sub = sequence[i:i+length]
            if sequence.count(sub) > 1:
                repeats.append(sub)
    return Counter(repeats)

def analyze_composition(sequence):
    return Counter(sequence)

def main():
    sequences = read_fasta('data/sequences.fasta')
    print(f"Analyzing {len(sequences)} sequences...\n")

    all_repeats = Counter()
    
    for id, seq in sequences.items():
        print(f"--- Sequence {id} ---")
        print(f"Length: {len(seq)}")
        comp = analyze_composition(seq)
        # Sort by most common
        top_aa = comp.most_common(3)
        print(f"Most common AA: {top_aa}")
        
        repeats = find_repeats(seq)
        all_repeats.update(repeats)
        
        # Check for specific motifs (e.g., G-X-G, common in bioactive regions)
        gxg = re.findall(r'G.G', seq)
        if gxg:
            print(f"G-X-G motifs found: {gxg}")
        
        # C-C patterns (disulfide potential)
        cc = re.findall(r'C.C', seq)
        if cc:
            print(f"C-X-C motifs found: {cc}")
            
        print("\n")

    print("--- Top Repeated Motifs across all sequences ---")
    for motif, count in all_repeats.most_common(10):
        if count > 2:
            print(f"{motif}: {count}")

if __name__ == "__main__":
    main()
