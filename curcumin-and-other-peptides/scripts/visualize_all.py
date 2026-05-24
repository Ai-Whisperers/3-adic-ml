import os
import sys
import re
from visualize_peptide import fetch_from_afdb, generate_visualization_html

def read_uniprot_ids(fasta_path):
    ids = []
    if not os.path.exists(fasta_path):
        print(f"Error: {fasta_path} not found.")
        return ids
        
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract ID between pipes: >sp|C6L7V9|CURS3_CURLO
                match = re.search(r'\|([^|]+)\|', line)
                if match:
                    ids.append(match.group(1))
                else:
                    # Fallback for IDs without pipes
                    id_part = line[1:].split()[0]
                    ids.append(id_part)
    return ids

def main():
    # Assume we run from project root
    fasta_path = os.path.join('data', 'sequences.fasta')
    output_dir = 'visualizations'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ids = read_uniprot_ids(fasta_path)
    print(f"Found {len(ids)} sequences in {fasta_path}: {', '.join(ids)}")
    
    summary_html = [
        "<html><head><title>Peptide Visualization Gallery</title>",
        "<style>body { font-family: sans-serif; padding: 20px; }",
        ".gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }",
        ".card { border: 1px solid #ccc; padding: 15px; border-radius: 8px; text-align: center; text-decoration: none; color: #333; }",
        ".card:hover { border-color: #1a73e8; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }</style>",
        "</head><body><h1>Peptide Visualization Gallery</h1><div class='gallery'>"
    ]
    
    for uid in ids:
        output_file = os.path.join(output_dir, f"{uid}.html")
        print(f"\n--- Processing {uid} ---")
        
        result = fetch_from_afdb(uid)
        if result:
            data, fmt = result
            html = generate_visualization_html(
                data, fmt, 
                title=f"AlphaFold Prediction: {uid}", 
                source="AlphaFold Database", 
                id_str=uid
            )
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Saved visualization to {output_file}")
            summary_html.append(f"<a href='{uid}.html' class='card'><h3>{uid}</h3><p>View 3D Structure</p></a>")
        else:
            print(f"Skipping {uid} (No structure found)")
            
    summary_html.append("</div></body></html>")
    
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write("\n".join(summary_html))
        
    print(f"\nPipeline complete. Gallery available at {output_dir}/index.html")

if __name__ == "__main__":
    main()
