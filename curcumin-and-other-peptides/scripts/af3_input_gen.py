import json
import argparse
import sys

def create_af3_json(name, sequences, seeds=[1]):
    """
    Creates a JSON input file for AlphaFold 3.
    sequences is a list of dicts: {'type': 'protein', 'id': 'A', 'sequence': '...'}
    """
    job = {
        "name": name,
        "modelSeeds": seeds,
        "sequences": [],
        "dialect": "alphafold3",
        "version": 1
    }
    
    for seq_info in sequences:
        entity = {}
        stype = seq_info['type'].lower()
        if stype == 'protein':
            entity = {"protein": {"id": seq_info['id'], "sequence": seq_info['sequence']}}
        elif stype == 'dna':
            entity = {"dna": {"id": seq_info['id'], "sequence": seq_info['sequence']}}
        elif stype == 'rna':
            entity = {"rna": {"id": seq_info['id'], "sequence": seq_info['sequence']}}
        elif stype == 'ligand':
            entity = {"ligand": {"id": seq_info['id'], "ligand": seq_info['sequence']}} # sequence here is CCD code or SMILES?
        
        job["sequences"].append(entity)
        
    return job

def main():
    parser = argparse.ArgumentParser(description='Generate AlphaFold 3 JSON input.')
    parser.add_argument('--name', type=str, required=True, help='Job name')
    parser.add_argument('--seq', type=str, required=True, help='Amino acid sequence')
    parser.add_argument('--id', type=str, default='A', help='Chain ID')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    sequences = [{'type': 'protein', 'id': args.id, 'sequence': args.seq}]
    job_json = create_af3_json(args.name, sequences)
    
    output_file = args.output or f"{args.name}_af3.json"
    
    with open(output_file, 'w') as f:
        json.dump(job_json, f, indent=2)
        
    print(f"AlphaFold 3 input JSON saved to {output_file}")
    print("You can upload this file to the AlphaFold 3 Server or run it with local AlphaFold 3.")

if __name__ == "__main__":
    main()
