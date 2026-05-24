import argparse
import sys
import os
import subprocess

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Set PYTHONPATH to include scripts directory for imports
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.getcwd(), "scripts")
        
        subprocess.run(command, check=True, shell=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Peptide Research Pipeline - CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    subparsers.add_parser('analyze', help='Run pattern analysis on sequences.fasta')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate 3D visualizations')
    viz_parser.add_argument('--id', type=str, help='UniProt ID for single visualization')
    viz_parser.add_argument('--all', action='store_true', help='Visualize all sequences in FASTA')

    # AF3 command
    af3_parser = subparsers.add_parser('af3-gen', help='Generate AlphaFold 3 JSON input')
    af3_parser.add_argument('--name', type=str, required=True, help='Job name')
    af3_parser.add_argument('--seq', type=str, required=True, help='Amino acid sequence')

    args = parser.parse_args()

    if args.command == 'analyze':
        run_command("python3 scripts/analyze_patterns.py", "Analyzing peptide patterns")
    
    elif args.command == 'visualize':
        if args.all:
            run_command("python3 scripts/visualize_all.py", "Generating batch visualizations")
        elif args.id:
            run_command(f"python3 scripts/visualize_peptide.py --id {args.id} --output visualizations/{args.id}.html", f"Visualizing {args.id}")
        else:
            print("Please specify --id <ID> or --all")
    
    elif args.command == 'af3-gen':
        run_command(f"python3 scripts/af3_input_gen.py --name {args.name} --seq {args.seq}", f"Generating AF3 JSON for {args.name}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
