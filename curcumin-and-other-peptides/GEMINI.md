# Project Peptides: Bioactive Peptides in Ginger & Curcumin

## Overview
This project explores the proteomic landscape of *Curcuma longa* (Turmeric) and *Zingiber officinale* (Ginger), specifically focusing on bioactive peptides. Unlike traditional secondary metabolites like curcumin and gingerols, these peptides offer unique structural interactions with cellular signaling pathways.

## Research Goals
1. **Proteomic Profiling:** Identify and map peptide sequences from Ginger and Curcumin.
2. **Stability Analysis:** Investigate proteolytic stability and antioxidant properties.
3. **EV Cargo:** Analyze the peptide content of Ginger-derived extracellular vesicles.
4. **Computational Design:** Utilize generative models and latent space representations to design stable, bioactive peptide variants.

## Tech Stack (Proposed)
- **Language:** Python
- **Libraries:** Biopython, PyTorch/TensorFlow, RDKit (if applicable), Proteomics tools (e.g., Pyteomics)
- **Modeling:** Protein Language Models (e.g., ESM-2), VAEs/GANs for sequence generation.

## Directory Structure
- `docs/`: Research papers, notes, and literature reviews.
- `data/`: Genomic sequences, proteomic mass spec data (FASTA, etc.).
- `models/`: Trained models and scripts for peptide design.
- `scripts/`: Data processing and analysis scripts.
    - `visualize_peptide.py`: Fetches and renders 3D structures from AlphaFold DB.
    - `visualize_all.py`: Batch processes all sequences in `data/sequences.fasta` into a gallery.
    - `af3_input_gen.py`: Generates JSON inputs for AlphaFold 3 predictions.

## Usage
The project uses a unified CLI entry point `main.py` for all tasks.

### Core Commands
- **Analyze Patterns:** `python3 main.py analyze`
- **Visualize Batch:** `python3 main.py visualize --all`
- **Visualize ID:** `python3 main.py visualize --id <UNIPROT_ID>`
- **Generate AF3 Input:** `python3 main.py af3-gen --name <NAME> --seq <SEQUENCE>`

For detailed instructions on 3D visualization and AlphaFold 3 integration, see [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md).
