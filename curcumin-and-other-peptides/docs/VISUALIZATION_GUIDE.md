# Visualization and AlphaFold 3 Guide

This guide describes how to use the integrated 3D visualization and AlphaFold 3 tools within the Peptides project.

## Unified Pipeline Entry Point

The `main.py` script provides a single entry point for all research tasks.

### 1. 3D Visualization

We use `3Dmol.js` to render interactive structures. You can fetch known structures from the AlphaFold Database or visualize local predictions.

#### Visualize All Sequences
To generate a gallery for all peptides in `data/sequences.fasta`:
```bash
python3 main.py visualize --all
```
The result is available in `visualizations/index.html`.

#### Visualize a Specific ID
```bash
python3 main.py visualize --id C6L7V9
```

### 2. AlphaFold 3 Integration

AlphaFold 3 requires a specific JSON input format for its server or local execution.

#### Generate AF3 Input JSON
```bash
python3 main.py af3-gen --name "Ginger_Peptide_1" --seq "MGSLQAMRRA"
```
This generates `Ginger_Peptide_1_af3.json`.

#### Running Predictions
1. Go to the [AlphaFold Server](https://alphafoldserver.com/).
2. Upload the generated JSON file.
3. Once the prediction is complete, download the `.cif` or `.pdb` results.
4. Visualize the local result using the visualization script:
   ```bash
   python3 scripts/visualize_peptide.py --file path/to/downloaded_result.cif --output visualizations/my_prediction.html
   ```

## Directory Structure for Outputs
- `visualizations/`: Contains all generated HTML structure reports.
- `visualizations/index.html`: The central gallery for batch results.

## Technical Details
- **Fetcher:** `scripts/visualize_peptide.py` uses the AlphaFold EBI API to retrieve PDB files.
- **Renderer:** The HTML template utilizes `3Dmol.org` for WebGL-based hardware-accelerated rendering.
- **AF3 Schema:** `scripts/af3_input_gen.py` follows the Dialect 1, Version 1 schema required by AlphaFold 3.
