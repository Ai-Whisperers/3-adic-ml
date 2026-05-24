import os
import requests
import json
import argparse
import sys

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Peptide Visualization - {title}</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        #container {{ width: 100%; height: 600px; position: relative; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .controls {{ margin-top: 20px; padding: 15px; background: white; border-radius: 8px; }}
        h1 {{ color: #1a73e8; }}
        .info {{ color: #5f6368; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="info">
        <strong>Source:</strong> {source}<br>
        <strong>ID:</strong> {id}
    </div>
    <div id="container"></div>
    
    <div class="controls">
        <button onclick="viewer.setStyle({{}},{{cartoon:{{color: 'spectrum'}}}}); viewer.render();">Cartoon Spectrum</button>
        <button onclick="viewer.setStyle({{}},{{stick:{{}}}}); viewer.render();">Sticks</button>
        <button onclick="viewer.setStyle({{}},{{sphere:{{}}}}); viewer.render();">Spheres</button>
        <button onclick="viewer.setSurface(3Dmol.SurfaceType.VDW, {{opacity:0.5, color:'white'}}); viewer.render();">Surface</button>
        <button onclick="viewer.removeAllSurfaces(); viewer.render();">Clear Surface</button>
    </div>

    <script>
        let viewer;
        $(function() {{
            let element = $('#container');
            let config = {{ backgroundColor: 'white' }};
            viewer = $3Dmol.createViewer(element, config);
            
            let data = `{data}`;
            let format = '{format}';
            
            viewer.addModel(data, format);
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.zoomTo();
            viewer.render();
        }});
    </script>
</body>
</html>
"""

def fetch_from_afdb(uniprot_id):
    print(f"Fetching metadata for {uniprot_id}...")
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    response = requests.get(api_url)
    if response.status_code != 200 or not response.json():
        print(f"Error: Could not find structure for {uniprot_id}")
        return None
    
    metadata = response.json()[0]
    pdb_url = metadata.get('pdbUrl')
    if not pdb_url:
        print(f"Error: No PDB URL found for {uniprot_id}")
        return None
    
    print(f"Downloading PDB from {pdb_url}...")
    pdb_response = requests.get(pdb_url)
    return pdb_response.text, 'pdb'

def generate_visualization_html(data, fmt, title, source, id_str):
    # Escape backticks and backslashes for JS string
    safe_data = data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    
    return HTML_TEMPLATE.format(
        title=title,
        source=source,
        id=id_str,
        data=safe_data,
        format=fmt
    )

def main():
    parser = argparse.ArgumentParser(description='Visualize a peptide structure.')
    parser.add_argument('--id', type=str, help='UniProt ID to fetch from AlphaFold DB')
    parser.add_argument('--file', type=str, help='Local PDB or CIF file to visualize')
    parser.add_argument('--output', type=str, default='visualization.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    data = None
    fmt = 'pdb'
    title = "Peptide Structure"
    source = "Unknown"
    
    if args.id:
        result = fetch_from_afdb(args.id)
        if result:
            data, fmt = result
            title = f"AlphaFold Prediction: {args.id}"
            source = "AlphaFold Database"
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found.")
            sys.exit(1)
        with open(args.file, 'r') as f:
            data = f.read()
        fmt = 'pdb' if args.file.endswith('.pdb') else 'cif'
        title = f"Local File: {os.path.basename(args.file)}"
        source = "Local File"
    else:
        print("Please provide either --id or --file")
        sys.exit(1)
        
    if data:
        html_content = generate_visualization_html(data, fmt, title, source, args.id or os.path.basename(args.file))
        
        with open(args.output, 'w') as f:
            f.write(html_content)
        
        print(f"Visualization saved to {args.output}")
        print(f"Open this file in your browser to interact with the 3D model.")

if __name__ == "__main__":
    main()
