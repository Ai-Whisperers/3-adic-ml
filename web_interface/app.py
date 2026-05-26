import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
from pathlib import Path
import sys
import plotly.io as pio

# Setup imports
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from scripts.analysis.test_hierarchical_search import setup_model
from src.analysis.anomaly_detector import AnomalyDetector
from src.analysis.attribute_anomaly import compute_attribution
from prepare_codon_data import seq_to_ternary_index
from src.utils.poincare_renderer import render_poincare_disk
from src.core import TERNARY

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="web_interface/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("web_interface/static/index.html")

# Global model/detector
ckpt_path = "runs/v16.0_human_fine_tuning_20260523_094539/checkpoints/final.pt"
human_data_path = "data/human_tp53_indices.pt"

model = setup_model(ckpt_path)
normal_indices = torch.load(human_data_path, weights_only=True)
with torch.no_grad():
    mu_norm = model.get_mu_representations(normal_indices, torch.device("cpu"))
    z_norm, _ = model.projections.proj_A(mu_norm)

detector = AnomalyDetector(model)
detector.fit(z_norm, k=5, sigma_factor=2.0)

class Query(BaseModel):
    sequence: str

@app.post("/analyze")
async def analyze(query: Query):
    seq = query.sequence.upper().replace(" ", "")
    if len(seq) != 9:
        raise HTTPException(status_code=400, detail="Sequence must be 9 nuc long")
    
    try:
        idx = torch.tensor([seq_to_ternary_index(seq)])
        with torch.no_grad():
            mu = model.get_mu_representations(idx, torch.device("cpu"))
            z, _ = model.projections.proj_A(mu)
        
        results = detector.detect(z)
        
        # Generate Interactive Visualization
        plot_indices = torch.cat([idx, normal_indices[:100]])
        plot_z = torch.cat([z, z_norm[:100]])
        plot_vals = TERNARY.valuation(plot_indices)
        
        fig = render_poincare_disk(
            plot_z.numpy(), 
            plot_vals.numpy(), 
            plot_indices.numpy(),
            title="Latent Projection",
            c=1.0,
            show_tree=False
        )
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Segoe UI, sans-serif"),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.update_traces(
            marker=dict(size=7, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)'))
        )
        
        return {
            "status": "ANOMALY" if results["is_anomaly"][0] else "NORMAL",
            "dist": float(results["min_dist"][0]),
            "fig_json": json.loads(pio.to_json(fig))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attribute")
async def attribute(query: Query):
    seq = query.sequence.upper().replace(" ", "")
    if len(seq) != 9:
        raise HTTPException(status_code=400, detail="Sequence must be 9 nuc long")
    try:
        attribution, dist = compute_attribution(seq, model, detector)
        return {
            "attribution": attribution,
            "dist": dist
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
async def analyze_batch(file: UploadFile = File(...)):
    content = await file.read()
    seq = "".join([nuc for nuc in content.decode().upper() if nuc in 'ACGT'])
    
    results = []
    window_size = 9
    
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i+window_size]
        try:
            idx = torch.tensor([seq_to_ternary_index(window)])
            with torch.no_grad():
                mu = model.get_mu_representations(idx, torch.device("cpu"))
                z, _ = model.projections.proj_A(mu)
            res = detector.detect(z)
            if res["is_anomaly"][0]:
                results.append({"pos": i, "seq": window, "dist": float(res["min_dist"][0])})
        except Exception:
            continue
    return {"anomalies": results}
