function showTab(tab) {
    document.getElementById('single-tab').style.display = tab === 'single' ? 'block' : 'none';
    document.getElementById('batch-tab').style.display = tab === 'batch' ? 'block' : 'none';
}

async function analyze() {
    const seq = document.getElementById('seq').value;
    const resultDiv = document.getElementById('result');
    const attrDiv = document.getElementById('attribution');
    const plotDiv = document.getElementById('plot');
    
    resultDiv.innerHTML = "Analyzing...";
    attrDiv.innerHTML = "";
    plotDiv.innerHTML = "";
    
    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sequence: seq})
    });
    
    const attrResponse = await fetch('/attribute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sequence: seq})
    });
    
    if (!response.ok || !attrResponse.ok) {
        resultDiv.innerText = "Error: Sequence must be 9 nuc long";
        return;
    }
    
    const data = await response.json();
    const attrData = await attrResponse.json();
    
    resultDiv.innerHTML = `<p>Status: <strong style="color: ${data.status === 'ANOMALY' ? '#f87171' : '#4ade80'}">${data.status}</strong></p>`;
    
    let attrHtml = "<p>Sensitivity (Hotspots):</p><div style='display:flex; justify-content:space-between;'>";
    attrData.attribution.forEach((val, i) => {
        const intensity = Math.min(val * 5, 1);
        attrHtml += `<span style="background:rgba(248,113,113,${intensity}); padding:5px; border-radius:5px;">${seq[i]}</span>`;
    });
    attrHtml += "</div>";
    attrDiv.innerHTML = attrHtml;
    
    const layout = data.fig_json.layout;
    layout.paper_bgcolor = "rgba(0,0,0,0)";
    layout.plot_bgcolor = "rgba(0,0,0,0)";
    layout.font = { color: "#f8fafc" };
    layout.margin = { l: 20, r: 20, b: 20, t: 40 };
    
    Plotly.newPlot('plot', data.fig_json.data, layout, {responsive: true});
}

async function analyzeBatch() {
    const fileInput = document.getElementById('fastaFile');
    const resultDiv = document.getElementById('result');
    if (fileInput.files.length === 0) return;
    
    resultDiv.innerHTML = "Processing Batch...";
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    const response = await fetch('/analyze-batch', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    resultDiv.innerHTML = `<h3>Batch Report: Found ${data.anomalies.length} anomalies</h3>`;
    data.anomalies.slice(0, 5).forEach(a => {
        resultDiv.innerHTML += `<p>Pos ${a.pos}: ${a.seq} (Dist: ${a.dist.toFixed(4)})</p>`;
    });
}
