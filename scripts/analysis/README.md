# Analysis Scripts

`project_audit.py` is the reproducible audit entry point for this repository's saved runs.

Usage:

```bash
python3 scripts/analysis/project_audit.py \
  --run-dir runs/v7_large_20260324_013725 \
  --json-out local/analysis/project_audit.json \
  --text-out local/analysis/project_audit.txt
```

What it does:

- Ranks saved runs using `results.json`
- Loads a checkpoint and evaluates it on the full 19,683-state domain
- Reports exact input/output surface and parameter count
- Measures decoder-prior generation diversity
- Recomputes direction clustering ARI on the full domain
- Runs a transparent Monte Carlo scenario model for feasibility discussion

Important:

- The Monte Carlo block is a scenario model, not a benchmark or market forecast.
- The script evaluates what is supported by the repository today. It does not assume external-task generalization.
