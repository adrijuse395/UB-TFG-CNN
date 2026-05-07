# Analyzer Web

Web module for interactive experiment analysis.

## Files

- `index.html`: main dashboard (filters, KPIs, plots, table)
- `run.html`: run-focused page
- `assets/styles.css`: UI styles
- `assets/app.js`: frontend logic
- `data/results.json`: generated dataset from CSV runs

## Data loading

- Preferred mode: the analyzer reads `runs/run_*/results.csv` directly from the browser (live mode).
- Fallback mode: if live read is not possible, it uses `analyzer_web/data/results.json`.
- Optional legacy build command:

```bash
python3 scripts/build_web_analysis_data.py --runs-dir runs --output analyzer_web/data/results.json
```

## Open locally

Recommended:

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/analyzer_web/index.html`
- `http://localhost:8000/analyzer_web/run.html?run=run_YYYYMMDD_HHMMSS`

Notes:
- Run `http.server` from the project root so `../runs/` is available to `analyzer_web`.
- Dashboard auto-refresh checks for new CSV rows every ~8 seconds.
