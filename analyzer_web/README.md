# Analyzer Web

Web module for interactive experiment analysis.

## Files

- `index.html`: main dashboard (table, chart, multi-run compare)
- `pareto.html`: compression vs accuracy map (regions + Pareto frontier)
- `run.html`: run-focused page
- `assets/styles.css`: UI styles
- `assets/js/`: ES modules (`boot-dashboard.js` / `boot-run.js`, shared `data-loader.js`, `compare-tab.js`, etc.)
- `data/results.json`: generated dataset from CSV runs

## Data loading

- Preferred mode: the analyzer reads `runs/run_*/results.csv` directly from the browser (live mode).
- Fallback mode: if live read is not possible, it uses `analyzer_web/data/results.json`.
- Optional legacy build command:

```bash
python3 scripts/build_web_analysis_data.py --runs-dir runs --output analyzer_web/data/results.json
```

## Compare tab (Vega-Lite)

- Add **full runs** from the live dataset (every row in that run’s `results.csv`) or **upload** a `results.csv`. Runs from the picker are saved in `localStorage` until you remove them; uploads are session-only.
- With **multiple** datasets, series names are prefixed with your label so traces stay distinct.
- Renders with **Vega-Lite** and **Vega Themes** (defaults: thin connecting line + very small points for dense runs).
- Controls: **Visualization** (line + small points, line only, small scatter, step + points), **Point size** (tiny / small / medium), **Theme** (clean, ggplot2, Quartz, Urban Institute, Excel, dark), plus axis / series / title fields aligned with Chart.
- **Table** and **Chart** still use Plotly.

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
