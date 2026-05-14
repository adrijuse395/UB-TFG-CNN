# Analyzer Web

Web module for interactive experiment analysis.

## Files

- `index.html`: main dashboard (table, chart, multi-run compare)
- `pareto.html`: compression vs accuracy map (regions + Pareto frontier)
- `run.html`: run-focused page
- `assets/styles.css`: UI styles
- `assets/js/`: ES modules (`boot-dashboard.js` / `boot-run.js`, shared `data-loader.js`, `compare-tab.js`, etc.)

## Data loading

The analyzer **only** reads `runs/run_*/results.csv` over HTTP. There is **no** silent fallback to a frozen JSON: if the server cannot expose `runs/`, you fix the setup (see below) instead of looking at stale data.

Optional: export a JSON snapshot for a informe or sharing (not used by the web UI):

```bash
python3 scripts/build_web_analysis_data.py --runs-dir runs --output /path/to/snapshot.json
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
- Run `http.server` from the **project root** (parent of `analyzer_web/`). The loader resolves `runs/` from `analyzer_web/assets/js/data-loader.js` (`../../../runs/`) so it matches the repo layout even if the page URL is odd.
- Dashboard auto-refresh checks for new CSV rows every ~8 seconds.
