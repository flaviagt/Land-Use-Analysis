# Land-Use-Analysis
Project for DSC190: Data Science for Sustainable Development

## Kalimantan fire analysis (midterm EDA)
DSC 190 (Data Science for Sustainable Development) project: what happens to land in Kalimantan, Indonesia, in the five years after it burns — compared with similar land that did not burn.

**Research question.** Among land that was natural vegetation in the year before a fire, how do land-cover trajectories over the next five years differ between land that burned and comparable land that did not — separately for forest and non-forest vegetation?

**Data.** MapBiomas Indonesia land cover (Collection 4.1, annual, 30 m) and MapBiomas Fire Indonesia (Collection 2, annual burned area, 30 m), read in Google Earth Engine.

### Repository layout

| Path | What it is |
| --- | --- |
| `gee/01_fast_random_points.js` | **Main pipeline.** Earth Engine Code Editor script. Draws random points over Kalimantan, packs each point's variables into one integer code, prints a histogram of codes (fast, no Drive export). |
| `gee/02_full_resolution_exports.js` | Full-resolution version (exports to Google Drive). Much slower (hours); kept for the final report. |
| `gee/03_exploration_map_landcover.js` | Earth Engine script for the midterm exploration data: burned-area grid for 2015/2019 (map) and land cover at random points, 2000–2024. |
| `analysis/eda_exploration.py` | **Midterm report figures 2–5:** burned-area map (`figA`), land cover 2000–2024 (`figB`), post-fire land cover for forest (`figC`) and non-forest (`figD`). |
| `analysis/eda_fast.py` | Decodes `data/fast_results.json` into Tables 1–2 and Figures 1–4 in `outputs/` (Figure 1 is midterm Figure 1). |
| `analysis/eda_full_exports.py` | Same analysis for the Drive exports of script 02. |
| `analysis/laptop_geotiff_eda.py` | Alternative without Earth Engine: reads downloaded MapBiomas GeoTIFFs on a laptop. |
| `data/fast_results.json` | Raw histograms returned by script 01 (one entry per random-point batch). |
| `data/burn_grid_005deg.json`, `data/landcover_by_year.json` | Data returned by script 03. |
| `outputs/` | Figures (PNG) and tables (CSV). |

### Study design (short)

- **Cohorts:** pixels that burned in 2015 or 2019 and had **no fire in the 5 previous years** ("first fire").
- **Before/after:** land cover in **t−1** (before the fire) and **t+5** (five years after).
- **Comparison:** unburned pixels with the same pre-fire land cover and the same clean fire history.
- **Main confounder (planned for the final report):** distance to the nearest oil palm in t−1 (bands 0–2, 2–5, 5–15, ≥15 km).
- **"Oil palm" outcome:** classified oil palm in both t+4 and t+5 (filters one-year misclassification).
- **Sampling:** simple random points across Kalimantan (538,047 km²) → every point has equal weight.
- **Uncertainty:** 95% Wilson intervals (difference: Newcombe). These do not yet adjust for spatial clustering.

Association only — this design cannot show that fire *causes* conversion or who set any fire.

### How to reproduce

1. Open https://code.earthengine.google.com with a registered noncommercial project, paste `gee/01_fast_random_points.js`, set `WHAT` (`'c2015'`, `'c2019'` or `'fig1'`), `SEEDS` and `N_POINTS` (≈4,000 per seed keeps each request under the 5-minute interactive limit), run, and copy each printed JSON into `data/fast_results.json` under a key like `"c2019_31": {"n_points": 4000, "r": <printed JSON>}`.
2. `pip install pandas numpy matplotlib` then `python analysis/eda_fast.py` and `python analysis/eda_exploration.py` → `outputs/`.
