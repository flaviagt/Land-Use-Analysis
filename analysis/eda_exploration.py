# %% [markdown]
# # DSC 190 – basic data-exploration figures (midterm, revised)
# Inputs (all produced in Google Earth Engine, then saved into data/):
#   data/fast_results.json        random-point histograms (cohorts 2015, 2019 and Figure 1)
#   data/burn_grid_005deg.json    0.05° grid: land fraction and burned fraction in 2015 and 2019
#   data/landcover_by_year.json   random-point histograms of land-cover group, 2000–2024
# Outputs (outputs/):
#   figA_burned_map_2015_2019.png        where burning happened
#   figB_landcover_2000_2024.png         how Kalimantan's land cover changed
#   figC_postfire_forest.png             land cover 5 years later, land that was forest before the fire
#   figD_postfire_nonforest.png          same for non-forest natural vegetation

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)
AOI_KM2 = 538046.876
GROUP = {1: "Forest", 2: "Non-forest natural", 3: "Oil palm", 4: "Pulpwood",
         5: "Other agriculture", 6: "Other"}
ORDER = list(GROUP.values())
COLORS = dict(zip(ORDER, ["#1f7a3a", "#a6c96a", "#d4271e", "#7a5900", "#e9b44c", "#9aa0a6"]))
BURNED_C, UNBURNED_C = "#c2410c", "#64748b"
Z = 1.96
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


# %% Figure A – map of burned area, 2015 and 2019 ------------------------------------
gpath = DATA / "burn_grid_005deg.json"
if gpath.exists():
    G = json.loads(gpath.read_text())
    kal = np.array(G["kal"], int); res = G["res"]
    lon0, lat0 = G["lon0"], G["lat0"]
    ny, nx = kal.shape
    extent = [lon0 - res / 2, lon0 + (nx - 0.5) * res, lat0 - (ny - 0.5) * res, lat0 + res / 2]
    # burned share of each 0.05° cell (≈5.5 km), from 30 m MapBiomas Fire pixels, binned into classes
    bins = [0, 0.001, 0.02, 0.10, 0.25, 1.01]
    labels = ["not burned", "< 2%", "2–10%", "10–25%", "> 25%"]
    cmap = ListedColormap(["#e6e9e2", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"])
    norm = BoundaryNorm(bins, cmap.N)
    lat = lat0 - res * np.arange(ny)
    cell_km2 = (res * 111.32) ** 2 * np.cos(np.radians(lat))[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, y in zip(axes, ["2015", "2019"]):
        b = np.clip(np.array(G["b" + y], float), 0, 1)
        show = np.where(kal > 0, b, np.nan)
        km2 = float(np.nansum(show * cell_km2))
        cells = int(np.sum((show >= 0.02)))
        ax.imshow(show, extent=extent, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
        ax.set_title(f"Burned in {y}  (≈{km2:,.0f} km²)")
        ax.set_xlabel("Longitude (°E)")
        ax.set_xlim(108.5, 119.1); ax.set_ylim(-4.3, 4.4); ax.set_aspect("equal")
        ax.grid(color="#f0f0f0", lw=0.5)
    axes[0].set_ylabel("Latitude (°)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
    fig.legend(handles, labels, title="Share of each ≈5.5 km grid cell that burned", loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "figA_burned_map_2015_2019.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figure A saved")

# %% Figure B – land cover of Kalimantan, 2000–2024 ----------------------------------
lpath = DATA / "landcover_by_year.json"
if lpath.exists():
    L = json.loads(lpath.read_text())
    rows = []
    for batch in L.values():
        for yk, h in batch["hist"].items():
            for g, c in h.items():
                rows.append((int(yk[1:]), GROUP[int(g)], c))
    lb = pd.DataFrame(rows, columns=["year", "group", "count"]).groupby(["year", "group"])["count"].sum().unstack(fill_value=0)
    lb = lb.reindex(columns=ORDER, fill_value=0)
    share = lb.div(lb.sum(axis=1), axis=0) * 100
    share.round(2).to_csv(OUT / "figB_data_landcover_share.csv")
    YEARS = [2000, 2014, 2018, 2024]
    YCOL = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(ORDER)); w = 0.2
    for j, (yr, col) in enumerate(zip(YEARS, YCOL)):
        v = share.loc[yr, ORDER].values
        ax.bar(x + (j - 1.5) * w, v, w * 0.9, color=col, label=str(yr))
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 1.5) * w, vi + 0.8, f"{vi:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("% of Kalimantan's land area")
    ax.set_ylim(0, 78); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    ax.legend(title="Year", frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "figB_landcover_2000_2024.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figure B saved\n", share.loc[[2000, 2010, 2014, 2015, 2019, 2024]].round(1).to_string())

# %% Figures C and D – land cover five years after the fire, by pre-fire class ---------
raw = json.loads((DATA / "fast_results.json").read_text())
rows = []
for key, v in raw.items():
    if key.startswith("fig1"):
        continue
    coh = int(key.split("_")[0][1:])
    for code, cnt in v["r"]["hist"].items():
        rows.append((coh, int(code), cnt))
d = pd.DataFrame(rows, columns=["cohort", "code", "count"])
digit = lambda c, p: (c // 10 ** p) % 10
d = d.assign(prefire=digit(d.code, 8).map(GROUP), out5=digit(d.code, 7).map(GROUP), burned=digit(d.code, 6))
d["status"] = np.where(d.burned == 1, "Burned", "Unburned")
t = d.groupby(["cohort", "prefire", "status", "out5"])["count"].sum().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
recs = []
for (coh, pre, st), r in t.iterrows():
    n = int(r.sum())
    for g in ORDER:
        p, lo, hi = wilson(int(r[g]), n)
        recs.append(dict(cohort=coh, prefire=pre, status=st, out5=g, k=int(r[g]), n=n, pct=100 * p, lo=100 * lo, hi=100 * hi))
shares = pd.DataFrame(recs)
shares.round(2).to_csv(OUT / "figCD_data_postfire_shares.csv", index=False)

SHORT = {"Forest": "Forest", "Non-forest natural": "Non-forest\nnatural", "Oil palm": "Oil palm",
         "Pulpwood": "Pulpwood", "Other agriculture": "Other\nagriculture", "Other": "Other"}


def postfire_fig(pre, fname, letter):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(ORDER)); w = 0.38
    for ax, coh in zip(axes, [2015, 2019]):
        s = shares[(shares.cohort == coh) & (shares.prefire == pre)]
        for j, (st, col) in enumerate([("Burned", BURNED_C), ("Unburned", UNBURNED_C)]):
            ss = s[s.status == st].set_index("out5").reindex(ORDER)
            n = int(ss.n.iloc[0])
            ax.bar(x + (j - 0.5) * w, ss.pct, w * 0.92, color=col,
                   label=f"{st} in {coh} (n = {n:,} points)")
            ax.errorbar(x + (j - 0.5) * w, ss.pct, yerr=[np.clip(ss.pct - ss.lo, 0, None), np.clip(ss.hi - ss.pct, 0, None)],
                        fmt="none", ecolor="#333333", lw=0.8, capsize=2)
            if st == "Burned":
                for xi, v, h in zip(x, ss.pct, ss.hi):
                    ax.text(xi - 0.5 * w, h + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#7c2d12")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[g] for g in ORDER], fontsize=9)
        ax.set_title(f"{pre} in {coh - 1}  →  land cover in {coh + 5}")
        ax.set_ylim(0, 108); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("% of points")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


postfire_fig("Forest", "figC_postfire_forest.png", "C")
postfire_fig("Non-forest natural", "figD_postfire_nonforest.png", "D")
print(shares[shares.status == "Burned"].pivot_table(index=["cohort", "prefire"], columns="out5", values="pct").round(1).to_string())
print(shares.groupby(["cohort", "prefire", "status"]).n.first().to_string())
