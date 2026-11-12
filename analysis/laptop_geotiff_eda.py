# %% [markdown]
# # Kalimantan fires and land-cover change: laptop version (no Earth Engine)
# Basic exploratory analysis of the 2015 fire cohort using MapBiomas GeoTIFFs.
#
# Files to put in ./data (see download commands in the chat walkthrough):
#   indonesia_coverage_2010.tif   land cover 5 years before (optional, for "stable" check)
#   indonesia_coverage_2014.tif   land cover the year BEFORE the fire  (t-1)
#   indonesia_coverage_2020.tif   land cover 5 years AFTER the fire     (t+5)
#   burned_area_2010.tif ... burned_area_2014.tif   (optional: earlier fires, for the "first fire" rule)
#   burned_area_2015.tif          the fire year (t)
#
# Idea: instead of loading all ~1.5 billion 30 m pixels of Kalimantan, we read every
# STEP-th pixel in each direction (a systematic sample on a regular grid). Every
# sampled pixel keeps its full 30 m information; we just look at fewer of them.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)
BBOX = (108.5, -4.3, 119.1, 4.4)   # Kalimantan (lon_min, lat_min, lon_max, lat_max); Malaysian Borneo is empty in MapBiomas Indonesia
STEP = 5                           # read every 5th pixel -> ~150 m grid spacing. Use 10 if your laptop struggles.
FIRE_YEAR, BEFORE, AFTER = 2015, 2014, 2020

GROUP = {**{c: "Forest" for c in [3, 5, 6, 76]},
         **{c: "Non-forest natural" for c in [4, 10, 11, 12, 13]},
         35: "Oil palm", 9: "Pulpwood",
         **{c: "Other agriculture" for c in [14, 15, 18, 19, 20, 21, 36, 39, 40, 41]}}
ORDER = ["Forest", "Non-forest natural", "Oil palm", "Pulpwood", "Other agriculture", "Other"]
COLORS = dict(zip(ORDER, ["#1f7a3a", "#a6c96a", "#d4271e", "#7a5900", "#e9b44c", "#9aa0a6"]))

def read(name):
    """Read the Kalimantan box from a GeoTIFF, keeping every STEP-th pixel (nearest = no averaging)."""
    with rasterio.open(DATA / name) as src:
        win = from_bounds(*BBOX, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, out_shape=(int(win.height) // STEP, int(win.width) // STEP),
                       resampling=Resampling.nearest, boundless=True, fill_value=0)
        res_m = abs(src.res[0]) * 111_000 * STEP   # approx. spacing between sampled pixels in metres
    return arr, res_m

def burned(year):
    f = DATA / f"burned_area_{year}.tif"
    return (read(f.name)[0] > 0) if f.exists() else None

# %% Step 1 – load ----------------------------------------------------------------
lc_before, spacing_m = read(f"indonesia_coverage_{BEFORE}.tif")
lc_after, _ = read(f"indonesia_coverage_{AFTER}.tif")
fire = burned(FIRE_YEAR)
assert lc_before.shape == lc_after.shape == fire.shape, "rasters are not on the same grid"
print(f"Grid: {lc_before.shape}, one sampled pixel every ~{spacing_m:.0f} m")
print("Land-cover codes present (check against the legend!):", sorted(np.unique(lc_before)))

# 'first fire' rule (only if the earlier fire files were downloaded)
earlier = [b for b in (burned(y) for y in range(FIRE_YEAR - 5, FIRE_YEAR)) if b is not None]
first_fire_ok = ~np.any(earlier, axis=0) if earlier else np.ones_like(fire)
print(f"Using {len(earlier)} earlier fire years for the first-fire rule")

# %% Step 2 – build a table: one row per sampled pixel inside Kalimantan ----------------
valid = (lc_before > 0) & (lc_after > 0)                 # 0 = no data / outside Indonesia
df = pd.DataFrame({
    "pre": pd.Series(lc_before[valid]).map(GROUP).fillna("Other"),
    "post": pd.Series(lc_after[valid]).map(GROUP).fillna("Other"),
    "burned": fire[valid],
    "first_fire_ok": first_fire_ok[valid],
})
# distance from every pixel to the nearest oil palm in the year before the fire
dist_m = ndimage.distance_transform_edt(lc_before != 35) * spacing_m
df["dist_km"] = dist_m[valid] / 1000
df["dist_band"] = pd.cut(df.dist_km, [-1, 2, 5, 15, np.inf], labels=["0–2 km", "2–5 km", "5–15 km", "≥15 km"])
px_ha = (spacing_m ** 2) / 10_000                          # each sampled pixel stands for this many hectares
print(f"{len(df):,} sampled pixels; each represents ~{px_ha:.1f} ha")

# %% Table 1 – how much burned in 2015, by land cover the year before ------------------------
t1 = (df[df.burned].groupby("pre").size().reindex(ORDER).fillna(0).rename("pixels").to_frame())
t1["area_kha"] = (t1.pixels * px_ha / 1000).round(1)
t1["share_%"] = (100 * t1.pixels / t1.pixels.sum()).round(1)
t1.to_csv(OUT / "table1_burned_2015_by_prefire.csv")
print("\nTABLE 1: area burned in 2015 by land cover in 2014\n", t1)

# %% Figure 1 – same as Table 1, as a bar chart ------------------------------------------------
ax = t1.area_kha.plot(kind="bar", color=[COLORS[c] for c in t1.index], figsize=(7, 4))
ax.set(ylabel="Burned area in 2015 (thousand ha)", xlabel="Land cover in 2014",
       title="What was burning in Kalimantan in 2015?")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(OUT / "fig1_burned_by_prefire.png", dpi=200); plt.close()

# %% Figure 2 + Table 2 – where did land end up by 2020? Burned vs unburned ----------------------
study = df[df.pre.isin(["Forest", "Non-forest natural"]) & df.first_fire_ok]
study = study.assign(status=np.where(study.burned, "Burned 2015", "Unburned"))
t2 = (pd.crosstab([study.pre, study.status], study.post, normalize="index") * 100).round(1)
t2 = t2.reindex(columns=[c for c in ORDER if c in t2.columns])
t2.to_csv(OUT / "table2_landcover_2020_shares.csv")
print("\nTABLE 2: % of each group in each land cover in 2020\n", t2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, pre in zip(axes, ["Forest", "Non-forest natural"]):
    t2.loc[pre].plot(kind="bar", stacked=True, ax=ax, legend=False, color=[COLORS[c] for c in t2.columns])
    ax.set(title=f"Land cover in 2014: {pre}", ylabel="% of area in 2020", xlabel=""); ax.tick_params(axis="x", rotation=0)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, title="Land cover 2020", loc="center right")
fig.suptitle("Five years after the 2015 fires: burned vs unburned land")
plt.tight_layout(rect=[0, 0, 0.82, 0.93]); plt.savefig(OUT / "fig2_outcomes_2020.png", dpi=200); plt.close()

# %% Table 3 + Figure 3 – conversion to oil palm by distance to existing plantations ------------
t3 = (study.assign(palm=study.post.eq("Oil palm"))
           .groupby(["pre", "dist_band", "status"], observed=True)
           .agg(n=("palm", "size"), pct_oil_palm=("palm", lambda x: 100 * x.mean())).round(1))
t3.to_csv(OUT / "table3_palm_by_distance.csv")
print("\nTABLE 3: % oil palm in 2020 by distance to oil palm in 2014\n", t3)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, pre in zip(axes, ["Forest", "Non-forest natural"]):
    t3.loc[pre].pct_oil_palm.unstack("status").plot(kind="bar", ax=ax, color=["#d4271e", "#4a6fa5"])
    ax.set(title=f"Land cover in 2014: {pre}", xlabel="Distance to nearest oil palm in 2014", ylabel="% oil palm in 2020")
    ax.tick_params(axis="x", rotation=0)
fig.suptitle("Is the burned/unburned gap just about being near plantations?")
plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(OUT / "fig3_palm_by_distance.png", dpi=200); plt.close()

# %% Figure 4 – map of the 2015 burned area coloured by what it was in 2014 ----------------------
codes = {g: i for i, g in enumerate(ORDER, start=1)}
pre_img = np.vectorize(lambda c: codes.get(GROUP.get(c, "Other"), 6))(lc_before)
show = np.where(fire & (lc_before > 0), pre_img, np.nan)
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(np.where(lc_before > 0, 1, np.nan), cmap="Greys", vmin=0, vmax=4, extent=(BBOX[0], BBOX[2], BBOX[1], BBOX[3]))
from matplotlib.colors import ListedColormap
ax.imshow(show, cmap=ListedColormap([COLORS[g] for g in ORDER]), vmin=0.5, vmax=6.5,
          extent=(BBOX[0], BBOX[2], BBOX[1], BBOX[3]), interpolation="nearest")
ax.set(title="Area burned in 2015, coloured by land cover in 2014", xlabel="Longitude", ylabel="Latitude")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[g]) for g in ORDER], labels=ORDER, loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig4_map_burned_2015.png", dpi=200); plt.close()
print("Done. Figures and tables are in", OUT.resolve())
