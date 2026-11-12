# %% [markdown]
# # DSC 190 – Kalimantan post-fire land-cover trajectories: EDA
# Inputs (exported from kalimantan_fire_gee.js to Google Drive, put in ./data):
#   A_burned_area_by_prefire_class.csv   year, lc_prev, burned_ha
#   B_stratum_pixel_counts.csv           cohort, stratum, n_pixels
#   C_cohort_sample.csv                  cohort, stratum, burned, burned_later, dist_palm_m,
#                                        lc_m1, lc_0, lc_p1..lc_p5, lon, lat
# Outputs go to ./out (PNG figures + CSV tables).
# Run cell by cell in Jupyter / VS Code, or `python kalimantan_fire_eda.py`.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)
RNG = np.random.default_rng(190)
N_BOOT = 300  # bootstrap replicates for confidence intervals

# %% [markdown]
# ## 1. Land-cover class groups
# CHECK these codes against the STEP 0 printout and the official Collection 4.1 legend.
# Any code not listed is reported as "Unmapped" so nothing is silently miscounted.

# %%
GROUPS = {
    "Forest":              [3, 5, 6, 76],          # forest formation, mangrove, floodable, peat swamp forest
    "Non-forest natural":  [4, 10, 11, 12, 13],    # savanna, wetland, grassland, other natural vegetation (scrub)
    "Oil palm":            [35],
    "Pulpwood":            [9],
    "Other agriculture":   [14, 15, 18, 19, 20, 21, 36, 39, 40, 41],
    "Other":               [22, 23, 24, 25, 26, 29, 30, 31, 33],  # urban, bare, mining, aquaculture, water
}
CODE2GROUP = {c: g for g, codes in GROUPS.items() for c in codes}
OUTCOME_ORDER = ["Forest", "Non-forest natural", "Oil palm", "Pulpwood",
                 "Other agriculture", "Other", "Unmapped"]
COLORS = dict(zip(OUTCOME_ORDER, ["#1f7a3a", "#a6c96a", "#d4271e", "#7a5900",
                                   "#e9b44c", "#9aa0a6", "#000000"]))
PREFIRE = ["Forest", "Non-forest natural"]   # Decision B: study population

def to_group(s: pd.Series) -> pd.Series:
    return s.map(CODE2GROUP).fillna("Unmapped")

# %% [markdown]
# ## 2. Load and prepare the sample

# %%
area = pd.read_csv(DATA / "A_burned_area_by_prefire_class.csv")
counts = pd.read_csv(DATA / "B_stratum_pixel_counts.csv")
s = pd.read_csv(DATA / "C_cohort_sample.csv")

unmapped = sorted(set(pd.concat([s.filter(like="lc_").stack(), area.lc_prev]).astype(int))
                  - set(CODE2GROUP))
print("Codes not in GROUPS (fix the mapping if any are common):", unmapped)

# sampling weight = pixels in stratum / points sampled from stratum
n_samp = s.groupby(["cohort", "stratum"]).size().rename("n_sampled").reset_index()
w = counts.merge(n_samp, on=["cohort", "stratum"])
w["weight"] = w.n_pixels / w.n_sampled
s = s.merge(w[["cohort", "stratum", "weight"]], on=["cohort", "stratum"], how="left")
assert s.weight.notna().all(), "Some sampled strata have no pixel count – check export B"

for c in ["lc_m5", "lc_m1", "lc_0", "lc_p1", "lc_p2", "lc_p3", "lc_p4", "lc_p5"]:
    s[c + "_g"] = to_group(s[c])
s["prefire"] = s["lc_m1_g"]
s["outcome5"] = s["lc_p5_g"]
# "persistent" oil palm = classified oil palm in BOTH t+4 and t+5 (filters one-year noise)
s["palm5_stable"] = ((s.lc_p4 == 35) & (s.lc_p5 == 35)).astype(int)
s["dist_band"] = pd.cut(s.dist_palm_m, [-1, 2000, 5000, 15000, np.inf],
                        labels=["0–2 km", "2–5 km", "5–15 km", "≥15 km"])
def block_id(lon, lat, size_deg):
    return (np.floor(lon / size_deg).astype(int).astype(str) + "_" +
            np.floor(lat / size_deg).astype(int).astype(str))
s["cell"] = block_id(s.lon, s.lat, 0.09)     # ~10 km blocks (main)
s["cell50"] = block_id(s.lon, s.lat, 0.45)   # ~50 km blocks (sensitivity: big fire scars span many 10 km cells)
# natural vegetation already 5 years before the fire = not land that was recently cleared and regrew
s["stable_natural"] = s["lc_m5_g"].isin(PREFIRE).astype(int)
s["status"] = np.where(s.burned == 1, "Burned", "Unburned")

pop = s[s.prefire.isin(PREFIRE)].copy()
print(f"Sample: {len(s):,} points; study population (natural vegetation at t-1): {len(pop):,}")

# %% [markdown]
# ## 3. Weighted shares with spatial-block bootstrap CIs

# %%
def wmean(df, col):
    return np.average(df[col], weights=df.weight) if len(df) else np.nan

def block_boot(df, col, n_boot=N_BOOT):
    """95% CI for a weighted mean, resampling ~10 km cells (handles spatial clustering)."""
    if df.cell.nunique() < 5:
        return (np.nan, np.nan)
    per_cell = (df.assign(wy=df.weight * df[col])
                  .groupby("cell")[["wy", "weight"]].sum().to_numpy())
    # each replicate = how many times each cell is drawn (resampling cells with replacement)
    draws = RNG.multinomial(len(per_cell), np.full(len(per_cell), 1 / len(per_cell)), size=n_boot)
    stats = (draws @ per_cell[:, 0]) / (draws @ per_cell[:, 1])
    return tuple(np.percentile(stats, [2.5, 97.5]))

def block_boot_diff(df, col, cell_col="cell", n_boot=N_BOOT):
    """95% CI for weighted P(burned) - P(unburned), resampling blocks that contain BOTH groups."""
    d = df.assign(wy=df.weight * df[col])
    agg = d.pivot_table(index=cell_col, columns="burned", values=["wy", "weight"],
                        aggfunc="sum", fill_value=0)
    if agg.shape[0] < 5 or ("weight", 1) not in agg or ("weight", 0) not in agg:
        return (np.nan, np.nan)
    k = len(agg)
    draws = RNG.multinomial(k, np.full(k, 1 / k), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = (draws @ agg[("wy", 1)].to_numpy()) / (draws @ agg[("weight", 1)].to_numpy())
        pu = (draws @ agg[("wy", 0)].to_numpy()) / (draws @ agg[("weight", 0)].to_numpy())
    diff = 100 * (pb - pu)
    return tuple(np.nanpercentile(diff, [2.5, 97.5]))

# %% [markdown]
# ## Table 1 – sample sizes and population sizes

# %%
t1 = (pop.assign(ws=pop.weight * pop.stable_natural)
         .groupby(["cohort", "prefire", "status"])
         .agg(sample_points=("weight", "size"), est_pixels=("weight", "sum"), ws=("ws", "sum")))
t1["est_area_kha"] = t1.est_pixels * 0.09 / 1000   # 30 m pixel = 0.09 ha
t1["pct_natural_since_t-5"] = 100 * t1.ws / t1.est_pixels
t1 = t1.drop(columns=["est_pixels", "ws"]).round(1)
t1.to_csv(OUT / "table1_sample_sizes.csv")
print(t1)

# %% [markdown]
# ## Figure 1 – Annual burned area by pre-fire land cover (Kalimantan, 2001–2024)
# Question: how much of the burning happens on forest vs already non-forest land?

# %%
area["prefire"] = to_group(area.lc_prev)
f1 = (area.pivot_table(index="year", columns="prefire", values="burned_ha", aggfunc="sum")
          .reindex(columns=[g for g in OUTCOME_ORDER if g in set(area.prefire)]).fillna(0) / 1e6)
f1.to_csv(OUT / "fig1_data_burned_Mha.csv")
ax = f1.plot(kind="bar", stacked=True, figsize=(11, 4.5), width=0.85,
             color=[COLORS[c] for c in f1.columns])
ax.set(xlabel="Fire year", ylabel="Burned area (million ha)",
       title="Burned area in Kalimantan by land cover in the year before the fire")
ax.legend(title="Land cover in t−1", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "fig1_burned_area_by_prefire.png", dpi=200); plt.close()

# %% [markdown]
# ## Figure 2 – Land cover 5 years after the fire year, burned vs unburned
# Question: where does burned natural vegetation end up, compared with similar unburned land?

# %%
rows = []
for (coh, pre, st), g in pop.groupby(["cohort", "prefire", "status"]):
    shares = g.groupby("outcome5").weight.sum() / g.weight.sum()
    for o, v in shares.items():
        rows.append(dict(cohort=coh, prefire=pre, status=st, outcome=o, share=v))
f2 = pd.DataFrame(rows)
f2.to_csv(OUT / "fig2_data_outcome_shares.csv", index=False)

cohorts = sorted(pop.cohort.unique())
fig, axes = plt.subplots(1, len(PREFIRE), figsize=(12, 4.8), sharey=True)
for ax, pre in zip(axes, PREFIRE):
    d = (f2[f2.prefire == pre]
         .assign(col=lambda x: x.status + "\n" + x.cohort.astype(str))
         .pivot_table(index="col", columns="outcome", values="share", aggfunc="sum")
         .reindex(columns=[o for o in OUTCOME_ORDER if o in set(f2.outcome)]).fillna(0))
    d.plot(kind="bar", stacked=True, ax=ax, legend=False, width=0.8,
           color=[COLORS[c] for c in d.columns])
    ax.set(title=f"Pre-fire: {pre}", xlabel="", ylabel="Share of area at t+5")
    ax.tick_params(axis="x", rotation=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Land cover at t+5", loc="center right")
fig.suptitle("Five-year outcomes for first-fire cohorts vs unburned comparison land")
plt.tight_layout(rect=[0, 0, 0.84, 0.95]); plt.savefig(OUT / "fig2_outcomes_t5.png", dpi=200); plt.close()

# %% [markdown]
# ## Table 2 – P(persistent oil palm at t+5): burned vs unburned, with block-bootstrap CIs
# Also a sensitivity check that drops comparison pixels that burned later (t+1..t+5).

# %%
def table2(df, label):
    out = []
    for (coh, pre), g in df.groupby(["cohort", "prefire"]):
        b, u = g[g.burned == 1], g[g.burned == 0]
        pb, pu = wmean(b, "palm5_stable"), wmean(u, "palm5_stable")
        lo_b, hi_b = block_boot(b, "palm5_stable")
        lo_u, hi_u = block_boot(u, "palm5_stable")
        d10 = block_boot_diff(g, "palm5_stable", "cell")
        d50 = block_boot_diff(g, "palm5_stable", "cell50")
        out.append(dict(spec=label, cohort=coh, prefire=pre, n=len(g),
                        p_burned=pb, ci_burned=f"[{lo_b:.3f}, {hi_b:.3f}]",
                        p_unburned=pu, ci_unburned=f"[{lo_u:.3f}, {hi_u:.3f}]",
                        diff_pp=100 * (pb - pu),
                        diff_ci_10km=f"[{d10[0]:.1f}, {d10[1]:.1f}]",
                        diff_ci_50km=f"[{d50[0]:.1f}, {d50[1]:.1f}]",
                        ratio=pb / pu if pu else np.nan))
    return pd.DataFrame(out)

t2 = pd.concat([
    table2(pop, "main"),
    table2(pop[(pop.burned == 1) | (pop.burned_later == 0)], "controls never burned t+1..t+5"),
    table2(pop[pop.stable_natural == 1], "natural vegetation since t-5"),
])
t2.round(4).to_csv(OUT / "table2_palm_burned_vs_unburned.csv", index=False)
print(t2.round(4).to_string(index=False))

# %% [markdown]
# ## Figure 3 – Is the burned/unburned gap just "distance to existing plantations"?
# Same comparison within bands of distance to oil palm in t-1 (key confounder).

# %%
rows = []
for (pre, band, st), g in pop.groupby(["prefire", "dist_band", "status"], observed=True):
    lo, hi = block_boot(g, "palm5_stable")
    rows.append(dict(prefire=pre, dist_band=band, status=st,
                     p=wmean(g, "palm5_stable"), lo=lo, hi=hi, n=len(g)))
f3 = pd.DataFrame(rows)
f3.to_csv(OUT / "fig3_data_palm_by_distance.csv", index=False)

fig, axes = plt.subplots(1, len(PREFIRE), figsize=(12, 4.5), sharey=True)
bands = list(pop.dist_band.cat.categories)
for ax, pre in zip(axes, PREFIRE):
    for i, (st, col) in enumerate([("Burned", "#d4271e"), ("Unburned", "#4a6fa5")]):
        d = f3[(f3.prefire == pre) & (f3.status == st)].set_index("dist_band").reindex(bands)
        x = np.arange(len(bands)) + (i - 0.5) * 0.3
        ax.bar(x, 100 * d.p, width=0.3, color=col, label=st)
        ax.errorbar(x, 100 * d.p, yerr=[100 * (d.p - d.lo), 100 * (d.hi - d.p)],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(range(len(bands)), bands)
    ax.set(title=f"Pre-fire: {pre}", xlabel="Distance to nearest oil palm in t−1",
           ylabel="% oil palm at t+4 and t+5")
axes[0].legend()
fig.suptitle("Conversion to oil palm by distance to existing plantations (cohorts pooled, 95% block-bootstrap CI)")
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(OUT / "fig3_palm_by_distance.png", dpi=200); plt.close()

# %% [markdown]
# ## Figure 4 – How long after the fire does oil palm appear?
# Among burned pixels that are persistent oil palm at t+5: first year from which they stay oil palm.
# Short lags fit "clearing for planting"; long lags fit later/opportunistic use (suggestive only).

# %%
lc_cols = ["lc_0", "lc_p1", "lc_p2", "lc_p3", "lc_p4", "lc_p5"]
def first_stable_palm(row):
    vals = row[lc_cols].to_numpy()
    for k in range(len(vals)):
        if (vals[k:] == 35).all():
            return k
    return np.nan

NATURAL_CODES = set(GROUPS["Forest"]) | set(GROUPS["Non-forest natural"])
def first_non_natural(row):
    """First year (0..5) the pixel leaves natural vegetation = proxy for when it was CLEARED.
    Young oil palm can take years to be mapped as oil palm, so the palm lag alone overstates
    the gap between fire and planting; comparing both lags separates clearing from detection."""
    for k, c in enumerate(lc_cols):
        if row[c] not in NATURAL_CODES:
            return k
    return np.nan

conv = pop[(pop.burned == 1) & (pop.palm5_stable == 1)].copy()
conv["lag_palm"] = conv.apply(first_stable_palm, axis=1)
conv["lag_clear"] = conv.apply(first_non_natural, axis=1)

def lag_shares(col):
    return (conv.groupby(["prefire", col]).weight.sum()
                .groupby(level=0).transform(lambda x: x / x.sum()).unstack(0)
                .reindex(range(6)).fillna(0))
f4_clear, f4_palm = lag_shares("lag_clear"), lag_shares("lag_palm")
pd.concat({"left_natural": f4_clear, "stable_oil_palm": f4_palm}, axis=1).to_csv(OUT / "fig4_data_lag_shares.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (d, ttl) in zip(axes, [(f4_clear, "Year land first leaves natural vegetation"),
                               (f4_palm, "Year land becomes stable oil palm")]):
    (100 * d).plot(kind="bar", ax=ax, width=0.8, color=[COLORS[c] for c in d.columns], legend=False)
    ax.set(title=ttl, xlabel="Years after fire year (0 = fire year)", ylabel="% of converted area")
    ax.tick_params(axis="x", rotation=0)
axes[0].legend(title="Pre-fire")
fig.suptitle("Timing of conversion among burned land that is stable oil palm at t+5")
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(OUT / "fig4_lag_to_palm.png", dpi=200); plt.close()
print("Converted burned sample points per pre-fire group:", conv.groupby("prefire").size().to_dict())

print("Done. Figures and tables are in", OUT.resolve())
