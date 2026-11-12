# %% [markdown]
# # DSC 190 – EDA from the FAST Earth Engine run (random-point histograms)
# Input: data/fast_results.json  = {"c2015_11": {"n_points": 4000, "r": {"cohort":2015,"n":..,"hist":{code:count}}}, ...}
#        optional data/A_burned_area_by_prefire_class.csv (exact Figure 1 from the batch export)
# Each random point is an equal-probability draw from Kalimantan (538,047 km2), so every
# point represents AOI_KM2 / (total points drawn for that cohort) km2. No weights needed.
#
# Code layout (9 digits): pre | out5 | burned | burned_later | natural_at_t-5 | dist_band | palm_t4t5 | lag_clear | lag_palm
#   groups: 1 Forest, 2 Non-forest natural, 3 Oil palm, 4 Pulpwood, 5 Other agriculture, 6 Other
#   dist_band: 0 = <2 km, 1 = 2–5 km, 2 = 5–15 km, 3 = >=15 km from oil palm in t-1
#   lags: 0..5 years after the fire year, 6 = never

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)
AOI_KM2 = 538046.876
GROUP = {1: "Forest", 2: "Non-forest natural", 3: "Oil palm", 4: "Pulpwood",
         5: "Other agriculture", 6: "Other"}
ORDER = list(GROUP.values())
COLORS = dict(zip(ORDER, ["#1f7a3a", "#a6c96a", "#d4271e", "#7a5900", "#e9b44c", "#9aa0a6"]))
DBAND = {0: "0–2 km", 1: "2–5 km", 2: "5–15 km", 3: "≥15 km"}
Z = 1.96

# %% Load and decode -------------------------------------------------------------
raw = json.loads((DATA / "fast_results.json").read_text())
rows, drawn, errors = [], {}, []
for key, v in raw.items():
    if key.startswith("fig1"):
        continue
    coh = int(key.split("_")[0][1:])
    if "error" in v:
        errors.append((key, v["error"])); continue
    drawn[coh] = drawn.get(coh, 0) + v["n_points"]
    for code, cnt in v["r"]["hist"].items():
        rows.append((coh, int(code), cnt))
print("Batches with errors (excluded):", errors)
d = pd.DataFrame(rows, columns=["cohort", "code", "count"])
digit = lambda c, p: (c // 10 ** p) % 10
d = d.assign(prefire=digit(d.code, 8).map(GROUP), out5=digit(d.code, 7).map(GROUP),
             burned=digit(d.code, 6), burned_later=digit(d.code, 5), natural_t5=digit(d.code, 4),
             dist_band=digit(d.code, 3).map(DBAND), palm=digit(d.code, 2),
             lag_clear=digit(d.code, 1), lag_palm=digit(d.code, 0))
d["status"] = np.where(d.burned == 1, "Burned", "Unburned")
km2_per_pt = {c: AOI_KM2 / n for c, n in drawn.items()}
print("Points drawn per cohort:", drawn, "| km2 per point:", {c: round(v, 2) for c, v in km2_per_pt.items()})

def prop_ci(k, n):
    """Wilson 95% interval for a proportion."""
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den; h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)

def diff_ci(k1, n1, k0, n0):
    """Newcombe (Wilson-score) 95% interval for p1 - p0."""
    p1, l1, u1 = prop_ci(k1, n1); p0, l0, u0 = prop_ci(k0, n0)
    dlt = p1 - p0
    return dlt, dlt - np.sqrt((p1 - l1)**2 + (u0 - p0)**2), dlt + np.sqrt((u1 - p1)**2 + (p0 - l0)**2)

# %% Table 1 – sample sizes and estimated area --------------------------------------
t1 = d.groupby(["cohort", "prefire", "status"]).agg(points=("count", "sum")).reset_index()
t1["est_area_kha"] = t1.apply(lambda r: r.points * km2_per_pt[r.cohort] * 100 / 1000, axis=1).round(0)
nat5 = d[d.natural_t5 == 1].groupby(["cohort", "prefire", "status"])["count"].sum()
t1["pct_natural_since_t-5"] = (100 * t1.set_index(["cohort", "prefire", "status"]).index.map(nat5).fillna(0)
                               / t1.points).round(1)
t1.to_csv(OUT / "table1_sample_sizes.csv", index=False)
print("\nTABLE 1\n", t1.to_string(index=False))

# %% Figure 2 – land cover at t+5 ---------------------------------------------------
f2 = (d.groupby(["cohort", "prefire", "status", "out5"])["count"].sum()
       .groupby(level=[0, 1, 2]).transform(lambda x: x / x.sum()).rename("share").reset_index())
f2.to_csv(OUT / "fig2_data_outcome_shares.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, pre in zip(axes, ["Forest", "Non-forest natural"]):
    p = (f2[f2.prefire == pre].assign(col=lambda x: x.status + "\n" + x.cohort.astype(str))
         .pivot_table(index="col", columns="out5", values="share", aggfunc="sum")
         .reindex(columns=[o for o in ORDER if o in set(f2.out5)]).fillna(0))
    (100 * p).plot(kind="bar", stacked=True, ax=ax, legend=False, width=0.8, color=[COLORS[c] for c in p.columns])
    ax.set(title=f"Land cover before the fire (t−1): {pre}", xlabel="", ylabel="% of area at t+5")
    ax.tick_params(axis="x", rotation=0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, title="Land cover at t+5", loc="center right")
fig.suptitle("Where land ends up five years after the fire year (first-time burns vs. unburned land)")
plt.tight_layout(rect=[0, 0, 0.84, 0.94]); plt.savefig(OUT / "fig2_outcomes_t5.png", dpi=200); plt.close()

# %% Table 2 – P(stable oil palm at t+5), burned vs unburned -------------------------
def table2(df, spec):
    out = []
    for (coh, pre), g in df.groupby(["cohort", "prefire"]):
        b, u = g[g.burned == 1], g[g.burned == 0]
        kb, nb = b.loc[b.palm == 1, "count"].sum(), b["count"].sum()
        ku, nu = u.loc[u.palm == 1, "count"].sum(), u["count"].sum()
        pb, lb, ub = prop_ci(kb, nb); pu, lu, uu = prop_ci(ku, nu); dd, dl, du = diff_ci(kb, nb, ku, nu)
        out.append(dict(spec=spec, cohort=coh, prefire=pre, n_burned=nb, n_unburned=nu,
                        pct_burned=f"{100*pb:.1f} [{100*lb:.1f}, {100*ub:.1f}]",
                        pct_unburned=f"{100*pu:.1f} [{100*lu:.1f}, {100*uu:.1f}]",
                        diff_pp=f"{100*dd:.1f} [{100*dl:.1f}, {100*du:.1f}]",
                        ratio=round(pb / pu, 1) if pu else np.nan))
    return pd.DataFrame(out)
t2 = pd.concat([table2(d, "main"),
                table2(d[(d.burned == 1) | (d.burned_later == 0)], "controls never burned t+1..t+5"),
                table2(d[d.natural_t5 == 1], "natural vegetation since t-5")])
t2.to_csv(OUT / "table2_palm_burned_vs_unburned.csv", index=False)
print("\nTABLE 2\n", t2.to_string(index=False))

# %% Figure 3 – by distance to existing oil palm --------------------------------------
rows3 = []
for (pre, band, st), g in d.groupby(["prefire", "dist_band", "status"]):
    k, n = g.loc[g.palm == 1, "count"].sum(), g["count"].sum()
    p, lo, hi = prop_ci(k, n); rows3.append(dict(prefire=pre, dist_band=band, status=st, p=p, lo=lo, hi=hi, n=n))
f3 = pd.DataFrame(rows3); f3.to_csv(OUT / "fig3_data_palm_by_distance.csv", index=False)
bands = list(DBAND.values())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, pre in zip(axes, ["Forest", "Non-forest natural"]):
    for i, (st, col) in enumerate([("Burned", "#d4271e"), ("Unburned", "#4a6fa5")]):
        q = f3[(f3.prefire == pre) & (f3.status == st)].set_index("dist_band").reindex(bands)
        x = np.arange(len(bands)) + (i - 0.5) * 0.3
        ax.bar(x, 100 * q.p, width=0.3, color=col, label=st)
        ax.errorbar(x, 100 * q.p, yerr=[np.clip(100 * (q.p - q.lo), 0, None), np.clip(100 * (q.hi - q.p), 0, None)],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    nb = f3[(f3.prefire == pre) & (f3.status == "Burned")].set_index("dist_band").n.reindex(bands).fillna(0)
    nu = f3[(f3.prefire == pre) & (f3.status == "Unburned")].set_index("dist_band").n.reindex(bands).fillna(0)
    ax.set_xticks(range(len(bands)), [f"{b}\nn={int(x1)} / {int(x0)}" for b, x1, x0 in zip(bands, nb, nu)])
    ax.set(title=f"Before the fire: {pre}", xlabel="Distance to nearest oil palm in t−1", ylabel="% stable oil palm at t+5")
axes[0].legend()
fig.suptitle(f"Conversion to oil palm by distance to existing plantations (cohort {' + '.join(map(str, sorted(drawn)))}; 95% Wilson CI; n = burned / unburned points)")
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(OUT / "fig3_palm_by_distance.png", dpi=200); plt.close()

# %% Figure 4 – timing ---------------------------------------------------------------
conv = d[(d.burned == 1) & (d.palm == 1)]
def lag_share(col):
    return (conv.groupby(["prefire", col])["count"].sum().groupby(level=0).transform(lambda x: x / x.sum())
                .unstack(0).reindex(range(6)).fillna(0))
fc, fp = lag_share("lag_clear"), lag_share("lag_palm")
pd.concat({"left_natural": fc, "stable_oil_palm": fp}, axis=1).to_csv(OUT / "fig4_data_lag_shares.csv")
n_conv = conv.groupby("prefire")["count"].sum().to_dict()
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (q, ttl) in zip(axes, [(fc, "Year land first stops being natural vegetation"), (fp, "Year land becomes stable oil palm")]):
    (100 * q).plot(kind="bar", ax=ax, width=0.8, color=[COLORS[c] for c in q.columns], legend=False)
    ax.set(title=ttl, xlabel="Years after the fire year (0 = fire year)", ylabel="% of converted burned area"); ax.tick_params(axis="x", rotation=0)
axes[0].legend([f"{k} (n={v})" for k, v in n_conv.items()], title="Before the fire")
fig.suptitle(f"Timing of conversion among burned land that is oil palm at t+5 (cohort {' + '.join(map(str, sorted(drawn)))})")
plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(OUT / "fig4_lag_to_palm.png", dpi=200); plt.close()
print("\nConverted burned points by pre-fire group:", n_conv)

# %% Figure 1 – annual burned area by pre-fire land cover (exact, from export A if available)
a_path = DATA / "A_burned_area_by_prefire_class.csv"
fig1_keys = [k for k, v in raw.items() if k.startswith("fig1") and "r" in v]
if not a_path.exists() and fig1_keys:
    # fallback: random-point estimate. value 0 = not burned that year, 1..6 = pre-fire group of a burned point
    n_tot = sum(raw[k]["n_points"] for k in fig1_keys)
    ha_per_pt = AOI_KM2 * 100 / n_tot
    recs = []
    for k in fig1_keys:
        for yk, h in raw[k]["r"]["hist"].items():
            for g, cnt in h.items():
                if int(g) > 0:
                    recs.append(dict(year=int(yk[1:]), prefire=GROUP[int(g)], burned_ha=cnt * ha_per_pt, pts=cnt))
    a = pd.DataFrame(recs)
    f1 = (a.pivot_table(index="year", columns="prefire", values="burned_ha", aggfunc="sum")
           .reindex(index=range(2001, 2025), columns=[o for o in ORDER if o in set(a.prefire)]).fillna(0) / 1e6)
    f1.round(4).to_csv(OUT / "fig1_data_burned_Mha.csv")
    ax = f1.plot(kind="bar", stacked=True, figsize=(11, 4.5), width=0.85, color=[COLORS[c] for c in f1.columns])
    ax.set(xlabel="Fire year", ylabel="Burned area (million ha, estimated)",
           title=f"Burned area in Kalimantan by land cover in the year before the fire (estimate from {n_tot:,} random points)")
    ax.legend(title="Land cover in t−1", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(OUT / "fig1_burned_area_by_prefire.png", dpi=200); plt.close()
    print(f"\nFIGURE 1 (random-point estimate, {n_tot} points, {ha_per_pt:.0f} ha/point)\n", f1.round(3).to_string())
if a_path.exists():
    from collections import defaultdict
    CODE2G = {**{c: "Forest" for c in [3, 5, 6, 76]}, **{c: "Non-forest natural" for c in [4, 10, 11, 12, 13]},
              35: "Oil palm", 9: "Pulpwood", **{c: "Other agriculture" for c in [14, 15, 18, 19, 20, 21, 36, 39, 40, 41]}}
    a = pd.read_csv(a_path); a["prefire"] = a.lc_prev.map(CODE2G).fillna("Other")
    f1 = a.pivot_table(index="year", columns="prefire", values="burned_ha", aggfunc="sum").reindex(columns=[o for o in ORDER if o in set(a.prefire)]).fillna(0) / 1e6
    f1.round(4).to_csv(OUT / "fig1_data_burned_Mha.csv")
    ax = f1.plot(kind="bar", stacked=True, figsize=(11, 4.5), width=0.85, color=[COLORS[c] for c in f1.columns])
    ax.set(xlabel="Fire year", ylabel="Burned area (million ha)", title="Burned area in Kalimantan by land cover in the year before the fire")
    ax.legend(title="Land cover in t−1", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(OUT / "fig1_burned_area_by_prefire.png", dpi=200); plt.close()
    print("\nFIGURE 1 data (Mha)\n", f1.round(3).to_string())
print("Done ->", OUT.resolve())
