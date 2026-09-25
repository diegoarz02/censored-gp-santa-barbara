# %% [markdown]
# # Notebook 2 — Bayesian censored spatial modelling
#
# **Project**: Probabilistic soil-contamination mapping under detection-limit censoring at the
# Santa Bárbara mercury mine, Huancavelica, Peru.
#
# **Research question**
#
# > When a large fraction of environmental measurements arrive as "below the detection limit", what
# > can still be asserted about risk across a territory, and with how much confidence?
#
# **The conceptual answer that organises everything**: treating censoring changes a conclusion only
# when the regulatory threshold and the detection limit are of the same order of magnitude. Measured
# across the whole OEFA soil archive, the median ECA/LOD ratio is 220 for mercury, 35 for lead, 14
# for arsenic and **2.8 for cadmium**.
#
# **Contents**
#
# | Section | Milestone |
# |---|---|
# | 1–4 | Setup, exploratory analysis, spatial structure, anisotropy (H2) |
# | 5 | Prior elicitation and prior predictive checks (H3) |
# | 6 | Parameter recovery on simulated fields (H4) |
# | 7 | Main hierarchical censored GP models (H5) |
# | 8 | Masking experiment on real complete fields (H6) |
# | 9 | Spatial cross-validation on the real data (H7) |
# | 10 | Geochemical background versus mining input (H8) |
# | 11 | Exceedance maps (H9) |
# | 12 | Monitoring design (H10) |
# | 13 | Coregionalisation and preferential sampling (H11) |
# | 14 | Publication maps (H12) |

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260906
rng = np.random.default_rng(SEED)

PROJ = Path(r"c:\Users\diego\OneDrive\Documentos\ANALISIS BAYESIANO\proyecto_metales_bayesiano")
DIR_FINAL = PROJ / "data" / "final"
DIR_FIG = PROJ / "figuras"
DIR_OUT = PROJ / "outputs"
DIR_IDATA = DIR_OUT / "idata"
DIR_COV = PROJ / "data" / "fase1" / "covariables"
for d in (DIR_FIG, DIR_OUT, DIR_IDATA):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ / "notebooks" / "_src"))

CRS_UTM = "EPSG:32718"
ECA = {"As": (50.0, 50.0, 140.0), "Pb": (70.0, 140.0, 800.0), "Cd": (1.4, 10.0, 22.0),
       "Hg": (6.6, 6.6, 24.0), "Ba": (750.0, 500.0, 2000.0)}
TRUTH_FIELDS = ["Hg", "Pb", "As"]          # H6 masking experiment
MAIN_METALS = ["Hg", "Pb", "As", "Ba", "Cd"]  # H5 main models
N_CPU = os.cpu_count()

print("Python", sys.version.split()[0], "| PyMC", pm.__version__, "| ArviZ", az.__version__)
print("CPUs available:", N_CPU)
print("Seed:", SEED)

# %%
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM

# Kept as a dictionary and applied through a function because third-party imports overwrite it.
# `import preliz` silently sets savefig.bbox back to "tight", which crops each figure to its content
# and shifts a 190 mm figure to 193 mm. Any import that touches matplotlib styling has to be
# followed by apply_style().
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # 8 pt minimum at final print size, ticks and legends included. The figure audit
    # (src/auditar_figuras.py) measured the previous values on the PDF and they were below the
    # journal floor across the whole set.
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 0.9,
    "savefig.dpi": 600, "savefig.bbox": "standard",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 140,
}


def apply_style():
    plt.rcParams.update(RC)


apply_style()
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]


def save_fig(fig, stem, dpi_tiff=600):
    """Export to vector PDF and TIFF, and verify the width that actually reached disk."""
    apply_style()                      # defend against any import that reset the style
    fig.savefig(DIR_FIG / f"{stem}.pdf")
    fig.savefig(DIR_FIG / f"{stem}.tif", dpi=dpi_tiff, pil_kwargs={"compression": "tiff_lzw"})
    declared = fig.get_size_inches()[0] * 25.4
    from PIL import Image
    actual = Image.open(DIR_FIG / f"{stem}.tif").size[0] / dpi_tiff * 25.4
    flag = "" if abs(actual - declared) < 0.6 else f"  <-- CHECK, declared {declared:.0f} mm"
    print(f"saved: {stem} | width {actual:.0f} mm | {dpi_tiff} dpi{flag}")


def to_latex(df, name, caption, label, floatfmt=3):
    d2 = df.copy()
    for c in d2.columns:
        if pd.api.types.is_float_dtype(d2[c]):
            d2[c] = d2[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    body = d2.to_latex(index=False, escape=True,
                       column_format="l" * 2 + "r" * max(0, d2.shape[1] - 2))
    (DIR_FIG / f"{name}.tex").write_text(
        "\\begin{table}[htbp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label
        + "}\n\\footnotesize\n" + body + "\\end{table}\n", encoding="utf-8")


def save_table(df, name, caption=None, label=None, index=False, floatfmt=3):
    df.to_csv(DIR_FIG / f"{name}.csv", index=index, encoding="utf-8")
    if caption:
        to_latex(df.reset_index() if index else df, name, caption, label or name, floatfmt)
    print(f"table: {name}.csv" + (" + .tex" if caption else ""))


def scale_bar(ax, length_km=1.0, label=None, pos=(0.06, 0.04)):
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    dx, dy = np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]
    xa, ya = x0 + pos[0] * dx, y0 + pos[1] * dy
    ax.plot([xa, xa + length_km], [ya, ya], color="k", lw=1.4, solid_capstyle="butt", zorder=10)
    ax.text(xa + length_km / 2, ya + 0.015 * dy, label or f"{length_km:g} km",
            ha="center", va="bottom", fontsize=8, zorder=10)


def north_arrow(ax, pos=(0.88, 0.86), length=0.06):
    ax.annotate("N", xy=(pos[0], pos[1] + length), xytext=pos, xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="bottom", fontsize=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, mutation_scale=7))


def cache_idata(name, build, force=False):
    """Fit once and reuse. Keeps the notebook idempotent across reruns."""
    p = DIR_IDATA / f"{name}.nc"
    if p.exists() and not force:
        print(f"[cache] loaded {name}")
        return az.from_netcdf(p)
    t0 = time.time()
    idata = build()
    idata.to_netcdf(p)
    print(f"[fit] {name} in {time.time()-t0:.0f} s")
    return idata

# %% [markdown]
# ## 2. Data

# %%
df = pd.read_csv(DIR_FINAL / "santa_barbara_soil.csv")
workings = pd.read_csv(DIR_FINAL / "mine_workings.csv")
print("rows:", len(df), "| locations:", df.location_id.nunique(), "| analytes:", df.analyte.nunique())

# One row per location with the covariates
loc = (df.drop_duplicates("location_id")
       .set_index("location_id")[["easting", "northing", "lon", "lat", "elev_oefa", "elev_dem",
                                  "slope_deg", "aspect_deg", "stratum", "sample_type", "sector",
                                  "dist_waste_dump_m", "dist_adit_m", "dist_city_m", "coord_flag"]]
       .sort_index())
print("locations table:", loc.shape)

# Coordinates in kilometres, centred: the GP length scale is then read directly in km
XY = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
XY = XY - XY.mean(0)
DIST = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1))
iu = np.triu_indices(len(XY), 1)
print(f"pairwise distance (km): min {DIST[iu].min():.3f}, median {np.median(DIST[iu]):.3f}, "
      f"max {DIST[iu].max():.3f}")

# Covariates for the GP mean, standardised
COVARS = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
loc["log_dist_dump"] = np.log(loc["dist_waste_dump_m"].clip(lower=1.0))
loc["log_dist_city"] = np.log(loc["dist_city_m"])
COV = loc[COVARS].to_numpy(float)
COV = (COV - COV.mean(0)) / COV.std(0)
print("covariates (standardised):", COVARS)

IS_BACKGROUND = (loc["stratum"] == "background").to_numpy()
IS_COMPOSITE = (loc["sample_type"] == "composite").to_numpy()
IS_UNLABELLED = (loc["stratum"] == "unlabelled").to_numpy()
print(f"background {IS_BACKGROUND.sum()} | potential interest {(~IS_BACKGROUND & ~IS_UNLABELLED).sum()} "
      f"| unlabelled {IS_UNLABELLED.sum()}")
print("stratum equals support:", bool((IS_BACKGROUND == IS_COMPOSITE).all()))


def metal_arrays(analyte):
    """Return the observation arrays for one analyte, aligned with the location table."""
    g = df[df.analyte == analyte].set_index("location_id").loc[loc.index]
    cens = g["censored"].to_numpy(bool)
    val = g["value"].to_numpy(float)
    lod = float(g["reported_limit"].iloc[0]) if np.isfinite(g["reported_limit"].iloc[0]) else np.nan
    res = float(g["resolution"].iloc[0])
    return dict(value=val, censored=cens, lod=lod, resolution=res,
                log_value=np.where(cens, np.nan, np.log(np.where(cens, 1.0, val))),
                eca=ECA.get(analyte, (np.nan,) * 3))

# %% [markdown]
# ## 3. Exploratory analysis (H2)
#
# ### 3.1 Descriptive statistics by stratum
#
# The stratified design is the reason this site can answer a question Cuajone could not: it lets the
# geochemical background be estimated from data rather than assumed.

# %%
rows = []
for a in MAIN_METALS + ["Co", "Sb", "Ag", "Bi", "Ni"]:
    for st in ["background", "potential_interest", "all"]:
        g = df[df.analyte == a]
        if st != "all":
            g = g[g.stratum == st]
        if not len(g):
            continue
        q = g["value"].dropna()
        thr = ECA.get(a, (np.nan,))[0]
        rows.append({
            "analyte": a, "stratum": st, "n": len(g),
            "pct_censored": 100 * g["censored"].mean(),
            "gm_quantified": float(np.exp(np.log(q).mean())) if len(q) else np.nan,
            "median_quantified": q.median() if len(q) else np.nan,
            "max_quantified": q.max() if len(q) else np.nan,
            "sd_log": float(np.std(np.log(q))) if len(q) > 2 else np.nan,
            "pct_above_eca": 100 * (g["value"] > thr).sum() / len(g) if np.isfinite(thr) else np.nan,
        })
desc = pd.DataFrame(rows).round(3)
print(desc[desc.stratum == "all"].to_string(index=False))
save_table(desc, "TABLE4_descriptives_by_stratum",
           caption="Descriptive statistics by analyte and sampling stratum. Concentrations in "
                   "mg kg$^{-1}$; sd$_{\\log}$ is the standard deviation of the natural logarithm of "
                   "the quantified values.",
           label="tab:desc")

# %% [markdown]
# ### 3.2 A confounding the design forces on us
#
# The background points are not interspersed among the potential-interest points: they form two
# compact clusters. So the stratum is partly a *location* label, and the stratum effect will be
# partly absorbed by the spatial field. This is quantified rather than asserted.

# %%
def _n_clusters(d, thresh):
    """Number of single-linkage clusters at a distance threshold. Descriptive only."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dd = d.copy()
    np.fill_diagonal(dd, 0.0)
    return int(fcluster(linkage(squareform(dd, checks=False), "single"), thresh, "distance").max())


# Nearest neighbour within the same stratum against nearest in the other stratum. If the background
# points were interspersed the two would be similar; if segregated, the same-stratum neighbour is
# much closer.
D_ns = DIST.copy()
np.fill_diagonal(D_ns, np.inf)
nn_same_bg = D_ns[np.ix_(IS_BACKGROUND, IS_BACKGROUND)].min(1)
nn_other_bg = D_ns[np.ix_(IS_BACKGROUND, ~IS_BACKGROUND)].min(1)
seg_ratio = float(np.median(nn_other_bg) / np.median(nn_same_bg))
n_bg_clusters = _n_clusters(DIST[np.ix_(IS_BACKGROUND, IS_BACKGROUND)], 0.3)

print(f"background point, nearest background neighbour : median {np.median(nn_same_bg)*1000:6.0f} m")
print(f"background point, nearest other-stratum point  : median {np.median(nn_other_bg)*1000:6.0f} m")
print(f"segregation ratio                              : {seg_ratio:.1f}x")
print(f"background clusters (single linkage at 300 m)  : {n_bg_clusters}")
print("\nThe background points sit in a small number of compact clusters rather than being\n"
      "interspersed among the potential-interest points, so the stratum label carries spatial\n"
      "information as well as design information. The model can still separate the two because the\n"
      "spatial field is continuous while the stratum enters as a step, but that separation rests on\n"
      "the smoothness assumption. It is reported as a limitation, not as an estimate free of doubt.")

# %% [markdown]
# ### 3.3 Distributions and log-normality

# %%
from scipy import stats

# One analyte per file: at most two panels per figure, and a histogram per analyte is easier to
# place in a manuscript than a six-panel block that has to be reproduced at full page width.
for a in MAIN_METALS + ["Sb"]:
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.72), constrained_layout=True)
    g = df[df.analyte == a]
    q = np.log(g["value"].dropna())
    ax.hist(q, bins=18, color=OI[0], edgecolor="k", linewidth=0.3, alpha=0.85)
    thr = ECA.get(a, (np.nan,))[0]
    if np.isfinite(thr):
        ax.axvline(np.log(thr), color=OI[1], lw=1.1, ls="--", label=f"ECA {thr:g}")
    lod = g["reported_limit"].iloc[0]
    if np.isfinite(lod):
        ax.axvline(np.log(lod), color="0.3", lw=1.1, ls=":", label=f"LOD {lod:g}")
    ax.set_xlabel(f"log {a} (mg kg$^{{-1}}$)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
    ax.text(0.97, 0.95, f"{a}\n{100 * g['censored'].mean():.0f}% censored",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    save_fig(fig, f"FIG2_{a}_log_distribution")
    plt.show()

# %%
for a in TRUTH_FIELDS:
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.85), constrained_layout=True)
    q = np.log(df[df.analyte == a]["value"].dropna())
    stats.probplot(q, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=2.2, markerfacecolor=OI[0],
                          markeredgecolor="k", markeredgewidth=0.2, linestyle="none")
    ax.get_lines()[1].set(color=OI[1], lw=1.0)
    ax.set_title("")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel(f"Observed log {a}")
    sw = stats.shapiro(q)
    ax.text(0.03, 0.95, f"{a}\nShapiro-Wilk p = {sw.pvalue:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    save_fig(fig, f"FIG3_{a}_qq")
    plt.show()
print("\nThese three analytes have no censored values at all, so unlike the previous phase of this\n"
      "project the quantile plot uses the complete distribution and not a truncated upper tail.")

# %% [markdown]
# ## 4. Spatial structure (H2)
#
# ### 4.1 Empirical variogram
#
# Computed on the four analytes with **zero censoring**, where no substitution is involved and the
# empirical variogram is therefore unbiased. `use_nugget=True` is essential: `scikit-gstat` fixes
# the nugget at zero by default, which forces the fitted model through the origin and understates
# the noise fraction.

# %%
import skgstat as skg

VARIO_METALS = ["Hg", "Pb", "As", "Ba"]
vario_rows, variograms = [], {}
for a in VARIO_METALS:
    y = np.log(metal_arrays(a)["value"])
    for model in ["exponential", "spherical", "matern"]:
        V = skg.Variogram(XY, y, model=model, n_lags=15, maxlag=0.5 * DIST[iu].max(),
                          use_nugget=True, normalize=False)
        d = V.describe()
        vario_rows.append({"analyte": a, "model": model, "nugget": d["nugget"], "sill": d["sill"],
                           "effective_range_km": d["effective_range"],
                           "nugget_ratio": d["nugget"] / d["sill"] if d["sill"] else np.nan,
                           "rmse": V.rmse})
        if model == "exponential":
            variograms[a] = V
vario = pd.DataFrame(vario_rows).round(4)
print(vario.to_string(index=False))
save_table(vario, "TABLE5_variogram_fits",
           caption="Empirical variogram fits on the log-concentration of the four uncensored "
                   "analytes. The nugget ratio is the proportion of variance not spatially "
                   "structured.", label="tab:vario")

best = vario.loc[vario.groupby("analyte")["rmse"].idxmin()]
print("\nbest-fitting model per analyte:")
print(best[["analyte", "model", "effective_range_km", "nugget_ratio", "rmse"]].to_string(index=False))

# %% [markdown]
# ### 4.2 Directional variogram
#
# `skgstat.DirectionalVariogram` fails with recent SciPy releases, so the direction-restricted
# Matheron estimator is implemented directly: for each pair the azimuth is computed, pairs are kept
# when their direction falls within a tolerance of the target azimuth, and the classic estimator
# $\hat\gamma(h)=\frac{1}{2N(h)}\sum (z_i-z_j)^2$ is evaluated on the retained pairs.

# %%
def directional_variogram(xy, z, azimuth_deg, tol_deg=22.5, n_lags=12, maxlag=None):
    """Matheron estimator restricted to pairs within tol_deg of the target azimuth."""
    i, j = np.triu_indices(len(xy), 1)
    dv = xy[j] - xy[i]
    h = np.hypot(dv[:, 0], dv[:, 1])
    # azimuth measured clockwise from north, folded to [0, 180)
    az_ = (np.degrees(np.arctan2(dv[:, 0], dv[:, 1])) + 180.0) % 180.0
    target = azimuth_deg % 180.0
    diff = np.minimum(np.abs(az_ - target), 180.0 - np.abs(az_ - target))
    keep = diff <= tol_deg
    maxlag = maxlag or 0.5 * h.max()
    edges = np.linspace(0, maxlag, n_lags + 1)
    sq = 0.5 * (z[i] - z[j]) ** 2
    out = []
    for k in range(n_lags):
        m = keep & (h > edges[k]) & (h <= edges[k + 1])
        out.append({"lag_km": 0.5 * (edges[k] + edges[k + 1]), "gamma": sq[m].mean() if m.sum() else np.nan,
                    "n_pairs": int(m.sum())})
    return pd.DataFrame(out)


AZIMUTHS = [0, 45, 90, 135]

# The maximum lag has to be common to all directions, and that is dictated by the *narrow* side of
# the domain. Santa Bárbara is 2.37 km east-west and 5.37 km north-south, so beyond about 1.2 km
# there are simply no east-west pairs: the domain is not that wide. Comparing "sills" across
# directions over a lag axis that only some directions can reach measures the shape of the survey
# area, not anisotropy of the field. Both versions are computed below to make that explicit.
EXT_EW = float(np.ptp(XY[:, 0]))
EXT_NS = float(np.ptp(XY[:, 1]))
MAXLAG_COMMON = 0.5 * min(EXT_EW, EXT_NS)
print(f"domain extent: {EXT_EW:.2f} km E-W x {EXT_NS:.2f} km N-S")
print(f"common maximum lag (half the narrow side): {MAXLAG_COMMON:.2f} km")

dir_rows = []
for a in VARIO_METALS:
    y = np.log(metal_arrays(a)["value"])
    for azm in AZIMUTHS:
        for tag, ml in [("naive", 0.5 * DIST[iu].max()), ("comparable", MAXLAG_COMMON)]:
            dv = directional_variogram(XY, y, azm, n_lags=12 if tag == "naive" else 8, maxlag=ml)
            dv["analyte"], dv["azimuth"], dv["lag_scheme"] = a, azm, tag
            dir_rows.append(dv)
dirvar = pd.concat(dir_rows, ignore_index=True)
save_table(dirvar.round(4), "TABLE_S1_directional_variogram")


def anisotropy_table(scheme, q=0.6):
    out = []
    for a in VARIO_METALS:
        s, npair = {}, {}
        for azm in AZIMUTHS:
            g = dirvar[(dirvar.analyte == a) & (dirvar.azimuth == azm) & (dirvar.lag_scheme == scheme)]
            outer = g[g.lag_km > g.lag_km.quantile(q)]
            s[azm] = outer["gamma"].mean()
            npair[azm] = int(outer["n_pairs"].sum())
        vals = np.array([s[k] for k in AZIMUTHS], float)
        out.append({"analyte": a, **{f"gamma_{k}deg": s[k] for k in AZIMUTHS},
                    "min_pairs_outer": min(npair.values()),
                    "anisotropy_ratio": np.nanmax(vals) / np.nanmin(vals)})
    return pd.DataFrame(out).round(3)


aniso_naive = anisotropy_table("naive", q=0.66)
aniso = anisotropy_table("comparable", q=0.6)
print("\nNaive version (each direction taken to half its own maximum lag):")
print(aniso_naive.to_string(index=False))
print("\nComparable version (all directions capped at the common maximum lag):")
print(aniso.to_string(index=False))
save_table(aniso, "TABLE6_anisotropy",
           caption="Directional variogram sills and anisotropy ratios for the uncensored analytes, "
                   "with all azimuths restricted to a common maximum lag of "
                   f"{MAXLAG_COMMON:.2f} km imposed by the east-west extent of the survey area.",
           label="tab:aniso")
save_table(aniso_naive, "TABLE_S2_anisotropy_naive")

ANISO_MAX = float(aniso["anisotropy_ratio"].max())
USE_ANISOTROPIC = ANISO_MAX > 2.0
worst = aniso.loc[aniso["anisotropy_ratio"].idxmax(), "analyte"]
print(f"\nmaximum anisotropy ratio with comparable lags: {ANISO_MAX:.2f} (analyte {worst})")
print(f"naive maximum would have been {aniso_naive['anisotropy_ratio'].max():.2f}, inflated by the "
      f"elongated shape of the survey area")
print(f"decision: use an {'anisotropic (ARD)' if USE_ANISOTROPIC else 'isotropic'} kernel "
      f"(threshold fixed at 2.0 before inspecting the data)")

# Which direction is most continuous? The smallest semivariance marks the direction of greatest
# continuity, and that determines what an axis-aligned kernel has to represent.
mean_gamma = aniso[[f"gamma_{k}deg" for k in AZIMUTHS]].mean()
print("\nmean semivariance by azimuth across analytes:")
for k in AZIMUTHS:
    print(f"  {k:3d} deg : {mean_gamma[f'gamma_{k}deg']:.2f}")
print(f"most continuous direction: {AZIMUTHS[int(np.argmin(mean_gamma.values))]} deg "
      f"(north-south, along the valley and its drainage)")
print("An axis-aligned ARD kernel with separate length scales for easting and northing captures\n"
      "this. A full rotation angle was considered and rejected: with 114 points it adds a poorly\n"
      "identified parameter for a secondary effect.")

# %% [markdown]
# ### 4.3 Moran's I
#
# A second, model-free check of spatial autocorrelation, using a k-nearest-neighbour weights matrix.

# %%
from libpysal.weights import KNN
from esda.moran import Moran

W = KNN.from_array(XY, k=8)
W.transform = "r"
moran_rows = []
for a in VARIO_METALS + ["Cd"]:
    m = metal_arrays(a)
    if m["censored"].all():
        continue
    y = np.where(m["censored"], np.log(m["lod"] / 2), np.log(m["value"]))
    mi = Moran(y, W, permutations=999)
    moran_rows.append({"analyte": a, "morans_I": mi.I, "expected_I": mi.EI,
                       "z_score": mi.z_sim, "p_value": mi.p_sim,
                       "note": "L/2 substitution used only for this exploratory statistic"
                               if m["censored"].any() else ""})
moran = pd.DataFrame(moran_rows).round(4)
print(moran.to_string(index=False))
save_table(moran, "TABLE7_morans_i",
           caption="Moran's I of the log-concentration using an 8-nearest-neighbour row-standardised "
                   "weights matrix, with pseudo p-values from 999 permutations.", label="tab:moran")

# %% [markdown]
# ### 4.4 Variogram figure

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.34), constrained_layout=True)

ax = axes[0]
for k, a in enumerate(VARIO_METALS):
    V = variograms[a]
    ax.plot(V.bins, V.experimental, "o", ms=2.8, color=OI[k], mec="k", mew=0.2, label=a)
    xh = np.linspace(0.01, V.bins.max(), 200)
    ax.plot(xh, V.fitted_model(xh), "-", color=OI[k], lw=0.9)
ax.set_xlabel("Lag distance (km)")
ax.set_ylabel(r"Semivariance $\hat\gamma(h)$ (log mg kg$^{-1}$)$^2$")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=2)
ax.text(0.97, 0.05, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

ax = axes[1]
a = "Hg"
for k, azm in enumerate(AZIMUTHS):
    g = dirvar[(dirvar.analyte == a) & (dirvar.azimuth == azm)]
    ax.plot(g.lag_km, g.gamma, "-o", ms=2.6, lw=0.9, color=OI[k], mec="k", mew=0.2,
            label=f"{azm}$^\\circ$")
ax.set_xlabel("Lag distance (km)")
ax.set_ylabel(r"Semivariance $\hat\gamma(h)$ (log mg kg$^{-1}$)$^2$")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=2, title="Azimuth", title_fontsize=8)
ax.text(0.97, 0.05, "(b)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

save_fig(fig, "FIG4_variograms")
plt.show()

# %% [markdown]
# ### 4.5 What the exploratory analysis settles
#
# The numbers above fix three modelling decisions that the rest of the notebook inherits.

# %%
h2 = {
    "n_locations": int(len(XY)),
    "domain_extent_km": float(DIST[iu].max()),
    "median_pair_distance_km": float(np.median(DIST[iu])),
    "domain_extent_ew_km": EXT_EW, "domain_extent_ns_km": EXT_NS,
    "anisotropy_maxlag_common_km": MAXLAG_COMMON,
    "anisotropy_ratio_max": ANISO_MAX,
    "anisotropy_ratio_max_naive": float(aniso_naive["anisotropy_ratio"].max()),
    "kernel": "anisotropic ARD Matern 5/2" if USE_ANISOTROPIC else "isotropic Matern 5/2",
    "variogram_best": best.set_index("analyte")[["model", "effective_range_km", "nugget_ratio"]].to_dict("index"),
    "morans_I": moran.set_index("analyte")[["morans_I", "p_value"]].to_dict("index"),
    "background_segregation_ratio": seg_ratio,
}
(DIR_OUT / "h2_spatial_structure.json").write_text(json.dumps(h2, indent=2), encoding="utf-8")
print(json.dumps(h2, indent=2))
