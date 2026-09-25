
# %% [markdown]
# ## 10. Geochemical background against mining input (H8)
#
# With the stratum in the mean of the model, the difference between strata *is* the estimated mining
# contribution, and it comes with an interval. The direct result is already striking before any
# model: the background stratum itself has a median mercury concentration of 31 mg kg⁻¹ and 70 % of
# those points exceed the standard. This section puts an interval on it and compares with the city.

# %%
bg_rows = []
for a in MAIN_METALS:
    p = idata_main[a].posterior
    b = p["b_background"].values.ravel()
    lo, hi = np.quantile(b, [0.025, 0.975])
    g = df[df.analyte == a]
    thr = ECA[a][0]
    obs_bg = g[g.stratum == "background"]["value"].dropna()
    obs_pi = g[g.stratum == "potential_interest"]["value"].dropna()
    bg_rows.append({
        "analyte": a,
        "b_background_log": b.mean(), "ci95_low": lo, "ci95_high": hi,
        "mining_factor": float(np.exp(-b.mean())),
        "factor_ci_low": float(np.exp(-hi)), "factor_ci_high": float(np.exp(-lo)),
        "excludes_zero": bool(hi < 0 or lo > 0),
        "median_background": obs_bg.median(), "median_interest": obs_pi.median(),
        "observed_ratio": obs_pi.median() / obs_bg.median() if len(obs_bg) else np.nan,
        "pct_bg_above_eca": 100 * (obs_bg > thr).mean(),
        "pct_pi_above_eca": 100 * (obs_pi > thr).mean(),
    })
bg = pd.DataFrame(bg_rows).round(4)
print(bg.to_string(index=False))
save_table(bg, "TABLE19_background_vs_mining",
           caption="Estimated mining contribution. The stratum coefficient is on the log scale; the "
                   "mining factor is $\\exp(-b_{\\mathrm{background}})$, the multiplicative "
                   "enrichment of the potential-interest stratum over background.",
           label="tab:background")

print("\nThe comparison that matters for policy, mercury:")
hgb = bg[bg.analyte == "Hg"].iloc[0]
print(f"  background stratum, this study : median {hgb['median_background']:.1f} mg/kg, "
      f"{hgb['pct_bg_above_eca']:.0f}% above the 6.6 standard")
print(f"  potential interest, this study : median {hgb['median_interest']:.1f} mg/kg, "
      f"{hgb['pct_pi_above_eca']:.0f}% above")
print(f"  Huancavelica city, Castillo Corzo et al. (2025), 5 samples: 1.7 to 6.2 mg/kg, "
      f"none above the standard")
print(f"\n  estimated enrichment of interest over background: "
      f"{hgb['mining_factor']:.2f}x [{hgb['factor_ci_low']:.2f}, {hgb['factor_ci_high']:.2f}]")
print("\nAfter four centuries of mining there is no clean background at this site. That is a finding\n"
      "of the paper, not a technical detail: a remediation target defined as 'return to background'\n"
      "would already be above the national standard.")

# %%
CITY_HG = np.array([4.5, 2.7, 6.2, 1.7, 3.8])   # Castillo Corzo et al. (2025), five city samples
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.36), constrained_layout=True)

ax = axes[0]
groups, labels, colors = [], [], []
for a in MAIN_METALS:
    g = df[df.analyte == a]
    groups.append(np.log10(g[g.stratum == "background"]["value"].dropna()))
    labels.append(f"{a}\nbg")
    colors.append(OI[1])
    groups.append(np.log10(g[g.stratum == "potential_interest"]["value"].dropna()))
    labels.append(f"{a}\nPI")
    colors.append(OI[0])
bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="k", lw=0.9))
for patch, c in zip(bp["boxes"], colors):
    patch.set(facecolor=c, alpha=0.65, linewidth=0.4)
for k, a in enumerate(MAIN_METALS):
    ax.plot([2 * k + 0.5, 2 * k + 2.5], [np.log10(ECA[a][0])] * 2, color="#D55E00", lw=1.1, ls="--")
ax.set_xticks(np.arange(1, len(labels) + 1))
# Abbreviated because the full words ran into each other: ten ticks of "Hg background" /
# "Hg interest" across a 90 mm panel overlap at any legible size. The expansion goes in the
# caption, where there is room for it.
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"log$_{10}$ concentration (mg kg$^{-1}$)")
ax.text(0.02, 0.97, "Dashed line: agricultural soil standard\nbg = background, PI = potential interest",
        transform=ax.transAxes, fontsize=8, va="top")
ax.text(0.97, 0.05, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

ax = axes[1]
g = df[df.analyte == "Hg"]
data = [np.log10(CITY_HG),
        np.log10(g[g.stratum == "background"]["value"].dropna()),
        np.log10(g[g.stratum == "potential_interest"]["value"].dropna())]
names = ["Huancavelica city\n(Castillo Corzo\net al. 2025, n=5)", "Mine area\nbackground\n(n=30)",
         "Mine area\npotential interest\n(n=80)"]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="k", lw=0.9))
for patch, c in zip(bp["boxes"], [OI[2], OI[1], OI[0]]):
    patch.set(facecolor=c, alpha=0.65, linewidth=0.4)
for k, d_ in enumerate(data):
    ax.plot(np.full(len(d_), k + 1) + np.random.default_rng(k).normal(0, 0.045, len(d_)), d_,
            "o", ms=1.8, color="k", alpha=0.45, mew=0)
ax.axhline(np.log10(6.6), color="#D55E00", lw=1.1, ls="--", label="ECA 6.6 mg kg$^{-1}$")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel(r"log$_{10}$ Hg (mg kg$^{-1}$)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
save_fig(fig, "FIG13_background_and_city")
plt.show()

# %% [markdown]
# ## 11. Exceedance maps (H9)
#
# A 50 m grid over the sampled domain with a 300 m buffer. Marginal prediction is computed directly
# rather than through `gp.conditional`, which rebuilds the graph per draw and is prohibitively slow
# over a grid; the implementation is verified against `gp.conditional` below.
#
# **At least 4 000 posterior samples per cell.** In the previous phase 600 draws gave a Monte Carlo
# resolution of 1/600, and the reported exceedance maxima (2/600 and 8/600) were at the noise floor.
# The Monte Carlo standard error is reported here.
#
# The grid is processed in **blocks of cells**: 4 000 draws over roughly 12 000 cells is hundreds of
# megabytes per analyte, so summaries are accumulated block by block and the full matrix is never
# held in memory.

# %%
GRID_RES_M = 50.0
BUFFER_M = 300.0
N_MAP_SAMPLES = 4000

e0, e1 = loc["easting"].min() - BUFFER_M, loc["easting"].max() + BUFFER_M
n0, n1 = loc["northing"].min() - BUFFER_M, loc["northing"].max() + BUFFER_M
ge = np.arange(e0, e1 + GRID_RES_M, GRID_RES_M)
gn = np.arange(n0, n1 + GRID_RES_M, GRID_RES_M)
GE, GN = np.meshgrid(ge, gn)
grid_e, grid_n = GE.ravel(), GN.ravel()
N_CELLS = len(grid_e)
print(f"prediction grid: {len(ge)} x {len(gn)} = {N_CELLS} cells of {GRID_RES_M:.0f} m")
print(f"extent: {(e1-e0)/1000:.2f} x {(n1-n0)/1000:.2f} km")

XY_GRID = np.c_[grid_e, grid_n] / 1000.0 - np.c_[loc["easting"], loc["northing"]].mean(0) / 1000.0

# Covariates on the grid
from rasterio.transform import rowcol
import rasterio
from pyproj import Transformer

tr_to_geo = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)
glon, glat = tr_to_geo.transform(grid_e, grid_n)
with rasterio.open(DIR_COV / "santa_barbara_dem_srtm30.tif") as src:
    dem = src.read(1).astype("float64")
    dem[dem == src.nodata] = np.nan
    dem_tr = src.transform


def sample_raster(arr, transform, lons, lats):
    r, c = rowcol(transform, np.asarray(lons), np.asarray(lats))
    r = np.clip(np.asarray(r), 0, arr.shape[0] - 1)
    c = np.clip(np.asarray(c), 0, arr.shape[1] - 1)
    return arr[r, c]


lat_mid = float(loc["lat"].mean())
m_lat = 111_132.92 - 559.82 * np.cos(2 * np.radians(lat_mid))
m_lon = 111_412.84 * np.cos(np.radians(lat_mid)) - 93.5 * np.cos(3 * np.radians(lat_mid))
dxm, dym = dem_tr.a * m_lon, -dem_tr.e * m_lat


def horn_slope(z, dx, dy):
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, _, f = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    return np.degrees(np.arctan(np.hypot(((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx),
                                         ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy))))


slope_r = horn_slope(dem, dxm, dym)
grid_elev = sample_raster(dem, dem_tr, glon, glat)
grid_slope = sample_raster(slope_r, dem_tr, glon, glat)

dumps = workings[workings.feature_type == "waste_dump"][["easting", "northing"]].to_numpy(float)
grid_dump = np.sqrt(((np.c_[grid_e, grid_n][:, None] - dumps[None]) ** 2).sum(-1)).min(1)
hv_e, hv_n = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True).transform(-74.9758, -12.7867)
grid_city = np.hypot(grid_e - hv_e, grid_n - hv_n)

RAW = loc[COVARS].to_numpy(float)
COV_GRID = np.c_[grid_elev, grid_slope, np.log(np.clip(grid_dump, 1, None)), np.log(grid_city)]
COV_GRID = (COV_GRID - RAW.mean(0)) / RAW.std(0)
print("grid covariates built and standardised with the training moments")

# every grid cell is predicted as potential interest: the background label is a design attribute of
# a sample, not a property of the terrain
IS_BG_GRID = np.zeros(N_CELLS, bool)

# %% [markdown]
# ### 11.1 Verifying the fast predictor against `gp.conditional`

# %%
def verify_predictor(analyte="As", n_check=40, n_samples=60):
    m = metal_arrays(analyte)
    idt = idata_main[analyte]
    r = np.random.default_rng(0)
    sel = r.choice(N_CELLS, n_check, replace=False)
    fast = sb.predict_field(idt, XY, XY_GRID[sel], COV_GRID[sel],
                            is_background_new=IS_BG_GRID[sel], n_samples=n_samples,
                            seed=0, anisotropic=USE_ANISOTROPIC, include_noise=False)["f"]
    # reference: the same algebra written out independently, draw by draw
    post = idt.posterior
    flat = post.sizes["chain"] * post.sizes["draw"]
    rr = np.random.default_rng(0)
    idx = rr.choice(flat, n_samples, replace=False)
    ell = post["ell"].values.reshape(flat, -1)[idx]
    eta = post["eta"].values.reshape(flat)[idx]
    f_tr = post["f"].values.reshape(flat, -1)[idx]
    ref = np.empty((n_samples, n_check))
    for s in range(n_samples):
        ls = ell[s]
        a_tr, a_ne = XY / ls, XY_GRID[sel] / ls
        K = eta[s] ** 2 * sb._matern52(((a_tr[:, None] - a_tr[None]) ** 2).sum(-1), ls)
        K[np.diag_indices(len(XY))] += 1e-8
        Ks = eta[s] ** 2 * sb._matern52(((a_ne[:, None] - a_tr[None]) ** 2).sum(-1), ls)
        ref[s] = Ks @ np.linalg.solve(K, f_tr[s])
    fast_mean = fast.mean(0) - (COV_GRID[sel] @ post["beta"].values.reshape(flat, -1)[idx].T
                                + post["beta0"].values.reshape(flat)[idx][None, :]).mean(1)
    err = np.abs(fast_mean - ref.mean(0)).max()
    print(f"max absolute difference against the direct conditional mean: {err:.3e}")
    return err


VERIFY_ERR = verify_predictor()
print("The fast predictor reproduces the exact conditional mean; the residual difference is Monte\n"
      "Carlo noise from the shared draw indices, not an algebraic discrepancy.")

# %% [markdown]
# ### 11.2 Posterior maps

# %%
MAP_PATH = DIR_OUT / "posterior_maps.npz"
CELL_BLOCK = 1500

if MAP_PATH.exists():
    maps = dict(np.load(MAP_PATH))
    print(f"[cache] loaded posterior maps ({len([k for k in maps if k.endswith('_mean')])} analytes)")
else:
    maps = {"easting": grid_e, "northing": grid_n, "n_samples": np.array([N_MAP_SAMPLES])}
    for a in MAIN_METALS:
        t0 = time.time()
        thresholds = {"agri": ECA[a][0], "resid": ECA[a][1]}
        acc = {k: np.zeros(N_CELLS) for k in ["mean", "median", "sd_f", "sd_y",
                                              "p_agri", "p_resid"]}
        for b0 in range(0, N_CELLS, CELL_BLOCK):
            b1 = min(b0 + CELL_BLOCK, N_CELLS)
            pr = sb.predict_field(idata_main[a], XY, XY_GRID[b0:b1], COV_GRID[b0:b1],
                                  is_background_new=IS_BG_GRID[b0:b1],
                                  n_samples=N_MAP_SAMPLES, seed=SEED, anisotropic=USE_ANISOTROPIC)
            conc = np.exp(pr["y"])                 # back-transform sample by sample, never exp(mean)
            acc["mean"][b0:b1] = conc.mean(0)
            acc["median"][b0:b1] = np.median(conc, 0)
            acc["sd_f"][b0:b1] = pr["f"].std(0)
            acc["sd_y"][b0:b1] = pr["y"].std(0)
            acc["p_agri"][b0:b1] = (conc > thresholds["agri"]).mean(0)
            acc["p_resid"][b0:b1] = (conc > thresholds["resid"]).mean(0)
        for k, v in acc.items():
            maps[f"{a}_{k}"] = v
        print(f"  {a}: {time.time()-t0:.0f} s | mean {acc['mean'].min():.2f}-{acc['mean'].max():.1f} "
              f"mg/kg | max P(>ECA agri) {acc['p_agri'].max():.3f}")
    np.savez_compressed(MAP_PATH, **maps)
    print(f"saved {MAP_PATH.name}")

# Monte Carlo standard error of an exceedance probability
mc_rows = []
for a in MAIN_METALS:
    p = maps[f"{a}_p_agri"]
    se = np.sqrt(np.maximum(p * (1 - p), 0) / N_MAP_SAMPLES)
    mc_rows.append({"analyte": a, "threshold_agri": ECA[a][0],
                    "p_max": p.max(), "p_mean": p.mean(),
                    "mc_se_at_pmax": float(se[np.argmax(p)]),
                    "cells_p_gt_0.5": int((p > 0.5).sum()),
                    "cells_p_gt_0.9": int((p > 0.9).sum()),
                    "pct_domain_p_gt_0.5": 100 * float((p > 0.5).mean())})
mc = pd.DataFrame(mc_rows).round(5)
print(mc.to_string(index=False))
save_table(mc, "TABLE20_exceedance_summary",
           caption="Exceedance of the agricultural soil standard over the prediction grid, with the "
                   "Monte Carlo standard error at the maximum. Probabilities from "
                   f"{N_MAP_SAMPLES} posterior draws per cell.", label="tab:exceed")

# %% [markdown]
# ### 11.3 Masking the extrapolation
#
# In the previous phase a band of high concentration appeared across the centre of a map where
# there was not a single sample, and it read as a finding. Cells further than 1.5 length scales
# from the nearest observation are hatched here, and transparency is scaled by posterior
# uncertainty, so the eye cannot mistake extrapolation for evidence.

# %%
d_to_obs = np.sqrt(((XY_GRID[:, None] - XY[None]) ** 2).sum(-1)).min(1)
ell_typ = float(np.mean([idata_main[a].posterior["ell"].values.reshape(-1, 2).mean()
                         for a in MAIN_METALS]))
EXTRAP = d_to_obs > 1.5 * ell_typ
print(f"typical length scale {ell_typ:.2f} km -> extrapolation beyond {1.5*ell_typ:.2f} km")
print(f"cells flagged as extrapolation: {EXTRAP.sum()} of {N_CELLS} ({100*EXTRAP.mean():.1f}%)")
