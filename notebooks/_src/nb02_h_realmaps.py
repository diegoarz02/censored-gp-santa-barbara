
# %% [markdown]
# ## 14. Publication maps (H12) — SUPERSEDED by `src/build_maps_f3.py`
#
# **This section no longer owns the map figures.** From run F3 they are drawn by
# `src/build_maps_f3.py` under the H2 rules: locator placed by measuring where the data are, no
# summary box over the surface it exists to show, hatched extrapolation, 8 pt minimum, and the
# decision threshold as a colour-bar tick rather than floating text.
#
# It is kept because its numbers feed the summary, but its figures are overwritten. Two sources
# writing `FIG16`/`FIG17`/`FIG18` meant whichever ran last won: a notebook rerun silently
# reintroduced 31 text collisions that had already been fixed. **After any notebook run, execute
# `python src/build_maps_f3.py`.**
#
# Three versions are produced and kept, so the choice can be made with the outputs in view rather
# than in the abstract:
#
# * **Version A** — a real OpenStreetMap/CartoDB basemap via `contextily`, with hillshade and the
#   sampling points on top. Requires an internet connection and **attribution in the caption**.
# * **Version B** — no external dependency at all: hillshade from the downloaded DEM, INEI district
#   boundaries, and place names labelled from the OEFA field descriptions. Fully reproducible
#   offline, and the version to use if OpenStreetMap coverage of Huancavelica turns out to be thin.
# * **Version C** — version A plus streets, the Ichu river and place names from `osmnx`.
#
# Every version carries: the city of Huancavelica labelled, the sectors, the waste dumps, the 114
# sampling points distinguished by stratum and censoring status, a scale bar, a north arrow, the
# declared CRS, a latitude/longitude graticule and a Peru locator inset with Huancavelica
# highlighted.

# %%
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from matplotlib_scalebar.scalebar import ScaleBar

INTERNET_OK = True
try:
    import contextily as cx
except Exception as _e:                                             # noqa: BLE001
    cx = None
    INTERNET_OK = False
    print("contextily unavailable:", _e)

EXTENT = (float(e0), float(e1), float(n0), float(n1))
print(f"map extent UTM 18S: {EXTENT[0]:.0f}-{EXTENT[1]:.0f} E, {EXTENT[2]:.0f}-{EXTENT[3]:.0f} N")

# Hillshade from the DEM, reprojected onto the prediction grid
ls_ = LightSource(azdeg=315, altdeg=45)
dem_grid = grid_elev.reshape(len(gn), len(ge))
HILLSHADE = ls_.hillshade(dem_grid, vert_exag=2.0, dx=GRID_RES_M, dy=GRID_RES_M)

# Place names taken from the OEFA descriptions, positioned at the centroid of the points mentioning
# each one. Derived from the data, not typed in from a map.
PLACES = {}
for name, pattern in [("Santa Barbara", "Santa Bárbara"), ("Yanamina", "Yanamina"),
                      ("Suytococha", "Suytococha"), ("Carniceria", "Carnicería"),
                      ("Cumallipata", "Cumallipata")]:
    m_ = df.drop_duplicates("location_id")
    sel = m_["description"].astype(str).str.contains(pattern, case=False, na=False)
    if sel.sum() >= 3:
        PLACES[name] = (float(m_.loc[sel, "easting"].mean()), float(m_.loc[sel, "northing"].mean()))
print("place labels derived from the field descriptions:", list(PLACES))

HVCA = (float(hv_e), float(hv_n))


def peru_inset(fig, ax, loc_box=(0.02, 0.62, 0.26, 0.34)):
    """Locator inset of Peru with the Huancavelica region highlighted."""
    import geopandas as gpd
    axins = fig.add_axes([ax.get_position().x0 + loc_box[0] * ax.get_position().width,
                          ax.get_position().y0 + loc_box[1] * ax.get_position().height,
                          loc_box[2] * ax.get_position().width,
                          loc_box[3] * ax.get_position().height])
    try:
        dep = gpd.read_file(f"zip://{GEOREF / 'Departamental INEI 2023 geogpsperu SuyoPomalia.zip'}")
        dep = dep.to_crs("EPSG:4326")
        dep.plot(ax=axins, facecolor="0.92", edgecolor="0.55", linewidth=0.25)
        col = next((c for c in dep.columns if dep[c].astype(str)
                    .str.contains("HUANCAVELICA", case=False, na=False).any()), None)
        if col:
            hv = dep[dep[col].astype(str).str.contains("HUANCAVELICA", case=False, na=False)]
            hv.plot(ax=axins, facecolor="#D55E00", edgecolor="k", linewidth=0.3)
    except Exception as exc:                                        # noqa: BLE001
        axins.text(0.5, 0.5, "Peru", ha="center", va="center", fontsize=8)
        print("  locator inset without boundaries:", repr(exc)[:80])
    axins.plot(-74.9758, -12.7867, "*", color="k", ms=4, mew=0)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Peru", fontsize=8, pad=1)
    for sp in axins.spines.values():
        sp.set_linewidth(0.4)
    return axins


GEOREF = PROJ.parent / "GEOREFERENCIAS"


def graticule(ax, n=4):
    """Latitude/longitude graticule drawn over projected axes."""
    tr_geo = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
    lons = np.round(np.linspace(glon.min(), glon.max(), n), 3)
    lats = np.round(np.linspace(glat.min(), glat.max(), n), 3)
    for lo_ in lons:
        xs, ys = tr_geo.transform(np.full(50, lo_), np.linspace(glat.min(), glat.max(), 50))
        ax.plot(np.array(xs) / 1000, np.array(ys) / 1000, color="0.6", lw=0.3, ls=":", zorder=1)
        ax.annotate(f"{abs(lo_):.2f}°W", (xs[0] / 1000, ax.get_ylim()[0]), fontsize=8,
                    color="0.35", ha="center", va="bottom", xytext=(0, 1),
                    textcoords="offset points")
    for la_ in lats:
        xs, ys = tr_geo.transform(np.linspace(glon.min(), glon.max(), 50), np.full(50, la_))
        ax.plot(np.array(xs) / 1000, np.array(ys) / 1000, color="0.6", lw=0.3, ls=":", zorder=1)
        ax.annotate(f"{abs(la_):.2f}°S", (ax.get_xlim()[0], ys[0] / 1000), fontsize=8,
                    color="0.35", ha="left", va="center", xytext=(1, 0),
                    textcoords="offset points")


def annotate_map(ax, *, points=True, labels=True, dumps_on=True):
    if dumps_on:
        w = workings[workings.feature_type == "waste_dump"]
        ax.scatter(w.easting / 1000, w.northing / 1000, marker="x", s=13, c="0.15",
                   linewidth=0.7, zorder=6, label="Waste dump")
    if points:
        for st, mk, col, lab in [("potential_interest", "o", OI[0], "Potential interest"),
                                 ("background", "s", OI[1], "Background"),
                                 ("unlabelled", "^", OI[2], "Unlabelled")]:
            s = loc[loc.stratum == st]
            ax.scatter(s.easting / 1000, s.northing / 1000, marker=mk, s=8, facecolor=col,
                       edgecolor="k", linewidth=0.2, zorder=7, label=lab)
    if labels:
        for nm, (px, py) in PLACES.items():
            ax.annotate(nm, (px / 1000, py / 1000), fontsize=8, color="k", zorder=8,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))
        ax.annotate("Huancavelica", (HVCA[0] / 1000, HVCA[1] / 1000), fontsize=8,
                    fontweight="bold", zorder=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.4", lw=0.3, alpha=0.85))
    ax.set_xlabel("Easting UTM 18S, EPSG:32718 (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    scale_bar(ax, 1.0)
    north_arrow(ax)


# %% [markdown]
# ### 14.1 Version B — fully reproducible, no external service
#
# Built first because it is the version that always works and therefore the safe default for the
# manuscript.

# %%
fig = plt.figure(figsize=(W15, W15 * 1.15), constrained_layout=True)
ax = fig.add_subplot(111)
ax.imshow(HILLSHADE, cmap="gray", extent=[e0 / 1000, e1 / 1000, n0 / 1000, n1 / 1000],
          origin="lower", alpha=0.85, zorder=0, interpolation="bilinear")
cs = ax.contour(GE / 1000, GN / 1000, dem_grid, levels=8, colors="0.35", linewidths=0.25, zorder=2)
ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
annotate_map(ax)
graticule(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=4, fontsize=8)
peru_inset(fig, ax)
ax.set_xlim(e0 / 1000, e1 / 1000)
ax.set_ylim(n0 / 1000, n1 / 1000)
save_fig(fig, "FIG16_site_map_versionB_offline")
plt.show()

# %% [markdown]
# ### 14.2 Version A — real basemap through contextily
#
# **Attribution**: basemap © OpenStreetMap contributors, tiles by CARTO. This must appear in the
# figure caption of any published version.

# %%
BASEMAP_OK = False
if cx is not None:
    try:
        fig = plt.figure(figsize=(W15, W15 * 1.15), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlim(e0, e1)
        ax.set_ylim(n0, n1)
        cx.add_basemap(ax, crs=CRS_UTM, source=cx.providers.CartoDB.Positron, attribution=False,
                       zoom=14)
        # redraw in kilometre units on top of the basemap
        ax.set_xlim(e0, e1)
        ax.set_ylim(n0, n1)
        ax.imshow(HILLSHADE, cmap="gray", extent=[e0, e1, n0, n1], origin="lower", alpha=0.25,
                  zorder=1, interpolation="bilinear")
        for st, mk, col, lab in [("potential_interest", "o", OI[0], "Potential interest"),
                                 ("background", "s", OI[1], "Background"),
                                 ("unlabelled", "^", OI[2], "Unlabelled")]:
            s = loc[loc.stratum == st]
            ax.scatter(s.easting, s.northing, marker=mk, s=9, facecolor=col, edgecolor="k",
                       linewidth=0.2, zorder=7, label=lab)
        w = workings[workings.feature_type == "waste_dump"]
        ax.scatter(w.easting, w.northing, marker="x", s=14, c="0.1", linewidth=0.7, zorder=6,
                   label="Waste dump")
        for nm, (px, py) in PLACES.items():
            ax.annotate(nm, (px, py), fontsize=8, ha="center", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
        ax.annotate("Huancavelica", HVCA, fontsize=8, fontweight="bold", ha="center",
                    va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.4", lw=0.3, alpha=0.85))
        ax.add_artist(ScaleBar(1, "m", location="lower left", box_alpha=0.7, font_properties={"size": 5}))
        north_arrow(ax)
        ax.set_xlabel("Easting UTM 18S, EPSG:32718 (m)")
        ax.set_ylabel("Northing UTM 18S (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=4,
                  fontsize=8)
        peru_inset(fig, ax)
        save_fig(fig, "FIG16_site_map_versionA_basemap")
        plt.show()
        BASEMAP_OK = True
    except Exception as exc:                                        # noqa: BLE001
        print("version A could not be produced:", repr(exc)[:200])
        print("This is exactly why version B exists and is the default for the manuscript.")

# %% [markdown]
# ### 14.3 Version C — basemap plus OpenStreetMap features

# %%
OSM_OK = False
try:
    import osmnx as ox
    bbox_geo = (glon.min(), glat.min(), glon.max(), glat.max())
    t0 = time.time()
    roads = ox.features_from_bbox(bbox=bbox_geo, tags={"highway": True})
    water = ox.features_from_bbox(bbox=bbox_geo, tags={"waterway": True})
    print(f"OSM features downloaded in {time.time()-t0:.0f} s: "
          f"{len(roads)} highway, {len(water)} waterway")
    roads = roads.to_crs(CRS_UTM)
    water = water.to_crs(CRS_UTM)

    fig = plt.figure(figsize=(W15, W15 * 1.15), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.imshow(HILLSHADE, cmap="gray", extent=[e0, e1, n0, n1], origin="lower", alpha=0.7,
              zorder=0, interpolation="bilinear")
    roads.plot(ax=ax, color="0.35", linewidth=0.35, zorder=3)
    water.plot(ax=ax, color="#0072B2", linewidth=0.5, zorder=4)
    for st, mk, col, lab in [("potential_interest", "o", OI[0], "Potential interest"),
                             ("background", "s", OI[1], "Background"),
                             ("unlabelled", "^", OI[2], "Unlabelled")]:
        s = loc[loc.stratum == st]
        ax.scatter(s.easting, s.northing, marker=mk, s=9, facecolor=col, edgecolor="k",
                   linewidth=0.2, zorder=7, label=lab)
    w = workings[workings.feature_type == "waste_dump"]
    ax.scatter(w.easting, w.northing, marker="x", s=14, c="0.1", linewidth=0.7, zorder=6,
               label="Waste dump")
    for nm, (px, py) in PLACES.items():
        ax.annotate(nm, (px, py), fontsize=8, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    ax.annotate("Huancavelica", HVCA, fontsize=8, fontweight="bold", ha="center", va="center",
                zorder=8, bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.4", lw=0.3,
                                    alpha=0.85))
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.add_artist(ScaleBar(1, "m", location="lower left", box_alpha=0.7,
                           font_properties={"size": 5}))
    north_arrow(ax)
    ax.set_xlabel("Easting UTM 18S, EPSG:32718 (m)")
    ax.set_ylabel("Northing UTM 18S (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=4, fontsize=8)
    peru_inset(fig, ax)
    save_fig(fig, "FIG16_site_map_versionC_osm")
    plt.show()
    OSM_OK = True
except Exception as exc:                                            # noqa: BLE001
    print("version C could not be produced:", repr(exc)[:200])

print(f"\nmap versions produced: B (offline) always; A (basemap) {BASEMAP_OK}; C (OSM) {OSM_OK}")

# %% [markdown]
# ### 14.4 Posterior mean and uncertainty
#
# Two panels per figure at most, and the extrapolation is masked: cells beyond 1.5 length scales
# from the nearest sample are hatched, so interpolation between clusters cannot be mistaken for
# evidence.

# %%
def map_panel(ax, values, cmap, label, *, log10=False, mask_extrap=True, vmin=None, vmax=None):
    v = values.reshape(len(gn), len(ge)).copy()
    if log10:
        v = np.log10(np.maximum(v, 1e-6))
    im = ax.pcolormesh(GE / 1000, GN / 1000, v, cmap=cmap, shading="auto", rasterized=True,
                       vmin=vmin, vmax=vmax, zorder=2)
    if mask_extrap:
        me = EXTRAP.reshape(len(gn), len(ge))
        ax.contourf(GE / 1000, GN / 1000, me.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["////"], zorder=3)
        ax.contour(GE / 1000, GN / 1000, me.astype(float), levels=[0.5], colors="k",
                   linewidths=0.3, zorder=4)
    ax.scatter(loc.easting / 1000, loc.northing / 1000, s=2.2, c="k", marker="o", lw=0, zorder=7)
    ax.set_xlabel("Easting UTM 18S, EPSG:32718 (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    scale_bar(ax, 1.0)
    north_arrow(ax)
    return im


for a in MAIN_METALS:
    fig, axes = plt.subplots(1, 2, figsize=(W15, W15 * 1.05), constrained_layout=True)
    im = map_panel(axes[0], maps[f"{a}_mean"], "viridis", "", log10=True)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.55, pad=0.02)
    cb.set_label(f"log$_{{10}}$ {a} posterior mean (mg kg$^{{-1}}$)")
    axes[0].text(0.97, 0.015, "(a)", transform=axes[0].transAxes, ha="right", va="bottom",
                 fontweight="bold")

    im = map_panel(axes[1], maps[f"{a}_sd_f"], "cividis", "")
    cb = fig.colorbar(im, ax=axes[1], shrink=0.55, pad=0.02)
    cb.set_label("Posterior sd of the latent field (log mg kg$^{-1}$)")
    axes[1].text(0.97, 0.015, "(b)", transform=axes[1].transAxes, ha="right", va="bottom",
                 fontweight="bold")
    save_fig(fig, f"FIG17_{a}_mean_and_uncertainty")
    plt.show()

# %% [markdown]
# ### 14.5 Exceedance probability
#
# For cadmium both thresholds are mapped, agricultural (1.4) and residential (10), and presented as
# a **regulatory sensitivity analysis**: the site has dwellings within 3 km and also farmland and
# high-Andean wetlands, so which standard applies is a decision, not a fact. Both are shown.

# %%
for a in MAIN_METALS:
    fig, axes = plt.subplots(1, 2, figsize=(W15, W15 * 1.05), constrained_layout=True)
    for ax, key, thr, tag in [(axes[0], "p_agri", ECA[a][0], "agricultural"),
                              (axes[1], "p_resid", ECA[a][1], "residential / parks")]:
        im = map_panel(ax, maps[f"{a}_{key}"], "magma", "", vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
        cb.set_label(f"P({a} > {thr:g} mg kg$^{{-1}}$)")
        ax.text(0.03, 0.985, f"{tag}\nthreshold {thr:g} mg kg$^{{-1}}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    axes[0].text(0.97, 0.015, "(a)", transform=axes[0].transAxes, ha="right", va="bottom",
                 fontweight="bold")
    axes[1].text(0.97, 0.015, "(b)", transform=axes[1].transAxes, ha="right", va="bottom",
                 fontweight="bold")
    save_fig(fig, f"FIG18_{a}_exceedance")
    plt.show()

# %% [markdown]
# ## 15. Closing summary of the notebook

# %%
summary = {
    "seed": SEED,
    "n_locations": int(len(XY)),
    "kernel": h2["kernel"],
    "anisotropy_ratio": ANISO_MAX,
    "recovery_test_passes": bool(RECOVERY_PASSES),
    "all_main_models_converge": bool(diag["passes"].all()),
    "stratum_support_separable": bool(SEPARABLE),
    "masking_cells": int(len(design)),
    "masking_divergence_slopes_significant": int(slopes["diverges"].sum()),
    "coverage_80pct_all_replicates_won": bool(CAL80_ALL_WON),
    "coverage_80pct_comparisons": int(len(CAL80)),
    "coverage_80pct_max_p": float(CAL80["p_wilcoxon"].max()),
    "picp95_at_80pct": {f"{r.metal}|{r.model}": float(r.picp95) for r in
                        h6[h6["level"] == 0.8].groupby(["metal", "model"], as_index=False)
                        ["picp95"].mean().itertuples()},
    "masking_slopes_total": int(len(slopes)),
    "cv_best_by_joint_score": {m: str(g.loc[g["joint_log_score"].idxmax(), "model"])
                               for m, g in cvsum.groupby("metal")},
    "n_map_cells": int(N_CELLS),
    "map_samples": int(N_MAP_SAMPLES),
    "grid_resolution_m": float(GRID_RES_M),
    "exceedance_max": {a: float(maps[f"{a}_p_agri"].max()) for a in MAIN_METALS},
    "pct_domain_above_half": {a: float(100 * (maps[f"{a}_p_agri"] > 0.5).mean())
                              for a in MAIN_METALS},
    "lmc_identified": bool(lmc["identified"].any()),
    "map_versions": {"A_basemap": BASEMAP_OK, "B_offline": True, "C_osm": OSM_OK},
    "monitoring_reduction_20_points_pct": float(red.loc[20, "reduction_pct"]),
    "predictor_verification_max_error": float(VERIFY_ERR),
}
(DIR_OUT / "nb02_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
