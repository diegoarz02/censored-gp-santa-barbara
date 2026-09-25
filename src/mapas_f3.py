"""Bloque 2 — the thirteen map figures, redrawn under the H2 rules. No MCMC: the posterior is on
disk, so this only re-renders.

What changed against the previous run, and why each one was a defect and not a preference:

* **H2.1** the Peru locator sat in the middle of the data, covering ground between Santa Bárbara
  and the western contours. It now goes to the emptiest quadrant, measured.
* **H2.2** no map said where Huancavelica is, and the city is half the argument: the 114 points sit
  between 1.5 and 6.9 km from the main square, 55 of them inside 3 km.
* **H2.3** the hillshade swallowed the points. Context is never the subject.
* **H2.4** CartoDB started demanding an API key and stamped "API KEY REQUIRED" across version A.
  OpenTopoMap needs no key and its contours suit terrain at 4 100-4 500 m.
* **H2.5** the legend floated outside and extrapolation was not degraded, so a cell 1.7 km from any
  sample looked exactly as authoritative as one 100 m away.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import DATA, FIG, OUT, PROJ, RUN_ID, observe  # noqa: E402

warnings.filterwarnings("ignore")

CRS_UTM = "EPSG:32718"
HVCA_LONLAT = (-74.9758, -12.7867)          # Huancavelica main square
ECA_AGRI = {"Hg": 6.6, "Pb": 70.0, "As": 50.0, "Ba": 750.0, "Cd": 1.4}
# Verified against RECURSOS/normativa/DS-011-2017-MINAM_ECA_Suelo.pdf, not against a
# secondary citation: the three land uses are agricultural / residential-park /
# commercial-industrial-extractive. Cadmium is 1.4 / 10 / 22, so the residential value is
# 10 and not 22 as the run plan stated.
ECA_RESID = {"Hg": 6.6, "Pb": 140.0, "As": 50.0, "Ba": 500.0, "Cd": 10.0}
ECA_INDUS = {"Hg": 24.0, "Pb": 800.0, "As": 140.0, "Ba": 2000.0, "Cd": 22.0}
MAIN = ["Hg", "Pb", "As", "Ba", "Cd"]


# ------------------------------------------------------------------------------------ data load
def load():
    mp = dict(np.load(OUT / "posterior_maps.npz"))
    df = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    work = pd.read_csv(DATA / "final" / "mine_workings.csv")

    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
    hv = to_utm.transform(*HVCA_LONLAT)

    ge = np.unique(mp["easting"])
    gn = np.unique(mp["northing"])
    shape = (len(gn), len(ge))

    # distance of every grid cell to the nearest sample, in metres
    XY = np.c_[loc["easting"], loc["northing"]]
    G = np.c_[mp["easting"], mp["northing"]]
    d_obs = np.sqrt(((G[:, None] - XY[None]) ** 2).sum(-1)).min(1)

    # hillshade from the elevation already sampled on the grid, if it is there
    hill = None
    if "grid_elev" in mp:
        hill = LightSource(azdeg=315, altdeg=45).hillshade(
            mp["grid_elev"].reshape(shape), vert_exag=2.0, dx=50.0, dy=50.0)
    else:
        dem = PROJ / "data" / "fase1" / "covariables" / "santa_barbara_dem_srtm30.tif"
        if dem.exists():
            try:
                import rasterio
                from pyproj import Transformer as T2
                to_geo = T2.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)
                lon, lat = to_geo.transform(mp["easting"], mp["northing"])
                with rasterio.open(dem) as src:
                    z = np.array([v[0] for v in src.sample(np.c_[lon, lat])], float)
                hill = LightSource(azdeg=315, altdeg=45).hillshade(
                    z.reshape(shape), vert_exag=2.0, dx=50.0, dy=50.0)
            except Exception as exc:                                # noqa: BLE001
                print("hillshade unavailable:", repr(exc)[:120])

    places = {}
    m_ = df.drop_duplicates("location_id")
    for name, pat in [("Santa Bárbara", "Santa Bárbara"), ("Yanamina", "Yanamina"),
                      ("Suytococha", "Suytococha"), ("Carnicería", "Carnicería")]:
        s = m_["description"].astype(str).str.contains(pat, case=False, na=False)
        if s.sum() >= 3:
            places[name] = (float(m_.loc[s, "easting"].mean()), float(m_.loc[s, "northing"].mean()))

    return dict(mp=mp, df=df, loc=loc, work=work, hv=hv, ge=ge, gn=gn, shape=shape,
                d_obs=d_obs, hill=hill, places=places,
                extent=(ge.min(), ge.max(), gn.min(), gn.max()))


# --------------------------------------------------------------------------- shared map furniture
def base_axes(D, ax, *, hill_alpha=0.35, scale_x=0.62):
    """Everything every map shares: hillshade, workings, city, place names, scale, north arrow."""
    e0, e1, n0, n1 = D["extent"]
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect("equal")
    if D["hill"] is not None:
        ax.imshow(D["hill"], cmap="gray", extent=(e0, e1, n0, n1), origin="lower",
                  alpha=hill_alpha, zorder=1, interpolation="bilinear")
    w = D["work"]
    if "feature_type" in w:
        wd = w[w.feature_type == "waste_dump"]
        ax.scatter(wd.easting, wd.northing, marker="x", s=16, c="#3a3835", linewidth=0.8,
                   zorder=6, label="Waste dump")
    for nm, (px, py) in D["places"].items():
        ax.annotate(nm, (px, py), fontsize=ef.FS_MIN, ha="center", va="center", zorder=8, color="#2b2926",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    hv_e, hv_n = D["hv"]
    ax.scatter([hv_e], [hv_n], marker="*", s=90, facecolor="#f5d90a", edgecolor="k",
               linewidth=0.5, zorder=9)
    ax.annotate("Huancavelica", (hv_e, hv_n), xytext=(0, -9), textcoords="offset points",
                fontsize=ef.FS_MIN, fontweight="bold", ha="center", va="top", zorder=9,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#52514e", lw=0.3, alpha=0.9))
    ax.set_xlabel("Easting UTM 18S (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ef.utm_km_ticks(ax)
    _scale_bar(ax, frac_x=scale_x)
    _north(ax)


def _scale_bar(ax, km=1.0, frac_x=0.62, frac_y=0.035):
    """Scale bar bottom-RIGHT by default.

    It used to sit bottom-left, which is where the summary text box goes, and the audit caught the
    pair overlapping on all six exceedance maps ('with' x 'km', 'P' x 'km'). Two objects that both
    want the quietest corner cannot both have it.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xa = x0 + frac_x * (x1 - x0)
    ya = y0 + frac_y * (y1 - y0)
    ax.plot([xa, xa + km * 1000], [ya, ya], color="k", lw=1.6, solid_capstyle="butt", zorder=10)
    ax.text(xa + km * 500, ya + 0.012 * (y1 - y0), f"{km:g} km", ha="center", va="bottom",
            fontsize=ef.FS_MIN, zorder=10)


def _north(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.94 * (x1 - x0)
    y = y0 + 0.86 * (y1 - y0)
    ax.annotate("N", xy=(x, y + 0.05 * (y1 - y0)), xytext=(x, y), ha="center", va="center",
                fontsize=ef.FS_MIN, fontweight="bold", zorder=10,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))


# INEI 2023 administrative boundaries, supplied by Diego in `ANALISIS BAYESIANO/GEOREFERENCIAS/`
# as zipped shapefiles. geopandas reads them straight out of the archive, so nothing is unpacked
# into the project. Read once and cached: the district layer is 1 891 polygons.
GEO_DIR = PROJ.parent / "GEOREFERENCIAS"
_GEO_CACHE = {}


def boundaries(level: str = "Departamental"):
    """`Departamental` (25), `Provincial` or `Distrital` (1 891), EPSG:4326."""
    if level in _GEO_CACHE:
        return _GEO_CACHE[level]
    import geopandas as gpd
    z = GEO_DIR / f"{level} INEI 2023 geogpsperu SuyoPomalia.zip"
    if not z.exists():
        _GEO_CACHE[level] = None
        return None
    try:
        g = gpd.read_file(f"zip://{z.as_posix()}")
    except Exception as exc:                                        # noqa: BLE001
        print(f"boundaries({level}) unavailable: {exc!r}"[:140])
        g = None
    _GEO_CACHE[level] = g
    return g


def peru_inset_drawer(D):
    """What goes inside the locator box. Placement is `add_locator_inset`'s job, not this one."""
    def draw(iax):
        g = boundaries("Departamental")
        if g is not None:
            g.plot(ax=iax, facecolor="#eceae7", edgecolor="#b9b5af", linewidth=0.25, zorder=1)
            col = "DEPARTAMEN"
            hv = g[g[col].astype(str).str.upper().str.contains("HUANCAVELICA", na=False)]
            if len(hv):
                hv.plot(ax=iax, facecolor="#f6c9b5", edgecolor="#d62728", linewidth=0.5, zorder=2)
        iax.plot([HVCA_LONLAT[0]], [HVCA_LONLAT[1]], marker="v", ms=4.0, color="#d62728",
                 markeredgecolor="k", markeredgewidth=0.3, zorder=4)
        iax.set_xlim(-82.0, -68.0)
        iax.set_ylim(-19.0, -1.0)
        iax.set_aspect("equal")
        iax.text(0.06, 0.04, "Peru", transform=iax.transAxes, ha="left", va="bottom",
                 fontsize=ef.FS_MIN, color=ef.C_REF)
        iax.set_facecolor("white")
    return draw


def district_lines(ax, D, lw=0.4):
    """District boundaries over the map extent. The audit asked for them by name: without them the
    map shows contamination but not who administers the ground it sits on."""
    g = boundaries("Distrital")
    if g is None:
        return 0
    e0, e1, n0, n1 = D["extent"]
    gg = g.to_crs(CRS_UTM)
    sub = gg.cx[e0:e1, n0:n1]
    if not len(sub):
        return 0
    sub.boundary.plot(ax=ax, edgecolor="#6f6b66", linewidth=lw, zorder=3, alpha=0.8)
    dcol = "DISTRITO" if "DISTRITO" in sub.columns else sub.columns[-2]
    for _, r in sub.iterrows():
        c = r.geometry.representative_point()
        if e0 < c.x < e1 and n0 < c.y < n1:
            ax.annotate(str(r[dcol]).title(), (c.x, c.y), fontsize=ef.FS_MIN, color="#6f6b66",
                        ha="center", va="center", zorder=8, alpha=0.85)
    return len(sub)


def extrapolation_overlay(ax, D, ell_km=1.03, factor=1.5):
    """Beyond `factor` length scales from the nearest sample there is no evidence, and the map
    must not look as if there were. Hatched, not merely faded, so it survives greyscale printing."""
    e0, e1, n0, n1 = D["extent"]
    mask = (D["d_obs"] > factor * ell_km * 1000).reshape(D["shape"])
    ax.contourf(np.linspace(e0, e1, D["shape"][1]), np.linspace(n0, n1, D["shape"][0]),
                mask.astype(float), levels=[0.5, 1.5], colors="none",
                hatches=["////"], zorder=5)
    ax.contour(np.linspace(e0, e1, D["shape"][1]), np.linspace(n0, n1, D["shape"][0]),
               mask.astype(float), levels=[0.5], colors=["#52514e"], linewidths=0.5, zorder=5)
    return float(mask.mean())


# =============================================================================================
# H2.4 — site maps, three versions. Version B never depends on the network.
# =============================================================================================
def site_map_offline(D):
    """Version B: DEM hillshade only. The manuscript default, because it has no external
    dependency that can break between submission and print."""
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 1.30), constrained_layout=True)
    base_axes(D, ax, hill_alpha=0.35)
    district_lines(ax, D)
    loc = D["loc"]
    for st, mk, col, lab in [("potential_interest", "o", ef.C_CENSORED, "Potential interest"),
                             ("background", "s", ef.C_KRIGING, "Background"),
                             ("unlabelled", "^", ef.C_REF, "Unlabelled")]:
        s = loc[loc.stratum == st]
        if len(s):
            ax.scatter(s.easting, s.northing, marker=mk, s=11, facecolor=col, edgecolor="k",
                       linewidth=0.25, zorder=7, label=lab)
    XY = np.c_[loc["easting"], loc["northing"]]
    dhv = np.sqrt(((XY - np.array(D["hv"])) ** 2).sum(1)) / 1000.0
    # Below the map, not "upper right": on this extent the north end of the valley still has
    # points under it, so any in-axes corner sits on top of real data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), frameon=False, ncol=2,
              fontsize=ef.FS_MIN, columnspacing=1.0)
    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    # Diego removed the Peru locator from this figure on review (2026-09-24): it does not belong
    # on the manuscript's main site map. `peru_inset_drawer` stays in the module in case a
    # different figure wants a country locator later; it is simply not called here any more.
    info = ef.save_fig(fig, "FIG16_site_map_versionB_offline")
    plt.close(fig)
    return {**info, "d_min_km": float(dhv.min()), "d_max_km": float(dhv.max()),
            "n_within_3km": int((dhv < 3).sum())}


def site_map_basemap(D, provider="OpenTopoMap"):
    """Version A: a real basemap. CartoDB now demands a key and stamped 'API KEY REQUIRED' across
    the previous version; OpenTopoMap needs none and its contours suit terrain above 4 000 m.

    Network calls get three attempts with growing waits: after a laptop wakes from suspend the
    connection takes a while to come back, and today that aborts the cell.
    """
    import time
    try:
        import contextily as cx
    except Exception as exc:                                        # noqa: BLE001
        return {"ok": False, "why": f"contextily unavailable: {exc!r}"[:150]}

    sources = {"OpenTopoMap": cx.providers.OpenTopoMap,
               "Esri.WorldTopoMap": cx.providers.Esri.WorldTopoMap,
               "OpenStreetMap.Mapnik": cx.providers.OpenStreetMap.Mapnik}
    attribution = {"OpenTopoMap": "Map data © OpenStreetMap contributors, SRTM · "
                                  "map style © OpenTopoMap (CC-BY-SA)",
                   "Esri.WorldTopoMap": "Basemap © Esri",
                   "OpenStreetMap.Mapnik": "© OpenStreetMap contributors"}

    last = None
    for name in [provider] + [k for k in sources if k != provider]:
        for attempt in range(3):
            try:
                ef.apply_style()
                fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 1.30), constrained_layout=True)
                e0, e1, n0, n1 = D["extent"]
                ax.set_xlim(e0, e1)
                ax.set_ylim(n0, n1)
                cx.add_basemap(ax, crs=CRS_UTM, source=sources[name], attribution=False, zoom=14)
                ax.set_xlim(e0, e1)
                ax.set_ylim(n0, n1)
                base_axes(D, ax, hill_alpha=0.22)   # lighter: the basemap already carries relief
                loc = D["loc"]
                for st, mk, col, lab in [("potential_interest", "o", ef.C_CENSORED,
                                          "Potential interest"),
                                         ("background", "s", ef.C_KRIGING, "Background"),
                                         ("unlabelled", "^", ef.C_REF, "Unlabelled")]:
                    s = loc[loc.stratum == st]
                    if len(s):
                        ax.scatter(s.easting, s.northing, marker=mk, s=11, facecolor=col,
                                   edgecolor="k", linewidth=0.25, zorder=7, label=lab)
                # Below the map, not "upper right": Diego caught the legend sitting on top of the
                # Santa Barbara point cluster in the north end of the valley (same fix as version
                # B). The attribution line goes below that, not fighting it for the same corner.
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), frameon=False, ncol=2,
                          fontsize=ef.FS_MIN, columnspacing=1.0)
                # No message_title: body-of-article figure, message goes in the LaTeX caption.
                ax.text(0.5, -0.145, attribution[name], transform=ax.transAxes,
                        fontsize=ef.FS_MIN, ha="center", va="top", color=ef.C_REF)
                info = ef.save_fig(fig, "FIG16_site_map_versionA_basemap")
                plt.close(fig)
                return {"ok": True, "provider": name, **info}
            except (OSError, TimeoutError, ValueError) as exc:
                # Only network and tile errors are retried. A broad `except Exception` here once
                # swallowed a NameError from this very function and reported it as "provider
                # failed", so version A silently kept the old file with the API-key watermark for
                # hours. A retry loop must never catch bugs in its own body.
                last = exc
                plt.close("all")
                time.sleep(2 ** attempt)
        print(f"  provider {name} failed after 3 attempts: {repr(last)[:110]}")
    return {"ok": False, "why": repr(last)[:200]}


# =============================================================================================
# H2.5 + H4.2 — posterior mean with uncertainty, and exceedance at each land use
# =============================================================================================
def mean_and_uncertainty(D, analyte):
    """Two panels: posterior median concentration, and predictive uncertainty.

    Extrapolation is hatched on both. Fading alone was not enough in the previous run: a cell
    1.7 km from any sample looked as authoritative as one 100 m away, and the eye does not read
    transparency as a claim about evidence.
    """
    ef.apply_style()
    mp, sh = D["mp"], D["shape"]
    e0, e1, n0, n1 = D["extent"]
    # `mean` and `median` are stored in mg/kg, back-transformed sample by sample; `sd_f` and
    # `sd_y` are on the log scale. The array names do not carry the unit, which is how this was
    # misread once — see OBSERVACIONES.
    med = mp[f"{analyte}_median"].reshape(sh)
    sdy = mp[f"{analyte}_sd_y"].reshape(sh)

    fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.62), constrained_layout=True)
    for ax, arr, lab, cmap, panel in [
            (axes[0], med, f"posterior median {analyte} (mg kg$^{{-1}}$), red line = ECA "
                           f"{ECA_AGRI[analyte]:g}", ef.CMAP_MAG, "a"),
            (axes[1], sdy, "predictive sd (log units)", "cividis", "b")]:
        base_axes(D, ax, hill_alpha=0.18, scale_x=0.05)
        kw = dict(extent=(e0, e1, n0, n1), origin="lower", zorder=2, interpolation="bilinear")
        if panel == "a":
            from matplotlib.colors import LogNorm
            im = ax.imshow(arr, cmap=cmap, norm=LogNorm(vmin=max(arr.min(), 1e-3),
                                                        vmax=arr.max()), alpha=0.85, **kw)
        else:
            im = ax.imshow(arr, cmap=cmap, alpha=0.85, **kw)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label(lab, fontsize=ef.FS_MIN)
        cb.ax.tick_params(labelsize=ef.FS_MIN)
        if panel == "a":
            thr = ECA_AGRI[analyte]
            try:
                # Line only. Writing "ECA <value>" beside the bar put the text on top of the
                # bar's own tick labels; the audit caught it as 'k' x '750' on barium.
                cb.ax.axhline(thr, color="#d62728", lw=1.4)
            except Exception:                                       # noqa: BLE001
                pass
        frac = extrapolation_overlay(ax, D)
        ef.panel_label(ax, panel)
        ax.tick_params(labelsize=ef.FS_MIN)
        if panel == "b":
            ax.set_ylabel("")

    axes[0].scatter(D["loc"].easting, D["loc"].northing, s=4, facecolor="none",
                    edgecolor="k", linewidth=0.25, zorder=7)
    hatch = mpatches.Patch(facecolor="white", edgecolor=ef.C_REF, hatch="////",
                           label=f"extrapolation ({100 * frac:.1f} %)")
    # Upper right: the scale bar owns the bottom-right corner now, and the audit caught the two
    # overlapping ('km' x 'domain)') on all five analytes.
    axes[1].legend(handles=[hatch], loc="upper right", frameon=True, framealpha=0.92,
                   edgecolor="#cfccc7", fontsize=ef.FS_MIN)
    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    info = ef.save_fig(fig, f"FIG17_{analyte}_mean_and_uncertainty")
    plt.close(fig)
    return {**info, "extrap_frac": frac}


def exceedance(D, analyte, land_use="agricultural"):
    """P(concentration > ECA) for one land use, with the 0.5 decision threshold marked.

    Cadmium is drawn for both agricultural (1.4) and residential (10) because that pair *is* the
    argument of the paper: the same metal, the same soil, the same detection limit, inside the
    critical ECA/LOD band under one land use and far outside it under the other.
    """
    ef.apply_style()
    key = {"agricultural": "p_agri", "residential": "p_resid"}[land_use]
    thr = (ECA_AGRI if land_use == "agricultural" else ECA_RESID)[analyte]
    p = D["mp"][f"{analyte}_{key}"].reshape(D["shape"])
    e0, e1, n0, n1 = D["extent"]

    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 1.28), constrained_layout=True)
    base_axes(D, ax, hill_alpha=0.18)
    im = ax.imshow(p, cmap=ef.CMAP_MAG, vmin=0, vmax=1, extent=(e0, e1, n0, n1),
                   origin="lower", zorder=2, alpha=0.88, interpolation="bilinear")
    ax.contour(np.linspace(e0, e1, D["shape"][1]), np.linspace(n0, n1, D["shape"][0]),
               p, levels=[0.5], colors=["#d62728"], linewidths=0.9, zorder=4)
    frac = extrapolation_overlay(ax, D)
    ax.scatter(D["loc"].easting, D["loc"].northing, s=4, facecolor="none", edgecolor="k",
               linewidth=0.25, zorder=7)

    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(f"P({analyte} > {thr:g} mg kg$^{{-1}}$)", fontsize=ef.FS_MIN)
    cb.ax.tick_params(labelsize=ef.FS_MIN)
    ef.mark_half_on_colorbar(cb)

    above = 100 * (p > 0.5).mean()
    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    # No summary box and no locator inset here. Both covered the surface the figure exists
    # to show, and both said what a caption says better. A figure earns every object drawn
    # on top of its data.
    suffix = "" if land_use == "agricultural" else "_residential"
    info = ef.save_fig(fig, f"FIG18_{analyte}_exceedance{suffix}")
    plt.close(fig)
    return {**info, "pct_above_half": above, "threshold": thr, "land_use": land_use}


# =============================================================================================
# Rejilla 2x2 de excedencia — idea de Diego, y el argumento es bueno: cuatro PDF separados
# obligan al revisor a recordar; una rejilla con los mismos limites geograficos deja ver de un
# vistazo si los focos de As, Hg, Pb, Cd y Ba comparten la misma huella espacial desde la mina.
# 2026-09-25, pedido de Diego: se agrega As (antes solo Hg/Pb/Cd/Ba) y la 6a celda del grid 2x3
# se usa para Cd residencial en vez de quedar vacia -- aporta el contraste agricola/residencial
# de cadmio (la tesis del articulo) sin gastar una figura aparte.
# =============================================================================================
def exceedance_grid(D, analytes=("As", "Hg", "Pb", "Cd", "Ba"), land_use="agricultural"):
    """Cinco metales en una rejilla 2x3, mismos limites y misma escala de color.

    La sexta celda es cadmio bajo el uso residencial (ECA 10, no 1.4): mismo metal, mismo suelo,
    mismo limite de deteccion, bajo el otro umbral regulatorio -- el contraste que sostiene el
    argumento del articulo sobre por que el criterio ECA/LD importa.

    Los puntos de muestreo van encima como cruces negras, tambien por sugerencia de Diego: sin
    ellos no se distingue donde el modelo tiene soporte de datos duros y donde esta interpolando
    por pura covarianza. Es la diferencia entre un mapa que informa y uno que solo decora.

    Sin `message_title`: esta figura lleva su mensaje en el caption de LaTeX, no impreso encima
    del contenido -- pedido explicito de Diego para esta figura en particular.
    """
    ef.apply_style()
    key_agri = "p_agri"
    e0, e1, n0, n1 = D["extent"]
    panels = [(a, "agricultural", key_agri, ECA_AGRI[a]) for a in analytes]
    panels.append(("Cd", "residential", "p_resid", ECA_RESID["Cd"]))

    fig, axes = plt.subplots(2, 3, figsize=(ef.W2, ef.W2 * 0.70), constrained_layout=True)
    im = None
    for ax, (a, lu, key, thr) in zip(axes.ravel(), panels):
        p = D["mp"][f"{a}_{key}"].reshape(D["shape"])
        if D["hill"] is not None:
            ax.imshow(D["hill"], cmap="gray", extent=(e0, e1, n0, n1), origin="lower",
                      alpha=0.16, zorder=1, interpolation="bilinear")
        im = ax.imshow(p, cmap=ef.CMAP_MAG, vmin=0, vmax=1, extent=(e0, e1, n0, n1),
                       origin="lower", zorder=2, alpha=0.9, interpolation="bilinear")
        ax.contour(np.linspace(e0, e1, D["shape"][1]), np.linspace(n0, n1, D["shape"][0]),
                   p, levels=[0.5], colors=["#d62728"], linewidths=0.8, zorder=4)
        # Los 114 puntos reales, como cruces finas: donde no hay cruces, el mapa es covarianza.
        ax.scatter(D["loc"].easting, D["loc"].northing, marker="+", s=9, c="k",
                   linewidths=0.45, zorder=7)
        ax.set_xlim(e0, e1)
        ax.set_ylim(n0, n1)
        ax.set_aspect("equal")
        # Shorter than the 2x2 version's title: each panel is now 1/3 of the width, not 1/2, and
        # "... % > 0.5" ran into the next panel's title (the audit caught two real collisions).
        # The ">0.5" threshold is already on the colorbar and the red contour, so it is redundant
        # here anyway.
        label = f"{a} resid." if lu == "residential" else a
        ax.set_title(f"{label}, ECA {thr:g}, {100 * (p > 0.5).mean():.0f}%",
                     loc="left", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xticks([503000, 504000, 505000])
        ax.set_xticklabels(["503", "504", "505"])
        ax.set_yticks([8580000, 8582000, 8584000])
        ax.set_yticklabels(["8580", "8582", "8584"])
    for ax in axes[1]:
        ax.set_xlabel("Easting UTM 18S (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Northing UTM 18S (km)")
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02, shrink=0.85)
    cb.set_label("P(exceeds the standard)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ef.mark_half_on_colorbar(cb)
    info = ef.save_fig(fig, "FIG22_exceedance_grid")
    plt.close(fig)
    return info
