"""Experimento 2 de la fase de blindaje de revista: propaga la sensibilidad al prior a la
DECISION real -- % del dominio que excede el estandar agricola (1.4) y residencial (10.0) de Cd --
no solo a los parametros del kernel (eso ya estaba en TABLE31_prior_sensitivity.csv).

h_sensibilidad_priors.py / prior_worker.py solo guardan el RESUMEN del posterior (medias/sd), no
el posterior completo, asi que no se puede recomputar excedencia desde ahi -- hacen falta ajustes
nuevos. Reutiliza exactamente `build_censored_gp(..., prior_family=family)` (mismo mecanismo que
prior_worker.py) y el protocolo de esa misma skill (draws=800/tune=1200/chains=4/target_accept=0.95,
"secundario" segun el propio comentario de h_sensibilidad_priors.py) -- no el protocolo KW_MAIN,
para quedar comparable con el resto del analisis de sensibilidad ya publicado, no para ahorrar.

Replica el grid de prediccion oficial (50 m, buffer 300 m, DEM+distancias) linea por linea de
nb02.py seccion 11, y usa el mismo N_MAP_SAMPLES=4000 para que "% del dominio" sea comparable al
57.4%/6.3% ya publicado.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import rasterio      # noqa: E402
from rasterio.transform import rowcol  # noqa: E402
from pyproj import Transformer  # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DIR_COV = PROJ / "data" / "fase1" / "covariables"
CK_DIR = OUT / "prior_cd_exceedance"
CK_DIR.mkdir(parents=True, exist_ok=True)

CRS_UTM = "EPSG:32718"
ECA_CD_AGRI, ECA_CD_RESID = 1.4, 10.0
FAMILIES = ["default", "pc", "vague"]
SEED = 20260908                 # misma semilla que prior_worker.py
KW = dict(draws=800, tune=1200, chains=4, target_accept=0.95, cores=4)
GRID_RES_M, BUFFER_M, N_MAP_SAMPLES = 50.0, 300.0, 4000


def build_context():
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    workings = pd.read_csv(PROJ / "data" / "final" / "mine_workings.csv")
    loc = (df.drop_duplicates("location_id")
           .set_index("location_id")[["easting", "northing", "lon", "lat", "elev_oefa",
                                      "slope_deg", "stratum", "sample_type",
                                      "dist_waste_dump_m", "dist_city_m"]]
           .sort_index())
    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)
    covars = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
    loc = loc.assign(log_dist_dump=np.log(loc["dist_waste_dump_m"].clip(lower=1.0)),
                     log_dist_city=np.log(loc["dist_city_m"]))
    cov = loc[covars].to_numpy(float)
    cov_mean, cov_std = cov.mean(0), cov.std(0)
    cov = (cov - cov_mean) / cov_std
    is_bg = (loc["stratum"] == "background").to_numpy()
    is_comp = (loc["sample_type"] == "composite").to_numpy()
    ell_params, _, _ = sb.ell_prior_params(xy)

    g = df[df.analyte == "Cd"].set_index("location_id").loc[loc.index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(value))
    min_det = float(np.nanmin(value)) if np.isfinite(value).any() else None

    # --- grid de prediccion, identico a nb02.py seccion 11 ---
    e0, e1 = loc["easting"].min() - BUFFER_M, loc["easting"].max() + BUFFER_M
    n0, n1 = loc["northing"].min() - BUFFER_M, loc["northing"].max() + BUFFER_M
    ge = np.arange(e0, e1 + GRID_RES_M, GRID_RES_M)
    gn = np.arange(n0, n1 + GRID_RES_M, GRID_RES_M)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    xy_grid = np.c_[grid_e, grid_n] / 1000.0 - xy_grid_center(loc)

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

    cov_grid = np.c_[grid_elev, grid_slope, np.log(np.clip(grid_dump, 1, None)), np.log(grid_city)]
    cov_grid = (cov_grid - cov_mean) / cov_std
    is_bg_grid = np.zeros(len(grid_e), bool)

    return dict(xy=xy, cov=cov, is_bg=is_bg, is_comp=is_comp, ell_params=ell_params,
               value=value, censored=censored, resolution=resolution, lod=lod, min_det=min_det,
               xy_grid=xy_grid, cov_grid=cov_grid, is_bg_grid=is_bg_grid,
               grid_e=grid_e, grid_n=grid_n)


def xy_grid_center(loc):
    return np.c_[loc["easting"], loc["northing"]].mean(0) / 1000.0


def fit_and_predict(ctx, family):
    ck = CK_DIR / f"Cd_{family}.npz"
    if ck.exists():
        z = np.load(ck)
        return dict(z)

    extra = {}
    if family == "pc":
        obs = np.log(ctx["value"][~ctx["censored"]]) if (~ctx["censored"]).any() else np.log(ctx["value"])
        lam, U, mass = sb.pc_rate_from_bound(float(np.nanmean(obs)))
        extra = dict(pc_lambda=lam)

    model = sb.build_censored_gp(
        ctx["xy"], ctx["cov"], np.where(ctx["censored"], np.nan, ctx["value"]), ctx["censored"],
        ctx["resolution"], ctx["lod"], ell_params=ctx["ell_params"], is_background=ctx["is_bg"],
        is_composite=ctx["is_comp"], anisotropic=True, limit_as_parameter=True,
        min_detected=ctx["min_det"], prior_family=family, **extra)

    t0 = time.time()
    idata = sb.fit_model(model, seed=SEED, **KW)
    dt = time.time() - t0
    diag = sb.diagnose(idata)
    print(f"Cd/{family}: {dt:.0f}s, R-hat={diag['rhat_max']:.4f}, ESS bulk={diag['ess_bulk_min']:.0f}, "
          f"div={diag['divergences']}")

    pr = sb.predict_field(idata, ctx["xy"], ctx["xy_grid"], ctx["cov_grid"],
                          is_background_new=ctx["is_bg_grid"], n_samples=N_MAP_SAMPLES,
                          seed=SEED, anisotropic=True)
    conc = np.exp(pr["y"])
    p_agri = (conc > ECA_CD_AGRI).mean(0)
    p_resid = (conc > ECA_CD_RESID).mean(0)
    mean_conc = conc.mean(0)
    width95 = np.quantile(conc, 0.975, axis=0) - np.quantile(conc, 0.025, axis=0)

    out = dict(p_agri=p_agri, p_resid=p_resid, mean_conc=mean_conc, width95=width95,
              seconds=dt, rhat_max=diag["rhat_max"], ess_bulk_min=diag["ess_bulk_min"],
              ess_tail_min=diag["ess_tail_min"], divergences=diag["divergences"])
    np.savez_compressed(ck, **out)
    return out


def main():
    ctx = build_context()
    results = {}
    for family in FAMILIES:
        results[family] = fit_and_predict(ctx, family)

    rows = []
    for family, r in results.items():
        rows.append({
            "family": family, "seconds": float(r["seconds"]),
            "pct_domain_above_half_agri": float(100 * (r["p_agri"] > 0.5).mean()),
            "pct_domain_above_half_resid": float(100 * (r["p_resid"] > 0.5).mean()),
            "mean_width95_mgkg": float(r["width95"].mean()),
            "rhat_max": float(r["rhat_max"]), "ess_bulk_min": float(r["ess_bulk_min"]),
            "ess_tail_min": float(r["ess_tail_min"]), "divergences": int(r["divergences"]),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_prior_sens_Cd_exceedance.csv", index=False, encoding="utf-8")

    # concordancia espacial: donde default dice "excedente" (p_agri>0.5), lo dicen tambien pc/vague?
    base = results["default"]["p_agri"] > 0.5
    agree = {}
    for family in ["pc", "vague"]:
        other = results[family]["p_agri"] > 0.5
        agree[family] = float((base == other).mean())

    txt = f"""# Sensibilidad del prior propagada a la decisión de excedencia de Cd (2026-10-26)

Pedido de Diego: ver si el 57.4 % agrícola / 6.3 % residencial de `main.tex` (línea 538) es robusto
al prior, no solo los parámetros del kernel (`TABLE31_prior_sensitivity.csv`, ya publicado). Tres
ajustes nuevos de Cd (default/PC/vago), posterior completo, mismo grid de 50 m / {len(ctx['grid_e'])}
celdas que el mapa oficial, N={N_MAP_SAMPLES} muestras por celda.

## % del dominio con P(excede)>0.5, por prior

{tab.round(3).to_markdown(index=False)}

**Valor ya publicado para comparar**: 57.4 % agrícola, 6.3 % residencial (`main.tex` línea 538,
`TABLE20_exceedance_summary.csv`, prior "default").

## Concordancia espacial (agrícola): ¿las mismas celdas se marcan excedentes bajo otro prior?

{chr(10).join(f'- default vs. {f}: {100*v:.1f} % de las celdas coinciden en si excede (p>0.5) o no' for f, v in agree.items())}

## Dictamen

{"El % de dominio excedente varía poco entre priors (diferencia < 2 puntos porcentuales) y la concordancia espacial es alta (>95%) -- la cifra 57.4%/6.3% es robusta al prior." if tab['pct_domain_above_half_agri'].max() - tab['pct_domain_above_half_agri'].min() < 2 and min(agree.values()) > 0.95 else "El % de dominio excedente o la ubicación espacial cambian de forma no trivial entre priors -- ver la tabla y los números de concordancia arriba antes de afirmar que 57.4%/6.3% es robusto al prior."}

## Archivos

`figuras/TABLE_prior_sens_Cd_exceedance.csv`, checkpoints en `outputs/prior_cd_exceedance/`.
"""
    (OUT / "H_prior_sens_Cd_exceedance.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print("escrito: outputs/H_prior_sens_Cd_exceedance.md, figuras/TABLE_prior_sens_Cd_exceedance.csv")


if __name__ == "__main__":
    main()
