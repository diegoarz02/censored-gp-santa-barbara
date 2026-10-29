"""T1.2 (segunda parte) del blindaje Q1/Q2: sensibilidad al soporte muestral (Copilot #5, ChatGPT).

El modelo oficial trata el ruido de las muestras compuestas (estrato de fondo) con un factor
MULTIPLICATIVO sobre sigma_n: sd_i = sigma_n*exp(log_tau*c_i), log_tau~Normal(0,0.5) (sbmodel.py
L.187-188). Eso ata la escala de ruido de las compuestas a la de las simples salvo por un solo
parametro con un prior centrado en razon=1.

Opcion (a) del prompt -- excluir las compuestas del ajuste -- se descarto tras verificar los datos:
para Cd, `is_composite` esta PERFECTAMENTE confundido con `stratum=="background"` (30/30 filas
coinciden exacto) y las 30 son 100% censuradas. Excluir las compuestas deja CERO puntos del estrato
de fondo, lo que vuelve `b_background` no identificable (sin datos de fondo con que estimarlo) --
no seria una prueba de sensibilidad al soporte, seria otro modelo distinto por una razon ajena
(colinealidad total, no tamano de muestra). Documentado como decision, no como atajo.

Opcion (b), la implementada aqui -- nugget diferenciado SIN el amarre multiplicativo -- mantiene
los 114 puntos (b_background sigue identificable) y relaja la unica restriccion fuerte del modelo
oficial: en vez de sd_compuesta=sigma_n*exp(log_tau) (log_tau con prior centrado en razon=1),
se usa sigma_n_compuesta ~ HalfNormal(sd_sigma) COMPLETAMENTE LIBRE, sin atarla a sigma_tot/rho de
las simples. Si el resultado (parametros espaciales, limit_ratio, excedencia de Cd) no cambia,
es evidencia de robustez real a esa eleccion de parametrizacion, no una comparacion contaminada
por quitar datos.

Protocolo: igual al oficial KW_MAIN (draws=1000, tune=1500, chains=4, target_accept=0.99).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm    # noqa: E402
import pytensor.tensor as pt  # noqa: E402
import rasterio       # noqa: E402
from rasterio.transform import rowcol  # noqa: E402
from pyproj import Transformer  # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DIR_COV = PROJ / "data" / "fase1" / "covariables"
CK_DIR = OUT / "sens_soporte_cd"
CK_DIR.mkdir(parents=True, exist_ok=True)

CRS_UTM = "EPSG:32718"
ECA_CD_AGRI, ECA_CD_RESID = 1.4, 10.0
SEED = 20260908
DRAWS, TUNE, CHAINS, TARGET_ACCEPT = 1000, 1500, 4, 0.99
GRID_RES_M, BUFFER_M, N_MAP_SAMPLES = 50.0, 300.0, 4000
SD_BETA, SD_SIGMA, SD_BETA0 = sb.SD_BETA, sb.SD_SIGMA, sb.SD_BETA0


def build_diffnugget_model(xy, cov, y_value, censored, resolution, lod, *, ell_params,
                           is_background, is_composite, min_detected):
    """Copia de build_censored_gp con el bloque de ruido reemplazado por un nugget diferenciado
    LIBRE por tipo de muestra (no amarrado multiplicativamente). Todo lo demas -- medias, campo
    espacial, verosimilitud censurada con limit_ratio -- es identico al modelo oficial."""
    n, k = len(xy), cov.shape[1]
    censored = np.asarray(censored, bool)
    y_value = np.asarray(y_value, float)
    is_composite = np.asarray(is_composite, bool)

    obs = np.where(censored, np.nan, np.log(np.where(censored, 1.0, y_value)))
    y_bar = float(np.nanmean(obs))

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=y_bar, sigma=SD_BETA0)
        beta = pm.Normal("beta", mu=0.0, sigma=SD_BETA, shape=k)
        mean_terms = beta0 + pt.dot(cov, beta)
        b_bg = pm.Normal("b_background", mu=0.0, sigma=SD_BETA)
        mean_terms = mean_terms + b_bg * np.asarray(is_background, float)

        ell = pm.InverseGamma("ell", alpha=ell_params["alpha"], beta=ell_params["beta"], shape=2)
        sigma_tot = pm.HalfNormal("sigma_tot", sigma=SD_SIGMA)
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        eta = pm.Deterministic("eta", sigma_tot * pt.sqrt(rho))
        # nombrada "sigma_n" (no "sigma_n_simple") porque sb.predict_field lee ese nombre exacto
        # del posterior al reconstruir el campo sobre la grilla -- mismo rol que en el modelo oficial.
        sigma_n = pm.Deterministic("sigma_n", sigma_tot * pt.sqrt(1.0 - rho))
        # nugget de las compuestas: LIBRE, sin amarre multiplicativo a sigma_tot/rho
        sigma_n_composite = pm.HalfNormal("sigma_n_composite", sigma=SD_SIGMA)
        sd_i = pt.where(pt.as_tensor(is_composite), sigma_n_composite, sigma_n)

        cov_func = eta ** 2 * pm.gp.cov.Matern52(2, ls=ell)
        gp = pm.gp.Latent(cov_func=cov_func)
        f = gp.prior("f", X=xy)
        mu = pm.Deterministic("mu", mean_terms + f)

        dist = pm.Normal.dist(mu=mu, sigma=sd_i)
        det = ~censored
        if det.any():
            v_full = np.where(det, y_value, 1.0)
            half = resolution / 2.0
            lo_full = np.log(np.maximum(v_full - half, np.finfo(float).tiny))
            hi_full = np.log(v_full + half)
            logp_det = sb.logdiffexp(pm.logcdf(dist, hi_full)[det], pm.logcdf(dist, lo_full)[det])
            pm.Potential("logp_detected", logp_det.sum())
        if censored.any():
            r_max = float(min_detected / lod)
            r = pm.TruncatedNormal("limit_ratio", mu=3.0, sigma=1.0, lower=1.0, upper=r_max,
                                   initval=min(3.0, 0.5 * (1.0 + r_max)))
            log_limit = pm.Deterministic("log_limit", pt.log(lod * r))
            pm.Potential("logp_censored", pm.logcdf(dist, log_limit)[censored].sum())
    return model


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
    min_det = float(np.nanmin(value))

    e0, e1 = loc["easting"].min() - BUFFER_M, loc["easting"].max() + BUFFER_M
    n0, n1 = loc["northing"].min() - BUFFER_M, loc["northing"].max() + BUFFER_M
    ge = np.arange(e0, e1 + GRID_RES_M, GRID_RES_M)
    gn = np.arange(n0, n1 + GRID_RES_M, GRID_RES_M)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    xy_grid = np.c_[grid_e, grid_n] / 1000.0 - np.c_[loc["easting"], loc["northing"]].mean(0) / 1000.0

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


def main():
    ctx = build_context()
    ck = CK_DIR / "Cd_diffnugget.npz"
    if ck.exists():
        out = dict(np.load(ck))
    else:
        model = build_diffnugget_model(
            ctx["xy"], ctx["cov"], np.where(ctx["censored"], np.nan, ctx["value"]), ctx["censored"],
            ctx["resolution"], ctx["lod"], ell_params=ctx["ell_params"], is_background=ctx["is_bg"],
            is_composite=ctx["is_comp"], min_detected=ctx["min_det"])
        t0 = time.time()
        idata = sb.fit_model(model, seed=SEED, draws=DRAWS, tune=TUNE, chains=CHAINS,
                             cores=CHAINS, target_accept=TARGET_ACCEPT)
        dt = time.time() - t0
        diag = sb.diagnose(idata)
        print(f"Cd/diffnugget: {dt:.0f}s, R-hat={diag['rhat_max']:.4f}, ESS bulk={diag['ess_bulk_min']:.0f}, "
              f"div={diag['divergences']}")

        pr = sb.predict_field(idata, ctx["xy"], ctx["xy_grid"], ctx["cov_grid"],
                              is_background_new=ctx["is_bg_grid"], n_samples=N_MAP_SAMPLES,
                              seed=SEED, anisotropic=True)
        conc = np.exp(pr["y"])
        p_agri = (conc > ECA_CD_AGRI).mean(0)
        p_resid = (conc > ECA_CD_RESID).mean(0)

        post = idata.posterior
        out = dict(
            seconds=dt, rhat_max=diag["rhat_max"], ess_bulk_min=diag["ess_bulk_min"],
            ess_tail_min=diag["ess_tail_min"], divergences=diag["divergences"],
            pct_agri=float(100 * (p_agri > 0.5).mean()), pct_resid=float(100 * (p_resid > 0.5).mean()),
            ell_mean=post["ell"].values.mean(0).mean(0), eta_mean=float(post["eta"].values.mean()),
            sigma_tot_mean=float(post["sigma_tot"].values.mean()), rho_mean=float(post["rho"].values.mean()),
            sigma_n_simple_mean=float(post["sigma_n"].values.mean()),
            sigma_n_composite_mean=float(post["sigma_n_composite"].values.mean()),
            limit_ratio_mean=float(post["limit_ratio"].values.mean()),
            b_background_mean=float(post["b_background"].values.mean()),
        )
        np.savez_compressed(ck, **out)

    # --- comparacion contra el modelo oficial (main_Cd.nc) ---
    import arviz as az
    off = az.from_netcdf(OUT / "idata" / "main_Cd.nc")
    offpost = off.posterior
    official = dict(
        pct_agri=57.4,  # publicado; ver TABLE20_exceedance_summary.csv / posterior_maps.npz
        ell_mean=offpost["ell"].values.mean(0).mean(0), eta_mean=float(offpost["eta"].values.mean()),
        sigma_tot_mean=float(offpost["sigma_tot"].values.mean()), rho_mean=float(offpost["rho"].values.mean()),
        sigma_n_mean=float(offpost["sigma_n"].values.mean()),
        limit_ratio_mean=float(offpost["limit_ratio"].values.mean()),
        b_background_mean=float(offpost["b_background"].values.mean()),
    )

    rows = [
        dict(cantidad="% dominio agricola P>0.5", oficial=official["pct_agri"], diffnugget=out["pct_agri"]),
        dict(cantidad="ell[0] (km)", oficial=official["ell_mean"][0], diffnugget=out["ell_mean"][0]),
        dict(cantidad="ell[1] (km)", oficial=official["ell_mean"][1], diffnugget=out["ell_mean"][1]),
        dict(cantidad="eta=sigma_f", oficial=official["eta_mean"], diffnugget=out["eta_mean"]),
        dict(cantidad="sigma_tot", oficial=official["sigma_tot_mean"], diffnugget=out["sigma_tot_mean"]),
        dict(cantidad="rho", oficial=official["rho_mean"], diffnugget=out["rho_mean"]),
        dict(cantidad="sigma_n (simple)", oficial=official["sigma_n_mean"], diffnugget=out["sigma_n_simple_mean"]),
        dict(cantidad="sigma_n (compuesta)", oficial=np.nan, diffnugget=out["sigma_n_composite_mean"]),
        dict(cantidad="limit_ratio r", oficial=official["limit_ratio_mean"], diffnugget=out["limit_ratio_mean"]),
        dict(cantidad="b_background", oficial=official["b_background_mean"], diffnugget=out["b_background_mean"]),
    ]
    tab = pd.DataFrame(rows)
    tab["dif_absoluta"] = (tab["diffnugget"] - tab["oficial"]).abs()
    tab.to_csv(FIG / "TABLE_sens_soporte_Cd.csv", index=False, encoding="utf-8")
    print("\n" + tab.round(4).to_string(index=False))
    print(f"\nR-hat={out['rhat_max']:.4f} ESS_bulk={out['ess_bulk_min']:.0f} div={out['divergences']}")
    print("\nescrito: figuras/TABLE_sens_soporte_Cd.csv")


if __name__ == "__main__":
    main()
