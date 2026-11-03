"""Flip L=LD para Cd (orden de revision externa) -- F2, F3, F4.

Reutiliza el idata YA AJUSTADO en outputs/idata/flip_LD_Cd.nc (ver F1, generado por
scratchpad/f1_refit_cd_LD.py: L=LD fijo, protocolo oficial, R-hat=1.0051, 0 divergencias, 156s).
NO reajusta nada aqui -- solo prediccion posterior (sb.predict_field) en 3 resoluciones, igual al
patron ya validado de t2_1_area_excedida_posterior.py (F3) y q1_t4_resolucion_area.py (F4).

F2: regenera las 3 figuras del Cd con los MISMOS nombres, reutilizando mapas_f3.py sin tocarlo --
se construye un dict D equivalente al oficial pero con las 4 claves Cd_* reemplazadas por las del
modelo L=LD, y se llama a M.mean_and_uncertainty/M.exceedance tal cual.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import rasterio      # noqa: E402
from rasterio.transform import rowcol  # noqa: E402
from pyproj import Transformer  # noqa: E402
import arviz as az   # noqa: E402

import sbmodel as sb  # noqa: E402
import mapas_f3 as M  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DIR_COV = PROJ / "data" / "fase1" / "covariables"
CK_DIR = OUT / "flip_ld_cd"

CRS_UTM = "EPSG:32718"
ECA_AGRI, ECA_RESID = 1.4, 10.0
SEED = 20260906
N_MAP_SAMPLES, CELL_BLOCK = 4000, 1500
THRESHOLDS_T = [0.5, 0.8, 0.95]
COVARS = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]


def build_grid(loc, workings, grid_res_m, buffer_m, cov_mean, cov_std, xy_center):
    e0, e1 = loc["easting"].min() - buffer_m, loc["easting"].max() + buffer_m
    n0, n1 = loc["northing"].min() - buffer_m, loc["northing"].max() + buffer_m
    ge = np.arange(e0, e1 + grid_res_m, grid_res_m)
    gn = np.arange(n0, n1 + grid_res_m, grid_res_m)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    xy_grid = np.c_[grid_e, grid_n] / 1000.0 - xy_center

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
    return grid_e, grid_n, xy_grid, cov_grid, is_bg_grid


def shared_setup():
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    workings = pd.read_csv(PROJ / "data" / "final" / "mine_workings.csv")
    loc = (df.drop_duplicates("location_id")
           .set_index("location_id")[["easting", "northing", "lon", "lat", "elev_oefa",
                                      "slope_deg", "stratum", "sample_type",
                                      "dist_waste_dump_m", "dist_city_m"]]
           .sort_index())
    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)
    loc = loc.assign(log_dist_dump=np.log(loc["dist_waste_dump_m"].clip(lower=1.0)),
                     log_dist_city=np.log(loc["dist_city_m"]))
    raw = loc[COVARS].to_numpy(float)
    cov_mean, cov_std = raw.mean(0), raw.std(0)
    xy_center = np.c_[loc["easting"], loc["northing"]].mean(0) / 1000.0
    return df, workings, loc, xy, cov_mean, cov_std, xy_center


def predict_at_resolution(idata, xy, loc, workings, cov_mean, cov_std, xy_center,
                          grid_res_m, buffer_m=300.0, keep_draws=False):
    grid_e, grid_n, xy_grid, cov_grid, is_bg_grid = build_grid(
        loc, workings, grid_res_m, buffer_m, cov_mean, cov_std, xy_center)
    n_cells = len(grid_e)
    median = np.zeros(n_cells)
    sd_y = np.zeros(n_cells)
    p_agri = np.zeros(n_cells)
    p_resid = np.zeros(n_cells)
    width95 = np.zeros(n_cells)
    count_agri_draw = np.zeros(N_MAP_SAMPLES)
    count_resid_draw = np.zeros(N_MAP_SAMPLES)

    t0 = time.time()
    for b0 in range(0, n_cells, CELL_BLOCK):
        b1 = min(b0 + CELL_BLOCK, n_cells)
        pr = sb.predict_field(idata, xy, xy_grid[b0:b1], cov_grid[b0:b1],
                              is_background_new=is_bg_grid[b0:b1],
                              n_samples=N_MAP_SAMPLES, seed=SEED, anisotropic=True)
        y = pr["y"]                      # (N_MAP_SAMPLES, block) en escala log
        conc = np.exp(y)
        median[b0:b1] = np.median(conc, axis=0)
        sd_y[b0:b1] = y.std(axis=0)
        width95[b0:b1] = np.quantile(conc, 0.975, axis=0) - np.quantile(conc, 0.025, axis=0)
        exceeds_agri = conc > ECA_AGRI
        exceeds_resid = conc > ECA_RESID
        p_agri[b0:b1] = exceeds_agri.mean(0)
        p_resid[b0:b1] = exceeds_resid.mean(0)
        count_agri_draw += exceeds_agri.sum(1)
        count_resid_draw += exceeds_resid.sum(1)
    dt = time.time() - t0

    area_frac_agri_draw = count_agri_draw / n_cells
    area_frac_resid_draw = count_resid_draw / n_cells
    print(f"  resolucion {grid_res_m:.0f}m: {n_cells} celdas, {dt:.0f}s")
    return dict(grid_e=grid_e, grid_n=grid_n, n_cells=n_cells, median=median, sd_y=sd_y,
               p_agri=p_agri, p_resid=p_resid, width95=width95,
               area_frac_agri_draw=area_frac_agri_draw, area_frac_resid_draw=area_frac_resid_draw,
               seconds=dt)


def summarize(arr):
    return dict(mean=float(arr.mean()), median=float(np.median(arr)), sd=float(arr.std()),
               ci90_lo=float(np.percentile(arr, 5)), ci90_hi=float(np.percentile(arr, 95)),
               ci95_lo=float(np.percentile(arr, 2.5)), ci95_hi=float(np.percentile(arr, 97.5)))


def three_quantities_row(res_label, r):
    row = {"resolucion_m": res_label, "n_cells": r["n_cells"], "seconds": r["seconds"]}
    for t in THRESHOLDS_T:
        row[f"pct_domain_agri_gt_{t}"] = float(100 * (r["p_agri"] > t).mean())
        row[f"pct_domain_resid_gt_{t}"] = float(100 * (r["p_resid"] > t).mean())
    row["E_area_agri_pct"] = float(100 * r["p_agri"].mean())
    row["E_area_resid_pct"] = float(100 * r["p_resid"].mean())
    s_agri = summarize(r["area_frac_agri_draw"])
    s_resid = summarize(r["area_frac_resid_draw"])
    for k, v in s_agri.items():
        row[f"area_frac_agri_{k}_pct"] = v * 100
    for k, v in s_resid.items():
        row[f"area_frac_resid_{k}_pct"] = v * 100
    return row


def main():
    idata = az.from_netcdf(OUT / "idata" / "flip_LD_Cd.nc")
    df, workings, loc, xy, cov_mean, cov_std, xy_center = shared_setup()

    print("=== F3/F4: prediccion en 3 resoluciones (sin reajustar el modelo) ===")
    r50 = predict_at_resolution(idata, xy, loc, workings, cov_mean, cov_std, xy_center, 50.0)
    assert r50["n_cells"] == 7381, f"grid 50m no coincide con el oficial: {r50['n_cells']}"
    r100 = predict_at_resolution(idata, xy, loc, workings, cov_mean, cov_std, xy_center, 100.0)
    r25 = predict_at_resolution(idata, xy, loc, workings, cov_mean, cov_std, xy_center, 25.0)

    rows = [three_quantities_row(50, r50), three_quantities_row(100, r100),
           three_quantities_row(25, r25)]
    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_LD_principal_Cd.csv", index=False, encoding="utf-8")
    print("\n" + tab.round(3).to_string(index=False))

    np.savez_compressed(CK_DIR / "maps_50m.npz", **{k: v for k, v in r50.items()
                                                     if isinstance(v, np.ndarray)})

    # ----------------------------------------------------------------- F2: 3 figuras, mismos nombres
    print("\n=== F2: regenerando FIG17/FIG18 del Cd con el modelo L=LD ===")
    D = M.load()
    D_ld = dict(D)
    mp_ld = dict(D["mp"])
    mp_ld["Cd_median"] = r50["median"]
    mp_ld["Cd_mean"] = r50["median"]          # mean_and_uncertainty solo usa _median y _sd_y
    mp_ld["Cd_sd_y"] = r50["sd_y"]
    mp_ld["Cd_sd_f"] = r50["sd_y"]            # no se usa en estas 2 funciones, se deja coherente
    mp_ld["Cd_p_agri"] = r50["p_agri"]
    mp_ld["Cd_p_resid"] = r50["p_resid"]
    D_ld["mp"] = mp_ld

    info17 = M.mean_and_uncertainty(D_ld, "Cd")
    info18a = M.exceedance(D_ld, "Cd", "agricultural")
    info18r = M.exceedance(D_ld, "Cd", "residential")
    print("FIG17_Cd_mean_and_uncertainty:", info17)
    print("FIG18_Cd_exceedance:", info18a)
    print("FIG18_Cd_exceedance_residential:", info18r)

    print("\nescrito: figuras/TABLE_LD_principal_Cd.csv, FIG17_Cd_mean_and_uncertainty.pdf, "
          "FIG18_Cd_exceedance.pdf, FIG18_Cd_exceedance_residential.pdf, "
          "outputs/flip_ld_cd/maps_50m.npz")


if __name__ == "__main__":
    main()
