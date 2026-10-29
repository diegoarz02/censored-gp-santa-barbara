"""Q1-T4 (ronda Q1): sensibilidad de la proporcion de area excedida del Cd a la resolucion del
grid (hoy 50m, 7381 celdas). Reutiliza EXACTAMENTE la logica de T2.1
(src/t2_1_area_excedida_posterior.py) parametrizando GRID_RES_M -- mismo idata ya ajustado
(main_Cd.nc), sin reajustar el modelo de datos, solo re-prediciendo con sb.predict_field sobre
grids de 100m (mas grueso) y 25m (mas fino), ademas del de 50m ya existente como referencia.
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
import arviz as az   # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DIR_COV = PROJ / "data" / "fase1" / "covariables"

CRS_UTM = "EPSG:32718"
ECA_CD_AGRI, ECA_CD_RESID = 1.4, 10.0
COVARS = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
SEED = 20260906
N_MAP_SAMPLES = 4000
BUFFER_M = 300.0
CELL_BLOCK = 1500
RESOLUTIONS_M = [100.0, 50.0, 25.0]   # 50m = control de regresion contra T2.1/el mapa oficial


def build_context_at_resolution(grid_res_m):
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

    e0, e1 = loc["easting"].min() - BUFFER_M, loc["easting"].max() + BUFFER_M
    n0, n1 = loc["northing"].min() - BUFFER_M, loc["northing"].max() + BUFFER_M
    ge = np.arange(e0, e1 + grid_res_m, grid_res_m)
    gn = np.arange(n0, n1 + grid_res_m, grid_res_m)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    n_cells = len(grid_e)

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
    is_bg_grid = np.zeros(n_cells, bool)

    return dict(xy=xy, n_cells=n_cells, xy_grid=xy_grid, cov_grid=cov_grid, is_bg_grid=is_bg_grid)


def run_at_resolution(grid_res_m):
    ctx = build_context_at_resolution(grid_res_m)
    idata = az.from_netcdf(OUT / "idata" / "main_Cd.nc")
    n_cells = ctx["n_cells"]

    p_agri_cell = np.zeros(n_cells)
    p_resid_cell = np.zeros(n_cells)
    count_agri_draw = np.zeros(N_MAP_SAMPLES)
    count_resid_draw = np.zeros(N_MAP_SAMPLES)

    t0 = time.time()
    for b0 in range(0, n_cells, CELL_BLOCK):
        b1 = min(b0 + CELL_BLOCK, n_cells)
        pr = sb.predict_field(idata, ctx["xy"], ctx["xy_grid"][b0:b1], ctx["cov_grid"][b0:b1],
                              is_background_new=ctx["is_bg_grid"][b0:b1],
                              n_samples=N_MAP_SAMPLES, seed=SEED, anisotropic=True)
        conc = np.exp(pr["y"])
        exceeds_agri = conc > ECA_CD_AGRI
        exceeds_resid = conc > ECA_CD_RESID
        p_agri_cell[b0:b1] = exceeds_agri.mean(0)
        p_resid_cell[b0:b1] = exceeds_resid.mean(0)
        count_agri_draw += exceeds_agri.sum(1)
        count_resid_draw += exceeds_resid.sum(1)
    dt = time.time() - t0

    area_frac_agri = count_agri_draw / n_cells
    area_frac_resid = count_resid_draw / n_cells

    def summarize(arr, prefix):
        return {f"{prefix}_mean": float(arr.mean()), f"{prefix}_median": float(np.median(arr)),
               f"{prefix}_sd": float(arr.std()),
               f"{prefix}_ci95_lo": float(np.percentile(arr, 2.5)),
               f"{prefix}_ci95_hi": float(np.percentile(arr, 97.5))}

    row = {"grid_res_m": grid_res_m, "n_cells": n_cells, "seconds": dt,
          "pct_domain_agri_gt_0.5": float(100 * (p_agri_cell > 0.5).mean()),
          "pct_domain_resid_gt_0.5": float(100 * (p_resid_cell > 0.5).mean())}
    row.update(summarize(area_frac_agri, "area_frac_agri"))
    row.update(summarize(area_frac_resid, "area_frac_resid"))
    print(f"res={grid_res_m:.0f}m ({n_cells} celdas): {dt:.0f}s | %dominio agri t=0.5={row['pct_domain_agri_gt_0.5']:.2f}% "
          f"| E[area] agri={row['area_frac_agri_mean']*100:.2f}% IC95=[{row['area_frac_agri_ci95_lo']*100:.2f},"
          f"{row['area_frac_agri_ci95_hi']*100:.2f}]%", flush=True)
    return row


def main():
    rows = [run_at_resolution(r) for r in RESOLUTIONS_M]
    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_Q1_T4_resolucion_area_Cd.csv", index=False, encoding="utf-8")

    # control de regresion: la fila de 50m debe coincidir con T2.1/el mapa oficial
    ref = tab[tab.grid_res_m == 50.0].iloc[0]
    print(f"\ncontrol de regresion (50m, debe ser 7381 celdas y 57.40%): "
          f"n_cells={ref['n_cells']:.0f}, %dominio={ref['pct_domain_agri_gt_0.5']:.4f}%")

    print("\n" + tab.round(4).to_string(index=False))
    print("\nescrito: figuras/TABLE_Q1_T4_resolucion_area_Cd.csv")


if __name__ == "__main__":
    main()
