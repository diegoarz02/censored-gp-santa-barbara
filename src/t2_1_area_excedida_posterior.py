"""T2.1 del blindaje Q1/Q2 (prompt de Cowork, 2026-10-28) -- LA TAREA MAS IMPORTANTE.

Ajuste del profe Luis #2 + Copilot #4 + ChatGPT G: el manuscrito reporta "57.4% del dominio con
P(excedencia)>0.5" como una cifra puntual, pero eso mezcla dos preguntas distintas. Este script
separa y calcula TRES cantidades, reutilizando el idata oficial ya guardado (main_{metal}.nc,
SIN reajustar el modelo) y replicando EXACTO el grid/seed/protocolo de
`notebooks/_src/nb02_f_maps.py` (el que de hecho genero outputs/posterior_maps.npz, el mapa
oficial publicado):

  (1) % del dominio con P(c>ECA)>t, para t en {0.5, 0.8, 0.95}  (clasificacion binaria por celda)
  (2) E[fraccion de area en excedencia] = media sobre draws de (fraccion de celdas que exceden)
  (3) Distribucion posterior COMPLETA de esa fraccion de area (una por cada uno de los N_MAP_SAMPLES
      draws) -- media, mediana, SD, IC90, IC95.

(1) viene de agregar sobre CELDAS (promediando sobre draws primero, por celda). (2)-(3) vienen de
agregar sobre DRAWS (promediando sobre celdas primero, por draw) -- son ejes de agregacion
distintos y por eso main.tex hoy las mezcla en una sola cifra.

Clave de reproducibilidad: `sb.predict_field(..., seed=SEED)` elige los MISMOS N_MAP_SAMPLES
indices de draw del posterior sin importar que bloque de celdas se le pase (ver
`notebooks/_src/sbmodel.py` L.333-337: el rng se siembra fresco en cada llamada y el primer
consumo es exactamente esa eleccion de indices) -- por eso se puede acumular, bloque a bloque sin
jamas tener la matriz completa (N_MAP_SAMPLES x N_CELLS) en memoria, EXACTAMENTE como ya hace
nb02_f_maps.py para el mapa oficial, pero guardando ademas el conteo por draw en vez de descartarlo.

NO se reajusta ningun modelo: predict_field es muestreo predictivo posterior sobre un idata ya
ajustado, no MCMC nuevo -- cumple la regla dura #4 (reutilizar antes que recalcular).
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
ECA = {"As": (50.0, 50.0, 140.0), "Pb": (70.0, 140.0, 800.0), "Cd": (1.4, 10.0, 22.0),
      "Hg": (6.6, 6.6, 24.0), "Ba": (750.0, 500.0, 2000.0)}
MAIN_METALS = ["Hg", "Pb", "As", "Ba", "Cd"]
COVARS = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
SEED = 20260906                 # SEED de nb02.py, el que de hecho genero posterior_maps.npz
GRID_RES_M, BUFFER_M, N_MAP_SAMPLES, CELL_BLOCK = 50.0, 300.0, 4000, 1500
THRESHOLDS_T = [0.5, 0.8, 0.95]


def build_shared_context():
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
    ge = np.arange(e0, e1 + GRID_RES_M, GRID_RES_M)
    gn = np.arange(n0, n1 + GRID_RES_M, GRID_RES_M)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    n_cells = len(grid_e)
    print(f"grid oficial: {len(ge)} x {len(gn)} = {n_cells} celdas de {GRID_RES_M:.0f} m "
          f"(buffer {BUFFER_M:.0f} m) -- debe ser 7381 para igualar el mapa publicado")

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

    # diagnostico de extrapolacion (T3.4): distancia de cada celda a la observacion mas cercana
    d_to_obs = np.sqrt(((xy_grid[:, None] - xy[None]) ** 2).sum(-1)).min(1)

    return dict(xy=xy, n_cells=n_cells, xy_grid=xy_grid, cov_grid=cov_grid,
               is_bg_grid=is_bg_grid, grid_e=grid_e, grid_n=grid_n, d_to_obs=d_to_obs)


def per_metal(ctx, metal):
    idata = az.from_netcdf(OUT / "idata" / f"main_{metal}.nc")
    n_cells = ctx["n_cells"]
    thr_agri, thr_resid = ECA[metal][0], ECA[metal][1]

    p_agri_cell = np.zeros(n_cells)
    p_resid_cell = np.zeros(n_cells)
    count_agri_draw = np.zeros(N_MAP_SAMPLES)     # acumulador por draw, a traves de bloques
    count_resid_draw = np.zeros(N_MAP_SAMPLES)

    t0 = time.time()
    for b0 in range(0, n_cells, CELL_BLOCK):
        b1 = min(b0 + CELL_BLOCK, n_cells)
        pr = sb.predict_field(idata, ctx["xy"], ctx["xy_grid"][b0:b1], ctx["cov_grid"][b0:b1],
                              is_background_new=ctx["is_bg_grid"][b0:b1],
                              n_samples=N_MAP_SAMPLES, seed=SEED, anisotropic=True)
        conc = np.exp(pr["y"])                     # (N_MAP_SAMPLES, block_size)
        exceeds_agri = conc > thr_agri
        exceeds_resid = conc > thr_resid
        p_agri_cell[b0:b1] = exceeds_agri.mean(0)
        p_resid_cell[b0:b1] = exceeds_resid.mean(0)
        count_agri_draw += exceeds_agri.sum(1)
        count_resid_draw += exceeds_resid.sum(1)
    dt = time.time() - t0

    area_frac_agri_draw = count_agri_draw / n_cells      # (N_MAP_SAMPLES,) -- la cantidad (3)
    area_frac_resid_draw = count_resid_draw / n_cells

    def summarize(arr):
        return dict(mean=float(arr.mean()), median=float(np.median(arr)), sd=float(arr.std()),
                   ci90_lo=float(np.percentile(arr, 5)), ci90_hi=float(np.percentile(arr, 95)),
                   ci95_lo=float(np.percentile(arr, 2.5)), ci95_hi=float(np.percentile(arr, 97.5)))

    row = {"metal": metal, "seconds": dt}
    for t in THRESHOLDS_T:
        row[f"pct_domain_P_agri_gt_{t}"] = float(100 * (p_agri_cell > t).mean())
        row[f"pct_domain_P_resid_gt_{t}"] = float(100 * (p_resid_cell > t).mean())
    s_agri = summarize(area_frac_agri_draw)
    s_resid = summarize(area_frac_resid_draw)
    for k, v in s_agri.items():
        row[f"area_frac_agri_{k}"] = v
    for k, v in s_resid.items():
        row[f"area_frac_resid_{k}"] = v

    print(f"{metal}: {dt:.0f}s | cls t=0.5/0.8/0.95 agri = "
          f"{row['pct_domain_P_agri_gt_0.5']:.2f}/{row['pct_domain_P_agri_gt_0.8']:.2f}/"
          f"{row['pct_domain_P_agri_gt_0.95']:.2f}% | E[area] agri = {s_agri['mean']*100:.2f}% "
          f"IC95=[{s_agri['ci95_lo']*100:.2f},{s_agri['ci95_hi']*100:.2f}]%")

    return row, p_agri_cell, p_resid_cell, area_frac_agri_draw, area_frac_resid_draw


def main():
    ctx = build_shared_context()
    assert ctx["n_cells"] == 7381, f"la grilla no coincide con la oficial (7381): {ctx['n_cells']}"

    rows = []
    arrays = {}
    for metal in MAIN_METALS:
        row, p_agri, p_resid, area_agri, area_resid = per_metal(ctx, metal)
        rows.append(row)
        arrays[metal] = dict(p_agri_cell=p_agri, p_resid_cell=p_resid,
                             area_frac_agri_draw=area_agri, area_frac_resid_draw=area_resid)

    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_area_excedida_posterior.csv", index=False, encoding="utf-8")
    np.savez_compressed(OUT / "area_excedida_posterior_arrays.npz",
                        **{f"{m}_{k}": v for m, d in arrays.items() for k, v in d.items()},
                        easting=ctx["grid_e"], northing=ctx["grid_n"], d_to_obs=ctx["d_to_obs"])

    # chequeo de consistencia contra el mapa oficial ya publicado (posterior_maps.npz)
    official = dict(np.load(OUT / "posterior_maps.npz"))
    print("\n=== Chequeo de consistencia contra outputs/posterior_maps.npz (oficial) ===")
    for metal in MAIN_METALS:
        p_official = official[f"{metal}_p_agri"]
        p_new = arrays[metal]["p_agri_cell"]
        pct_official = 100 * (p_official > 0.5).mean()
        pct_new = 100 * (p_new > 0.5).mean()
        max_cell_diff = float(np.abs(p_official - p_new).max())
        print(f"  {metal}: %dominio t=0.5 oficial={pct_official:.4f}% vs nuevo={pct_new:.4f}% | "
              f"dif. maxima por celda={max_cell_diff:.4f}")

    print("\n" + tab.round(4).to_string(index=False))
    print("\nescrito: figuras/TABLE_area_excedida_posterior.csv, "
          "outputs/area_excedida_posterior_arrays.npz")


if __name__ == "__main__":
    main()
