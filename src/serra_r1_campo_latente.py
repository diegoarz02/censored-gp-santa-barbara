"""SERRA-R1 T2 -- diagnostico de convergencia del campo latente y de las cantidades que usan los
mapas, en las 5 metales principales del sitio, SIN MCMC nuevo. Post-proceso puro sobre los idata
ya ajustados.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_serra as es  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import OUT  # noqa: E402
from flip_ld_cd_f2_f4 import build_grid, shared_setup  # noqa: E402

OUT_DIR = OUT / "serra_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDATA_FILES = {
    "As": "main_As.nc", "Ba": "main_Ba.nc", "Pb": "main_Pb.nc",
    "Hg": "probe_Hg_strict_2026-10-26_ta0999.nc", "Cd": "flip_LD_Cd.nc",
}
ECA_AGRI = {"Hg": 6.6, "Pb": 70.0, "As": 50.0, "Ba": 750.0, "Cd": 1.4}
AREA_METALS = ["Cd", "Hg", "Pb", "As"]       # T2.2.a: Ba excluido por diseno de la tarea


class _IdentityChoiceRNG:
    """Fuerza a sb.predict_field a usar TODOS los draws en orden natural (chain-mayor), para
    conservar la estructura de cadena que R-hat/ESS necesitan -- su propio rng.choice() por
    defecto baraja los indices, lo que destruiria la mitad de cadena que usa el split-R-hat. El
    ruido de realizacion (rng.standard_normal) si sigue siendo aleatorio: eso es la incertidumbre
    predictiva real, no algo que deba preservar orden."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    def choice(self, n, size=None, replace=True):
        return np.arange(n)

    def standard_normal(self, size):
        return self._rng.standard_normal(size)


def field_diag_row(idata, metal, varname):
    post = idata.posterior
    da = post[varname]
    rhat = az.rhat(idata, var_names=[varname])[varname].values.ravel()
    ess_b = az.ess(idata, var_names=[varname], method="bulk")[varname].values.ravel()
    ess_t = az.ess(idata, var_names=[varname], method="tail")[varname].values.ravel()
    return {
        "metal": metal, "variable": varname, "n_dims": len(rhat),
        "rhat_min": float(np.min(rhat)), "rhat_q05": float(np.quantile(rhat, 0.05)),
        "rhat_q50": float(np.quantile(rhat, 0.50)), "rhat_q95": float(np.quantile(rhat, 0.95)),
        "rhat_max": float(np.max(rhat)),
        "ess_bulk_min": float(np.min(ess_b)), "ess_bulk_q05": float(np.quantile(ess_b, 0.05)),
        "ess_bulk_q50": float(np.quantile(ess_b, 0.50)), "ess_bulk_q95": float(np.quantile(ess_b, 0.95)),
        "ess_bulk_max": float(np.max(ess_b)),
        "ess_tail_min": float(np.min(ess_t)), "ess_tail_max": float(np.max(ess_t)),
        "pct_rhat_ge_1.01": float(100 * np.mean(rhat >= 1.01)),
        "pct_ess_bulk_lt_400": float(100 * np.mean(ess_b < 400)),
        "pct_ess_tail_lt_400": float(100 * np.mean(ess_t < 400)),
    }


def t2_1(idatas):
    rows = []
    for metal, idata in idatas.items():
        post = idata.posterior
        field_var = "f" if "f" in post else "f_rotated_"
        rows.append(field_diag_row(idata, metal, field_var))
        if "mu" in post:
            rows.append(field_diag_row(idata, metal, "mu"))
        else:
            rows.append({"metal": metal, "variable": "mu", "n_dims": 0,
                        "rhat_min": None, "nota": "mu no esta guardado en este idata"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "campo_latente_diag.csv", index=False, encoding="utf-8")
    return df


def predict_grid_full_chain(idata, xy, loc, workings, cov_mean, cov_std, xy_center,
                            grid_res_m=50.0, buffer_m=300.0, seed=0):
    """Predice en la grilla oficial conservando (chain, draw) -- TODOS los draws, sin submuestreo
    aleatorio, reutilizando build_grid()/predict_field() ya validados. Devuelve Y (log-escala,
    con ruido de observacion, la misma cantidad que alimenta los mapas publicados) y MU
    (media condicional, sin ruido) con forma (chain, draw, n_cells)."""
    grid_e, grid_n, xy_grid, cov_grid, is_bg_grid = build_grid(
        loc, workings, grid_res_m, buffer_m, cov_mean, cov_std, xy_center)
    n_chain = idata.posterior.sizes["chain"]
    n_draw = idata.posterior.sizes["draw"]
    flat = n_chain * n_draw
    pr = sb.predict_field(idata, xy, xy_grid, cov_grid, is_background_new=is_bg_grid,
                          n_samples=flat, seed=seed, rng=_IdentityChoiceRNG(seed),
                          anisotropic=True, include_noise=True, return_mean=True)
    Y = pr["y"].reshape(n_chain, n_draw, -1)
    MU = pr["mu"].reshape(n_chain, n_draw, -1)
    return grid_e, grid_n, Y, MU


def t2_2a(idatas, df, workings, cov_mean, cov_std, xy_center, xy):
    ck_path = OUT_DIR / "cantidades_mapa_diag_area.csv"
    done = pd.read_csv(ck_path) if ck_path.exists() else pd.DataFrame(columns=["metal"])
    rows = list(done.to_dict("records"))
    for metal in AREA_METALS:
        if metal in done.get("metal", pd.Series(dtype=str)).values:
            print(f"  [T2.2a] {metal}: [cache]", flush=True)
            continue
        t0 = time.time()
        idata = idatas[metal]
        loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
        grid_e, grid_n, Y, MU = predict_grid_full_chain(idata, xy, loc, workings, cov_mean,
                                                         cov_std, xy_center, seed=20260906)
        conc = np.exp(Y)
        area_frac = (conc > ECA_AGRI[metal]).mean(axis=-1)    # (chain, draw)
        idata_area = az.from_dict({"posterior": {"x": area_frac}})
        rhat = float(az.rhat(idata_area)["x"].values)
        ess_b = float(az.ess(idata_area, method="bulk")["x"].values)
        ess_t = float(az.ess(idata_area, method="tail")["x"].values)
        mcse_mean = float(az.mcse(idata_area, method="mean")["x"].values)
        flat = area_frac.ravel()
        median, q025, q975 = np.median(flat), np.quantile(flat, 0.025), np.quantile(flat, 0.975)
        mcse_median = float(az.mcse(idata_area, method="quantile", prob=0.5)["x"].values)
        mcse_q025 = float(az.mcse(idata_area, method="quantile", prob=0.025)["x"].values)
        mcse_q975 = float(az.mcse(idata_area, method="quantile", prob=0.975)["x"].values)
        dt = time.time() - t0
        rows.append({"metal": metal, "n_cells": Y.shape[-1], "n_chain": Y.shape[0],
                    "n_draw": Y.shape[1], "eca_agri": ECA_AGRI[metal],
                    "area_frac_mean_pct": 100 * float(area_frac.mean()),
                    "area_frac_median_pct": 100 * median, "area_frac_q025_pct": 100 * q025,
                    "area_frac_q975_pct": 100 * q975,
                    "rhat": rhat, "ess_bulk": ess_b, "ess_tail": ess_t,
                    "mcse_mean_pct": 100 * mcse_mean, "mcse_median_pct": 100 * mcse_median,
                    "mcse_q025_pct": 100 * mcse_q025, "mcse_q975_pct": 100 * mcse_q975,
                    "seconds": dt})
        print(f"  [T2.2a] {metal}: {dt:.0f}s area_frac_median={100*median:.2f}% "
             f"rhat={rhat:.4f} ess_bulk={ess_b:.0f}", flush=True)
        pd.DataFrame(rows).to_csv(ck_path, index=False, encoding="utf-8")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(ck_path, index=False, encoding="utf-8")
    return df_out, None


def t2_2b(df, workings, cov_mean, cov_std, xy_center, xy, idatas):
    ck = OUT_DIR / "cantidades_mapa_diag_cd_cellwise.csv"
    if ck.exists():
        out = pd.read_csv(ck)
        unstable = out["unstable_0.5"].to_numpy(bool)
        print(f"  [T2.2b] [cache] {ck}")
        return dict(n_cells=len(out), pct_unstable=100 * float(unstable.mean()),
                   mcse_min=float(out["mcse"].min()), mcse_max=float(out["mcse"].max()),
                   mcse_median=float(out["mcse"].median()))
    idata_cd = idatas["Cd"]
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    grid_e, grid_n, Y, MU = predict_grid_full_chain(idata_cd, xy, loc, workings, cov_mean,
                                                     cov_std, xy_center, seed=20260906)
    conc = np.exp(Y)
    exceeds = (conc > ECA_AGRI["Cd"]).astype(float)           # (chain, draw, cell)
    p_exc = exceeds.mean(axis=(0, 1))                        # (cell,)
    n_cells = exceeds.shape[-1]
    mcse_cell = np.empty(n_cells)
    for c in range(n_cells):
        idata_c = az.from_dict({"posterior": {"x": exceeds[:, :, c]}})
        mcse_cell[c] = float(az.mcse(idata_c, method="mean")["x"].values)
    unstable = np.abs(p_exc - 0.5) < 2 * mcse_cell
    out = pd.DataFrame({"grid_e": grid_e, "grid_n": grid_n, "p_exceedance": p_exc,
                       "mcse": mcse_cell, "unstable_0.5": unstable})
    out.to_csv(OUT_DIR / "cantidades_mapa_diag_cd_cellwise.csv", index=False, encoding="utf-8")
    pct_unstable = 100 * float(unstable.mean())
    print(f"  [T2.2b] Cd: {pct_unstable:.2f}% de {n_cells} celdas con clasificacion inestable "
         f"(|P-0.5| < 2*MCSE)")
    return dict(n_cells=n_cells, pct_unstable=pct_unstable, mcse_min=float(mcse_cell.min()),
               mcse_max=float(mcse_cell.max()), mcse_median=float(np.median(mcse_cell)))


SUPPORT_M = 100.0   # soporte de datos exigido a las celdas de "fondo" y "cerca de labores"


def select_locations(workings, loc, grid_e, grid_n):
    """6 celdas de la grilla de 50 m, con reglas declaradas de antemano (no ajustadas a ningun
    resultado): con soporte de datos = a <= SUPPORT_M del dato mas cercano.
      - cerca de labores: las 2 celdas con soporte mas cercanas a una labor minera real
        (mine_workings.csv: bocaminas y desmontes);
      - fondo: las 2 celdas con soporte MAS LEJANAS de cualquier labor;
      - borde de extrapolacion: las 2 celdas mas lejanas de cualquier dato."""
    G = np.c_[grid_e, grid_n]
    XY = loc[["easting", "northing"]].to_numpy(float)
    WK = workings[["easting", "northing"]].to_numpy(float)
    d_obs = np.sqrt(((G[:, None] - XY[None]) ** 2).sum(-1)).min(1)
    d_work = np.sqrt(((G[:, None] - WK[None]) ** 2).sum(-1)).min(1)
    sup = np.flatnonzero(d_obs <= SUPPORT_M)
    near = sup[np.argsort(d_work[sup])[:2]]
    background = sup[np.argsort(-d_work[sup])[:2]]
    extrap = np.argsort(-d_obs)[:2]
    idx6 = np.concatenate([background, near, extrap])
    labels6 = ["background_1", "background_2", "near_working_1", "near_working_2",
              "extrapolation_edge_1", "extrapolation_edge_2"]
    return idx6, labels6, d_obs, d_work


def predict_cells(idata, xy, xy_grid, cov_grid, is_bg_grid, idx, seed):
    n_chain = idata.posterior.sizes["chain"]
    n_draw = idata.posterior.sizes["draw"]
    pr = sb.predict_field(idata, xy, xy_grid[idx], cov_grid[idx],
                          is_background_new=is_bg_grid[idx], n_samples=n_chain * n_draw,
                          seed=seed, rng=_IdentityChoiceRNG(seed), anisotropic=True,
                          include_noise=True, return_mean=True)
    return pr["mu"].reshape(n_chain, n_draw, -1)


def t2_2c(df, workings, cov_mean, cov_std, xy_center, xy, idatas):
    ck = OUT_DIR / "cantidades_mapa_diag_ubicaciones.csv"
    if ck.exists():
        print(f"  [T2.2c] [cache] {ck}")
        return pd.read_csv(ck)
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    grid_e, grid_n, xy_grid, cov_grid, is_bg_grid = build_grid(
        loc, workings, 50.0, 300.0, cov_mean, cov_std, xy_center)
    idx6, labels6, d_obs, d_work = select_locations(workings, loc, grid_e, grid_n)
    print(f"  [T2.2c] celdas elegidas (indice, E, N, d_obs, d_labor): "
          + "; ".join(f"{lab}=({i},{grid_e[i]:.0f},{grid_n[i]:.0f},{d_obs[i]:.0f}m,{d_work[i]:.0f}m)"
                      for i, lab in zip(idx6, labels6)), flush=True)

    # la media condicional de una celda no depende de que otras celdas se predigan a la vez:
    # se comprueba (Cd) comparando 6 celdas contra 6 + 100 celdas al azar, mismo seed.
    rng_chk = np.random.default_rng(1)
    extra = rng_chk.choice(np.setdiff1d(np.arange(len(grid_e)), idx6), 100, replace=False)
    mu_a = predict_cells(idatas["Cd"], xy, xy_grid, cov_grid, is_bg_grid, idx6, 20260906)
    mu_b = predict_cells(idatas["Cd"], xy, xy_grid, cov_grid, is_bg_grid,
                         np.concatenate([idx6, extra]), 20260906)[:, :, :6]
    print(f"  [T2.2c] invarianza al subconjunto de celdas (Cd): max|dMU| = "
          f"{float(np.abs(mu_a - mu_b).max()):.2e}", flush=True)

    rows = []
    for metal in ["As", "Ba", "Cd", "Hg", "Pb"]:
        MU = predict_cells(idatas[metal], xy, xy_grid, cov_grid, is_bg_grid, idx6, 20260906)
        for k, (i, lab) in enumerate(zip(idx6, labels6)):
            vals = MU[:, :, k]                             # (chain, draw)
            idata_v = az.from_dict({"posterior": {"x": vals}})
            rows.append({"metal": metal, "location_label": lab, "d_obs_m": float(d_obs[i]),
                        "d_work_m": float(d_work[i]), "easting": float(grid_e[i]),
                        "northing": float(grid_n[i]), "mu_mean": float(vals.mean()),
                        "rhat": float(az.rhat(idata_v)["x"].values),
                        "ess_bulk": float(az.ess(idata_v, method="bulk")["x"].values),
                        "ess_tail": float(az.ess(idata_v, method="tail")["x"].values)})
        print(f"  [T2.2c] {metal} listo", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ck, index=False, encoding="utf-8")
    return out


def t2_1_por_dimension(idatas):
    """Valores crudos por dimension (R-hat, ESS bulk, ESS tail) del campo latente f y de eta(s)=mu
    en las 114 ubicaciones: lo que dibuja FigS1 y lo que permite recalcular campo_latente_diag.csv."""
    ck = OUT_DIR / "campo_latente_por_dimension.csv"
    if ck.exists():
        print(f"[cache] {ck}")
        return pd.read_csv(ck)
    frames = []
    for metal, idata in idatas.items():
        post = idata.posterior
        for var in (["f" if "f" in post else "f_rotated_"] + (["mu"] if "mu" in post else [])):
            rhat = az.rhat(idata, var_names=[var])[var].values.ravel()
            ess_b = az.ess(idata, var_names=[var], method="bulk")[var].values.ravel()
            ess_t = az.ess(idata, var_names=[var], method="tail")[var].values.ravel()
            frames.append(pd.DataFrame({"metal": metal, "variable": var, "dim": np.arange(len(rhat)),
                                       "rhat": rhat, "ess_bulk": ess_b, "ess_tail": ess_t}))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(ck, index=False, encoding="utf-8")
    return out


def fig_s1(dim_df):
    es.apply_style_serra()
    import matplotlib.pyplot as plt
    f = dim_df[dim_df.variable.isin(["f", "f_rotated_"])]
    metals = ["As", "Ba", "Pb", "Hg", "Cd"]
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.46),
                             constrained_layout=True)
    ess_top = float(np.ceil(f["ess_bulk"].max() / 200.0) * 200.0)   # cubre TODO el rango observado
    for ax, col, bins, ref, xlab, letter in [
            (axes[0], "rhat", np.linspace(0.998, 1.012, 29), 1.01, r"$\hat{R}$", "a"),
            (axes[1], "ess_bulk", np.linspace(0, ess_top, int(ess_top / 100) + 1), 400,
             "ESS bulk", "b")]:
        for m in metals:
            v = f[f.metal == m][col].to_numpy()
            ax.hist(v, bins=bins, histtype="step", lw=1.3, color=es.METAL_COLOR[m],
                    ls=es.ef.METAL_LS[m], label=m)
        ax.axvline(ref, color="k", lw=1.0, ls="--")
        ax.set_xlabel(xlab, fontsize=es.FS_LABEL)
        ax.set_ylabel("number of latent-field dimensions", fontsize=es.FS_LABEL)
        es.panel_label(ax, letter)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5, frameon=False, fontsize=es.FS_LEGEND)
    info = es.save_fig_serra(fig, "FigS1_latent_diagnostics", 174)
    plt.close(fig)
    return info


def consolidar(area_df, cellwise, loc6_df):
    """cantidades_mapa_diag.csv: formato largo con TODO lo de T2.2 (menos el detalle por celda,
    que sigue en cantidades_mapa_diag_cd_cellwise.csv) -- el nombre y la forma de un solo archivo
    que pedia la tarea. Pura concatenacion de los CSV ya escritos, sin recalculo."""
    rows = []
    for _, r in area_df.iterrows():
        for stat in ["area_frac_median_pct", "area_frac_q025_pct", "area_frac_q975_pct", "rhat",
                     "ess_bulk", "ess_tail", "mcse_median_pct", "mcse_q025_pct", "mcse_q975_pct"]:
            rows.append({"quantity": "area_excedida_ECA_agricola", "metal": r["metal"],
                         "location_label": "", "statistic": stat, "value": r[stat]})
    for stat in ["n_cells", "pct_unstable", "mcse_min", "mcse_median", "mcse_max"]:
        rows.append({"quantity": "p_excedencia_celda", "metal": "Cd", "location_label": "",
                     "statistic": stat, "value": cellwise[stat]})
    for _, r in loc6_df.iterrows():
        for stat in ["d_obs_m", "d_work_m", "mu_mean", "rhat", "ess_bulk", "ess_tail"]:
            rows.append({"quantity": "ubicacion_representativa_mu", "metal": r["metal"],
                         "location_label": r["location_label"], "statistic": stat,
                         "value": r[stat]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cantidades_mapa_diag.csv", index=False, encoding="utf-8")
    return out


def main():
    idatas = {m: az.from_netcdf(OUT / "idata" / f) for m, f in IDATA_FILES.items()}
    df, workings, loc, xy, cov_mean, cov_std, xy_center = shared_setup()

    print("=== T2.1: diagnostico del campo latente (f, mu) por metal ===")
    ck1 = OUT_DIR / "campo_latente_diag.csv"
    if ck1.exists():
        field_df = pd.read_csv(ck1)
        print(f"[cache] {ck1}")
    else:
        field_df = t2_1(idatas)
    print(field_df.to_string(index=False))
    dim_df = t2_1_por_dimension(idatas)
    chk = dim_df[dim_df.variable.isin(["f", "f_rotated_"])].groupby("metal").agg(
        rhat_max=("rhat", "max"), ess_bulk_min=("ess_bulk", "min"))
    ref = field_df[field_df.variable.isin(["f", "f_rotated_"])].set_index("metal")
    print("coherencia por-dimension vs resumen: max|d rhat_max| = "
          f"{float((chk.rhat_max - ref.rhat_max).abs().max()):.2e}, max|d ess_bulk_min| = "
          f"{float((chk.ess_bulk_min - ref.ess_bulk_min).abs().max()):.2e}")

    print("\n=== T2.2a: area excedida por draw, conservando cadena ===")
    area_df, _ = t2_2a(idatas, df, workings, cov_mean, cov_std, xy_center, xy)

    print("\n=== T2.2b: MCSE por celda, Cd ===")
    cellwise = t2_2b(df, workings, cov_mean, cov_std, xy_center, xy, idatas)

    print("\n=== T2.2c: 6 ubicaciones representativas ===")
    loc6_df = t2_2c(df, workings, cov_mean, cov_std, xy_center, xy, idatas)

    print("\n=== FigS1 ===")
    info_fig = fig_s1(dim_df)

    pd.DataFrame([cellwise]).to_csv(OUT_DIR / "cantidades_mapa_diag_cd_summary.csv", index=False,
                                    encoding="utf-8")
    consolidar(area_df, cellwise, loc6_df)
    print("\nescrito: outputs/serra_r1/campo_latente_diag.csv, campo_latente_por_dimension.csv, "
         "cantidades_mapa_diag.csv (consolidado), cantidades_mapa_diag_area.csv, "
         "cantidades_mapa_diag_cd_cellwise.csv, cantidades_mapa_diag_cd_summary.csv, "
         "cantidades_mapa_diag_ubicaciones.csv, figuras/serra/FigS1_latent_diagnostics.pdf")


if __name__ == "__main__":
    main()
