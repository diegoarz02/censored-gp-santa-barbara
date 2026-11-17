"""SERRA-R2 -- resultados de Hg con el ajuste reforzado (probe_Hg_strict_2026-10-26_ta0999.nc) en vez
del ajuste estandar (main_Hg.nc). Post-proceso puro de draws existentes: cero MCMC nuevo.

Etapas (cada una es reanudable: si su salida existe, se salta):
  hiper   hiperparametros, covariables, P(tau<1), aliasing, prior-posterior, PPC, factor del estrato,
          diseno de monitoreo y zona de extrapolacion de Hg, antes (main_Hg.nc) y despues (reforzado),
          con autovalidacion del "antes" contra los CSV ya existentes
  mapas   productos de mapa de Hg con el pipeline de nb02_f_maps.py (misma grilla, semilla, N_MAP_SAMPLES,
          bloques y predict_field); valida primero el pipeline con el ajuste estandar en el bloque 0
  tablas  tabla antes/despues y clasificacion celda a celda
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sbmodel as sb  # noqa: E402
import t2_1_area_excedida_posterior as T2  # noqa: E402
from config import FIG, OUT  # noqa: E402
from flip_ld_cd_f2_f4 import shared_setup  # noqa: E402

OUT_DIR = OUT / "serra_r2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDATA_OLD = OUT / "idata" / "main_Hg.nc"
IDATA_NEW = OUT / "idata" / "probe_Hg_strict_2026-10-26_ta0999.nc"
SEED = 20260906            # SEED de nb02.py (el de posterior_maps.npz); config.SEED es otra corrida
N_MAP_SAMPLES, CELL_BLOCK = 4000, 1500
COVARS = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
OTHER_IDATA = {"Pb": "main_Pb.nc", "As": "main_As.nc", "Ba": "main_Ba.nc",
               "Cd (r libre, main_Cd)": "main_Cd.nc", "Cd (L=LD, flip)": "flip_LD_Cd.nc"}
ECA_HG = 6.6


def load(path):
    return az.from_netcdf(path)


# ------------------------------------------------------------------------------------------- hiper
def hyper_row(idata):
    p = idata.posterior
    row = {}
    for nm in ["ell", "eta", "sigma_n", "sigma_tot", "rho", "beta0", "b_background", "log_tau",
               "limit_ratio"]:
        if nm not in p:
            continue
        v = p[nm].values.reshape(-1, *p[nm].shape[2:])
        if v.ndim == 2 and v.shape[1] == 2:
            for j, ax_ in enumerate(["E", "N"]):
                row[f"{nm}_{ax_}"] = float(v[:, j].mean())
        else:
            row[nm] = float(v.ravel().mean())
    b = p["beta"].values.reshape(-1, len(COVARS))
    for j, c in enumerate(COVARS):
        lo, hi = np.quantile(b[:, j], [0.025, 0.975])
        row[f"beta_{c}"] = float(b[:, j].mean())
        row[f"beta_{c}_lo"] = float(lo)
        row[f"beta_{c}_hi"] = float(hi)
        row[f"beta_{c}_sig"] = bool(lo > 0 or hi < 0)
    return row


def alias_row(idata):
    p = idata.posterior
    b, t = p["b_background"].values.ravel(), p["log_tau"].values.ravel()
    lo_b, hi_b = np.quantile(b, [0.025, 0.975])
    lo_t, hi_t = np.quantile(t, [0.025, 0.975])
    return {"b_background": b.mean(), "b_ci95_low": lo_b, "b_ci95_high": hi_b, "log_tau": t.mean(),
            "tau_ci95_low": lo_t, "tau_ci95_high": hi_t,
            "posterior_corr": float(np.corrcoef(b, t)[0, 1]),
            "p_logtau_lt0": float((t < 0).mean())}


def prior_post_rows(idata, prior_tab):
    p = idata.posterior
    out = {}
    for nm in ["ell", "sigma_tot", "rho", "b_background", "log_tau"]:
        v = p[nm].values
        v = v.reshape(-1, *v.shape[2:])
        v = v[:, 0] if v.ndim == 2 else v.ravel()
        w_post = float(np.diff(np.quantile(v, [0.025, 0.975]))[0])
        w_pri = float(prior_tab.loc[nm, "prior_width95"])
        out[nm] = (w_pri, w_post, w_post / w_pri)
    return out


def ppc_rows(idata, metal_arr, is_composite, n_rep=600, seed=SEED):
    r = np.random.default_rng(seed)
    p = idata.posterior
    flat = p.sizes["chain"] * p.sizes["draw"]
    idx = r.choice(flat, n_rep, replace=n_rep > flat)
    mu = p["mu"].values.reshape(flat, -1)[idx]
    sn = p["sigma_n"].values.reshape(flat)[idx]
    tau = np.exp(p["log_tau"].values.reshape(flat)[idx]) if "log_tau" in p else np.ones(n_rep)
    sd = sn[:, None] * np.where(is_composite[None, :], tau[:, None], 1.0)
    yrep = mu + r.standard_normal(mu.shape) * sd
    if "log_limit" in p:
        loglim = p["log_limit"].values.reshape(flat)[idx]
    elif np.isfinite(metal_arr["lod"]):
        loglim = np.full(n_rep, np.log(metal_arr["lod"]))
    else:
        loglim = np.full(n_rep, -np.inf)
    cens_rep = yrep < loglim[:, None]
    obs_det = np.log(metal_arr["value"][~metal_arr["censored"]])
    obs = {"median_log": np.median(obs_det), "sd_log": np.std(obs_det), "max_log": obs_det.max(),
           "pct_censored": 100 * metal_arr["censored"].mean()}
    out = {}
    for k, fn in [("median_log", np.median), ("sd_log", np.std), ("max_log", np.max)]:
        rep = np.array([fn(yrep[s][~cens_rep[s]]) if (~cens_rep[s]).any() else np.nan
                        for s in range(n_rep)])
        out[k] = (float(obs[k]), float(np.nanmean(rep)), float(np.nanmean(rep >= obs[k])))
    rep_c = 100 * cens_rep.mean(1)
    out["pct_censored"] = (float(obs["pct_censored"]), float(rep_c.mean()),
                           float(np.mean(rep_c >= obs["pct_censored"])))
    return out


def design_run(ell_post, eta_post, sigma_n_post, extrap_mask, xy, xy_grid, k=20):
    """Diseno voraz de nb02.py (L.2290-2400), sin cambios algebraicos."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    n_cells = len(xy_grid)
    cand = xy_grid[~extrap_mask & (np.arange(n_cells) % 7 == 0)]

    def kmat(A, B):
        a, b = A / ell_post, B / ell_post
        return eta_post ** 2 * sb._matern52(((a[:, None] - b[None]) ** 2).sum(-1), ell_post)

    def posterior_sd(xy_obs, xy_target):
        K = kmat(xy_obs, xy_obs)
        K[np.diag_indices(len(xy_obs))] += sigma_n_post ** 2
        Ks = kmat(xy_target, xy_obs)
        L = np.linalg.cholesky(K)
        v = solve_triangular(L, Ks.T, lower=True, check_finite=False)
        return np.sqrt(np.maximum(eta_post ** 2 - (v ** 2).sum(0), 1e-12))

    def sd_after_each_candidate(xy_obs, target):
        K = kmat(xy_obs, xy_obs)
        K[np.diag_indices(len(xy_obs))] += sigma_n_post ** 2
        c = cho_factor(K, lower=True)
        Kt, Kc = kmat(target, xy_obs), kmat(cand, xy_obs)
        var_t = np.maximum(eta_post ** 2 - (Kt * cho_solve(c, Kt.T).T).sum(1), 1e-12)
        var_c = np.maximum(eta_post ** 2 - (Kc * cho_solve(c, Kc.T).T).sum(1), 1e-12)
        cov_tc = kmat(target, cand) - Kt @ cho_solve(c, Kc.T)
        new_var = var_t[:, None] - cov_tc ** 2 / (var_c + sigma_n_post ** 2)[None, :]
        return np.sqrt(np.maximum(new_var, 1e-12)).mean(0)

    chosen, current = [], xy.copy()
    trace = [float(posterior_sd(current, xy_grid).mean())]
    for _ in range(k):
        sd_each = sd_after_each_candidate(current, xy_grid)
        sd_each[chosen] = np.inf
        j = int(sd_each.argmin())
        chosen.append(j)
        current = np.vstack([current, cand[j]])
        trace.append(float(sd_each[j]))
    return chosen, np.array(trace), cand


def stage_hiper():
    f_hip = OUT_DIR / "hg_hiperparametros_antes_despues.csv"
    if f_hip.exists():
        print(f"[cache] {f_hip.name}")
        return
    df, workings, loc, xy, cov_mean, cov_std, xy_center = shared_setup()
    ctx = T2.build_shared_context()
    idt = {"antes": load(IDATA_OLD), "despues": load(IDATA_NEW)}
    rows = []

    def add(tabla, cantidad, a, d, nota=""):
        rows.append({"tabla": tabla, "cantidad": cantidad, "antes_main_Hg": a,
                     "despues_reforzado": d, "diferencia": (d - a) if isinstance(a, float) else "",
                     "nota": nota})

    # ---- TABLE11 + intervalos de las covariables
    h = {k: hyper_row(v) for k, v in idt.items()}
    ref11 = pd.read_csv(FIG / "TABLE11_posterior_parameters.csv").set_index("analyte").loc["Hg"]
    chk = max(abs(h["antes"][c] - float(ref11[c])) for c in ref11.index
              if c in h["antes"] and not c.endswith("_sig") and np.isfinite(ref11[c]))
    print(f"[autovalidacion] TABLE11 Hg (antes) reproducida: max|dif| = {chk:.2e} (redondeo a 4 dec.)")
    for c in h["antes"]:
        if c.endswith("_sig"):
            add("TABLE11", c, str(h["antes"][c]), str(h["despues"][c]))
        else:
            add("TABLE11 / covariables", c, h["antes"][c], h["despues"][c])

    # ---- P(tau<1), aliasing
    al = {k: alias_row(v) for k, v in idt.items()}
    t11 = pd.read_csv(FIG / "TABLE_T1_1_varianza_marginal.csv").set_index("metal").loc["Hg"]
    print(f"[autovalidacion] P(log tau<0) Hg antes = {al['antes']['p_logtau_lt0']:.4f} "
          f"(TABLE_T1_1: {float(t11['p_logtau_lt0']):.4f})")
    for c in al["antes"]:
        add("TABLE12 / P(tau<1)", c, al["antes"][c], al["despues"][c],
            "el articulo cita P(tau<1)=0.28 para Hg" if c == "p_logtau_lt0" else "")

    # ---- TABLE13 prior-posterior (el ancho del prior no depende del ajuste: se lee de la TABLE13)
    pt = pd.read_csv(FIG / "TABLE13_prior_posterior.csv")
    pt = pt[pt.analyte == "Hg"].set_index("parameter")
    pp = {k: prior_post_rows(v, pt) for k, v in idt.items()}
    for nm in pp["antes"]:
        add("TABLE13", f"{nm}: posterior_width95", pp["antes"][nm][1], pp["despues"][nm][1])
        add("TABLE13", f"{nm}: width_ratio", pp["antes"][nm][2], pp["despues"][nm][2])

    # ---- TABLE14 PPC
    g = df[df.analyte == "Hg"].set_index("location_id").loc[loc.index]
    marr = {"value": g["value"].to_numpy(float), "censored": g["censored"].to_numpy(bool),
            "lod": float(g["reported_limit"].iloc[0]) if np.isfinite(g["reported_limit"].iloc[0])
            else np.nan}
    is_comp = (loc["sample_type"] == "composite").to_numpy()
    pc = {k: ppc_rows(v, marr, is_comp) for k, v in idt.items()}
    ref14 = pd.read_csv(FIG / "TABLE14_posterior_predictive.csv")
    ref14 = ref14[ref14.analyte == "Hg"].set_index("statistic")
    print("[autovalidacion] PPC Hg antes vs TABLE14 (p_bayes): " + ", ".join(
        f"{s}: {pc['antes'][s][2]:.4f}/{float(ref14.loc[s, 'p_bayes']):.4f}" for s in pc["antes"]))
    for s in pc["antes"]:
        add("TABLE14 PPC", f"{s}: replicated_mean", pc["antes"][s][1], pc["despues"][s][1])
        add("TABLE14 PPC", f"{s}: p_bayes", pc["antes"][s][2], pc["despues"][s][2])

    # ---- TABLE19 factor del estrato (mismo b_background que el aliasing)
    b_old, b_new = al["antes"]["b_background"], al["despues"]["b_background"]
    add("TABLE19", "mining_factor = exp(-b_background)", float(np.exp(-b_old)), float(np.exp(-b_new)))
    add("TABLE19", "excludes_zero", str(bool(al["antes"]["b_ci95_high"] < 0 or al["antes"]["b_ci95_low"] > 0)),
        str(bool(al["despues"]["b_ci95_high"] < 0 or al["despues"]["b_ci95_low"] > 0)))

    # ---- anisotropia recuperada
    for k_, v in idt.items():
        e = v.posterior["ell"].values.reshape(-1, 2)
        h[k_]["aniso_NE"] = float(e[:, 1].mean() / e[:, 0].mean())
    add("anisotropia", "razon ell_N/ell_E", h["antes"]["aniso_NE"], h["despues"]["aniso_NE"])

    # ---- extrapolacion
    xyg, d_obs = ctx["xy_grid"], ctx["d_to_obs"]
    ell = {}
    for lab, f in [("Hg_std", IDATA_OLD), ("Hg_ref", IDATA_NEW)] + [
            (k, OUT / "idata" / v) for k, v in OTHER_IDATA.items()]:
        ell[lab] = float(load(f).posterior["ell"].values.reshape(-1, 2).mean())
    pb_as_ba = [ell["Pb"], ell["As"], ell["Ba"]]
    var = {"A. original de nb02 (Hg estandar, Cd r libre)":
           np.mean([ell["Hg_std"]] + pb_as_ba + [ell["Cd (r libre, main_Cd)"]]),
           "B. Hg reforzado, resto igual (la unica diferencia de R2)":
           np.mean([ell["Hg_ref"]] + pb_as_ba + [ell["Cd (r libre, main_Cd)"]]),
           "C. Hg reforzado y Cd L=LD (no es parte de R2)":
           np.mean([ell["Hg_ref"]] + pb_as_ba + [ell["Cd (L=LD, flip)"]]),
           "D. constante fija de mapas_f3.py (la que dibujan las figuras)": 1.03}
    erows = []
    for k_, v in var.items():
        m = d_obs > 1.5 * v
        erows.append({"variante": k_, "ell_typ_km": v, "umbral_km": 1.5 * v,
                      "celdas_extrapoladas": int(m.sum()), "pct_dominio": float(100 * m.mean())})
    pd.DataFrame(erows).to_csv(OUT_DIR / "hg_extrapolacion.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(erows).round(4).to_string(index=False))
    ell_typ_A = list(var.values())[0]
    ell_typ_B = list(var.values())[1]

    # ---- diseno de monitoreo (usa los parametros de Hg y la mascara de extrapolacion)
    def design_for(idata, ell_typ):
        p = idata.posterior
        ell_post = p["ell"].values.reshape(-1, 2).mean(0)
        eta_post = float(p["eta"].mean())
        sn_post = float(p["sigma_n"].mean())
        return design_run(ell_post, eta_post, sn_post, d_obs > 1.5 * ell_typ, ctx["xy"], xyg)

    ch_a, tr_a, cand_a = design_for(idt["antes"], ell_typ_A)
    cache = np.load(OUT / "monitoring_design.npz")
    same = list(cache["chosen20"]) == ch_a
    print(f"[autovalidacion] diseno 'antes' recalculado = monitoring_design.npz: puntos iguales = {same}; "
          f"max|dif traza| = {np.abs(cache['sd_trace'] - tr_a).max():.2e}")
    ch_b, tr_b, cand_b = design_for(idt["despues"], ell_typ_B)
    ch_c, tr_c, cand_c = design_for(idt["despues"], ell_typ_A)       # solo el cambio del ajuste, misma mascara
    red = lambda t: 100 * (t[0] - t) / t[0]            # noqa: E731
    dz = pd.DataFrame({"n_new_points": np.arange(21), "mean_sd_antes": tr_a,
                       "reduction_pct_antes": red(tr_a),
                       "mean_sd_despues_ell_typ_B": tr_b, "reduction_pct_despues_ell_typ_B": red(tr_b),
                       "mean_sd_despues_misma_mascara": tr_c,
                       "reduction_pct_despues_misma_mascara": red(tr_c)})
    dz.to_csv(OUT_DIR / "hg_diseno_monitoreo_antes_despues.csv", index=False, encoding="utf-8")
    for kk in (5, 10, 20):
        add("TABLE21 diseno", f"reduccion % con {kk} puntos nuevos", float(red(tr_a)[kk]),
            float(red(tr_b)[kk]), "despues con ell_typ_B (mascara recalculada)")
    for kk in (5, 10, 20):
        ov = len({tuple(cand_a[j]) for j in ch_a[:kk]} & {tuple(cand_b[j]) for j in ch_b[:kk]})
        add("TABLE22 diseno", f"puntos coincidentes entre antes y despues, primeros {kk}", float(kk),
            float(ov), f"{ov} de {kk} puntos coinciden (por coordenadas)")

    add("parametros de diseno", "ell_E (km)", float(h["antes"]["ell_E"]), float(h["despues"]["ell_E"]))
    add("parametros de diseno", "ell_N (km)", float(h["antes"]["ell_N"]), float(h["despues"]["ell_N"]))
    add("parametros de diseno", "eta", float(h["antes"]["eta"]), float(h["despues"]["eta"]))
    add("parametros de diseno", "sigma_n", float(h["antes"]["sigma_n"]), float(h["despues"]["sigma_n"]))

    # ---- intervalos de las covariables, los cinco metales (frase del articulo sobre las covariables)
    crow = []
    for lab, f in [("Hg (estandar, main_Hg)", IDATA_OLD), ("Hg (reforzado)", IDATA_NEW)] + [
            (k, OUT / "idata" / v) for k, v in OTHER_IDATA.items()]:
        b = load(f).posterior["beta"].values.reshape(-1, len(COVARS))
        for j, c in enumerate(COVARS):
            lo, hi = np.quantile(b[:, j], [0.025, 0.975])
            crow.append({"modelo": lab, "covariable": c, "media": float(b[:, j].mean()),
                         "ic95_bajo": float(lo), "ic95_alto": float(hi),
                         "excluye_cero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(crow).to_csv(OUT_DIR / "hg_covariables_ic95_cinco_metales.csv", index=False,
                              encoding="utf-8")
    pd.DataFrame(rows).to_csv(f_hip, index=False, encoding="utf-8")
    print(f"escrito: {f_hip.name}, hg_extrapolacion.csv, hg_diseno_monitoreo_antes_despues.csv, "
          f"hg_covariables_ic95_cinco_metales.csv")


# ------------------------------------------------------------------------------------------ mapas
def map_products(idata, ctx, only_blocks=None, seed=SEED, quiet=False):
    n_cells = ctx["n_cells"]
    acc = {k: np.zeros(n_cells) for k in ["mean", "median", "sd_f", "sd_y", "p_agri", "p_resid"]}
    cnt_a, cnt_r = np.zeros(N_MAP_SAMPLES), np.zeros(N_MAP_SAMPLES)
    for bi, b0 in enumerate(range(0, n_cells, CELL_BLOCK)):
        if only_blocks is not None and bi not in only_blocks:
            continue
        b1 = min(b0 + CELL_BLOCK, n_cells)
        pr = sb.predict_field(idata, ctx["xy"], ctx["xy_grid"][b0:b1], ctx["cov_grid"][b0:b1],
                              is_background_new=ctx["is_bg_grid"][b0:b1], n_samples=N_MAP_SAMPLES,
                              seed=seed, anisotropic=True)
        conc = np.exp(pr["y"])
        acc["mean"][b0:b1] = conc.mean(0)
        acc["median"][b0:b1] = np.median(conc, 0)
        acc["sd_f"][b0:b1] = pr["f"].std(0)
        acc["sd_y"][b0:b1] = pr["y"].std(0)
        acc["p_agri"][b0:b1] = (conc > ECA_HG).mean(0)
        acc["p_resid"][b0:b1] = (conc > ECA_HG).mean(0)
        cnt_a += (conc > ECA_HG).sum(1)
        cnt_r += (conc > ECA_HG).sum(1)
        if not quiet:
            print(f"    bloque {bi}: celdas {b0}-{b1 - 1}", flush=True)
    return acc, cnt_a / n_cells, cnt_r / n_cells


def stage_mapas():
    f_npz = OUT_DIR / "posterior_maps_Hg_refuerzo.npz"
    if f_npz.exists():
        print(f"[cache] {f_npz.name}")
        return
    ctx = T2.build_shared_context()
    assert ctx["n_cells"] == 7381
    official = dict(np.load(OUT / "posterior_maps.npz"))

    f_val = OUT_DIR / "validacion_pipeline_bloque0.csv"
    if not f_val.exists():
        t0 = time.time()
        print("[validacion] pipeline con main_Hg.nc, bloque 0 (celdas 0-1499) contra posterior_maps.npz")
        acc, _, _ = map_products(load(IDATA_OLD), ctx, only_blocks={0})
        rows = []
        for k in acc:
            d = np.abs(acc[k][:CELL_BLOCK] - official[f"Hg_{k}"][:CELL_BLOCK])
            rows.append({"clave": f"Hg_{k}", "celdas": CELL_BLOCK, "max_abs_dif": float(d.max()),
                         "max_rel_dif": float((d / np.maximum(np.abs(official[f"Hg_{k}"][:CELL_BLOCK]), 1e-12)).max())})
        v = pd.DataFrame(rows)
        v.to_csv(f_val, index=False, encoding="utf-8")
        print(v.to_string(index=False), f"\n  ({time.time() - t0:.0f} s)")
        assert v["max_abs_dif"].max() < 1e-6, "el pipeline NO reproduce posterior_maps.npz: no se sigue"

    t0 = time.time()
    print("[mapas] pipeline con el ajuste reforzado, 7381 celdas")
    idata = load(IDATA_NEW)
    acc, area_a, area_r = map_products(idata, ctx)
    out = {"easting": ctx["grid_e"], "northing": ctx["grid_n"], "n_samples": np.array([N_MAP_SAMPLES]),
           "seed": np.array([SEED]), "source_idata": np.array(IDATA_NEW.name),
           "Hg_area_frac_agri_draw": area_a, "Hg_area_frac_resid_draw": area_r}
    for k, v in acc.items():
        out[f"Hg_{k}"] = v
    np.savez_compressed(f_npz, **out)
    print(f"[mapas] {time.time() - t0:.0f} s -> {f_npz.name}")


def stage_ruido():
    """Ruido Monte Carlo de las cifras de la Tabla 9: mismo pipeline con otras semillas (no cambia el
    ajuste ni la grilla), para saber cuanto de la diferencia antes/despues es mas grande que el ruido."""
    f_out = OUT_DIR / "hg_ruido_montecarlo_semillas.csv"
    if f_out.exists():
        print(f"[cache] {f_out.name}")
        return
    ctx = T2.build_shared_context()
    rows = []
    for lab, path in [("antes (main_Hg.nc)", IDATA_OLD), ("despues (reforzado)", IDATA_NEW)]:
        idata = load(path)
        for seed in [SEED, SEED + 1, SEED + 2, SEED + 3]:
            t0 = time.time()
            acc, area_a, _ = map_products(idata, ctx, seed=seed, quiet=True)
            p = acc["p_agri"]
            rows.append({"ajuste": lab, "semilla": seed, "pct_P_gt_0.5": float(100 * (p > 0.5).mean()),
                         "pct_P_gt_0.8": float(100 * (p > 0.8).mean()),
                         "pct_P_gt_0.95": float(100 * (p > 0.95).mean()),
                         "area_excedida_mediana_pct": float(100 * np.median(area_a)),
                         "segundos": time.time() - t0})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(f_out, index=False, encoding="utf-8")
    print(f"escrito: {f_out.name}")


# Tabla 4 del articulo (tab:conv, main_FINAL_v15.tex L.336-346): lo que imprime, por metal.
TAB4_ARTICULO = {"Hg": (1.0059, 467, 693, IDATA_NEW.name), "Pb": (1.0068, 752, 1272, "main_Pb.nc"),
                 "As": (1.0064, 913, 1205, "main_As.nc"), "Ba": (1.0035, 1026, 1323, "main_Ba.nc"),
                 "Cd": (1.0051, 767, 845, "flip_LD_Cd.nc")}


def stage_tabla4():
    f_out = OUT_DIR / "tabla4_convergencia_articulo.csv"
    if f_out.exists():
        print(f"[cache] {f_out.name}")
        return
    rows = []
    for metal, (r_art, eb_art, et_art, fname) in TAB4_ARTICULO.items():
        d = sb.diagnose(load(OUT / "idata" / fname))
        rows.append({"metal": metal, "idata": fname, "rhat_max_calculado": d["rhat_max"],
                     "rhat_max_articulo": r_art, "ess_bulk_calculado": d["ess_bulk_min"],
                     "ess_bulk_articulo": eb_art, "ess_tail_calculado": d["ess_tail_min"],
                     "ess_tail_articulo": et_art, "divergencias": d["divergences"],
                     "coincide": bool(round(d["rhat_max"], 4) == r_art and round(d["ess_bulk_min"]) == eb_art
                                      and round(d["ess_tail_min"]) == et_art)})
    t = pd.DataFrame(rows)
    t.to_csv(f_out, index=False, encoding="utf-8")
    print(t.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------ tablas
def summarize(arr):
    return dict(mean=float(arr.mean()), median=float(np.median(arr)),
                ci95_lo=float(np.percentile(arr, 2.5)), ci95_hi=float(np.percentile(arr, 97.5)))


def stage_tablas():
    f_out = OUT_DIR / "hg_antes_despues.csv"
    new = dict(np.load(OUT_DIR / "posterior_maps_Hg_refuerzo.npz", allow_pickle=True))
    old = dict(np.load(OUT / "posterior_maps.npz"))
    arr_old = dict(np.load(OUT / "area_excedida_posterior_arrays.npz"))
    rows = []

    def add(q, a, d, art="", impreso=""):
        rows.append({"cantidad": q, "antes_main_Hg": a, "despues_reforzado": d,
                     "diferencia": d - a, "en_el_articulo": art, "impreso": impreso})

    p_old, p_new = old["Hg_p_agri"], new["Hg_p_agri"]
    for t in (0.5, 0.8, 0.95):
        a = float(100 * (p_old > t).mean())
        d = float(100 * (p_new > t).mean())
        add(f"% del dominio con P(Hg > 6.6) > {t} (agricola = residencial: mismo ECA)", a, d,
            "Tabla 9 (tab:exc), fila Hg" + (" y texto" if t == 0.5 else ""),
            f"{a:.1f} -> {d:.1f}")
    sa_old = summarize(arr_old["Hg_area_frac_agri_draw"] * 100)
    sa_new = summarize(new["Hg_area_frac_agri_draw"] * 100)
    for k, lab in [("median", "mediana"), ("ci95_lo", "cuantil 2.5 %"), ("ci95_hi", "cuantil 97.5 %"),
                   ("mean", "media")]:
        add(f"area excedida (% del dominio), {lab}", sa_old[k], sa_new[k], "no impreso para Hg",
            f"{sa_old[k]:.1f} -> {sa_new[k]:.1f}")
    for key, lab in [("median", "mediana posterior de la concentracion (mg/kg)"),
                     ("mean", "media posterior de la concentracion (mg/kg)")]:
        add(f"{lab}: minimo en la grilla", float(old[f"Hg_{key}"].min()), float(new[f"Hg_{key}"].min()))
        add(f"{lab}: maximo en la grilla", float(old[f"Hg_{key}"].max()), float(new[f"Hg_{key}"].max()))
    add("sd predictivo (log): media sobre la grilla", float(old["Hg_sd_y"].mean()),
        float(new["Hg_sd_y"].mean()), "Fig. 8 (barra de sd)")
    add("sd predictivo (log): maximo", float(old["Hg_sd_y"].max()), float(new["Hg_sd_y"].max()))
    add("P(exceder) maxima en la grilla", float(p_old.max()), float(p_new.max()))
    add("celdas con P > 0.5", float((p_old > 0.5).sum()), float((p_new > 0.5).sum()))
    add("celdas con P > 0.9", float((p_old > 0.9).sum()), float((p_new > 0.9).sum()))
    pd.DataFrame(rows).to_csv(f_out, index=False, encoding="utf-8")

    agree = []
    for t in (0.5, 0.8, 0.95):
        a, d = p_old > t, p_new > t
        agree.append({"umbral_P": t, "acuerdo_celda_a_celda_pct": float(100 * (a == d).mean()),
                      "celdas_que_cambian": int((a != d).sum()),
                      "de_excedente_a_no": int((a & ~d).sum()), "de_no_a_excedente": int((~a & d).sum())})
    ag = pd.DataFrame(agree)
    ag.to_csv(OUT_DIR / "hg_acuerdo_clasificacion.csv", index=False, encoding="utf-8")
    dif = {"corr_log_mediana": float(np.corrcoef(np.log(old["Hg_median"]), np.log(new["Hg_median"]))[0, 1]),
           "max_abs_dif_log_mediana": float(np.abs(np.log(old["Hg_median"]) - np.log(new["Hg_median"])).max()),
           "max_abs_dif_P": float(np.abs(p_old - p_new).max()),
           "max_abs_dif_sd_y": float(np.abs(old["Hg_sd_y"] - new["Hg_sd_y"]).max()),
           "mediana_dif_rel_mediana_conc_pct": float(100 * np.median(np.abs(new["Hg_median"] / old["Hg_median"] - 1)))}
    pd.DataFrame([dif]).to_csv(OUT_DIR / "hg_diferencias_mapas.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(ag.round(4).to_string(index=False))
    print(dif)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hiper", "mapas", "tablas", "ruido", "tabla4", "todo"], default="todo")
    a = ap.parse_args()
    if a.stage in ("hiper", "todo"):
        stage_hiper()
    if a.stage in ("mapas", "todo"):
        stage_mapas()
    if a.stage in ("tablas", "todo"):
        stage_tablas()
    if a.stage in ("ruido", "todo"):
        stage_ruido()
    if a.stage in ("tabla4", "todo"):
        stage_tabla4()
