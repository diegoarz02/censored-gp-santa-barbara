"""Q1-T1 (ronda Q1, la brecha principal): efecto del muestreo preferencial sobre la excedencia
del Cd.

Objecion de revisor: delta excluye cero (verificado abajo, para Cd especificamente -- el
delta=-1.724 de outputs/B1_preferencial_restringido.md es de **Hg**, el metal con el que se
corrio h_cierre_pendientes.preferential() en esa fase; Cd nunca se habia ajustado con este modelo
conjunto. Aclaracion importante: no son el mismo numero, hay que medir el de Cd aparte).

Este script:
1. Re-implementa el modelo conjunto de h_cierre_pendientes.preferential() (las MISMAS ecuaciones,
   copiadas verbatim: GP isotropico comun a la respuesta y al conteo de muestreo, delta de
   acoplamiento) para Cd, en las dos variantes ya validadas en el proyecto (grid completo nb=6, y
   restringido a soporte nb=14/mult=0.5) -- la funcion original solo devuelve un resumen, no el
   idata, asi que aqui se duplica la especificacion para poder guardar el posterior completo y
   usarlo en la prediccion de la grilla.
2. Con el idata guardado, predice el campo f sobre la grilla OFICIAL de excedencia (50m, 7381
   celdas, misma mascara que T2.1) usando la MISMA aproximacion de prediccion puntual
   (condicional del GP, varianza diagonal, sin covarianza cruzada entre celdas) que ya usa
   sb.predict_field para el modelo principal -- adaptada aqui porque este modelo conjunto no
   tiene covariables ni termino de fondo, asi que no calza con la firma de predict_field.
3. Calcula las mismas 3 cantidades de T2.1 para el Cd bajo el modelo preferencial, y compara
   contra el modelo estandar (outputs/area_excedida_posterior_arrays.npz, outputs/posterior_maps.npz).

Simplificacion declarada (igual que en el modelo original de h_cierre_pendientes.py): sin
covariables, sin anisotropia, sustitucion L/2 para los censurados en el ajuste del campo de
respuesta (no censura exacta) -- es un modelo de diagnostico para la pregunta del acoplamiento,
no un reemplazo del modelo principal del Cd.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm    # noqa: E402
import pytensor.tensor as pt  # noqa: E402
import rasterio       # noqa: E402
from rasterio.transform import rowcol  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.linalg import cho_solve  # noqa: E402

import sbmodel as sb            # noqa: E402
import masking_worker as mw     # noqa: E402
import h_cierre_pendientes as H  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DIR_COV = PROJ / "data" / "fase1" / "covariables"
CK_DIR = OUT / "q1_t1_preferencial_cd"
CK_DIR.mkdir(parents=True, exist_ok=True)

CRS_UTM = "EPSG:32718"
ECA_CD_AGRI, ECA_CD_RESID = 1.4, 10.0
GRID_RES_M, BUFFER_M, N_MAP_SAMPLES, CELL_BLOCK = 50.0, 300.0, 4000, 1500
SEED = 20260906
KW = dict(draws=1000, tune=1500, chains=4, cores=4, target_accept=0.95, seed=SEED)


def fit_preferential_with_idata(metal, restrict_to_support, support_mult=0.5, nb=14):
    """Copia verbatim de h_cierre_pendientes.preferential(), devolviendo el idata completo."""
    ctx, v, c, lod = H.ctx_metal(metal)
    y = np.log(np.where(c, lod / 2.0, v))
    xy = ctx["xy"]
    ep = ctx["ell_params"]

    xe = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nb + 1)
    ye = np.linspace(xy[:, 1].min(), xy[:, 1].max(), nb + 1)
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(cx, cy, indexing="ij")
    grid = np.c_[GX.ravel(), GY.ravel()]
    n_obs = counts.ravel()

    if restrict_to_support:
        ell_plugin = float(ep["beta"] / (ep["alpha"] - 1))
        d_near = np.sqrt(((grid[:, None] - xy[None]) ** 2).sum(-1)).min(1)
        keep = d_near <= support_mult * ell_plugin
        grid, n_obs = grid[keep], n_obs[keep]

    allpts = np.vstack([xy, grid])
    with pm.Model() as model:
        ell = pm.InverseGamma("ell", alpha=ep["alpha"], beta=ep["beta"])
        eta = pm.HalfNormal("eta", 0.7)
        sn = pm.HalfNormal("sigma_n", 0.7)
        gp = pm.gp.Latent(cov_func=eta ** 2 * pm.gp.cov.Matern52(2, ls=ell))
        f_all = gp.prior("f_all", X=allpts)
        f_obs = f_all[:len(xy)]
        f_grid = f_all[len(xy):]
        beta0 = pm.Normal("beta0", mu=float(y.mean()), sigma=1.0)
        pm.Normal("y_obs", mu=beta0 + f_obs, sigma=sn, observed=y)
        alpha = pm.Normal("alpha", 0.0, 2.0)
        delta = pm.Normal("delta", 0.0, 1.0)
        pm.Poisson("n_cells", mu=pt.exp(alpha + delta * f_grid), observed=n_obs)
        idata = pm.sample(nuts_sampler="nutpie", backend="numba", progressbar=False,
                          idata_kwargs={"log_likelihood": False}, **KW)
    return idata, xy


def predict_f_isotropic(idata, xy_train, xy_new, n_samples, seed, block_size=1500):
    """Prediccion condicional del GP isotropico (sin covariables ni termino de fondo) -- misma
    aproximacion de sb.predict_field (varianza diagonal, sin covarianza cruzada entre celdas
    nuevas), adaptada porque este modelo no tiene 'beta'/'b_background' que calcen con esa
    funcion."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    flat = n_chain * n_draw
    idx = rng.choice(flat, size=min(n_samples, flat), replace=n_samples > flat)

    def gv(name):
        return post[name].values.reshape(flat, *post[name].shape[2:])[idx]

    ell = gv("ell")       # escalar por draw (isotropico)
    eta = gv("eta")
    sigma_n = gv("sigma_n")
    beta0 = gv("beta0")
    f_obs = gv("f_all")[:, :len(xy_train)]

    n_tr, n_new = len(xy_train), len(xy_new)
    S = len(idx)
    bs = block_size or n_new
    Y = np.empty((S, n_new))

    dx_tt = ((xy_train[:, None, :] - xy_train[None, :, :]) ** 2).sum(-1)   # (n_tr, n_tr)
    dx_nt = ((xy_new[:, None, :] - xy_train[None, :, :]) ** 2).sum(-1)     # (n_new, n_tr)

    for s in range(S):
        ls = float(ell[s])
        Ktt = eta[s] ** 2 * sb._matern52(dx_tt / ls ** 2, 1.0)
        Ktt[np.diag_indices(n_tr)] += 1e-8
        Lc = np.linalg.cholesky(Ktt)
        a = cho_solve((Lc, True), f_obs[s])
        for b0 in range(0, n_new, bs):
            b1 = min(b0 + bs, n_new)
            Knt = eta[s] ** 2 * sb._matern52(dx_nt[b0:b1] / ls ** 2, 1.0)
            mu_f = Knt @ a
            from scipy.linalg import solve_triangular
            v = solve_triangular(Lc, Knt.T, lower=True, check_finite=False)
            var_f = np.maximum(eta[s] ** 2 - (v ** 2).sum(0), 1e-12)
            f_samp = mu_f + rng.standard_normal(b1 - b0) * np.sqrt(var_f)
            Y[s, b0:b1] = beta0[s] + f_samp + rng.standard_normal(b1 - b0) * sigma_n[s]
    return Y


def build_official_grid():
    """Mismo grid oficial que T2.1 (50m, 7381 celdas), reconstruido en coordenadas km centradas
    IGUAL que masking_worker/h_cierre_pendientes usan para xy (necesario para que la escala de
    distancias coincida con la del ell ajustado aqui)."""
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    loc = (df.drop_duplicates("location_id").set_index("location_id")
          [["easting", "northing"]].sort_index())
    e0, e1 = loc["easting"].min() - BUFFER_M, loc["easting"].max() + BUFFER_M
    n0, n1 = loc["northing"].min() - BUFFER_M, loc["northing"].max() + BUFFER_M
    ge = np.arange(e0, e1 + GRID_RES_M, GRID_RES_M)
    gn = np.arange(n0, n1 + GRID_RES_M, GRID_RES_M)
    GE, GN = np.meshgrid(ge, gn)
    grid_e, grid_n = GE.ravel(), GN.ravel()
    xy_grid = np.c_[grid_e, grid_n] / 1000.0 - np.c_[loc["easting"], loc["northing"]].mean(0) / 1000.0
    return xy_grid, len(grid_e)


def main():
    xy_grid, n_cells = build_official_grid()
    assert n_cells == 7381, f"grid no coincide con el oficial: {n_cells}"

    rows = []
    for tag, restrict in [("full_grid", False), ("restricted_support", True)]:
        ck = CK_DIR / f"Cd_{tag}.npz"
        if ck.exists():
            z = dict(np.load(ck))
        else:
            print(f"--- ajustando preferencial Cd, {tag} ---", flush=True)
            t0 = time.time()
            idata, xy = fit_preferential_with_idata("Cd", restrict_to_support=restrict)
            dl = np.asarray(idata.posterior["delta"].values).ravel()
            diag = sb.diagnose(idata, var_names=["beta0", "alpha", "delta", "eta", "sigma_n", "ell"])
            dt_fit = time.time() - t0
            print(f"  ajuste: {dt_fit:.0f}s, R-hat={diag['rhat_max']:.4f} div={diag['divergences']} "
                  f"delta_mean={dl.mean():.3f} IC95=[{np.percentile(dl,2.5):.3f},{np.percentile(dl,97.5):.3f}]",
                  flush=True)

            t0 = time.time()
            Y = predict_f_isotropic(idata, xy, xy_grid, N_MAP_SAMPLES, SEED, CELL_BLOCK)
            conc = np.exp(Y)
            p_agri = (conc > ECA_CD_AGRI).mean(0)
            p_resid = (conc > ECA_CD_RESID).mean(0)
            area_frac_agri = (conc > ECA_CD_AGRI).mean(1)
            area_frac_resid = (conc > ECA_CD_RESID).mean(1)
            dt_pred = time.time() - t0
            print(f"  prediccion grid: {dt_pred:.0f}s | %dominio agri t=0.5={100*(p_agri>0.5).mean():.2f}% "
                  f"| E[area] agri={100*area_frac_agri.mean():.2f}%", flush=True)

            z = dict(delta_mean=dl.mean(), delta_ci_low=np.percentile(dl, 2.5),
                    delta_ci_high=np.percentile(dl, 97.5),
                    excluye_cero=bool(np.percentile(dl, 2.5) > 0 or np.percentile(dl, 97.5) < 0),
                    rhat_max=diag["rhat_max"], divergences=diag["divergences"],
                    pct_domain_agri_gt_05=100 * (p_agri > 0.5).mean(),
                    pct_domain_resid_gt_05=100 * (p_resid > 0.5).mean(),
                    area_frac_agri_mean=area_frac_agri.mean(), area_frac_agri_median=np.median(area_frac_agri),
                    area_frac_agri_sd=area_frac_agri.std(),
                    area_frac_agri_ci95_lo=np.percentile(area_frac_agri, 2.5),
                    area_frac_agri_ci95_hi=np.percentile(area_frac_agri, 97.5),
                    area_frac_resid_mean=area_frac_resid.mean(), area_frac_resid_median=np.median(area_frac_resid),
                    area_frac_resid_ci95_lo=np.percentile(area_frac_resid, 2.5),
                    area_frac_resid_ci95_hi=np.percentile(area_frac_resid, 97.5),
                    seconds_fit=dt_fit, seconds_pred=dt_pred)
            np.savez_compressed(ck, **z)
        row = {"tag": tag, **{k: float(v) for k, v in z.items()}}
        rows.append(row)

    # ---- comparacion contra el modelo estandar ----
    std = np.load(OUT / "area_excedida_posterior_arrays.npz")
    std_agri = std["Cd_area_frac_agri_draw"]
    std_resid = std["Cd_area_frac_resid_draw"]
    official = np.load(OUT / "posterior_maps.npz")
    std_pct_agri = 100 * (official["Cd_p_agri"] > 0.5).mean()

    for row in rows:
        row["delta_std_pct_agri_t0.5"] = std_pct_agri
        row["delta_vs_estandar_pct_agri"] = row["pct_domain_agri_gt_05"] - std_pct_agri
        row["delta_vs_estandar_area_mean"] = row["area_frac_agri_mean"] - std_agri.mean()

    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_Q1_T1_preferencial_cd.csv", index=False, encoding="utf-8")
    print("\n" + tab.round(4).to_string(index=False))
    print(f"\nreferencia estandar: %dominio agri t=0.5={std_pct_agri:.2f}%, "
          f"E[area]={100*std_agri.mean():.2f}%")
    print("\nescrito: figuras/TABLE_Q1_T1_preferencial_cd.csv")


if __name__ == "__main__":
    main()
