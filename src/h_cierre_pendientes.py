"""Los tres pendientes que quedaban: mezcla bayesiana, Box-Cox y muestreo preferencial.

**H8.5b — mezcla gaussiana bayesiana (sílabo, semana 10).** El contraste fondo-minería no se resuelve
porque la etiqueta de estrato de OEFA está aliasada con la posición (R² = 0.840). La salida no es más
modelo sobre la etiqueta: es **dejar de usarla**. Se ajusta una mezcla de dos componentes sobre la
concentración en escala log —población de fondo y población afectada— y se compara la asignación
posterior contra la etiqueta oficial.

**H4.3 — Box-Cox con lambda estimado.** Diggle & Ribeiro (2007) §3.8, ec. (3.12): `Y* = (Y^λ − 1)/λ`
si `λ ≠ 0`, `log Y` si `λ = 0`. Hemos fijado `λ = 0` por convención. El plomo va de 17 a
58 000 mg/kg, factor 3 000: el supuesto log-normal está tenso. Se estima `λ` y se compara contra
`λ = 0` por PSIS-LOO.

**H4.4 — muestreo preferencial.** Diggle & Ribeiro §4.4.2, pp. 89-92: `λ(x) = exp{α + β·S(x)}`, y en
su Tabla 4.1 el sesgo cae sobre **la media**, que es justo la cifra de excedencia que reportamos.
Aquí la intensidad de muestreo se modela como función del campo latente y se estima el parámetro de
acoplamiento; `δ = 0` significa que no hay preferencialidad.
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))
warnings.filterwarnings("ignore")

import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import FIG, OUT, RUN_ID, log_attempt, observe, stamp_header  # noqa: E402

SEED = 20260908
KW = dict(draws=1000, tune=1500, chains=4, cores=4, target_accept=0.95, seed=SEED)


def ctx_metal(metal):
    ctx = mw._context()
    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    v = g["value"].to_numpy(float)
    c = g["censored"].to_numpy(bool)
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(v))
    return ctx, v, c, lod


# =============================================================================== H8.5b mezcla
def mixture(metal="Hg"):
    """Dos poblaciones sobre log concentración, con la proporción dependiendo del campo espacial.

    Deliberadamente **no usa la etiqueta de OEFA**: la compara contra la asignación posterior. Si
    los dos componentes separan y la asignación coincide con la etiqueta, la etiqueta está validada;
    si separan pero no coinciden, la mezcla dice algo que la etiqueta no; si no separan, se reporta
    como resultado negativo.
    """
    ctx, v, c, lod = ctx_metal(metal)
    y = np.log(np.where(c, lod / 2.0, v))
    xy = ctx["xy"]
    ep = ctx["ell_params"]
    with pm.Model() as model:
        # dos medias ordenadas, para que el componente 0 sea siempre el de fondo y no haya
        # intercambio de etiquetas entre cadenas
        mu = pm.Normal("mu_comp", mu=[y.mean() - 1.0, y.mean() + 1.0], sigma=1.5, shape=2,
                       transform=pm.distributions.transforms.ordered,
                       initval=np.array([y.mean() - 1.0, y.mean() + 1.0]))
        sd = pm.HalfNormal("sd_comp", sigma=1.0, shape=2)
        # la proporción varía en el espacio a través de un GP sobre el logit
        ell = pm.InverseGamma("ell", alpha=ep["alpha"], beta=ep["beta"])
        eta = pm.HalfNormal("eta_w", sigma=1.0)
        gp = pm.gp.Latent(cov_func=eta ** 2 * pm.gp.cov.Matern52(2, ls=ell))
        w_raw = gp.prior("w_raw", X=xy)
        a0 = pm.Normal("a0", 0.0, 1.0)
        p_aff = pm.Deterministic("p_affected", pm.math.sigmoid(a0 + w_raw))
        pm.Mixture("obs", w=pt.stack([1 - p_aff, p_aff], axis=1),
                   comp_dists=[pm.Normal.dist(mu[0], sd[0]), pm.Normal.dist(mu[1], sd[1])],
                   observed=y)
        idata = pm.sample(nuts_sampler="nutpie", backend="numba", progressbar=False,
                          idata_kwargs={"log_likelihood": False}, **KW)

    post = idata.posterior
    m = np.asarray(post["mu_comp"].values).reshape(-1, 2)
    p = np.asarray(post["p_affected"].values).reshape(-1, len(y)).mean(0)
    sep = float(np.mean(m[:, 1] - m[:, 0]))
    sep_lo, sep_hi = np.quantile(m[:, 1] - m[:, 0], [0.025, 0.975])
    lab = (ctx["loc"]["stratum"] == "potential_interest").to_numpy()
    agree = float(np.mean((p > 0.5) == lab))
    d = sb.diagnose(idata, var_names=["mu_comp", "sd_comp", "a0", "eta_w", "ell"])
    return {"metal": metal, "separacion_medias": sep, "sep_ci_low": float(sep_lo),
            "sep_ci_high": float(sep_hi), "separa": bool(sep_lo > 0),
            "acuerdo_con_etiqueta_oefa": agree,
            "pct_asignado_afectado": float(100 * (p > 0.5).mean()),
            "rhat_max": d["rhat_max"], "divergences": d["divergences"]}, p, lab


# ============================================================ B3 (corrida 3) — mezcla CENSURADA
def mixture_censored(metal="Cd", k=2, target_accept=0.99, tune=1500):
    """La mezcla de H8.5b, con la censura como verosimilitud en vez de sustituida por L/2.

    Permite aplicarla al cadmio (79.8 % censura), que es donde la pregunta fondo/afectado sigue
    más abierta y donde la versión con L/2 no era aplicable en rigor. `k` es el número de
    componentes (1, 2 o 3), para la comparación por PSIS-LOO que cierra la semana 10 del sílabo.

    Simplificación declarada frente al modelo principal: los detectados entran como observación
    puntual (`Normal.logpdf`), no como intervalo censurado por la resolución del instrumento — la
    censura por límite de detección sí se modela exactamente (`Normal.logcdf`). Interval-censurar
    también los detectados es posible pero encarece la verosimilitud de mezcla sin cambiar la
    pregunta que este hito responde (separan las poblaciones incorporando censura, sí o no).

    Para k=1 no hay proporción que estimar: es un Normal espacial censurado plano, la referencia
    nula de la comparación. Para k=3 los pesos de mezcla son constantes (Dirichlet), no espaciales,
    por tratabilidad — k=2 mantiene el diseño espacial original.

    `target_accept=0.99` (no el 0.95 del `KW` compartido): el primer ajuste de k=2/k=3 dio 72 y 46
    divergencias de 4000 — la mezcla ordenada con censura tiene una geometría más exigente que las
    otras funciones de este módulo cerca del límite donde los componentes casi se superponen.
    Subir solo el `target_accept` de esta función (no el de `KW`, que usan `mixture`/`boxcox`/
    `preferential` y ya está validado para ellas) es el primer paso estándar ante divergencias
    (Betancourt 2017) antes de tocar la parametrización del modelo.
    """
    ctx, v, c, lod = ctx_metal(metal)
    y_obs_log = np.log(np.maximum(v, 1e-6))     # solo se usa donde no está censurado
    log_lod = float(np.log(lod))
    xy = ctx["xy"]
    ep = ctx["ell_params"]
    n = len(v)
    det = ~c

    with pm.Model() as model:
        if k == 1:
            mu = pm.Normal("mu_comp", mu=float(y_obs_log[det].mean()), sigma=2.0, shape=1)
            sd = pm.HalfNormal("sd_comp", sigma=1.0, shape=1)
            logw = pt.zeros((n, 1))          # peso 1.0, en log
        else:
            init = np.linspace(y_obs_log[det].mean() - 1.0, y_obs_log[det].mean() + 1.0, k)
            mu = pm.Normal("mu_comp", mu=init, sigma=1.5, shape=k,
                           transform=pm.distributions.transforms.ordered, initval=init)
            sd = pm.HalfNormal("sd_comp", sigma=1.0, shape=k)
            if k == 2:
                ell = pm.InverseGamma("ell", alpha=ep["alpha"], beta=ep["beta"])
                eta = pm.HalfNormal("eta_w", sigma=1.0)
                gp = pm.gp.Latent(cov_func=eta ** 2 * pm.gp.cov.Matern52(2, ls=ell))
                w_raw = gp.prior("w_raw", X=xy)
                a0 = pm.Normal("a0", 0.0, 1.0)
                p1 = pm.Deterministic("p_affected", pm.math.sigmoid(a0 + w_raw))
                logw = pt.log(pt.stack([1 - p1, p1], axis=1) + 1e-12)
            else:
                weights = pm.Dirichlet("weights", a=np.ones(k))
                logw = pt.log(weights + 1e-12)[None, :] * pt.ones((n, 1))

        # log N(y | mu_j, sd_j) para cada componente j, en los N puntos -> (n, k)
        logp_comp = pt.stack([pm.logp(pm.Normal.dist(mu[j], sd[j]),
                                      pt.as_tensor(np.where(det, y_obs_log, 0.0)))
                              for j in range(k)], axis=1)
        logcdf_comp = pt.stack([pm.logcdf(pm.Normal.dist(mu[j], sd[j]), log_lod)
                                for j in range(k)], axis=1)
        per_point = pt.where(pt.as_tensor(det)[:, None],
                             logw + logp_comp, logw + logcdf_comp)
        loglik = pt.logsumexp(per_point, axis=1)
        pm.Deterministic("log_lik", loglik)
        pm.Potential("obs", loglik.sum())
        idata = pm.sample(nuts_sampler="nutpie", backend="numba", progressbar=False,
                          idata_kwargs={"log_likelihood": False},
                          **{**KW, "target_accept": target_accept, "tune": tune})

    ll = np.asarray(idata.posterior["log_lik"].values)
    import xarray as xr
    idata["log_likelihood"] = xr.DataTree(xr.Dataset(
        {"y": (("chain", "draw", "obs"), ll)},
        coords={"chain": idata.posterior.chain.values, "draw": idata.posterior.draw.values,
               "obs": np.arange(ll.shape[-1])}))
    loo = az.loo(idata, var_name="y")
    d = sb.diagnose(idata, var_names=["mu_comp", "sd_comp"])
    out = {"metal": metal, "k": k, "elpd_loo": float(loo.elpd), "se_loo": float(loo.se),
          "p_loo": float(loo.p), "rhat_max": d["rhat_max"], "divergences": d["divergences"]}
    if k >= 2:
        m = np.asarray(idata.posterior["mu_comp"].values).reshape(-1, k)
        sep = float(np.mean(m[:, -1] - m[:, 0]))
        sep_lo, sep_hi = np.quantile(m[:, -1] - m[:, 0], [0.025, 0.975])
        out.update(separacion_extremos=sep, sep_ci_low=float(sep_lo), sep_ci_high=float(sep_hi))
    p_aff = None
    if k == 2:
        p_aff = np.asarray(idata.posterior["p_affected"].values).reshape(-1, n).mean(0)
        lab = (ctx["loc"]["stratum"] == "potential_interest").to_numpy()
        out["acuerdo_con_etiqueta_oefa"] = float(np.mean((p_aff > 0.5) == lab))
    return out, p_aff


# =============================================================================== H4.3 Box-Cox
def boxcox(metal="Pb"):
    """`lambda` estimado frente a `lambda = 0`, comparados por PSIS-LOO."""
    ctx, v, c, lod = ctx_metal(metal)
    vv = np.where(c, lod / 2.0, v)
    xy, cov = ctx["xy"], ctx["cov"]
    ep = ctx["ell_params"]
    out = []
    for name, fixed in [("lambda = 0 (log)", 0.0), ("lambda estimado", None)]:
        with pm.Model() as model:
            lam = (pt.constant(0.0) if fixed is not None
                   else pm.Normal("lam", mu=0.0, sigma=0.3))
            # Box-Cox, Diggle & Ribeiro ec. (3.12); en lambda = 0 se reduce al logaritmo
            # La transformación **no puede ir en el dato observado**: `yt` depende de `lam`, y PyMC
            # rechaza una variable observada que dependa de otros nodos. La forma correcta es
            # escribir la verosimilitud sobre la escala original con su **jacobiano**:
            #     log p(y) = log N(y*(lambda) | mu, sigma) + (lambda - 1) log y
            # El término del jacobiano es lo que hace comparables dos valores de lambda; sin él, el
            # modelo elegiría el lambda que más comprime la escala, no el que mejor ajusta.
            yt = pt.switch(pt.abs(lam) < 1e-6, pt.log(vv), (vv ** lam - 1.0) / lam)
            beta0 = pm.Normal("beta0", mu=float(np.log(vv).mean()), sigma=2.0)
            beta = pm.Normal("beta", 0.0, 0.5, shape=cov.shape[1])
            ell = pm.InverseGamma("ell", alpha=ep["alpha"], beta=ep["beta"], shape=2)
            st = pm.HalfNormal("sigma_tot", 0.7)
            rho = pm.Beta("rho", 2.0, 2.0)
            eta = pm.Deterministic("eta", st * pt.sqrt(rho))
            sn = pm.Deterministic("sigma_n", st * pt.sqrt(1 - rho))
            gp = pm.gp.Latent(cov_func=eta ** 2 * pm.gp.cov.Matern52(2, ls=ell))
            f = gp.prior("f", X=xy)
            mu = beta0 + pt.dot(cov, beta) + f
            logp = pm.logp(pm.Normal.dist(mu=mu, sigma=sn), yt) + (lam - 1.0) * np.log(vv)
            pm.Deterministic("log_lik", logp)
            pm.Potential("obs", logp.sum())
            idata = pm.sample(nuts_sampler="nutpie", backend="numba", progressbar=False,
                              idata_kwargs={"log_likelihood": False}, **KW)
        # LOO desde la log-verosimilitud por punto que el propio modelo guardó
        ll = np.asarray(idata.posterior["log_lik"].values)
        import xarray as xr
        idata["log_likelihood"] = xr.DataTree(xr.Dataset(
            {"y": (("chain", "draw", "obs"), ll)},
            coords={"chain": idata.posterior.chain.values,
                    "draw": idata.posterior.draw.values,
                    "obs": np.arange(ll.shape[-1])}))
        loo = az.loo(idata, var_name="y")
        row = {"metal": metal, "modelo": name, "elpd_loo": float(loo.elpd),
               "se_loo": float(loo.se), "p_loo": float(loo.p)}
        if fixed is None:
            l_ = np.asarray(idata.posterior["lam"].values).ravel()
            row.update(lam_mean=float(l_.mean()),
                       lam_ci_low=float(np.quantile(l_, 0.025)),
                       lam_ci_high=float(np.quantile(l_, 0.975)),
                       lam_excluye_cero=bool(np.quantile(l_, 0.025) > 0
                                             or np.quantile(l_, 0.975) < 0))
        out.append(row)
    return out


# ======================================================================= H4.4 preferencial
def preferential(metal="Hg", restrict_to_support=False, support_mult=1.5, nb=6):
    """Modelo conjunto: la intensidad de muestreo comparte el campo latente con la respuesta.

    Formulación de Diggle & Ribeiro §4.4.2 en su versión moderna con parámetro de acoplamiento
    (Kanetaka & Shirota 2026): `delta = 0` significa que no hay muestreo preferencial. La
    aproximación discreta usa una cuadrícula de celdas con recuento de puntos, que es el estándar
    para ajustar un proceso de Cox log-gaussiano sin integrar la intensidad analíticamente.

    `restrict_to_support=True` (B1, corrida 3) cierra la circularidad señalada en el informe: el
    tercio oeste del dominio no tiene un solo dato, así que su `f_grid` es pura extrapolación del
    kernel, y si el modelo extrapola alto justo donde nadie muestreó, `delta` sale negativo por
    construcción, no porque haya preferencialidad real. Restringir las celdas de recuento a las
    que están dentro de `support_mult` rangos de correlación del punto real más cercano quita esa
    influencia: solo cuenta la geometría donde el campo está genuinamente informado por datos.
    """
    ctx, v, c, lod = ctx_metal(metal)
    y = np.log(np.where(c, lod / 2.0, v))
    xy = ctx["xy"]
    ep = ctx["ell_params"]

    # cuadrícula gruesa de recuentos: dónde midió OEFA y dónde no
    xe = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nb + 1)
    ye = np.linspace(xy[:, 1].min(), xy[:, 1].max(), nb + 1)
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(cx, cy, indexing="ij")
    grid = np.c_[GX.ravel(), GY.ravel()]
    n_obs = counts.ravel()

    if restrict_to_support:
        # rango de correlacion, estimado como la media del prior InverseGamma de ell (no depende
        # de ajustar nada primero: es la misma cantidad que ya calibra el prior del modelo
        # principal, ell_prior_params, sobre las distancias pareadas reales).
        ell_plugin = float(ep["beta"] / (ep["alpha"] - 1))
        d_near = np.sqrt(((grid[:, None] - xy[None]) ** 2).sum(-1)).min(1)
        keep = d_near <= support_mult * ell_plugin
        n_dropped = int((~keep).sum())
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
        # la intensidad de muestreo comparte el campo, escalado por delta
        alpha = pm.Normal("alpha", 0.0, 2.0)
        delta = pm.Normal("delta", 0.0, 1.0)
        pm.Poisson("n_cells", mu=pt.exp(alpha + delta * f_grid), observed=n_obs)
        idata = pm.sample(nuts_sampler="nutpie", backend="numba", progressbar=False,
                          idata_kwargs={"log_likelihood": False}, **KW)
    dl = np.asarray(idata.posterior["delta"].values).ravel()
    d = sb.diagnose(idata, var_names=["beta0", "alpha", "delta", "eta", "sigma_n", "ell"])
    out = {"metal": metal, "delta_mean": float(dl.mean()), "delta_sd": float(dl.std()),
          "delta_ci_low": float(np.quantile(dl, 0.025)),
          "delta_ci_high": float(np.quantile(dl, 0.975)),
          "excluye_cero": bool(np.quantile(dl, 0.025) > 0 or np.quantile(dl, 0.975) < 0),
          "p_delta_positivo": float((dl > 0).mean()),
          "rhat_max": d["rhat_max"], "divergences": d["divergences"],
          "restrict_to_support": restrict_to_support, "n_grid_cells": int(len(grid))}
    if restrict_to_support:
        out["ell_plugin_km"] = ell_plugin
        out["support_mult"] = support_mult
        out["n_grid_cells_dropped"] = n_dropped
    return out
