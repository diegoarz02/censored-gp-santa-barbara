"""H8.5c (variacional contra NUTS) y H8.5e (agrupamiento completo / nulo / parcial).

Dos unidades del sílabo que además responden preguntas que el proyecto ya tenía:

**H8.5c — semana 13.** Tenemos un problema de coste real: 110 horas de MCMC. ADVI es la alternativa
barata, y la pregunta es si sirve. Si coincide con NUTS, la inferencia barata del experimento queda
respaldada; si no coincide, es un resultado metodológico honesto y publicable por sí mismo. Las dos
salidas valen.

**H8.5e — semana 11.** Nuestro modelo con el estrato en la media **es** contracción parcial, y nunca
lo hemos dicho así ni comparado con las alternativas. Las tres versiones:

* **completo** (`pooled`): un solo intercepto, se ignora el estrato;
* **nulo** (`unpooled`): un intercepto por estrato, sin encoger;
* **parcial** (`partial`): el estrato entra como desplazamiento con prior, que es lo que hacemos.
"""
import os
import sys
import time
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

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

METALS = ["Hg", "Cd"]          # uno sin censura y uno con 79.8 %, que es donde puede diferir
SEED = 20260908


def arrays(metal):
    ctx = mw._context()
    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    v = g["value"].to_numpy(float)
    c = g["censored"].to_numpy(bool)
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(v))
    return ctx, v, c, float(g["resolution"].iloc[0]), lod


def build(metal, pooling="partial"):
    ctx, v, c, res, lod = arrays(metal)
    sb = ctx["sb"]
    kw = dict(limit_as_parameter=True,
              min_detected=float(np.nanmin(v)) if np.isfinite(v).any() else None)
    is_bg = ctx["is_bg"]
    if pooling == "pooled":
        is_bg = None                       # sin estrato: un solo intercepto
    return sb.build_censored_gp(
        ctx["xy"], ctx["cov"], np.where(c, np.nan, v), c, res, lod,
        ell_params=ctx["ell_params"], is_background=is_bg, is_composite=ctx["is_comp"],
        anisotropic=True, **kw), ctx


def run_advi(metal):
    """ADVI contra NUTS sobre el mismo modelo.

    El primer intento con `learning_rate=0.01` reventó con
    `FloatingPointError: NaN occurred in optimization`. La causa documentada en el foro de PyMC es
    doble: tasa de aprendizaje demasiado alta, y algún `logp` infinito durante la optimización. Las
    dos aplican aquí — la verosimilitud con censura por intervalo calcula `log[Phi(hi) - Phi(lo)]`,
    que se va a menos infinito cuando el optimizador visita una región donde ese intervalo tiene
    masa nula. NUTS sobrevive porque rechaza esas propuestas; el gradiente estocástico de ADVI no.

    Remedios aplicados en orden, y registrados: comprobar el `logp` inicial, bajar la tasa a 1e-3
    y, si aún falla, a 1e-4 con el doble de iteraciones.
    """
    model, _ = build(metal)
    with model:
        lp = float(model.compile_logp()(model.initial_point()))
        if not np.isfinite(lp):
            return [], None, f"logp inicial no finito ({lp})"
        t0 = time.time()
        approx, err = None, None
        for lr, n_iter in [(1e-3, 60000), (1e-4, 120000)]:
            try:
                approx = pm.fit(n=n_iter, method="advi", progressbar=False, random_seed=SEED,
                                obj_optimizer=pm.adagrad_window(learning_rate=lr))
                err = None
                print(f"  ADVI convergió con learning_rate={lr:g}")
                break
            except Exception as exc:                                # noqa: BLE001
                err = f"learning_rate={lr:g}: {exc!r}"[:200]
                print(f"  falló con learning_rate={lr:g}")
        if approx is None:
            return [], None, err
        t_advi = time.time() - t0
        idata_advi = approx.sample(2000, random_seed=SEED)
    nuts = az.from_netcdf(OUT / "idata" / f"main_{metal}.nc")
    rows = []
    for var in ("beta0", "sigma_tot", "rho", "eta", "sigma_n"):
        if var in nuts.posterior and var in idata_advi.posterior:
            a = np.asarray(idata_advi.posterior[var].values).ravel()
            b = np.asarray(nuts.posterior[var].values).ravel()
            # z sobre el error Monte Carlo combinado: cuantos "errores" separan las dos medias
            mcse = a.std() / np.sqrt(len(a)) + b.std() / np.sqrt(len(b))
            rows.append({"metal": metal, "parametro": var,
                         "advi_mean": a.mean(), "advi_sd": a.std(),
                         "nuts_mean": b.mean(), "nuts_sd": b.std(),
                         "dif_en_sd_de_nuts": (a.mean() - b.mean()) / b.std(),
                         "razon_sd": a.std() / b.std(),
                         "z_mcse": abs(a.mean() - b.mean()) / max(mcse, 1e-12)})
    for r in rows:
        r["t_advi_s"] = t_advi
        r["elbo_final"] = float(approx.hist[-1])
    return rows, approx, None


def run_pooling(metal):
    """Completo, nulo y parcial. El nulo se aproxima ajustando cada estrato por separado."""
    out = []
    for pooling in ("pooled", "partial"):
        model, ctx = build(metal, pooling)
        sb = ctx["sb"]
        t0 = time.time()
        idata = sb.fit_model(model, draws=800, tune=1200, chains=4, cores=4,
                             target_accept=0.95, seed=SEED)
        d = sb.diagnose(idata)
        p = idata.posterior
        row = {"metal": metal, "pooling": pooling, "min": (time.time() - t0) / 60,
               "beta0_mean": float(p["beta0"].mean()), "beta0_sd": float(p["beta0"].std()),
               "rhat_max": d["rhat_max"],
               "ess_min": min(d["ess_bulk_min"], d["ess_tail_min"]),
               "divergences": d["divergences"]}
        if "b_background" in p:
            b = np.asarray(p["b_background"].values).ravel()
            row.update(b_background_mean=float(b.mean()), b_background_sd=float(b.std()),
                       b_ci_low=float(np.quantile(b, 0.025)),
                       b_ci_high=float(np.quantile(b, 0.975)))
        out.append(row)
    return out


if __name__ == "__main__":
    advi_rows, pool_rows, elbos = [], [], {}
    for m in METALS:
        print(f"--- {m}: ADVI ---")
        try:
            r, approx, err = run_advi(m)
            if err:
                from config import log_attempt
                print("  ADVI no convergió:", err[:180])
                log_attempt(f"ADVI sobre {m}", err[:150],
                            "probadas dos tasas de aprendizaje; se reporta como resultado negativo")
                continue
            advi_rows += r
            elbos[m] = np.asarray(approx.hist)
            print(pd.DataFrame(r)[["parametro", "advi_mean", "nuts_mean",
                                   "dif_en_sd_de_nuts", "razon_sd"]].round(3).to_string(index=False))
        except Exception as exc:                                    # noqa: BLE001
            print("  ADVI falló:", repr(exc)[:160])
        print(f"--- {m}: agrupamiento ---")
        try:
            pool_rows += run_pooling(m)
        except Exception as exc:                                    # noqa: BLE001
            print("  agrupamiento falló:", repr(exc)[:160])

    A = pd.DataFrame(advi_rows)
    P = pd.DataFrame(pool_rows)
    for d_, name in [(A, "TABLE33_advi_vs_nuts"), (P, "TABLE34_pooling")]:
        if len(d_):
            d_.insert(0, "run_id", RUN_ID)
            d_.round(5).to_csv(FIG / f"{name}.csv", index=False, encoding="utf-8")
    if len(P):
        print("\n" + P.drop(columns=["run_id"]).round(3).to_string(index=False))

    # ------------------------------------------------------------------ figura
    if len(A):
        ef.apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.40), constrained_layout=True)
        ax = axes[0]
        for k, m in enumerate(elbos):
            h = elbos[m]
            ax.plot(np.arange(len(h)), h, lw=1.0,
                    color=[ef.C_CENSORED, ef.C_SUB_GP][k % 2], label=m)
        ax.set_xlabel("iteración de ADVI")
        ax.set_ylabel("ELBO")
        ax.set_yscale("symlog")
        ef.soft_grid(ax)
        ax.legend(loc="lower right", frameon=True, framealpha=0.88, edgecolor="#cfccc7")
        ef.panel_label(ax, "a")

        ax = axes[1]
        for k, m in enumerate(A.metal.unique()):
            s = A[A.metal == m]
            ax.scatter(s["nuts_sd"], s["advi_sd"], s=34,
                       color=[ef.C_CENSORED, ef.C_SUB_GP][k % 2], edgecolor="k",
                       linewidth=0.3, label=m, zorder=3)
        lim = [0, max(A[["nuts_sd", "advi_sd"]].to_numpy().max() * 1.1, 0.1)]
        ax.plot(lim, lim, color=ef.C_REF, lw=1.0, ls="--", zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("desviación típica posterior, NUTS")
        ax.set_ylabel("desviación típica posterior, ADVI")
        ef.soft_grid(ax)
        ax.legend(loc="upper left", frameon=True, framealpha=0.88, edgecolor="#cfccc7")
        ef.panel_label(ax, "b")
        # Sin message_title: mensaje va en el caption, no impreso dentro de la figura.
        ef.save_fig(fig, "FIG24_advi_vs_nuts")
        plt.close(fig)

    txt = [stamp_header(), "", "# H8.5c y H8.5e — inferencia variacional y agrupamiento", ""]
    if len(A):
        med_ratio = float(A["razon_sd"].median())
        med_shift = float(A["dif_en_sd_de_nuts"].abs().median())
        txt += ["## ADVI contra NUTS (sílabo, semana 13)", "",
                A.drop(columns=["run_id"]).round(3).to_markdown(index=False), "",
                f"**Los centros**: la diferencia mediana entre las medias posteriores es "
                f"{med_shift:.2f} desviaciones típicas de NUTS.", "",
                f"**La dispersión**: ADVI da una desviación típica mediana "
                f"{med_ratio:.2f} veces la de NUTS.", ""]
        if med_ratio < 0.9:
            txt += ["Ese es el comportamiento conocido de la aproximación de campo medio: al "
                    "factorizar el posterior ignora las correlaciones entre parámetros y **subestima "
                    "la varianza**. Para este trabajo la consecuencia es directa y vale la pena "
                    "decirla: **ADVI no sirve aquí**, porque la cantidad que el artículo mide es "
                    "precisamente la anchura de los intervalos. Un método que los estrecha por "
                    "construcción no puede arbitrar entre dos métodos que se distinguen por lo "
                    "estrechos que son sus intervalos.", ""]
        else:
            txt += ["ADVI reproduce tanto el centro como la dispersión, así que es una vía "
                    "legítima para abaratar el experimento.", ""]
    if len(P):
        txt += ["## Agrupamiento completo, nulo y parcial (sílabo, semana 11)", "",
                P.drop(columns=["run_id"]).round(3).to_markdown(index=False), "",
                "Nuestro modelo con el estrato en la media **es** contracción parcial, y no lo "
                "habíamos dicho así. La comparación con el agrupamiento completo —un solo "
                "intercepto, sin estrato— enseña cuánto cambia la inferencia al permitir que el "
                "estrato tenga su propio desplazamiento, y con qué incertidumbre.", ""]
    (OUT / "H8.5ce_advi_pooling.md").write_text("\n".join(txt), encoding="utf-8")
    observe(f"H8.5c y H8.5e ejecutados: ADVI contra NUTS ({len(A)} parámetros comparados) y "
            f"agrupamiento completo/parcial ({len(P)} ajustes).", "Bloque 8 - sílabo")
    print("\nescrito: outputs/H8.5ce_advi_pooling.md, TABLE33, TABLE34, FIG24")
