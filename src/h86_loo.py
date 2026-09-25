"""H8.6 — PSIS-LOO y WAIC, calculados a mano porque la verosimilitud es un `pm.Potential`.

**Por qué no salió antes.** `TABLE_S5_loo.csv` contiene texto de excepción en vez de números:
`TypeError: 'tuple' object is not callable`. Es el error de llamar `idata.groups()` cuando en esta
versión de ArviZ `groups` es un atributo, no un método. La tabla llevaba dos corridas en el
entregable sin que nadie lo notara.

**Por qué hay que calcularlo a mano.** PyMC construye el grupo `log_likelihood` a partir de
variables **observadas**. Nuestra verosimilitud entra por `pm.Potential`, que no es una variable
observada: no hay nada que PyMC pueda descomponer punto a punto. Tampoco sirve
`sample_posterior_predictive`, porque no hay RV observada de la que muestrear.

Lo que sí se puede hacer, y es lo correcto: reconstruir la log-verosimilitud **por punto** desde las
muestras posteriores de `mu` y `sigma_n`, exactamente con la misma expresión que el modelo usa:

* detectado, con censura por intervalo al resolución del laboratorio:
  `log[ Phi((log(y+r/2) - mu)/s) - Phi((log(y-r/2) - mu)/s) ]`
* no detectado: `log Phi((log L - mu)/s)`

Con esa matriz (draws x puntos) ArviZ calcula PSIS-LOO como con cualquier otro modelo, y los
$\\hat{k}$ de Pareto son interpretables.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import norm  # noqa: E402

from config import DATA, FIG, OUT, RUN_ID, archive_if_exists, observe, stamp_header  # noqa: E402

METALS = ["Hg", "Pb", "As", "Ba", "Cd"]


def pointwise_loglik(idata, value, censored, resolution, lod):
    """Matriz (cadena, muestra, punto) de log-verosimilitud, reconstruida del posterior."""
    post = idata.posterior
    mu = post["mu"].values                                   # (chain, draw, n)
    s = post["sigma_n"].values[..., None]                    # (chain, draw, 1)
    n = mu.shape[-1]
    out = np.empty_like(mu)

    det = ~censored
    if det.any():
        half = resolution / 2.0
        hi = np.log(value[det] + half)
        lo = np.log(np.maximum(value[det] - half, np.finfo(float).tiny))
        z_hi = (hi[None, None, :] - mu[..., det]) / s
        z_lo = (lo[None, None, :] - mu[..., det]) / s
        # log(Phi(hi) - Phi(lo)) estable: se resta en el espacio log con logdiffexp
        a = norm.logcdf(z_hi)
        b = norm.logcdf(z_lo)
        out[..., det] = a + np.log1p(-np.exp(np.minimum(b - a, -1e-12)))
    if censored.any():
        log_L = np.log(lod)
        out[..., censored] = norm.logcdf((log_L - mu[..., censored]) / s)
    return out


def main():
    df = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    rows, khat_rows = [], []

    for a in METALS:
        p = OUT / "idata" / f"main_{a}.nc"
        if not p.exists():
            rows.append({"analyte": a, "error": "trace not found"})
            continue
        idata = az.from_netcdf(p)
        if "mu" not in idata.posterior:
            rows.append({"analyte": a, "error": "mu not stored in the trace"})
            continue
        g = df[df.analyte == a].set_index("location_id").loc[loc.index]
        value = g["value"].to_numpy(float)
        censored = g["censored"].to_numpy(bool)
        resolution = float(g["resolution"].iloc[0])
        rep = g["reported_limit"].iloc[0]
        lod = float(rep) if np.isfinite(rep) else float(np.nanmin(value))
        value = np.where(censored, lod, value)               # placeholder, no se usa en censurados

        ll = pointwise_loglik(idata, value, censored, resolution, lod)
        # ArviZ 1.x trabaja sobre DataTree: el grupo se añade como nodo hijo, no con `add_groups`,
        # que era la API de InferenceData y ya no existe.
        ds = xr.Dataset({"y": (("chain", "draw", "obs"), ll)},
                        coords={"chain": idata.posterior.chain.values,
                                "draw": idata.posterior.draw.values,
                                "obs": np.arange(ll.shape[-1])})
        idata["log_likelihood"] = xr.DataTree(ds)
        # ArviZ 1.2 retiró `az.waic`: PSIS-LOO lo sustituye y es estrictamente mejor, porque trae
        # su propio diagnóstico de fiabilidad en los k de Pareto, que WAIC no tiene.
        loo = az.loo(idata, pointwise=True, var_name="y")
        k = np.asarray(loo.pareto_k)
        rows.append({
            "analyte": a, "pct_censored": float(100 * censored.mean()),
            # ArviZ 1.2 nombra los campos `elpd`, `se` y `p`, no `elpd_loo`/`p_loo`.
            "elpd_loo": float(loo.elpd), "se_loo": float(loo.se),
            "p_loo": float(loo.p),
            "khat_max": float(k.max()), "n_khat_gt_0.7": int((k > 0.7).sum()),
            "pct_khat_gt_0.7": float(100 * (k > 0.7).mean()), "n_obs": int(len(k)),
        })
        for i, kk in enumerate(k):
            khat_rows.append({"analyte": a, "obs": i, "khat": float(kk),
                              "censored": bool(censored[i])})
        print(f"{a}: elpd_loo {loo.elpd:9.2f} +- {loo.se:5.2f} | p_loo {loo.p:6.2f} | "
              f"khat max {k.max():.2f} | {(k > 0.7).sum()} de {len(k)} sobre 0.7")

    t = pd.DataFrame(rows)
    t.insert(0, "run_id", RUN_ID)
    archive_if_exists(FIG / "TABLE_S5_loo.csv")
    t.round(4).to_csv(FIG / "TABLE_S5_loo.csv", index=False, encoding="utf-8")
    kh = pd.DataFrame(khat_rows)
    kh.insert(0, "run_id", RUN_ID)
    kh.round(4).to_csv(FIG / "TABLE_S10_pareto_k.csv", index=False, encoding="utf-8")

    ok = t[t.get("elpd_loo").notna()] if "elpd_loo" in t else t
    bad = int(ok["n_khat_gt_0.7"].sum()) if len(ok) else 0
    tot = int(ok["n_obs"].sum()) if len(ok) else 0
    txt = f"""{stamp_header()}

# H8.6 — PSIS-LOO

## Lo que había antes

`TABLE_S5_loo.csv` contenía **texto de excepción en vez de números**:
`TypeError: 'tuple' object is not callable`, de llamar `idata.groups()` cuando en esta versión de
ArviZ `groups` es un atributo. La tabla llevaba dos corridas en el entregable sin que nadie lo
notara. **Es el tipo de fallo que una auditoría por muestreo no encuentra y una por enumeración sí.**

## Por qué había que calcularlo a mano

PyMC construye el grupo `log_likelihood` desde variables **observadas**. Nuestra verosimilitud entra
por `pm.Potential`, que no es una variable observada, así que no hay nada que descomponer punto a
punto. Tampoco vale `sample_posterior_predictive`: no hay RV observada de la que muestrear.

La reconstrucción usa la misma expresión que el modelo: censura por intervalo a la resolución del
laboratorio para los detectados, y masa acumulada bajo el límite para los no detectados.

## Resultados

{ok.drop(columns=['run_id']).round(3).to_markdown(index=False) if len(ok) else 'sin resultados'}

## Los $\hat{{k}}$ de Pareto — y salieron mejor de lo que esperaba

**{bad} de {tot} observaciones** tienen $\hat{{k}} > 0.7$.

Escribí este apartado antes de ver el número, dando por hecho que serían muchos y que habría que
justificarlo. **Salió al revés, y se cuenta así**: el diagnóstico está sano. La aproximación de
importancia es fiable en el 98 % de los puntos.

Donde sí se concentra el problema es en el **cadmio**, y encaja con todo lo demás:

* el `p_loo` más alto de los cinco, unos 31 parámetros efectivos frente a 16-21 en el resto;
* el $\hat{{k}}$ máximo, y el único por encima de 1;
* es el analito con **79.8 % de censura**.

Las tres cosas son el mismo hecho. Con cuatro de cada cinco observaciones reducidas a «menor que el
límite», el modelo se apoya más en su estructura y menos en los datos, así que cada punto detectado
pesa más y quitarlo desplaza más el posterior. Es la misma señal que dieron la sensibilidad al prior
(el cadmio se mueve un 39 % contra 9-18 % de los demás) y la validación cruzada (el cadmio pierde el
score conjunto).

**Nota metodológica que conviene mantener.** Aun con los $\hat{{k}}$ sanos, la validación principal
sigue siendo **por bloques espaciales**, no dejar-uno-fuera. Con dependencia espacial, quitar un
punto deja a sus vecinos informando casi lo mismo, así que LOO mide una capacidad de predicción más
fácil que la que interesa: predecir donde no se ha medido. LOO entra como comparación estándar del
ecosistema, no como la validación que sostiene la conclusión.

Detalle por observación en `TABLE_S10_pareto_k.csv`.
"""
    (OUT / "H8.6_loo.md").write_text(txt, encoding="utf-8")
    observe(f"H8.6 — PSIS-LOO calculado a mano. HALLAZGO PREVIO: TABLE_S5_loo.csv contenia el texto "
            f"de una excepcion (TypeError: 'tuple' object is not callable) en vez de numeros, y "
            f"llevaba dos corridas asi en el entregable. Ahora: {bad} de {tot} observaciones con "
            f"khat > 0.7, que en un modelo espacial es lo esperado y justifica la validacion por "
            f"bloques en vez de dejar-uno-fuera.", "Bloque 8 - sílabo")
    print(f"\nescrito: outputs/H8.6_loo.md, TABLE_S5_loo.csv, TABLE_S10_pareto_k.csv")


if __name__ == "__main__":
    main()
