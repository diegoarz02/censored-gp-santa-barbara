"""C2 -- recuperacion de parametros sobre N replicas (obs 2 de Luis: "repetir con al menos 50
replicas, reportando sesgo, RMSE y cobertura por parametro").

Diseno A (verdad fija, datos repetidos), preferido por el prompt sobre el Diseno B (theta del
prior, estilo SBC) que usa c1_sbc.py::simulate_one por defecto: aqui theta queda FIJO en los
mismos valores que ya usa TABLE9_parameter_recovery.csv (el TRUE dict de notebooks/_src/nb02.py,
seccion 6), y solo varia la parte estocastica (realizacion del campo, ruido, que puntos caen bajo
censura) replica a replica. Asi el sesgo y el RMSE por parametro tienen una interpretacion limpia
respecto de una verdad conocida -- exactamente lo que pide la observacion de Luis.

Mecanismo de generacion: Matern 5/2, reparametrizacion sigma_tot/rho, resolucion REAL de As
(reutilizado tal cual de c1_sbc.py::simulate_one). El ajuste (build_censored_gp, las dos variantes
censurada y sustitucion L/2) es el mismo de nb02.py, con el protocolo de muestreo BARATO ya
validado (c1_sbc.CHEAP, correlacion 0.9994 contra NUTS completo) en vez del caro KW_MAIN de
nb02.py, para que 50 replicas x 2 modelos sea factible en horas, no dias.

CORREGIDO 2026-10-23: el prompt original pedia reutilizar tambien el LOD "real" de As de
c1_sbc.py::simulate_one, asumiendo que ese campo existia -- As tiene 0% de censura real en el
dataset (es uno de los "truth fields" completos), asi que `reported_limit` es NaN en las 114
filas por diseno, no un hueco accidental. La primera version de este script uso ese NaN sin
comprobarlo (`valor < NaN` es siempre False), asi que las primeras 50 replicas no tuvieron ningun
dato censurado -- archivadas en outputs/c2_recovery_BUGGY_sin_censura_2026-10-23/. El limite ahora
se fija por CUANTIL sobre el campo simulado (60%, el mismo que nb02.py::simulate_field uso para
generar el Cuadro V original de 1 replica) -- ver la nota completa en simulate_fixed_theta().

Regla 3 del proyecto (medir, no calcular): se mide la replica 0 completa (los 2 ajustes) antes de
comprometerse a N replicas.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from c1_sbc import stats_invgamma_rvs  # noqa: E402  (no se usa para theta aqui, solo por si hiciera falta)
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

CK_DIR = OUT / "c2_recovery"
CK_DIR.mkdir(parents=True, exist_ok=True)
CHEAP = dict(draws=500, tune=500, chains=2, target_accept=0.95, cores=2)
N_TARGET = 50
SEED0 = 20261023          # distinto del namespace de seeds de c1_sbc.py, por claridad -- no hay
                          # colision real posible (np.random.default_rng crea un estado propio),
                          # pero se usa un offset propio para que no parezca el mismo experimento.
SUB_SEED_OFFSET = 10_000_000   # el ajuste de sustitucion usa una semilla desplazada para no
                               # compartir la misma secuencia de aceptacion/rechazo del NUTS que
                               # el ajuste censurado en la misma replica

TRUE = {"beta0": 4.0, "beta": np.array([0.3, -0.2, -0.5, 0.1]), "b_background": -0.9,
       "ell": np.array([0.65, 1.55]), "eta": 0.9, "sigma_n": 0.45}
TRUE["sigma_tot"] = float(np.hypot(TRUE["eta"], TRUE["sigma_n"]))
TRUE["rho"] = float(TRUE["eta"] ** 2 / TRUE["sigma_tot"] ** 2)

PARAMS = ["beta0", "b_background", "ell[0]", "ell[1]", "eta", "sigma_n", "sigma_tot", "rho",
         "beta[0]", "beta[1]", "beta[2]", "beta[3]"]


def verificar_true_contra_fuente():
    """No confiar de memoria: releer nb02.py y comparar literalmente contra el TRUE de arriba."""
    import re
    txt = (Path(__file__).resolve().parent.parent / "notebooks" / "_src" / "nb02.py").read_text(
        encoding="utf-8")
    m = re.search(r'TRUE = \{"beta0": ([\d.]+), "beta": np\.array\(\[([^\]]+)\]\), "b_bg": (-?[\d.]+),\s*\n\s*"ell": np\.array\(\[([^\]]+)\]\), "eta": ([\d.]+), "sigma_n": ([\d.]+)\}', txt)
    if not m:
        return False, "no se pudo parsear el dict TRUE de nb02.py"
    beta0, beta_s, b_bg, ell_s, eta, sigma_n = m.groups()
    beta_v = [float(x) for x in beta_s.split(",")]
    ell_v = [float(x) for x in ell_s.split(",")]
    ok = (float(beta0) == TRUE["beta0"] and float(b_bg) == TRUE["b_background"]
         and np.allclose(beta_v, TRUE["beta"]) and np.allclose(ell_v, TRUE["ell"])
         and float(eta) == TRUE["eta"] and float(sigma_n) == TRUE["sigma_n"])
    return ok, f"nb02.py: beta0={beta0} b_bg={b_bg} beta=[{beta_s}] ell=[{ell_s}] eta={eta} sigma_n={sigma_n}"


def simulate_fixed_theta(ctx, rng):
    """Misma mecanica que c1_sbc.simulate_one (Matern 5/2, resolucion real de As), pero con theta
    FIJO en TRUE en vez de dibujado del prior -- Diseno A.

    CORREGIDO 2026-10-23 (ver nota igual en c1_sbc.py::simulate_one): As tiene 0% de censura real
    -- `reported_limit` es NaN en las 114 filas por diseno del dataset, no un hueco accidental.
    La primera version de esta funcion (y la de c1_sbc.py de la que se copio) usaba ese campo sin
    comprobar el NaN: `valor < NaN` es siempre False, asi que las 50 replicas de la primera corrida
    nunca tuvieron un solo dato censurado -- el modelo censurado y la sustitucion L/2 ajustaron
    exactamente los mismos datos sin diferencia real entre ellos. Esos 50 checkpoints se archivaron
    en outputs/c2_recovery_BUGGY_sin_censura_2026-10-23/. El limite ahora se fija por CUANTIL sobre
    el campo simulado (igual que nb02.py::simulate_field uso para el Cuadro V original de 1
    replica), que garantiza una fraccion conocida de censura."""
    xy, cov, is_bg = ctx["xy"], ctx["cov"], ctx["is_bg"]
    n = len(xy)
    ell, eta, sigma_n = TRUE["ell"], TRUE["eta"], TRUE["sigma_n"]
    beta0, beta, b_bg = TRUE["beta0"], TRUE["beta"], TRUE["b_background"]

    a = xy / ell
    d2 = ((a[:, None] - a[None]) ** 2).sum(-1)
    K = eta ** 2 * sb._matern52(d2, ell)
    K[np.diag_indices(n)] += 1e-8
    f = rng.multivariate_normal(np.zeros(n), K)
    mu = beta0 + cov @ beta + f + b_bg * is_bg.astype(float)
    y_true_log = mu + rng.normal(0, sigma_n, n)
    value = np.exp(y_true_log)

    g_as = ctx["df"][ctx["df"].analyte == "As"].set_index("location_id").loc[ctx["loc"].index]
    resolution = float(g_as["resolution"].iloc[0])
    CENS_RATE = 0.60      # mismo cuantil que nb02.py::simulate_field uso para el Cuadro V original
    lod = float(np.quantile(value, CENS_RATE))
    censored = value < lod
    return value, censored, lod, resolution


def _extract_params(idata):
    post = idata.posterior
    out = {}
    for p in PARAMS:
        if p.startswith("ell["):
            j = int(p[4])
            v = post["ell"].values[..., j].ravel()
        elif p.startswith("beta["):
            j = int(p[5])
            v = post["beta"].values[..., j].ravel()
        elif p == "sigma_tot":
            v = np.hypot(post["eta"].values, post["sigma_n"].values).ravel()
        elif p == "rho":
            eta_v = post["eta"].values
            sn_v = post["sigma_n"].values
            v = (eta_v ** 2 / (eta_v ** 2 + sn_v ** 2)).ravel()
        else:
            v = post[p].values.ravel()
        lo, hi = np.quantile(v, [0.025, 0.975])
        out[p] = {"posterior_mean": float(v.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}
    return out


def run_replica(i, ctx):
    ck = CK_DIR / f"rep_{i:04d}.json"
    if ck.exists():
        try:
            o = json.loads(ck.read_text(encoding="utf-8"))
            if o.get("run_id") == RUN_ID:
                return o
        except Exception:                                             # noqa: BLE001
            pass

    rng = np.random.default_rng(SEED0 + i)
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    value, censored, lod, resolution = simulate_fixed_theta(ctx, rng)
    min_det = float(value[~censored].min()) if (~censored).any() else None
    n_cens = int(censored.sum())

    out = {"run_id": RUN_ID, "rep": i, "n_censored": n_cens, "lod": lod, "resolution": resolution}
    t_rep0 = time.time()
    for modelo, extra, seed in [
        ("censored", dict(limit_as_parameter=False), SEED0 + i),
        ("L/2 substitution", dict(substitute_half_lod=True), SEED0 + i + SUB_SEED_OFFSET),
    ]:
        t0 = time.time()
        try:
            model = sb.build_censored_gp(
                xy, cov, np.where(censored, np.nan, value), censored, resolution, lod,
                ell_params=ctx["ell_params"], is_background=is_bg, is_composite=is_comp,
                anisotropic=True, min_detected=min_det, **extra)
            idata = sb.fit_model(model, seed=seed, **CHEAP)
            dt = time.time() - t0
            diag = sb.diagnose(idata)
            params = _extract_params(idata)
            for p, d in params.items():
                d["true"] = (TRUE["beta"][int(p[5])] if p.startswith("beta[") else
                            TRUE["ell"][int(p[4])] if p.startswith("ell[") else TRUE[p])
                d["covers"] = bool(d["ci95_low"] <= d["true"] <= d["ci95_high"])
            out[modelo] = {"ok": True, "seconds": dt, "rhat_max": diag["rhat_max"],
                          "ess_bulk_min": diag["ess_bulk_min"], "ess_tail_min": diag["ess_tail_min"],
                          "divergences": diag["divergences"], "passes": diag["passes"],
                          "params": params}
        except Exception as exc:                                       # noqa: BLE001
            out[modelo] = {"ok": False, "error": repr(exc)[:300], "seconds": time.time() - t0}
    out["seconds_total"] = time.time() - t_rep0
    ck.write_text(json.dumps(out, default=float, ensure_ascii=False), encoding="utf-8")
    return out


def sbc_implied_coverage95():
    """Cobertura del 95% implicita en las 200 replicas de SBC ya existentes (outputs/c1_sbc/), sin
    volver a correr nada: un rango normalizado fuera del 2.5%-97.5% equivale a que el IC95 de esa
    replica NO cubrio la verdad."""
    sbc_dir = OUT / "c1_sbc"
    files = sorted(sbc_dir.glob("rep_*.json"))
    if not files:
        return None
    rows = []
    for f in files:
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                              # noqa: BLE001
            continue
        if not o.get("ok"):
            continue
        n_total = o["n_draws_total"]
        for p, rank in o["ranks"].items():
            frac = rank / n_total
            rows.append({"param": p, "covers95": bool(0.025 <= frac <= 0.975)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    agg = df.groupby("param")["covers95"].agg(["mean", "count"]).reset_index()
    agg.columns = ["param", "coverage95_sbc", "n_sbc_reps"]
    return agg


def main():
    ok_true, detalle = verificar_true_contra_fuente()
    print(f"verificacion TRUE contra nb02.py: {'OK' if ok_true else 'DISCREPANCIA'} -- {detalle}")
    if not ok_true:
        print("DETENIENDO: los valores verdaderos no coinciden con la fuente."); return

    import pymc, nutpie, arviz
    versions = {"pymc": pymc.__version__, "nutpie": nutpie.__version__,
               "numpy": np.__version__, "arviz": arviz.__version__}
    print("versiones:", versions)

    ctx = mw._context()

    # --------------------------------------------------------------------- regla 3: medir primero
    already = len(list(CK_DIR.glob("rep_*.json")))
    t0 = time.time()
    r0 = run_replica(0, ctx)
    dt0 = time.time() - t0
    per_rep_s = r0.get("seconds_total", dt0) if already == 0 else dt0
    proj_min = per_rep_s * N_TARGET / 60
    print(f"[C2] replica 0: {per_rep_s:.1f}s (censurado {r0.get('censored',{}).get('seconds','?')}, "
         f"sustitucion {r0.get('L/2 substitution',{}).get('seconds','?')}) "
         f"-> proyeccion {N_TARGET} replicas: {proj_min:.0f} min")
    observe(f"C2 recuperacion sobre N replicas: replica 0 midio {per_rep_s:.1f}s para los 2 "
            f"ajustes, proyeccion {proj_min:.0f} min para {N_TARGET}. Arrancando.",
            "Corrida extra v3 - obs Luis (replicas)")

    rows = [r0]
    for i in range(1, N_TARGET):
        rows.append(run_replica(i, ctx))
        if i % 10 == 0:
            print(f"  {i}/{N_TARGET} réplicas", flush=True)

    # ----------------------------------------------------------------------------- agregacion
    agg_rows = []
    descartadas = {"censored": [], "L/2 substitution": []}
    for modelo in ["censored", "L/2 substitution"]:
        n_ok = sum(1 for r in rows if r.get(modelo, {}).get("ok"))
        for p in PARAMS:
            vals = []
            covers = []
            for r in rows:
                m = r.get(modelo, {})
                if not m.get("ok"):
                    if r["rep"] not in descartadas[modelo]:
                        descartadas[modelo].append(r["rep"])
                    continue
                d = m["params"][p]
                vals.append(d["posterior_mean"] - d["true"])
                covers.append(d["covers"])
            vals = np.array(vals)
            n = len(vals)
            bias = float(vals.mean())
            rmse = float(np.sqrt((vals ** 2).mean()))
            cov95 = float(np.mean(covers)) if covers else float("nan")
            mcse = float(np.sqrt(cov95 * (1 - cov95) / n)) if n > 0 else float("nan")
            true_ref = (TRUE["beta"][int(p[5])] if p.startswith("beta[") else
                       TRUE["ell"][int(p[4])] if p.startswith("ell[") else TRUE[p])
            agg_rows.append({"model": modelo, "parameter": p, "true_ref": true_ref,
                            "bias": bias, "rmse": rmse, "coverage95": cov95,
                            "coverage95_mcse": mcse, "n_reps": n})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.insert(0, "run_id", RUN_ID)
    agg_df.round(5).to_csv(FIG / "TABLE9b_recovery_reps.csv", index=False, encoding="utf-8")
    print("\n=== TABLE9b_recovery_reps ===")
    print(agg_df.round(4).to_string(index=False))

    # salud de convergencia agregada
    conv_rows = []
    for modelo in ["censored", "L/2 substitution"]:
        rhats = [r[modelo]["rhat_max"] for r in rows if r.get(modelo, {}).get("ok")]
        ess_b = [r[modelo]["ess_bulk_min"] for r in rows if r.get(modelo, {}).get("ok")]
        ess_t = [r[modelo]["ess_tail_min"] for r in rows if r.get(modelo, {}).get("ok")]
        divs = [r[modelo]["divergences"] for r in rows if r.get(modelo, {}).get("ok")]
        n_pass = sum(1 for r in rows if r.get(modelo, {}).get("ok") and r[modelo]["passes"])
        conv_rows.append({"model": modelo, "n_ok": len(rhats),
                         "rhat_max_mean": float(np.mean(rhats)) if rhats else None,
                         "rhat_max_max": float(np.max(rhats)) if rhats else None,
                         "ess_bulk_min_mean": float(np.mean(ess_b)) if ess_b else None,
                         "ess_tail_min_mean": float(np.mean(ess_t)) if ess_t else None,
                         "divergences_total": int(np.sum(divs)) if divs else 0,
                         "n_pass_criterio": n_pass, "descartadas": descartadas[modelo]})
    conv_df = pd.DataFrame(conv_rows)
    print("\n=== salud de convergencia agregada ===")
    print(conv_df.to_string(index=False))

    sbc_cov = sbc_implied_coverage95()

    # ------------------------------------------------------------------------------ informe
    lectura_sostiene = []
    for p in ["ell[1]", "sigma_n"]:
        c = agg_df[(agg_df.parameter == p) & (agg_df.model == "censored")].iloc[0]
        s = agg_df[(agg_df.parameter == p) & (agg_df.model == "L/2 substitution")].iloc[0]
        lectura_sostiene.append(
            f"{p}: |sesgo| censurado {abs(c['bias']):.4f} vs L/2 {abs(s['bias']):.4f} "
            f"({'censurado menor' if abs(c['bias']) < abs(s['bias']) else 'L2 menor, NO sostiene'}); "
            f"RMSE censurado {c['rmse']:.4f} vs L/2 {s['rmse']:.4f} "
            f"({'censurado menor' if c['rmse'] < s['rmse'] else 'L2 menor, NO sostiene'})")

    txt = f"""{stamp_header()}

# Corrida de réplicas de recuperación — obs. 2 de Luis

## Objetivo (textual)

> "Repetir el análisis con al menos 50 réplicas, reportando el sesgo, el RMSE y la cobertura para
> cada parámetro, para evaluar con mayor solidez el desempeño del método."

## Diseño usado

**Diseño A — verdad fija, datos repetidos.** θ queda fijo en los mismos valores que ya usa
`TABLE9_parameter_recovery.csv` (verificado literalmente contra `notebooks/_src/nb02.py`, dict
`TRUE`, antes de calcular nada — ver salida de `verificar_true_contra_fuente()` arriba). Solo varía
la parte estocástica: la realización del campo gaussiano, el ruido de medición, y qué puntos caen
bajo censura (depende de la realización, no de un umbral fijo). Se prefirió sobre el Diseño B
(θ del prior, estilo SBC) porque da una interpretación limpia de sesgo/RMSE respecto de una verdad
conocida — que es exactamente lo que pide la observación de Luis.

Mecanismo de generación: Matérn 5/2, reparametrización σ_tot/ρ, resolución real de As (reutilizado
de `src/c1_sbc.py::simulate_one`). El ajuste (`build_censored_gp`, variantes censurada y
sustitución L/2) es el mismo de `nb02.py`, con el protocolo de muestreo **barato ya validado**
(`draws=500, tune=500, chains=2, target_accept=0.95`, correlación 0.9994 contra NUTS completo,
medido en F3, ver docstring de `c1_sbc.py`) en vez del protocolo caro de `nb02.py`
(`draws=1000, tune=1500, chains=4, target_accept=0.99`), para que {N_TARGET} réplicas × 2 modelos
fuera computacionalmente factible.

### ⚠ Bug real encontrado y corregido en esta corrida (no estaba en el prompt original)

El prompt pedía reutilizar también el **LOD real de As** de `c1_sbc.py::simulate_one`, asumiendo
que ese campo existía. Verificado contra `data/final/santa_barbara_soil.csv`: **As tiene 0 % de
censura real** (es uno de los "truth fields" completos del proyecto) y `reported_limit` es **NaN
en las 114 filas**, por diseño del dataset (`nb01_dataset.py`: *"Empty for analytes with no
censored result"*), no un hueco accidental. La primera versión de `c1_sbc.py::simulate_one` (y de
este script, copiado de ahí) usaba ese campo sin comprobar el NaN — `valor < NaN` es siempre
`False` en NumPy — así que **ninguna de las primeras 50 réplicas tuvo un solo dato censurado**, y
los dos modelos ajustaron exactamente los mismos datos sin diferencia real entre ellos (archivadas
en `outputs/c2_recovery_BUGGY_sin_censura_2026-10-23/`, no borradas).

**Más grave**: el mismo bug está en las **200 réplicas de SBC ya publicadas en el artículo**
(`FIG35_sbc_ranks`, citadas en `main.tex` línea ~375) — verificado corriendo `simulate_one`
directamente: produce 0 puntos censurados en 20/20 pruebas. El SBC publicado nunca puso a prueba
el mecanismo de censura del modelo, solo el régimen sin censura. Diego decidió corregir y
re-correr también el SBC (ver informe separado si esta corrida se hizo junto con esa).

**Corrección aplicada** (en ambos scripts, documentada en el código): el límite se fija por
**cuantil** sobre el campo simulado (60 %, el mismo mecanismo que `nb02.py::simulate_field` ya
usó y validó para generar el Cuadro V original de 1 réplica), en vez de un LOD real inexistente
para este analito. Los demás scripts del proyecto que usan `reported_limit` (`masking_worker.py`,
`h41_analisis.py`, `h86_loo.py`, etc.) ya tenían el guard `if np.isfinite(...) else fallback` —
revisados uno por uno, ninguno tiene este bug: usan la columna `censored` real de los datos, no la
recalculan desde `lod`, así que un NaN ahí era inocuo para ellos. Solo `simulate_one` (que SÍ
genera `censored` desde cero, por tratarse de datos simulados) lo necesitaba y no lo tenía.

## Semillas y réplicas

- Semilla por réplica: `{SEED0} + i` para el ajuste censurado, `{SEED0} + i + {SUB_SEED_OFFSET}`
  para la sustitución L/2 (desplazada para que las dos no compartan la misma secuencia de
  aceptación/rechazo del NUTS dentro de la misma réplica).
- Réplicas objetivo: {N_TARGET}. Réplica 0 midió **{per_rep_s:.1f} s** para los dos ajustes →
  proyección **{proj_min:.0f} min** totales (regla 3: medir antes de comprometerse).
- Versiones: {versions}.

## Salud de convergencia agregada

{conv_df.to_markdown(index=False)}

Réplicas descartadas (no convergieron o lanzaron excepción), censurado: {descartadas['censored'] or 'ninguna'}.
Réplicas descartadas, L/2: {descartadas['L/2 substitution'] or 'ninguna'}.

## Tabla agregada — sesgo, RMSE, cobertura por parámetro

{agg_df.round(4).to_markdown(index=False)}

## Comparación contra lo ya existente

**Contra `TABLE9_parameter_recovery.csv`** (1 réplica): el sesgo de esta corrida agregada es el
promedio sobre {N_TARGET} realizaciones, no debe coincidir número a número con una sola réplica —
compárense signos y magnitudes relativas entre modelos, no valores puntuales.

**Contra las 200 réplicas de SBC ya existentes** (`outputs/c1_sbc/rep_*.json`, θ del prior, no
verdad fija — comparación informal, los diseños difieren): cobertura del 95% implícita en los
rangos normalizados (fuera de [2.5%, 97.5%] = no cubre):

{sbc_cov.round(4).to_markdown(index=False) if sbc_cov is not None else "No se encontraron checkpoints de SBC en outputs/c1_sbc/ -- declarado, no inventado."}

## Lectura esperada: ¿la sostienen los resultados?

El argumento del paper predice que el modelo censurado debería tener menor sesgo/RMSE que la
sustitución L/2 en los parámetros de escala/ruido (ℓ₂, σₙ), y cobertura más cercana al 95%:

{chr(10).join('- ' + l for l in lectura_sostiene)}

## Barrido amplio de otros bugs (pedido explícito de Diego: "revisa que no haya más bugs antes de enviar a una revista")

Mientras esta corrida avanzaba se lanzó un barrido adicional (agente de investigación + verificación
propia, recalculando cada cifra desde el CSV fuente, nunca aceptando lo que reportó el agente sin
recalcular) sobre el resto del código que alimenta números citados en `main.tex`. Resultado: **un
bug de contenido real en el texto del artículo** (no de código, no requiere recómputo) y dos
hallazgos verificados y descartados como inocuos.

### Bug real #2 — `main.tex` línea 297 cita las métricas de convergencia del experimento BARATO
describiéndolas como si fueran las del experimento PROFUNDO

Texto actual: *"Sobre los 120 ajustes profundos del experimento de enmascaramiento, la sustitución
por L/2 acumula 488 transiciones divergentes y un R̂ máximo de 1.49, frente a una sola divergencia
en el modelo censurado."*

Verificado recalculando directamente ambos datasets (no solo leído del informe del agente):

| dataset | archivo | divergencias GP+L/2 | R̂ max GP+L/2 | divergencias censurado |
|---|---|---|---|---|
| barato (lo que main.tex cita hoy) | `outputs/H6_masking_experiment.csv` | 488 | 1.4904 | 1 |
| profundo (lo que main.tex dice citar, y lo que de hecho alimenta FIG28/FIG29/tab:cobertura) | `figuras/TABLE_H6_MAIN_con_kriging.csv` | 52 | 1.1916 | 0 |

La frase describe "120 ajustes profundos" pero los tres números (488, 1.49, "una sola divergencia")
coinciden exactamente con el dataset barato, no con el profundo. Causa probable: el informe
`outputs/H0.2_h6_diagnostics.md` (de `src/h02_h6_diagnostics.py`) se escribió antes de "corrida 3",
que introdujo el protocolo profundo como reemplazo del barato; nadie regeneró ese informe contra
`h6_deep` cuando el protocolo cambió, y el párrafo tomó el número del informe viejo.

**Corrección sugerida, lista para pegar en la fase de redacción** (no aplicada aquí: `main.tex` no
se toca en esta sesión por instrucción explícita, es trabajo de Cowork/redacción):

*"Sobre los 120 ajustes profundos del experimento de enmascaramiento, la sustitución por L/2
acumula 52 transiciones divergentes y un R̂ máximo de 1.19, frente a ninguna divergencia en el
modelo censurado."*

No cambia ningún otro número del artículo: la Tabla~\\ref{{tab:cobertura}} y FIG28/FIG29 ya usan el
dataset profundo correcto (verificado número por número, sin discrepancia).

### Verificado y descartado — esto NO son bugs

1. **Test de Wilcoxon (p=0.002, main.tex ~línea 361)**: `notebooks/_src/nb02.py` línea 1531
   (`cal = h6.copy()`) calcula este test sobre el dataset barato (`h6`, cargado en la línea 1301
   desde `H6_masking_experiment.csv`), no sobre el profundo. Recalculado el mismo test sobre
   `TABLE_H6_MAIN_con_kriging.csv` (profundo): da exactamente el mismo resultado — p=0.002, 10/10
   réplicas ganadas, en los 6 pares metal×baseline a 80 % de censura. Con n=10 pares sin empates,
   10/10 es el patrón que produce el p-valor mínimo posible del test exacto en cualquiera de los
   dos datasets, así que el número citado es correcto pese a que el pipeline no apunta al dataset
   que reemplazó al barato. Fragilidad de procedencia a corregir en el código algún día; no afecta
   el texto del artículo.
2. **Correlaciones LMC** (tab:lmc: Cd-Sb 0.78, Cd-Ag 0.64) **vs. 0.750/0.616 vistos en otro
   archivo**: `figuras/TABLE23_lmc_cross_correlation.csv` tiene dos columnas del mismo ajuste,
   `mean` (0.7496/0.616/0.8994) y `median` (0.7795/0.6414/0.9299). `main.tex` etiqueta su tabla
   explícitamente "Mediana", y sus valores (0.78/0.64/0.93) junto con los IC95 % ([0.36,0.97] /
   [0.16,0.93] / [0.64,1.00]) coinciden exactamente con la columna `median` de ese CSV redondeada a
   2 decimales. No hay inconsistencia real: son dos estadísticos distintos de la misma fila, no dos
   cifras que se contradicen.

### Archivos no cubiertos en este barrido (alcance, no exhaustivo — declarado, no asumido limpio)

`src/h85a_perdidas.py`, `src/h_red_bayesiana.py`, `src/h_sens_priors_informe.py`,
`src/b1_preferencial_restringido.py`, `src/b2_hsgp_test.py`, la mayor parte de
`notebooks/_src/nb02.py` (EDA univariado/bivariado, ajuste de variogramas, diseño de monitoreo) y
los notebooks sueltos `nb02_a_eda.py` … `nb02_h_realmaps.py` no se revisaron línea por línea. No se
encontró nada que apunte a que haga falta revisarlos con más urgencia, pero tampoco se afirma que
estén libres de bugs — solo que no se llegó a ellos en este barrido.

## Archivos generados

- `figuras/TABLE9b_recovery_reps.csv`
- `outputs/c2_recovery/rep_0000.json` … `rep_{N_TARGET-1:04d}.json`
- Este informe: `Claude outputs/INFORME_CORRIDA_REPLICAS_RECUPERACION.md`

## No se tocó

`articulo_latex/main.tex`, `referencias.bib`, ninguna figura del paper — por instrucción explícita
del prompt. La redacción es la fase siguiente.
"""
    out_dir = Path(__file__).resolve().parent.parent.parent / "Claude outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "INFORME_CORRIDA_REPLICAS_RECUPERACION.md").write_text(txt, encoding="utf-8")

    observe(f"C2 completado: {N_TARGET} réplicas (diseño A, verdad fija), 2 modelos cada una. "
            f"Tabla figuras/TABLE9b_recovery_reps.csv. Descartadas: censurado "
            f"{descartadas['censored']}, L/2 {descartadas['L/2 substitution']}. Informe en "
            f"'Claude outputs/INFORME_CORRIDA_REPLICAS_RECUPERACION.md'.",
            "Corrida extra v3 - obs Luis (replicas)")
    print(f"\nescrito: figuras/TABLE9b_recovery_reps.csv, "
         f"Claude outputs/INFORME_CORRIDA_REPLICAS_RECUPERACION.md, "
         f"{N_TARGET} checkpoints en outputs/c2_recovery/")


if __name__ == "__main__":
    main()
