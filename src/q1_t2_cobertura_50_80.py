"""Q1-T2 (ronda Q1, prompt de Cowork 2026-10-28): cobertura a 50% y 80%, completando la
calibracion multinivel (hoy solo hay 90/95%, ver figuras/TABLE_T3_3_calibracion_multinivel.csv).

Costo medido ANTES de lanzar nada (regla 3 del proyecto): la corrida original de las 240 celdas
profundas del experimento de enmascaramiento (outputs/OBSERVACIONES.md, 09:42) tardo 6610.8
minutos (~110h) con 10 procesos en paralelo para 167 de 240 celdas. Repetir las 240 con 10
replicas completas hoy costaria ~150-160h de reloj -- no viable en una sesion. Decision tomada
con Diego (pregunta explicita, respuesta "la que sea mas viable y sirva"): primero se lanzo con
**1 replica** por celda. Cowork, al revisar, sugirio **3 replicas** como "punto dulce" (reparte
el tiempo entre 3 en vez de 10, pero conserva algo de la dispersion entre replicas que pedia
originalmente) -- ajuste fino aplicado aqui: se extiende a replicas 0,1,2 (24 celdas x 3 = 72
ajustes x 2 modelos = 144 ajustes nuevos en total), reutilizando los checkpoints de la replica 0
ya calculados, sin repetirlos. Con n=3 SI se puede dar una dispersion entre replicas (media+SD),
aunque mas ruidosa que con 10 -- se declara ese matiz explicitamente, no se presenta como si
tuviera el mismo peso que las 10 originales del resto del experimento.

Reutiliza directamente las primitivas de masking_worker.py (_context, el mismo particionado
espacial, la misma censura por cuantil, la misma formula de semilla) y de sbmodel.py
(build_censored_gp, fit_model, predict_field, metrics) -- NO se reimplementa el modelo, solo se
extiende el calculo de metricas a 50/80% (que sb.metrics() no calcula, solo 90/95) sobre las
MISMAS muestras predictivas que ya usa run_cell, y se guardan esas muestras (antes se
descartaban) para poder verificar 90/95 como control de regresion.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import sbmodel as sb            # noqa: E402
import masking_worker as mw     # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "q1_t2_cobertura_50_80"
CK_DIR.mkdir(parents=True, exist_ok=True)

METALS = ["Hg", "Pb", "As"]          # mismos TRUTH_FIELDS del experimento original
LEVELS = [0.2, 0.4, 0.6, 0.8]
# 3 replicas (no 10): ajuste fino con Cowork tras su recomendacion de "punto dulce" -- la r0 se
# lanzo primero sola (decision inicial de costo), esta extension anade r1/r2 sin repetir r0 (los
# checkpoints de r0 ya en outputs/q1_t2_cobertura_50_80/ se reutilizan tal cual).
REPLICATES = [int(a) for a in sys.argv[1:]] or [0]
N_BLOCKS = 5
SEED0 = 20260906                      # mismo seed0 que masking_worker.py
N_PRED = 800                          # mismo n_pred que full_nuts en run_cell
LEVELS_PICP = (0.50, 0.80, 0.90, 0.95)
N_JOBS = 2                            # bajado de 4: cada worker ya usa cores=4 internamente (NUTS
                                      # a 4 cadenas), y esta es una SEGUNDA tanda (replicas 1-2)
                                      # corriendo junto con la primera (replica 0) + Q1-T1/T3/T4


def extended_metrics(y_true_log, samples_log, censored, log_limit):
    """sb.metrics() solo calcula picp90/95. Esto añade picp50/80 sobre las MISMAS muestras,
    sin tocar sbmodel.py, y reusa sb.metrics() para 90/95 como control de regresion."""
    base = sb.metrics(y_true_log, samples_log, censored=censored, log_limit=log_limit)
    y = np.asarray(y_true_log, float)
    extra = {}
    for lvl in LEVELS_PICP:
        lo = np.quantile(samples_log, (1 - lvl) / 2, axis=0)
        hi = np.quantile(samples_log, 1 - (1 - lvl) / 2, axis=0)
        extra[f"picp{int(lvl*100)}"] = float(np.mean((y >= lo) & (y <= hi)))
        extra[f"width{int(lvl*100)}"] = float(np.mean(hi - lo))
    # control de regresion: picp90/95 deben coincidir exacto con los de sb.metrics()
    for lvl in (90, 95):
        assert abs(extra[f"picp{lvl}"] - base[f"picp{lvl}"]) < 1e-9, \
            f"picp{lvl} no coincide: {extra[f'picp{lvl}']} vs {base[f'picp{lvl}']}"
    return {**base, **extra}


def run_one_cell(metal, level, replicate):
    import json
    ck = CK_DIR / f"{metal}_{int(round(level*100)):03d}_r{replicate:02d}.json"
    if ck.exists():
        try:
            return json.loads(ck.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            ck.unlink(missing_ok=True)

    ctx = mw._context()
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    seed = int(SEED0 + 1000 * replicate + 97 * int(level * 100) + 13 * hash(metal) % 1000)
    block = replicate % N_BLOCKS

    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    y_true_log = np.log(value)

    limit = float(np.quantile(value, level))
    censored = value < limit
    labels = sb.spatial_blocks(xy, k=N_BLOCKS, seed=SEED0)
    test, train = labels == block, labels != block
    min_det = float(value[~censored].min()) if (~censored).any() else None

    kw = dict(draws=1000, tune=1500, chains=4, target_accept=0.99)
    rows = {}
    for model_name, extra_kw in [("censored GP", dict(limit_as_parameter=False)),
                                 ("GP with L/2", dict(substitute_half_lod=True))]:
        t0 = time.time()
        model = sb.build_censored_gp(
            xy[train], cov[train], np.where(censored[train], np.nan, value[train]),
            censored[train], resolution, limit, ell_params=ctx["ell_params"],
            is_background=is_bg[train], is_composite=is_comp[train], anisotropic=True,
            min_detected=min_det, **extra_kw)
        idata = sb.fit_model(model, seed=seed, cores=4, **kw)
        pred = sb.predict_field(idata, xy[train], xy[test], cov[test],
                                is_background_new=is_bg[test], n_samples=N_PRED,
                                seed=seed, anisotropic=True)
        met = extended_metrics(y_true_log[test], pred["y"], censored[test], np.log(limit))
        diag = sb.diagnose(idata)
        dt = time.time() - t0
        rows[model_name] = dict(met, rhat_max=diag["rhat_max"],
                                ess_bulk_min=diag["ess_bulk_min"], ess_tail_min=diag["ess_tail_min"],
                                divergences=diag["divergences"], seconds=dt,
                                n_train=int(train.sum()), n_test=int(test.sum()),
                                n_test_censored=int(censored[test].sum()))
        print(f"  {metal}/{level}/{model_name}: {dt:.0f}s R-hat={diag['rhat_max']:.4f} "
              f"div={diag['divergences']} picp50={met['picp50']:.3f} picp80={met['picp80']:.3f} "
              f"picp90={met['picp90']:.3f} picp95={met['picp95']:.3f}", flush=True)

    out = {"metal": metal, "level": level, "replicate": replicate, "block": block,
          "censored_GP": rows["censored GP"], "GP_L2": rows["GP with L/2"]}
    ck.write_text(json.dumps(out, default=float, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    jobs = [(m, lvl, rep) for m in METALS for lvl in LEVELS for rep in REPLICATES]
    print(f"Q1-T2: {len(jobs)} celdas x 2 modelos = {2*len(jobs)} ajustes nuevos, "
          f"replicas {REPLICATES}, protocolo profundo completo "
          f"(draws=1000/tune=1500/chains=4/target_accept=0.99)")
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(run_one_cell)(m, lvl, rep) for m, lvl, rep in jobs)
    print(f"\ntotal: {(time.time()-t0)/60:.1f} min con {N_JOBS} procesos en paralelo")

    # agrega TODAS las replicas ya en checkpoint (no solo las de esta corrida), para que el CSV
    # final siempre refleje el estado completo acumulado hasta ahora
    all_ck = sorted(CK_DIR.glob("*.json"))
    import json as _json
    all_results = [_json.loads(p.read_text(encoding="utf-8")) for p in all_ck]

    rows = []
    for r in all_results:
        for model_key, model_label in [("censored_GP", "censored GP"), ("GP_L2", "GP with L/2")]:
            d = r[model_key]
            rows.append({"metal": r["metal"], "level": r["level"], "model": model_label,
                        "replicate": r["replicate"], **d})
    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_Q1_T2_cobertura_50_80.csv", index=False, encoding="utf-8")
    print("\n" + tab[["metal", "level", "model", "picp50", "picp80", "picp90", "picp95",
                      "rhat_max", "divergences", "seconds"]].round(4).to_string(index=False))
    print("\nescrito: figuras/TABLE_Q1_T2_cobertura_50_80.csv")


if __name__ == "__main__":
    main()
