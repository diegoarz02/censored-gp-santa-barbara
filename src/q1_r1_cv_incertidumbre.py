"""Q1-R1: incertidumbre de la comparacion de CV (SE pareado
entre pliegues) del Cd bajo L=LD contra los 3 baselines.

Puro post-procesamiento, SIN MCMC nuevo. Las 4 fuentes ya existen:
- censurado L=LD: outputs/cv_cd_LD/Cd_b{0..4}_LD.npz (muestras crudas, src/cv_cd_LD.py).
- GP con L/2, kriging L/2, media global L/2: outputs/cv_real_data.csv (scores ya agregados por
  pliegue para los 4 metales, calculados en notebooks/_src/nb02.py, misma particion
  k=5/seed0=20260906). Se filtra metal=='Cd' -- no se reajusta nada.

score_fold() es la copia EXACTA de notebooks/_src/nb02.py (y de src/cv_cd_LD.py, que ya la copio
igual). Las 4 fuentes comparten la misma particion espacial (verificado: mismo SEED=20260906,
mismo k=5 en los 3 scripts), asi que el apareamiento por bloque es valido.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "cv_cd_LD"
K_BLOCKS = 5
SEED0 = 20260906
METRICS = ["joint_log_score", "rmse_det", "picp95_det"]


def build_context():
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)
    g = df[df.analyte == "Cd"].set_index("location_id").loc[loc.index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    rep_lod = g["reported_limit"].iloc[0]
    lod = float(rep_lod) if np.isfinite(rep_lod) else float(np.nanmin(value))
    return dict(xy=xy, value=value, censored=censored, lod=lod)


def score_fold(samples_log, value, cens, te, lod):
    """Copia exacta de notebooks/_src/nb02.py::score_fold() -- no se reinventa la metrica."""
    y_true = np.where(cens[te], np.nan, np.log(np.where(cens[te], 1.0, value[te])))
    mu, sd = samples_log.mean(0), samples_log.std(0)
    det = ~cens[te]

    ll = np.empty(int(te.sum()))
    ll[det] = sps.norm.logpdf(y_true[det], mu[det], sd[det])
    if (~det).any():
        ll[~det] = sps.norm.logcdf(np.log(lod), mu[~det], sd[~det])
    out = {"n_test": int(te.sum()), "n_detected": int(det.sum()),
           "joint_log_score": float(ll.mean())}

    if det.any():
        s = samples_log[:, det]
        yd = y_true[det]
        out.update({"rmse_det": float(np.sqrt(np.mean((s.mean(0) - yd) ** 2)))})
        lo = np.quantile(s, 0.025, axis=0)
        hi = np.quantile(s, 0.975, axis=0)
        out["picp95_det"] = float(np.mean((yd >= lo) & (yd <= hi)))
    else:
        out["rmse_det"] = float("nan")
        out["picp95_det"] = float("nan")
    return out


def main():
    ctx = build_context()
    labels = sb.spatial_blocks(ctx["xy"], k=K_BLOCKS, seed=SEED0)

    rows_ld = []
    for b in range(K_BLOCKS):
        z = np.load(CK_DIR / f"Cd_b{b}_LD.npz")
        te = labels == b
        sc = score_fold(z["samples"], ctx["value"], ctx["censored"], te, ctx["lod"])
        sc["block"] = b
        sc["model"] = "censored GP (L=LD)"
        rows_ld.append(sc)
    ld = pd.DataFrame(rows_ld)

    cv = pd.read_csv(OUT / "cv_real_data.csv")
    cd = cv[cv.metal == "Cd"].copy()
    baselines = {"GP with L/2": "GP with L/2",
                 "ordinary kriging L/2": "ordinary kriging L/2",
                 "global mean with L/2": "global mean with L/2"}

    detail_rows = []
    for b in range(K_BLOCKS):
        row_ld = ld[ld.block == b].iloc[0]
        for model in baselines:
            row_b = cd[(cd.block == b) & (cd.model == model)].iloc[0]
            for met in METRICS:
                detail_rows.append({
                    "block": b, "baseline": model, "metric": met,
                    "censored_LD": row_ld[met], "baseline_value": row_b[met],
                    "n_test": int(row_ld["n_test"]), "n_detected": int(row_ld["n_detected"]),
                })
    detail = pd.DataFrame(detail_rows)
    detail["diff_LD_minus_baseline"] = detail["censored_LD"] - detail["baseline_value"]
    detail.to_csv(FIG / "TABLE_CV_Cd_LD_folds_detalle.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- SE pareado sobre los pliegues
    t_975_df4 = 2.776  # t de Student, 4 g.l., dos colas 95%
    summary_rows = []
    for model in baselines:
        for met in METRICS:
            d = detail[(detail.baseline == model) & (detail.metric == met)]["diff_LD_minus_baseline"]
            d = d.dropna()
            n = len(d)
            mean_d = float(d.mean())
            se_d = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            ci_lo = mean_d - t_975_df4 * se_d if n == 5 else float("nan")
            ci_hi = mean_d + t_975_df4 * se_d if n == 5 else float("nan")
            excluye_cero = bool((ci_lo > 0) or (ci_hi < 0)) if n == 5 else False
            summary_rows.append({
                "baseline": model, "metric": met, "n_folds": n,
                "mean_diff_LD_minus_baseline": mean_d, "se": se_d,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi, "ic_excluye_cero": excluye_cero,
            })
    summ = pd.DataFrame(summary_rows)

    print(detail.round(4).to_string(index=False))
    print()
    print(summ.round(4).to_string(index=False))

    jls = summ[summ.metric == "joint_log_score"]
    jls_mean = jls[jls.baseline == "global mean with L/2"].iloc[0]
    jls_krig = jls[jls.baseline == "ordinary kriging L/2"].iloc[0]
    jls_gp = jls[jls.baseline == "GP with L/2"].iloc[0]
    rmse = summ[summ.metric == "rmse_det"]
    picp = summ[summ.metric == "picp95_det"]

    def veredicto(row):
        if row["n_folds"] < 5:
            return "n < 5 pliegues (NaN en algun bloque sin detecciones) -- no se calcula IC t de 4 g.l."
        return ("la ventaja/desventaja es ESTABLE (el IC95% excluye 0)" if row["ic_excluye_cero"]
                else "dentro del RUIDO entre pliegues (el IC95% incluye 0, no se puede afirmar estabilidad)")

    report = f"""# Q1_CV_INCERTIDUMBRE.md -- SE pareado entre pliegues de la CV del Cd bajo L=LD (Q1-R1)

Orden de revision externa (Q1-R1), preparado tras la evaluacion de
la revision externa (objecion: el margen del censurado sobre la media global en `joint_log_score` es de
solo 0.02 -- sin variabilidad entre pliegues no se sabe si es ruido). Puro post-procesamiento:
el censurado L=LD reutiliza las muestras ya guardadas en `outputs/cv_cd_LD/` (`src/cv_cd_LD.py`),
los 3 baselines se leen de `outputs/cv_real_data.csv` (ya calculados por `notebooks/_src/nb02.py`
sobre la MISMA particion espacial k=5/seed0=20260906). Nada se reajusto.

## Detalle por pliegue x baseline x metrica

{detail.round(4).to_markdown(index=False)}

## Diferencia pareada (censurado L=LD menos baseline), media +- SE sobre los 5 pliegues

{summ.round(4).to_markdown(index=False)}

El IC usa t de Student con 4 g.l. (5 pliegues), no normal -- mas conservador con n tan chico.
Para RMSE y PICP95, el bloque 1 no tiene detecciones (n_detected=0: los 36 puntos del bloque son
todos censurados) y queda fuera del promedio (columna `n_folds` indica cuantos pliegues entraron
en cada fila).

## Veredicto de estabilidad

**joint_log_score, frente a la media global (el margen de 0.02 que se senalo):**
media de la diferencia pareada = {jls_mean['mean_diff_LD_minus_baseline']:.4f} +- {jls_mean['se']:.4f}
(IC95% [{jls_mean['ci95_lo']:.4f}, {jls_mean['ci95_hi']:.4f}]). {veredicto(jls_mean)}

**joint_log_score, frente a kriging L/2:** {jls_krig['mean_diff_LD_minus_baseline']:.4f} +-
{jls_krig['se']:.4f} (IC95% [{jls_krig['ci95_lo']:.4f}, {jls_krig['ci95_hi']:.4f}]).
{veredicto(jls_krig)}

**joint_log_score, frente a GP con L/2:** {jls_gp['mean_diff_LD_minus_baseline']:.4f} +-
{jls_gp['se']:.4f} (IC95% [{jls_gp['ci95_lo']:.4f}, {jls_gp['ci95_hi']:.4f}]). {veredicto(jls_gp)}

**RMSE** (positivo = el censurado L=LD empeora frente al baseline):

{rmse[['baseline','mean_diff_LD_minus_baseline','se','ci95_lo','ci95_hi','ic_excluye_cero']].round(4).to_markdown(index=False)}

**PICP95** (positivo = el censurado L=LD cubre mas):

{picp[['baseline','mean_diff_LD_minus_baseline','se','ci95_lo','ci95_hi','ic_excluye_cero']].round(4).to_markdown(index=False)}

No se toco `main.tex` ni `referencias.bib`. No se hizo push.
"""
    (OUT / "Q1_CV_INCERTIDUMBRE.md").write_text(report, encoding="utf-8")
    print("\nescrito: figuras/TABLE_CV_Cd_LD_folds_detalle.csv, outputs/Q1_CV_INCERTIDUMBRE.md")


if __name__ == "__main__":
    main()
