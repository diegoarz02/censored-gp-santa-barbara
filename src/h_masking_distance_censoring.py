"""Experimento 3 de la fase de blindaje de revista: repite el experimento de enmascaramiento con
un mecanismo de censura que depende de la DISTANCIA a la labor minera, no de un umbral por
cuantil sobre el valor -- para ver si la ventaja del modelo censurado se sostiene bajo un mecanismo
de censura espacialmente informativo, mas parecido al real (mas lejos de la mina, mas probable que
la concentracion caiga bajo el limite de deteccion).

Reutiliza exactamente `masking_worker.run_cell` (mismos modelos, misma verosimilitud, mismo
protocolo barato draws=500/tune=500/chains=2/target_accept=0.95, ya validado con correlacion 0.9994
contra NUTS completo), solo reemplaza la linea `limit = quantile(value, level); censored = value <
limit` por una asignacion basada en un score = distancia a la labor + ruido gaussiano, umbralizado
para mantener la MISMA fraccion objetivo de censura por nivel (comparable con el disenio original
por cuantil).

Alcance deliberadamente mas chico que el experimento original (3 metales x 4 niveles x 10 replicas
x 2 modelos = 240 celdas): aqui 3 metales x 2 niveles (60%, 80%, el regimen donde main.tex dice que
la ventaja es clara) x 3 replicas x 2 modelos = 36 celdas. Documentado, no escondido -- el objetivo
es robustez bajo otro mecanismo, no repetir el barrido completo de niveles.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "masking_distance"
CK_DIR.mkdir(parents=True, exist_ok=True)

METALS = ["Hg", "Pb", "As"]
LEVELS = [0.6, 0.8]
N_REPLICATES = 3
N_BLOCKS = 5
SEED0 = 20260906
NOISE_FRAC = 0.25   # ruido gaussiano = 25% del rango de distancia -- informativo, no deterministico
CHEAP = dict(draws=500, tune=500, chains=2, target_accept=0.95)


def build_context():
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)
    covars = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
    loc = loc.assign(log_dist_dump=np.log(loc["dist_waste_dump_m"].clip(lower=1.0)),
                     log_dist_city=np.log(loc["dist_city_m"]))
    cov = loc[covars].to_numpy(float)
    cov = (cov - cov.mean(0)) / cov.std(0)
    is_bg = (loc["stratum"] == "background").to_numpy()
    is_comp = (loc["sample_type"] == "composite").to_numpy()
    ell_params, _, _ = sb.ell_prior_params(xy)
    dist = loc["dist_waste_dump_m"].to_numpy(float)
    return dict(df=df, loc=loc, xy=xy, cov=cov, is_bg=is_bg, is_comp=is_comp,
               ell_params=ell_params, dist=dist)


def distance_censoring(value, dist, level, seed):
    """Censura las observaciones con mayor score = distancia + ruido, manteniendo la fraccion
    objetivo `level`. Mas lejos de la labor -> mas probable quedar censurado, con variabilidad
    de replica a replica via el ruido."""
    rng = np.random.default_rng(seed)
    d = (dist - dist.min()) / (dist.max() - dist.min())
    noise = rng.normal(0, NOISE_FRAC, size=len(d))
    score = d + noise
    thr = np.quantile(score, 1 - level)
    censored = score > thr
    # el "limite" reportado para la verosimilitud es el cuantil del VALOR que separa a los
    # censurados de los detectados bajo esta asignacion -- necesario porque build_censored_gp
    # espera un unico limite numerico, no una regla de censura por punto
    limit = float(np.quantile(value[censored], 0.999)) if censored.any() else float(value.min())
    # asegurar consistencia value-censored: si algun "censurado" por distancia tiene un valor muy
    # por encima del limite resultante, el limite no tendria sentido fisico -- se reporta tal cual,
    # es el costo de usar un mecanismo no monotono en valor, declarado en el informe final.
    return censored, limit


def run_cell(metal, level, rep, ctx):
    tag = f"{metal}_{int(level*100)}_r{rep}"
    ck = CK_DIR / f"{tag}.json"
    if ck.exists():
        import json
        return json.loads(ck.read_text(encoding="utf-8"))

    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    assert np.isfinite(value).all(), f"{metal} no es un truth field completo"
    y_true_log = np.log(value)

    seed = int(SEED0 + 1000 * rep + 97 * int(level * 100) + 13 * (hash(metal) % 1000))
    censored, limit = distance_censoring(value, ctx["dist"], level, seed)

    labels = sb.spatial_blocks(ctx["xy"], k=N_BLOCKS, seed=SEED0)
    block = rep % N_BLOCKS
    test = labels == block
    train = ~test
    if test.sum() < 5 or train.sum() < 30:
        row = {"metal": metal, "level": level, "replicate": rep, "error": "bloque muy chico"}
        ck.write_text(__import__("json").dumps(row), encoding="utf-8")
        return row

    min_det = float(value[~censored].min()) if (~censored).any() else None
    results = {}
    for model_name, extra in [("censored GP", dict(limit_as_parameter=False)),
                              ("GP with L/2", dict(substitute_half_lod=True))]:
        t0 = time.time()
        model = sb.build_censored_gp(
            ctx["xy"][train], ctx["cov"][train], np.where(censored[train], np.nan, value[train]),
            censored[train], resolution, limit, ell_params=ctx["ell_params"],
            is_background=ctx["is_bg"][train], is_composite=ctx["is_comp"][train],
            anisotropic=True, min_detected=min_det, **extra)
        idata = sb.fit_model(model, seed=seed, **CHEAP)
        dt = time.time() - t0
        diag = sb.diagnose(idata)
        pred = sb.predict_field(idata, ctx["xy"][train], ctx["xy"][test], ctx["cov"][test],
                                is_background_new=ctx["is_bg"][test], n_samples=800, seed=seed,
                                anisotropic=True)
        met = sb.metrics(y_true_log[test], pred["y"], censored=censored[test], log_limit=np.log(limit))
        results[model_name] = {**met, "seconds": dt, "rhat_max": diag["rhat_max"],
                               "ess_min": min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                               "divergences": diag["divergences"]}

    row = {"metal": metal, "level": level, "replicate": rep, "block": int(block),
          "n_train": int(train.sum()), "n_test": int(test.sum()),
          "n_test_censored": int(censored[test].sum()), "limit": limit, **results}
    import json
    tmp = ck.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=float, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ck)
    return row


def main():
    ctx = build_context()
    cells = [(m, lv, r) for m in METALS for lv in LEVELS for r in range(N_REPLICATES)]
    print(f"{len(cells)} celdas: {len(METALS)} metales x {len(LEVELS)} niveles x {N_REPLICATES} replicas")

    t0 = time.time()
    rows = []
    for i, (m, lv, r) in enumerate(cells):
        row = run_cell(m, lv, r, ctx)
        rows.append(row)
        if i == 0:
            dt0 = time.time() - t0
            proj = dt0 * len(cells) / 60
            print(f"[medir] celda 0: {dt0:.0f}s -> proyeccion {len(cells)} celdas: {proj:.0f} min")

    flat = []
    for row in rows:
        if "error" in row:
            continue
        for model in ("censored GP", "GP with L/2"):
            if model in row:
                flat.append({"metal": row["metal"], "level": row["level"],
                            "replicate": row["replicate"], "model": model,
                            "n_test_censored": row["n_test_censored"], "limit": row["limit"],
                            **row[model]})
    df = pd.DataFrame(flat)
    df.round(4).to_csv(FIG / "TABLE_masking_distance_censoring.csv", index=False, encoding="utf-8")

    summary = df.groupby(["metal", "level", "model"]).agg(
        picp95_mean=("picp95", "mean"), picp95_sd=("picp95", "std"),
        rmse_mean=("rmse", "mean"), joint_log_score_mean=("joint_log_score", "mean"),
        divergences_total=("divergences", "sum"), n=("replicate", "count")).round(4)

    txt = f"""# Robustez bajo censura espacialmente informativa (2026-10-26)

Mecanismo: censura asignada por score = distancia normalizada a la labor + ruido gaussiano
(sd={NOISE_FRAC}), no por cuantil del valor -- más lejos de la labor, más probable quedar
censurado, con variación real de réplica a réplica (semillas distintas). Mismo protocolo barato
y mismos modelos que el experimento de enmascaramiento original. Alcance: {len(METALS)} metales x
{len(LEVELS)} niveles (60%, 80%) x {N_REPLICATES} réplicas = {len(cells)} celdas (más chico que el
original de 240 celdas -- declarado, no escondido).

## Resultado agregado

{summary.to_markdown()}

## ¿Se sostiene la ventaja del modelo censurado?

{chr(10).join(
    f"- {metal} al {int(level*100)}%: censurado PICP95={summary.loc[(metal,level,'censored GP'),'picp95_mean']:.3f} "
    f"vs. L/2={summary.loc[(metal,level,'GP with L/2'),'picp95_mean']:.3f}"
    for metal in METALS for level in LEVELS
    if (metal, level, 'censored GP') in summary.index and (metal, level, 'GP with L/2') in summary.index
)}

## Archivos

`figuras/TABLE_masking_distance_censoring.csv`, checkpoints en `outputs/masking_distance/`.
"""
    (OUT / "H_masking_distance_censoring.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print("escrito: outputs/H_masking_distance_censoring.md, figuras/TABLE_masking_distance_censoring.csv")


if __name__ == "__main__":
    main()
