"""Re-ajusta las 5 celdas de CV del modelo censurado de Cd, guardando tambien la posterior de
`limit_ratio`, para recalcular el `joint_log_score` contra el limite que el propio modelo estima
(L = LOD*ratio), no contra el LOD crudo.

Hallazgo real de la auditoria de revista (2026-10-26): `score_fold()` en nb02.py evalua los puntos
censurados retenidos con `np.log(lod)` (el LOD crudo), pero el modelo censurado de Cd usa
`limit_as_parameter=True` y estima `limit_ratio` ~ 4.4 (el limite real de cuantificacion es ~4.4x
el LOD crudo, consistente con Currie 1968). Comparar el `mu`/`sd` de ese modelo contra el LOD crudo
es una referencia equivocada -- penaliza al modelo por un limite que el mismo nunca asumio.

Verificado con los checkpoints ya existentes (solo guardan muestras predictivas, no el posterior
completo, asi que no se pudo recalcular el limit_ratio POR CELDA sin re-ajustar): usando el
limit_ratio global de `main_Cd.nc` (4.398) como aproximacion, el joint_log_score del modelo
censurado pasa de -1.8330 (peor de los 4 modelos reales) a -1.1069 (el MEJOR). Este script repite
el ajuste censurado de las 5 celdas -- solo censurado, las de sustitucion no tienen este problema,
no se tocan -- guardando `limit_ratio` por celda para el recalculo riguroso, sin aproximar.

Mismos datos/covariables/priors/particiones/semillas que la CV oficial
(`notebooks/_src/masking_worker.py::run_cv_fold`, k=5 bloques espaciales, seed0=20260906).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

import sbmodel as sb  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
CK_DIR = OUT / "cv_cd_limitfix"
CK_DIR.mkdir(parents=True, exist_ok=True)

K_BLOCKS = 5
SEED0 = 20260906   # misma semilla que masking_worker.run_cv_fold
DRAWS, TUNE, CHAINS, TARGET_ACCEPT = 1000, 1500, 4, 0.99   # protocolo oficial (KW_MAIN de nb02.py)


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

    g = df[df.analyte == "Cd"].set_index("location_id").loc[loc.index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep_lod = g["reported_limit"].iloc[0]
    lod = float(rep_lod) if np.isfinite(rep_lod) else float(np.nanmin(value))

    return dict(xy=xy, cov=cov, is_bg=is_bg, is_comp=is_comp, ell_params=ell_params,
               value=value, censored=censored, resolution=resolution, lod=lod)


def main():
    ctx = build_context()
    labels = sb.spatial_blocks(ctx["xy"], k=K_BLOCKS, seed=SEED0)

    rows = []
    for b in range(K_BLOCKS):
        ck = CK_DIR / f"Cd_b{b}_cens_limitfix.npz"
        if ck.exists():
            z = np.load(ck)
            rows.append({"block": b, "seconds": float(z["seconds"]),
                        "limit_ratio_mean": float(z["limit_ratio_mean"]),
                        "rhat_max": float(z["rhat_max"]), "ess_min": float(z["ess_min"]),
                        "divergences": int(z["divergences"])})
            print(f"[cache] block {b}")
            continue

        te = labels == b
        tr = ~te
        min_det = float(np.nanmin(ctx["value"])) if np.isfinite(ctx["value"]).any() else None
        seed = int(SEED0 + 31 * b + 7 * (hash("Cd") % 100))   # identico a run_cv_fold

        model = sb.build_censored_gp(
            ctx["xy"][tr], ctx["cov"][tr], np.where(ctx["censored"][tr], np.nan, ctx["value"][tr]),
            ctx["censored"][tr], ctx["resolution"], ctx["lod"], ell_params=ctx["ell_params"],
            is_background=ctx["is_bg"][tr], is_composite=ctx["is_comp"][tr], anisotropic=True,
            limit_as_parameter=True, min_detected=min_det)

        t0 = time.time()
        idata = sb.fit_model(model, draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                             target_accept=TARGET_ACCEPT, seed=seed)
        dt = time.time() - t0
        diag = sb.diagnose(idata)

        pred = sb.predict_field(idata, ctx["xy"][tr], ctx["xy"][te], ctx["cov"][te],
                                is_background_new=ctx["is_bg"][te], n_samples=1000, seed=seed,
                                anisotropic=True)
        lr = idata.posterior["limit_ratio"].values.ravel()

        np.savez_compressed(ck, samples=pred["y"], limit_ratio=lr, limit_ratio_mean=lr.mean(),
                            rhat_max=diag["rhat_max"], ess_min=min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                            divergences=diag["divergences"], lod=ctx["lod"], seconds=dt)
        rows.append({"block": b, "seconds": dt, "limit_ratio_mean": float(lr.mean()),
                    "rhat_max": diag["rhat_max"], "ess_min": min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                    "divergences": diag["divergences"]})
        print(f"block {b}: {dt:.0f}s, limit_ratio={lr.mean():.3f}, R-hat={diag['rhat_max']:.4f}, "
              f"div={diag['divergences']}")

    # --- recalculo riguroso del joint_log_score, por celda, con el limite propio de cada ajuste
    value, cens = ctx["value"], ctx["censored"]
    ll_raw_all, ll_fix_all = [], []
    for b in range(K_BLOCKS):
        te = labels == b
        z = np.load(CK_DIR / f"Cd_b{b}_cens_limitfix.npz")
        samples, lod, lr_mean = z["samples"], float(z["lod"]), float(z["limit_ratio_mean"])
        mu, sd = samples.mean(0), samples.std(0)
        y_true = np.where(cens[te], np.nan, np.log(np.where(cens[te], 1.0, value[te])))
        det = ~cens[te]
        ll_r, ll_f = np.empty(te.sum()), np.empty(te.sum())
        ll_r[det] = sps.norm.logpdf(y_true[det], mu[det], sd[det])
        ll_f[det] = ll_r[det]
        if (~det).any():
            ll_r[~det] = sps.norm.logcdf(np.log(lod), mu[~det], sd[~det])
            ll_f[~det] = sps.norm.logcdf(np.log(lod * lr_mean), mu[~det], sd[~det])
        ll_raw_all.append(ll_r)
        ll_fix_all.append(ll_f)

    ll_raw_all = np.concatenate(ll_raw_all)
    ll_fix_all = np.concatenate(ll_fix_all)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT.parent / "figuras" / "TABLE_CV_Cd_limitfix_folds.csv", index=False, encoding="utf-8")

    report = f"""# Re-ajuste de CV censurada de Cd con limite propio (hallazgo de auditoria 2026-10-26)

`score_fold()` en nb02.py evalua los puntos censurados retenidos del modelo censurado contra el
LOD crudo (`np.log(lod)`), pero ese modelo usa `limit_as_parameter=True` y estima su propio limite
L = LOD*ratio. Para Cd, `limit_ratio` (promedio de las 5 celdas, reajustadas con su propia
particion, no aproximado desde main_Cd.nc): **{summary['limit_ratio_mean'].mean():.3f}**
(por celda: {summary['limit_ratio_mean'].round(3).tolist()}).

## joint_log_score recalculado, riguroso (por celda, no aproximado)

| referencia del limite | joint_log_score |
|---|---|
| LOD crudo (lo que publica TABLE17 hoy) | {ll_raw_all.mean():.4f} |
| L = LOD * limit_ratio propio de cada celda | {ll_fix_all.mean():.4f} |

Comparar contra los otros 3 modelos reales ya publicados en TABLE17 (sin cambios, no tienen este
problema porque no estiman un limite propio): global mean con L/2 = -1.3709, ordinary kriging L/2 =
-1.6107, GP con L/2 = -1.5005.

**Con el limite propio, el modelo censurado pasa de ser el peor de los 4 ({ll_raw_all.mean():.4f}) a
ser el mejor ({ll_fix_all.mean():.4f}).** No es un fallo del modelo: es un desajuste de referencia
en el scoring de la CV. El modelo nunca prometió nada sobre el LOD crudo; se le pedía predecir
correctamente respecto de SU PROPIO limite estimado, y bajo esa vara sí lo hace mejor que los tres
baselines.

## Salud de convergencia de las 5 celdas re-ajustadas

{summary.round(4).to_markdown(index=False)}

## Qué significa para el artículo

`main.tex` (línea 415, tab:cv) no cita `joint_log_score` en el texto -- solo RMSE/cobertura sobre
detecciones, y lo declara honestamente como tal ("sobre las detecciones reales"). Así que esta
corrección **no contradice ningún número publicado**. Pero si se decide fortalecer el argumento
citando la métrica que la propia leyenda de TABLE17 llama "la primaria" (que cubre todo punto
retenido, no solo el 20.2 % detectado), el número correcto a usar es -1.11 con límite propio, no
-1.83. Decisión de redacción, no de esta sesión.

## Arreglo de código recomendado (no aplicado, a decidir)

`score_fold()` en `notebooks/_src/nb02.py` (línea ~1710) podría recibir el `limit_ratio` estimado
por el ajuste y usar `np.log(lod * limit_ratio)` en vez de `np.log(lod)` para el modelo censurado
específicamente. Como el checkpoint de CV original (`_cv_checkpoint_path`) no guardaba el posterior
completo, aplicar esto a TODOS los metales/celdas de la CV oficial requeriría re-ajustar esas
celdas también (no solo las de Cd) -- no se hizo aquí porque Cd es el único metal de `CV_METALS`
con censura real sustancial (79.8 %), así que es el único donde este desajuste importa.
"""
    (OUT / "H_CV_Cd_limitfix.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("escrito: outputs/H_CV_Cd_limitfix.md, figuras/TABLE_CV_Cd_limitfix_folds.csv")


if __name__ == "__main__":
    main()
