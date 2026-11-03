"""C1: validacion cruzada del Cd bajo L=LD (tab:cv).

Reutiliza src/h_cv_cd_limit_fix.py CASI tal cual (mismo build_context, misma particion espacial
k=5/seed0=20260906, mismo protocolo KW_MAIN draws=1000/tune=1500/chains=4/target_accept=0.99) --
UN SOLO CAMBIO: limit_as_parameter=False (L=LD fijo = reported_limit, en vez de r libre). Se
escribe en un directorio de checkpoint NUEVO (outputs/cv_cd_LD/) para no pisar ni reutilizar por
error los checkpoints r-libre de h_cv_cd_limit_fix.py (que quedan intactos, es el registro
r-libre).

Metricas EXACTAS de notebooks/_src/nb02.py::score_fold() (rmse_det, mae_det, crps_det,
picp90/95_det, width90/95_det, joint_log_score) -- no reinventadas. Bajo L=LD no hay limit_ratio
que estimar, asi que el joint_log_score se evalua contra el LOD crudo sin la ambiguedad de
referencia que motivo H_CV_Cd_limitfix.md (una sola cifra, no dos).
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
FIG = PROJ / "figuras"
CK_DIR = OUT / "cv_cd_LD"
CK_DIR.mkdir(parents=True, exist_ok=True)

K_BLOCKS = 5
SEED0 = 20260906
DRAWS, TUNE, CHAINS, TARGET_ACCEPT = 1000, 1500, 4, 0.99


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
        out.update({
            "rmse_det": float(np.sqrt(np.mean((s.mean(0) - yd) ** 2))),
            "mae_det": float(np.mean(np.abs(s.mean(0) - yd))),
            "crps_det": float(sb.crps_samples(yd, s).mean()),
        })
        for lvl in (0.90, 0.95):
            lo = np.quantile(s, (1 - lvl) / 2, axis=0)
            hi = np.quantile(s, 1 - (1 - lvl) / 2, axis=0)
            out[f"picp{int(lvl*100)}_det"] = float(np.mean((yd >= lo) & (yd <= hi)))
            out[f"width{int(lvl*100)}_det"] = float(np.mean(hi - lo))
    return out


def main():
    ctx = build_context()
    labels = sb.spatial_blocks(ctx["xy"], k=K_BLOCKS, seed=SEED0)

    fold_rows, fold_scores = [], []
    for b in range(K_BLOCKS):
        ck = CK_DIR / f"Cd_b{b}_LD.npz"
        te = labels == b
        tr = ~te
        if ck.exists():
            z = np.load(ck)
            samples = z["samples"]
            fold_rows.append({"block": b, "seconds": float(z["seconds"]),
                              "rhat_max": float(z["rhat_max"]), "ess_min": float(z["ess_min"]),
                              "divergences": int(z["divergences"])})
            print(f"[cache] block {b}")
        else:
            min_det = float(np.nanmin(ctx["value"])) if np.isfinite(ctx["value"]).any() else None
            seed = int(SEED0 + 31 * b + 7 * (hash("Cd") % 100))

            model = sb.build_censored_gp(
                ctx["xy"][tr], ctx["cov"][tr], np.where(ctx["censored"][tr], np.nan, ctx["value"][tr]),
                ctx["censored"][tr], ctx["resolution"], ctx["lod"], ell_params=ctx["ell_params"],
                is_background=ctx["is_bg"][tr], is_composite=ctx["is_comp"][tr], anisotropic=True,
                limit_as_parameter=False, min_detected=min_det)

            t0 = time.time()
            idata = sb.fit_model(model, draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                                 target_accept=TARGET_ACCEPT, seed=seed)
            dt = time.time() - t0
            diag = sb.diagnose(idata)

            pred = sb.predict_field(idata, ctx["xy"][tr], ctx["xy"][te], ctx["cov"][te],
                                    is_background_new=ctx["is_bg"][te], n_samples=1000, seed=seed,
                                    anisotropic=True)
            samples = pred["y"]

            np.savez_compressed(ck, samples=samples, rhat_max=diag["rhat_max"],
                                ess_min=min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                                divergences=diag["divergences"], seconds=dt)
            fold_rows.append({"block": b, "seconds": dt, "rhat_max": diag["rhat_max"],
                              "ess_min": min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                              "divergences": diag["divergences"]})
            print(f"block {b}: {dt:.0f}s, R-hat={diag['rhat_max']:.4f}, div={diag['divergences']}",
                  flush=True)

        sc = score_fold(samples, ctx["value"], ctx["censored"], te, ctx["lod"])
        sc["block"] = b
        fold_scores.append(sc)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(FIG / "TABLE_CV_Cd_LD_folds.csv", index=False, encoding="utf-8")

    sdf = pd.DataFrame(fold_scores)

    def weighted(col, w):
        v, wt = sdf[col].to_numpy(float), sdf[w].to_numpy(float)
        ok = np.isfinite(v)
        return float(np.sum(v[ok] * wt[ok]) / np.sum(wt[ok])) if ok.any() else float("nan")

    agg = {
        "metal": "Cd", "model": "censored GP (L=LD)",
        "joint_log_score": weighted("joint_log_score", "n_test"),
        "rmse_det": float(np.sqrt(np.sum(sdf["n_detected"] * sdf["rmse_det"] ** 2)
                                  / sdf["n_detected"].sum())),
        "mae_det": weighted("mae_det", "n_detected"),
        "crps_det": weighted("crps_det", "n_detected"),
        "picp90_det": weighted("picp90_det", "n_detected"),
        "width90_det": weighted("width90_det", "n_detected"),
        "picp95_det": weighted("picp95_det", "n_detected"),
        "width95_det": weighted("width95_det", "n_detected"),
    }

    baselines = [
        {"metal": "Cd", "model": "global mean with L/2", "joint_log_score": -1.3709,
         "rmse_det": 3.01, "mae_det": 2.9294, "crps_det": 2.1911, "picp95_det": 0.3043,
         "width95_det": 5.3643},
        {"metal": "Cd", "model": "ordinary kriging L/2", "joint_log_score": -1.6107,
         "rmse_det": 2.7637, "mae_det": 2.6781, "crps_det": 1.8966, "picp95_det": 0.7391,
         "width95_det": 5.9533},
        {"metal": "Cd", "model": "GP with L/2", "joint_log_score": -1.5005,
         "rmse_det": 2.931, "mae_det": 2.8173, "crps_det": 2.0335, "picp95_det": 0.6087,
         "width95_det": 5.9206},
        {"metal": "Cd", "model": "censored GP (r libre, hoy citado)", "joint_log_score": -1.833,
         "rmse_det": 2.1862, "mae_det": 1.8309, "crps_det": 1.2657, "picp95_det": 0.8261,
         "width95_det": 7.4468},
    ]

    out_tab = pd.DataFrame([agg] + baselines)
    out_tab.to_csv(FIG / "TABLE_CV_Cd_LD.csv", index=False, encoding="utf-8")

    print("\n" + folds.round(4).to_string(index=False))
    print("\n" + out_tab.round(4).to_string(index=False))

    rmse_rank = out_tab.sort_values("rmse_det")
    cov_rank = out_tab.sort_values("picp95_det", ascending=False)
    jls_rank = out_tab.sort_values("joint_log_score", ascending=False)
    gana_rmse = rmse_rank.iloc[0]["model"] == "censored GP (L=LD)"
    gana_cov = cov_rank.iloc[0]["model"] == "censored GP (L=LD)"
    gana_jls = jls_rank.iloc[0]["model"] == "censored GP (L=LD)"

    report = f"""# CV_Cd_LD.md -- validacion cruzada del Cd bajo L=LD (tab:cv)

Orden de revision externa (C1). Reutiliza `src/h_cv_cd_limit_fix.py`
con un solo cambio (`limit_as_parameter=False`, L=LD fijo). Checkpoints en `outputs/cv_cd_LD/`
(no se tocan los de `outputs/cv_cd_limitfix/`, que quedan como el registro r-libre).

## Convergencia de las 5 celdas

{folds.round(4).to_markdown(index=False)}

## Metricas sobre las detecciones retenidas, L=LD vs. los 3 baselines + el r-libre ya citado

{out_tab.round(4).to_markdown(index=False)}

## Veredicto a/b/c

(a) RMSE: {'SI, L=LD sigue ganando' if gana_rmse else 'NO, L=LD deja de ganar'} ({agg['rmse_det']:.4f}
vs. el mejor baseline {rmse_rank.iloc[1]['rmse_det'] if gana_rmse else rmse_rank.iloc[0]['rmse_det']:.4f}).
(b) Cobertura PICP95: {'SI, L=LD sigue ganando' if gana_cov else 'NO, L=LD deja de ganar'}
({agg['picp95_det']:.4f}).
(c) joint_log_score: {'SI, L=LD sigue siendo el mejor' if gana_jls else 'NO, L=LD deja de ser el mejor'}
({agg['joint_log_score']:.4f}).

Bajo L=LD no hay `limit_ratio` que estimar, asi que el `joint_log_score` se evalua contra el LOD
crudo sin ambiguedad de referencia -- una sola cifra honesta, no las dos versiones (-1.83/-1.11)
que coexistian bajo r-libre (`H_CV_Cd_limitfix.md`).

**Comparacion directa contra el r-libre que hoy cita main.tex** (RMSE 2.1862->{agg['rmse_det']:.4f},
PICP95 0.8261->{agg['picp95_det']:.4f}, joint_log_score -1.833->{agg['joint_log_score']:.4f}).

No se toco `main.tex` ni `referencias.bib`. No se hizo push.
"""
    (OUT / "CV_Cd_LD.md").write_text(report, encoding="utf-8")
    print("\nescrito: figuras/TABLE_CV_Cd_LD.csv, TABLE_CV_Cd_LD_folds.csv, outputs/CV_Cd_LD.md")


if __name__ == "__main__":
    main()
