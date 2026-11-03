"""Q1-R2: robustez de la decision ambiental del Cd a los
priors, bajo L=LD (objecion: sigma_f cambia 86% entre priors; falta ver si la
CONCLUSION ambiental -- clasificacion de excedencia + intervalo del area -- se mantiene).

Reajusta el Cd bajo L=LD con las 3 especificaciones de prior (default/calibrada, pc, vague) --
reutiliza casi tal cual `src/prior_sens_cd_LD.py::run_prior_LD()` (mismo protocolo
draws=800/tune=1200/chains=4/target_accept=0.95, mismo seed=20260906), con UN SOLO CAMBIO: esta
vez persiste el idata COMPLETO (`prior_sens_cd_LD.py` solo guardo el resumen JSON). Checkpoint en
directorio NUEVO (outputs/q1_r2_prior_excedencia/) -- no se tocan los .json de
outputs/prior_sens_cd_LD/.

Para cada prior, predice en la grilla oficial (50 m, buffer 300 m, 7381 celdas) reutilizando
`predict_at_resolution()` de `src/flip_ld_cd_f2_f4.py` tal cual (misma rutina ya validada contra
el principal: 38.17%/39.82%/40.09%/[22.11,56.31] reproducidos exacto).
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402
import flip_ld_cd_f2_f4 as f2f4  # noqa: E402
from flip_ld_cd_f2_f4 import shared_setup, three_quantities_row  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "q1_r2_prior_excedencia"
CK_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "Q1R2-20261102"
SEED = 20260906
KW = dict(draws=800, tune=1200, chains=4, target_accept=0.95, cores=2)
FAMILIES = ["default", "pc", "vague"]

PRINCIPAL = dict(pct_domain_agri_gt_05=38.165560, E_area_agri_pct=39.822964,
                 area_frac_agri_median_pct=40.089419, area_frac_agri_ci95_lo_pct=22.110825,
                 area_frac_agri_ci95_hi_pct=56.307411)


def fit_prior_LD(family):
    """Copia de prior_sens_cd_LD.py::run_prior_LD(), con UN cambio: persiste el idata completo."""
    ck_nc = CK_DIR / f"Cd_{family}_LD.nc"
    ck_meta = CK_DIR / f"Cd_{family}_LD_meta.csv"
    if ck_nc.exists() and ck_meta.exists():
        print(f"[cache] {family}")
        return az.from_netcdf(ck_nc), pd.read_csv(ck_meta).iloc[0].to_dict()

    ctx = mw._context()
    sb = ctx["sb"]
    g = ctx["df"][ctx["df"].analyte == "Cd"].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(value))
    min_det = float(np.nanmin(value)) if np.isfinite(value).any() else None

    extra = {}
    if family == "pc":
        obs = np.log(value[~censored]) if (~censored).any() else np.log(value)
        lam, U, mass = sb.pc_rate_from_bound(float(np.nanmean(obs)))
        extra = dict(pc_lambda=lam)

    t0 = time.time()
    model = sb.build_censored_gp(
        ctx["xy"], ctx["cov"], np.where(censored, np.nan, value), censored, resolution, lod,
        ell_params=ctx["ell_params"], is_background=ctx["is_bg"], is_composite=ctx["is_comp"],
        anisotropic=True, limit_as_parameter=False, min_detected=min_det, prior_family=family,
        **extra)
    idata = sb.fit_model(model, seed=SEED, **KW)
    dt = time.time() - t0
    diag = sb.diagnose(idata)
    meta = dict(family=family, run_id=RUN_ID, seconds=dt, rhat_max=diag["rhat_max"],
                ess_min=min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                divergences=diag["divergences"])
    print(f"{family}: {dt:.0f}s R-hat={diag['rhat_max']:.4f} div={diag['divergences']}", flush=True)

    idata.to_netcdf(ck_nc)
    pd.DataFrame([meta]).to_csv(ck_meta, index=False, encoding="utf-8")
    return idata, meta


def main():
    df, workings, loc, xy, cov_mean, cov_std, xy_center = shared_setup()

    rows = []
    for family in FAMILIES:
        idata, meta = fit_prior_LD(family)
        # KW usa draws=800*chains=4=3200 (protocolo de sensibilidad), no 4000 como el principal --
        # flip_ld_cd_f2_f4.predict_at_resolution() preasigna arrays de tamano N_MAP_SAMPLES, asi que
        # hay que igualarlo al numero real de draws del posterior antes de llamarla (sin tocar el
        # archivo, que ya esta validado para el caso de 4000 draws del principal).
        n_post = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
        f2f4.N_MAP_SAMPLES = n_post
        r50 = f2f4.predict_at_resolution(idata, xy, loc, workings, cov_mean, cov_std, xy_center, 50.0)
        assert r50["n_cells"] == 7381, f"grid 50m no coincide: {r50['n_cells']}"
        row = three_quantities_row(50, r50)
        row.update(meta)
        rows.append(row)
        print(f"  {family}: pct_domain_agri_gt_0.5={row['pct_domain_agri_gt_0.5']:.2f}% "
              f"E_area={row['E_area_agri_pct']:.2f}% mediana={row['area_frac_agri_median_pct']:.2f}% "
              f"IC95=[{row['area_frac_agri_ci95_lo_pct']:.2f},{row['area_frac_agri_ci95_hi_pct']:.2f}]",
              flush=True)

    tab = pd.DataFrame(rows)
    keep = ["family", "run_id", "rhat_max", "ess_min", "divergences", "seconds",
            "pct_domain_agri_gt_0.5", "E_area_agri_pct", "area_frac_agri_median_pct",
            "area_frac_agri_ci95_lo_pct", "area_frac_agri_ci95_hi_pct",
            "pct_domain_resid_gt_0.5", "E_area_resid_pct", "area_frac_resid_median_pct",
            "area_frac_resid_ci95_lo_pct", "area_frac_resid_ci95_hi_pct"]
    tab = tab[keep]
    tab.to_csv(FIG / "TABLE_Q1_excedencia_por_prior.csv", index=False, encoding="utf-8")
    print("\n" + tab.round(3).to_string(index=False))

    row_default = tab[tab.family == "default"].iloc[0]
    max_spread_pct = float(tab["pct_domain_agri_gt_0.5"].max() - tab["pct_domain_agri_gt_0.5"].min())
    max_spread_median = float(tab["area_frac_agri_median_pct"].max()
                              - tab["area_frac_agri_median_pct"].min())
    clasificacion_estable = max_spread_pct < 10.0
    intervalo_estable = max_spread_median < 10.0

    comp_rows = []
    for _, r in tab.iterrows():
        comp_rows.append({
            "especificacion": r["family"],
            "pct_domain_gt_0.5": r["pct_domain_agri_gt_0.5"],
            "E_area_pct": r["E_area_agri_pct"],
            "mediana_pct": r["area_frac_agri_median_pct"],
            "IC95_lo": r["area_frac_agri_ci95_lo_pct"],
            "IC95_hi": r["area_frac_agri_ci95_hi_pct"],
        })
    comp_rows.append({"especificacion": "principal (flip_LD_Cd.nc, protocolo oficial)",
                      "pct_domain_gt_0.5": PRINCIPAL["pct_domain_agri_gt_05"],
                      "E_area_pct": PRINCIPAL["E_area_agri_pct"],
                      "mediana_pct": PRINCIPAL["area_frac_agri_median_pct"],
                      "IC95_lo": PRINCIPAL["area_frac_agri_ci95_lo_pct"],
                      "IC95_hi": PRINCIPAL["area_frac_agri_ci95_hi_pct"]})
    comp = pd.DataFrame(comp_rows)

    report = f"""# Q1_ROBUSTEZ_PRIOR_EXCEDENCIA.md -- robustez de la decision ambiental del Cd a los
priors, bajo L=LD (Q1-R2)

Orden de revision externa (Q1-R2), tras una objecion #5:
sigma_f cambia 86% entre priors (ver `outputs/PRIOR_SENS_Cd_LD.md`), pero eso no dice si la
CONCLUSION ambiental (clasificacion de excedencia agricola, ECA=1.4, y el intervalo del area) se
mueve o no. Reajuste de los 3 priors bajo L=LD con idata completo persistido (a diferencia de
`src/prior_sens_cd_LD.py`, que solo guardo el resumen JSON) -- mismo protocolo
(draws=800/tune=1200/chains=4/target_accept=0.95, seed=20260906). Prediccion en la grilla oficial
(50 m, buffer 300 m, 7381 celdas) con la misma rutina ya validada de
`src/flip_ld_cd_f2_f4.py::predict_at_resolution()`.

## Convergencia de los 3 reajustes

{tab[['family','rhat_max','ess_min','divergences','seconds']].round(4).to_markdown(index=False)}

## Las 3 cantidades del Cd agricola (ECA=1.4), por prior, vs. el principal

{comp.round(3).to_markdown(index=False)}

`principal` es `outputs/idata/flip_LD_Cd.nc` (protocolo oficial de produccion, draws/tune/target_accept
distintos a este reajuste de sensibilidad) -- se incluye como referencia, no como una 4ta
especificacion de prior.

## Veredicto de robustez

**Clasificacion de excedencia (`% dominio con P(exc)>0.5`):** rango entre priors =
{max_spread_pct:.2f} puntos porcentuales ({tab['pct_domain_agri_gt_0.5'].min():.2f}% a
{tab['pct_domain_agri_gt_0.5'].max():.2f}%). {'Se mantiene RAZONABLEMENTE ESTABLE (rango < 10 pp).' if clasificacion_estable else 'SE MUEVE de forma apreciable (rango >= 10 pp) -- la clasificacion de excedencia NO es robusta a la eleccion de prior.'}

**Intervalo del area (mediana de `area_frac_agri`):** rango entre priors =
{max_spread_median:.2f} puntos porcentuales ({tab['area_frac_agri_median_pct'].min():.2f}% a
{tab['area_frac_agri_median_pct'].max():.2f}%). {'Se mantiene RAZONABLEMENTE ESTABLE (rango < 10 pp).' if intervalo_estable else 'SE MUEVE de forma apreciable (rango >= 10 pp) -- el intervalo del area NO es robusto a la eleccion de prior.'}

Esto contrasta con `sigma_f` (69%/85.8% r-libre/L=LD entre calibrada y vaga, ver
`outputs/PRIOR_SENS_Cd_LD.md`): el hiperparametro individual es sensible al prior, pero
{'la CONCLUSION ambiental que de el depende (excedencia/area) se sostiene pese a eso -- son cosas distintas, y main.tex puede afirmar robustez ambiental a los priors con esta evidencia.' if (clasificacion_estable and intervalo_estable) else 'la CONCLUSION ambiental tambien se mueve de forma apreciable -- no se puede afirmar robustez ambiental a los priors con esta evidencia; main.tex deberia matizar o declarar esta sensibilidad como limitacion.'}

No se toco `main.tex` ni `referencias.bib`. No se hizo push.
"""
    (OUT / "Q1_ROBUSTEZ_PRIOR_EXCEDENCIA.md").write_text(report, encoding="utf-8")
    print("\nescrito: figuras/TABLE_Q1_excedencia_por_prior.csv, outputs/Q1_ROBUSTEZ_PRIOR_EXCEDENCIA.md")


if __name__ == "__main__":
    main()
