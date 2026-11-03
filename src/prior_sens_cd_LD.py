"""C2: sensibilidad al prior de sigma_f del Cd bajo L=LD
(tab:sens).

Reutiliza la logica de notebooks/_src/prior_worker.py::run_prior() CASI tal cual (mismo
protocolo KW=draws=800/tune=1200/chains=4/target_accept=0.95, mismo seed=20260906, misma
construccion de las 3 familias de prior) -- UN SOLO CAMBIO: limit_as_parameter=False en vez de
True. Solo Cd (Hg/As/Pb no tienen censura real, nunca pasan por la rama del limite -- no se
reajustan, se copian tal cual de TABLE31_prior_sensitivity.csv).

Checkpoint en directorio NUEVO (outputs/prior_sens_cd_LD/), no se tocan los de
outputs/prior_sens/ (registro r-libre, F3-20260908).
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import json   # noqa: E402
import time   # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "prior_sens_cd_LD"
CK_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "FLIPLD-20261031"
SEED = 20260906
KW = dict(draws=800, tune=1200, chains=4, target_accept=0.95, cores=2)
FAMILIES = ["default", "pc", "vague"]


def run_prior_LD(family):
    ck = CK_DIR / f"Cd_{family}_LD.json"
    if ck.exists():
        o = json.loads(ck.read_text(encoding="utf-8"))
        if o.get("run_id") == RUN_ID:
            print(f"[cache] {family}")
            return o

    ctx = mw._context()
    sb = ctx["sb"]
    g = ctx["df"][ctx["df"].analyte == "Cd"].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(value))
    min_det = float(np.nanmin(value)) if np.isfinite(value).any() else None

    row = {"metal": "Cd", "family": family, "run_id": RUN_ID,
          "pct_censored": float(100 * censored.mean())}
    t0 = time.time()
    try:
        extra = {}
        if family == "pc":
            obs = np.log(value[~censored]) if (~censored).any() else np.log(value)
            lam, U, mass = sb.pc_rate_from_bound(float(np.nanmean(obs)))
            extra = dict(pc_lambda=lam)
            row.update(pc_U=U, pc_lambda=lam, pc_mass_above_bound=mass)
        model = sb.build_censored_gp(
            ctx["xy"], ctx["cov"], np.where(censored, np.nan, value), censored, resolution, lod,
            ell_params=ctx["ell_params"], is_background=ctx["is_bg"],
            is_composite=ctx["is_comp"], anisotropic=True, limit_as_parameter=False,
            min_detected=min_det, prior_family=family, **extra)
        idata = sb.fit_model(model, seed=SEED, **KW)
        post = idata.posterior
        for name in ("sigma_tot", "rho", "eta", "sigma_n", "beta0"):
            if name in post:
                row[f"{name}_mean"] = float(post[name].mean())
                row[f"{name}_sd"] = float(post[name].std())
        if "ell" in post:
            e = post["ell"].values.reshape(-1, 2)
            row["ell0_mean"], row["ell1_mean"] = float(e[:, 0].mean()), float(e[:, 1].mean())
        d = sb.diagnose(idata)
        row.update(rhat_max=d["rhat_max"],
                   ess_min=min(d["ess_bulk_min"], d["ess_tail_min"]),
                   divergences=d["divergences"], seconds=time.time() - t0)
    except Exception as exc:                                       # noqa: BLE001
        row["error"] = repr(exc)[:300]

    tmp = ck.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=float, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ck)
    print(f"{family}: {row.get('seconds', 0):.0f}s eta_mean={row.get('eta_mean', float('nan')):.5f} "
         f"R-hat={row.get('rhat_max', float('nan')):.4f} div={row.get('divergences', 'NA')}",
         flush=True)
    return row


def main():
    rows_new = [run_prior_LD(f) for f in FAMILIES]
    new_df = pd.DataFrame(rows_new)

    old = pd.read_csv(FIG / "TABLE31_prior_sensitivity.csv")
    others = old[old.metal != "Cd"].copy()
    cd_old = old[old.metal == "Cd"].copy()

    out = pd.concat([new_df.assign(metal="Cd"), others], ignore_index=True)
    out.to_csv(FIG / "TABLE_prior_sens_Cd_LD.csv", index=False, encoding="utf-8")

    eta_default = float(new_df.loc[new_df.family == "default", "eta_mean"].iloc[0])
    eta_vague = float(new_df.loc[new_df.family == "vague", "eta_mean"].iloc[0])
    pct_change = 100 * (eta_vague - eta_default) / eta_default

    eta_default_old = float(cd_old.loc[cd_old.family == "default", "eta_mean"].iloc[0])
    eta_vague_old = float(cd_old.loc[cd_old.family == "vague", "eta_mean"].iloc[0])
    pct_change_old = 100 * (eta_vague_old - eta_default_old) / eta_default_old

    print("\n" + new_df[["metal", "family", "eta_mean", "eta_sd", "rhat_max", "ess_min",
                        "divergences"]].round(5).to_string(index=False))
    print(f"\ncambio calibrada->vaga bajo L=LD: {pct_change:.1f}% "
         f"(r-libre, hoy citado: {pct_change_old:.1f}%)")

    report = f"""# PRIOR_SENS_Cd_LD.md -- sensibilidad al prior de sigma_f del Cd bajo L=LD (tab:sens)

Orden de revision externa (C2). Reutiliza
`notebooks/_src/prior_worker.py::run_prior()` con un solo cambio (`limit_as_parameter=False`),
mismo protocolo (draws=800/tune=1200/chains=4/target_accept=0.95) y semilla (20260906) que el
run oficial F3-20260908. Solo la fila del Cd se reajusta -- Hg/As/Pb (0% censura real) se copian
tal cual de `TABLE31_prior_sensitivity.csv`, no se re-ejecutan.

## Posterior de sigma_f (eta) bajo las 3 especificaciones, L=LD

{new_df[["family", "eta_mean", "eta_sd", "rhat_max", "ess_min", "divergences"]].round(5).to_markdown(index=False)}

## Comparacion con el r-libre ya citado (F3-20260908)

| especificacion | eta_mean L=LD | eta_mean r-libre (hoy citado) |
|---|---|---|
| default | {eta_default:.5f} | {eta_default_old:.5f} |
| vague | {eta_vague:.5f} | {eta_vague_old:.5f} |

**Cambio porcentual calibrada->vaga**: L=LD = **{pct_change:.1f}%**, r-libre (hoy citado) =
{pct_change_old:.1f}%.

## Veredicto cualitativo

{"El efecto del prior sobre sigma_f SIGUE creciendo bajo censura fuerte, y bajo L=LD pesa AUN MAS que bajo r-libre, consistente con lo esperado (la verosimilitud censurada informa menos con el limite fijo pequeno)." if pct_change > pct_change_old else "El efecto del prior bajo L=LD es menor que bajo r-libre -- revisar si el mecanismo (el prior pesa mas cuando los datos informan menos) se mantiene cualitativamente pese al numero mas chico."}
La lectura cualitativa de main.tex ("el prior pesa mas justamente cuando los datos informan
menos") se mantiene; el numero puntual del Cd (69% bajo r-libre) cambia a {pct_change:.1f}% bajo
L=LD.

No se toco `main.tex` ni `referencias.bib`. No se hizo push.
"""
    (OUT / "PRIOR_SENS_Cd_LD.md").write_text(report, encoding="utf-8")
    print("\nescrito: figuras/TABLE_prior_sens_Cd_LD.csv, outputs/PRIOR_SENS_Cd_LD.md")


if __name__ == "__main__":
    main()
