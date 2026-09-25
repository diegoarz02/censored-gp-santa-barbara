"""Worker for the prior sensitivity analysis. Separate module because Windows spawns workers.

One fit per (analyte, prior family). Checkpointed per unit so a kill costs at most one fit, and so
this can share the machine with the main run without ever repeating work.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import masking_worker as mw  # noqa: E402

RUN_ID = "F3-20260908"
SEED = 20260908
KW = dict(draws=800, tune=1200, chains=4, target_accept=0.95, cores=2)


def ckpt_path(u: dict) -> Path:
    d = _HERE.parent.parent / "outputs" / "prior_sens"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{u['metal']}_{u['family']}.json"


def run_prior(u: dict) -> dict:
    ck = ckpt_path(u)
    if ck.exists():
        try:
            o = json.loads(ck.read_text(encoding="utf-8"))
            if o.get("run_id") == RUN_ID:
                return o
        except Exception:                                           # noqa: BLE001
            ck.unlink(missing_ok=True)

    ctx = mw._context()
    sb = ctx["sb"]
    metal, family = u["metal"], u["family"]
    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else float(np.nanmin(value))
    min_det = float(np.nanmin(value)) if np.isfinite(value).any() else None

    row = {"metal": metal, "family": family, "run_id": RUN_ID,
           "pct_censored": float(100 * censored.mean())}
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
            is_composite=ctx["is_comp"], anisotropic=True, limit_as_parameter=True,
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
                   divergences=d["divergences"])
    except Exception as exc:                                        # noqa: BLE001
        row["error"] = repr(exc)[:300]

    tmp = ck.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=float, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ck)
    return row
