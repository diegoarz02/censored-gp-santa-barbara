# -*- coding: utf-8 -*-
"""Worker for the H6 masking experiment.

This is a **module**, not a notebook cell, and that is not a stylistic choice. The experiment runs
across processes; on Windows `multiprocessing` uses *spawn*, so every worker re-imports the module
that defines the work function. A notebook cell cannot be re-imported. The previous phase of this
project hit exactly this: `RandomForest(n_jobs=-1)` made loky re-import the main module and
re-execute the whole script.

Everything that runs at import time here is cheap and side-effect free. The heavy lifting happens
inside `run_cell`, which each worker calls on its own share of the design grid.

Design of the experiment
------------------------
For each truth field (an analyte measured with **zero** censoring), each artificial censoring level
`c` and each replicate:

1. take the **real** measured values at the 114 locations — nothing is simulated;
2. censor them on the left at the `c` quantile of those real values. Censoring at a quantile is not
   arbitrary: the quantile *is* a fixed number, and a fixed laboratory limit producing `c` per cent
   non-detects is exactly what it represents;
3. hold out one whole spatial block and fit both models on the rest;
4. score against the true values at **every** held-out location, censored ones included.

Point 4 is the reason the experiment exists. Scoring only the detections would restrict the
comparison to the upper truncated tail, where the model that predicts higher wins automatically.
"""
from __future__ import annotations

import os
import numpy as np

# Keep every worker single-threaded in BLAS. Without this, N processes each spawn N BLAS threads
# and the machine thrashes. Set before numpy does any linear algebra.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

__all__ = ["run_cell", "make_design"]

_CACHE: dict = {}


def _context():
    """Load the shared data once per process and keep it. Spawned workers each pay this once."""
    if "ctx" in _CACHE:
        return _CACHE["ctx"]
    import sys
    from pathlib import Path
    import pandas as pd

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import sbmodel as sb

    proj = here.parent.parent
    df = pd.read_csv(proj / "data" / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()

    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)

    loc = loc.assign(log_dist_dump=np.log(loc["dist_waste_dump_m"].clip(lower=1.0)),
                     log_dist_city=np.log(loc["dist_city_m"]))
    covars = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
    cov = loc[covars].to_numpy(float)
    cov = (cov - cov.mean(0)) / cov.std(0)

    is_bg = (loc["stratum"] == "background").to_numpy()
    is_comp = (loc["sample_type"] == "composite").to_numpy()
    ell_params, _, _ = sb.ell_prior_params(xy)

    ctx = dict(sb=sb, df=df, loc=loc, xy=xy, cov=cov, is_bg=is_bg, is_comp=is_comp,
               ell_params=ell_params, proj=proj)
    _CACHE["ctx"] = ctx
    return ctx


def _checkpoint_dir(full_nuts=False):
    from pathlib import Path
    d = (Path(__file__).resolve().parent.parent.parent / "outputs"
         / ("h6_cells_fullnuts" if full_nuts else "h6_cells"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_path(cell, full_nuts=False):
    n = f"{cell['metal']}_{int(round(cell['level']*100)):03d}_r{cell['replicate']:02d}_b{cell['block']}"
    return _checkpoint_dir(full_nuts) / f"{n}.json"


def _write_json_atomic(path, obj):
    """Atomic write. A half-written file after a power cut would poison the resume."""
    import json as _json
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(obj, default=float), encoding="utf-8")
    tmp.replace(path)


def _cv_checkpoint_path(job):
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent.parent / "outputs" / "cv_folds"
    d.mkdir(parents=True, exist_ok=True)
    tag = "cens" if job["model"] == "censored GP" else "sub"
    return d / f"{job['metal']}_b{job['block']}_{tag}.npz"


def make_design(metals, levels, n_replicates, n_blocks=5):
    """The full experiment grid. One held-out block per replicate, rotating across replicates."""
    return [{"metal": m, "level": c, "replicate": r, "block": r % n_blocks}
            for m in metals for c in levels for r in range(n_replicates)]


def run_cell(cell, *, draws=500, tune=500, chains=2, target_accept=0.95, n_pred=800,
             n_blocks=5, seed0=20260906, full_nuts=False, cores=2):
    """Fit both models on one design cell and return one row per model.

    `full_nuts=True` switches to the same sampler settings as the main models. It is used on a
    subset of cells to validate that the cheap inference used inside the experiment does not change
    the conclusions.
    """
    ctx = _context()
    sb = ctx["sb"]
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]

    metal, level, rep, block = cell["metal"], cell["level"], cell["replicate"], cell["block"]
    seed = int(seed0 + 1000 * rep + 97 * int(level * 100) + 13 * hash(metal) % 1000)

    # Per-cell checkpoint, written as soon as the cell finishes. An interrupted run resumes
    # instead of repeating the whole experiment. Each worker writes its own file, so there is no
    # contention and no lock is needed.
    ck = _checkpoint_path(cell, full_nuts)
    if ck.exists():
        import json as _json
        try:
            return _json.loads(ck.read_text(encoding="utf-8"))
        except Exception:                                           # noqa: BLE001
            ck.unlink(missing_ok=True)

    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    assert np.isfinite(value).all(), f"{metal} is not a complete truth field"
    y_true_log = np.log(value)

    # Artificial censoring at the level-th quantile of the real values
    limit = float(np.quantile(value, level))
    censored = value < limit

    # Spatial blocks, fixed across the whole experiment so every cell sees the same partition
    labels = sb.spatial_blocks(xy, k=n_blocks, seed=seed0)
    test = labels == block
    train = ~test
    if test.sum() < 5 or train.sum() < 30:
        return []

    kw = (dict(draws=1000, tune=1500, chains=4, target_accept=0.99) if full_nuts
          else dict(draws=draws, tune=tune, chains=chains, target_accept=target_accept))

    min_det = float(value[~censored].min()) if (~censored).any() else None
    out = []

    # Classical baseline first: ordinary kriging on the L/2-substituted values. No MCMC, so it is
    # essentially free, and it is the literal practice the study is testing.
    try:
        y_sub = np.where(censored, np.log(limit / 2.0), y_true_log)
        draws, kinfo = sb.ordinary_kriging(xy[train], y_sub[train], xy[test],
                                           n_samples=n_pred, seed=seed)
        met = sb.metrics(y_true_log[test], draws, censored=censored[test],
                         log_limit=np.log(limit))
        out.append({**cell, "model": "ordinary kriging L/2", "limit": limit,
                    "n_train": int(train.sum()), "n_test": int(test.sum()),
                    "n_test_censored": int(censored[test].sum()), "full_nuts": full_nuts,
                    **met, "rhat_max": np.nan, "ess_min": np.nan, "divergences": np.nan,
                    "variogram": kinfo["variogram_model"]})
    except Exception as exc:                                        # noqa: BLE001
        out.append({**cell, "model": "ordinary kriging L/2", "limit": limit,
                    "error": repr(exc)[:200]})

    for model_name, extra in [("censored GP", dict(limit_as_parameter=False)),
                              ("GP with L/2", dict(substitute_half_lod=True))]:
        try:
            model = sb.build_censored_gp(
                xy[train], cov[train],
                np.where(censored[train], np.nan, value[train]), censored[train],
                resolution, limit, ell_params=ctx["ell_params"],
                is_background=is_bg[train], is_composite=is_comp[train],
                anisotropic=True, min_detected=min_det, **extra)
            idata = sb.fit_model(model, seed=seed, cores=cores, **kw)
            pred = sb.predict_field(idata, xy[train], xy[test], cov[test],
                                    is_background_new=is_bg[test], n_samples=n_pred,
                                    seed=seed, anisotropic=True)
            met = sb.metrics(y_true_log[test], pred["y"],
                             censored=censored[test], log_limit=np.log(limit))
            diag = sb.diagnose(idata)
            out.append({**cell, "model": model_name, "limit": limit,
                        "n_train": int(train.sum()), "n_test": int(test.sum()),
                        "n_test_censored": int(censored[test].sum()),
                        "full_nuts": full_nuts, **met,
                        "rhat_max": diag["rhat_max"], "ess_min": min(diag["ess_bulk_min"],
                                                                     diag["ess_tail_min"]),
                        "divergences": diag["divergences"]})
        except Exception as exc:                                    # noqa: BLE001
            out.append({**cell, "model": model_name, "limit": limit, "error": repr(exc)[:200]})

    _write_json_atomic(ck, out)
    return out


# `run_cell` is deliberately the only public entry point. Nothing executes on import, so a spawned
# worker re-importing this module pays only the cost of loading numpy and the CSV.
if __name__ == "__main__":  # pragma: no cover - smoke test only
    import json
    print(json.dumps(run_cell({"metal": "Hg", "level": 0.4, "replicate": 0, "block": 0},
                              draws=150, tune=150), indent=2, default=str))


# =============================================================================================
# H7: spatial cross-validation on the real data. Same module, same spawn-safety reasoning.
# =============================================================================================
def run_cv_fold(job, *, k_blocks=5, seed0=20260906, n_pred=1000,
                draws=1000, tune=1500, chains=4, target_accept=0.99, cores=4):
    """Fit one (analyte, block, model) combination of the real-data cross-validation.

    Kept in this module rather than in the notebook for the same reason as `run_cell`: the folds
    are run across processes and Windows re-imports the worker module in every one of them.
    """
    ctx = _context()
    sb = ctx["sb"]
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    metal, block, model_name = job["metal"], job["block"], job["model"]

    ckp = _cv_checkpoint_path(job)
    if ckp.exists():
        try:
            z = np.load(ckp, allow_pickle=False)
            return {"metal": metal, "block": block, "model": model_name,
                    "samples": z["samples"], "rhat_max": float(z["rhat_max"]),
                    "ess_min": float(z["ess_min"]), "divergences": int(z["divergences"]),
                    "lod": float(z["lod"])}
        except Exception:                                           # noqa: BLE001
            ckp.unlink(missing_ok=True)

    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep_lod = g["reported_limit"].iloc[0]
    lod = float(rep_lod) if np.isfinite(rep_lod) else float(np.nanmin(value))
    min_det = float(np.nanmin(value)) if np.isfinite(value).any() else None

    labels = sb.spatial_blocks(xy, k=k_blocks, seed=seed0)
    test, train = labels == block, labels != block
    seed = int(seed0 + 31 * block + 7 * (hash(metal) % 100))

    extra = (dict(limit_as_parameter=True, min_detected=min_det) if model_name == "censored GP"
             else dict(substitute_half_lod=True))
    model = sb.build_censored_gp(
        xy[train], cov[train], np.where(censored[train], np.nan, value[train]), censored[train],
        resolution, lod, ell_params=ctx["ell_params"], is_background=is_bg[train],
        is_composite=is_comp[train], anisotropic=True, **extra)
    idata = sb.fit_model(model, draws=draws, tune=tune, chains=chains, cores=cores,
                         target_accept=target_accept, seed=seed)
    pred = sb.predict_field(idata, xy[train], xy[test], cov[test],
                            is_background_new=is_bg[test], n_samples=n_pred, seed=seed,
                            anisotropic=True)
    diag = sb.diagnose(idata)
    out = {"metal": metal, "block": block, "model": model_name,
           "samples": pred["y"], "rhat_max": diag["rhat_max"],
           "ess_min": min(diag["ess_bulk_min"], diag["ess_tail_min"]),
           "divergences": diag["divergences"], "lod": lod}
    tmp = ckp.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, samples=out["samples"], rhat_max=out["rhat_max"],
                        ess_min=out["ess_min"], divergences=out["divergences"], lod=out["lod"])
    tmp.replace(ckp)
    return out
