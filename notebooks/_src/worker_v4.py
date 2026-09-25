"""H3.1 + H4.5 — one fit per unit, deeply sampled, checkpointed and resumable.

Two milestones share this worker because they ask for the same thing from different directions:

* **H3.1** re-runs the `GP with L/2` baseline, which produced 488 of the 489 divergences in the
  previous run. Its narrow intervals carry the headline, and an unconverged fit cannot support a
  claim about a method's calibration.
* **H4.5** replaces "more replicates" with "more depth". Every one of the 240 Bayesian fits had a
  minimum ESS below 400, median 148. Coverage and CRPS are tail functionals, so an ESS of 150 puts
  Monte Carlo error on the order of the effect being measured. Ten replicates already give a
  standard error of 0.062 against a difference of 0.40; depth is what is missing, not width.

Running both models deeply in the same pass is cheaper than two passes and gives a like-for-like
comparison: same sampler settings on both sides, so no one can say the baseline was handicapped.

One fit per unit, rather than both models per cell, so a kill loses at most one fit and the
scheduler can fill all workers evenly.
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
sys.path.insert(0, str(_HERE.parent.parent / "src"))

import masking_worker as mw  # noqa: E402

RUN_ID = "F4-20260923"     # corrida 3: solo se usa para checkpoints NUEVOS (kriging). Los 240
SEED0 = 20260923           # checkpoints de censored GP / GP with L/2 siguen con run_id F3-20260908
                            # y este modulo nunca los relee via _load(), asi que no se tocan.
KRIGING_MODEL = "ordinary kriging L/2"

# H3.1 prescribes these for the baseline; H4.5 wants the same depth on our own model so the two are
# compared on equal footing. 4 chains x 3000 iterations is four times the previous 2 x 1000.
DEEP = dict(draws=1000, tune=2000, chains=4, target_accept=0.99, cores=4)
N_PRED = 800
N_BLOCKS = 5

MODELS = {
    "censored GP": dict(limit_as_parameter=False),
    "GP with L/2": dict(substitute_half_lod=True),
}


def ckpt_dir() -> Path:
    d = _HERE.parent.parent / "outputs" / "h6_deep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_path(u: dict) -> Path:
    tag = {"censored GP": "cens", "GP with L/2": "sub",
           KRIGING_MODEL: "krig"}[u["model"]]
    return ckpt_dir() / (f"{u['metal']}_{int(round(u['level'] * 100)):03d}"
                         f"_r{u['replicate']:02d}_b{u['block']}_{tag}.json")


def _load(path: Path):
    """Open it, do not merely stat it: a truncated file exists and looks complete."""
    if not path.exists():
        return None
    try:
        o = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                               # noqa: BLE001
        path.unlink(missing_ok=True)
        return None
    if o.get("run_id") != RUN_ID:
        path.unlink(missing_ok=True)
        return None
    return o


def _save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float, ensure_ascii=False), encoding="utf-8")
    for attempt in range(3):                                        # OneDrive may hold the handle
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            import time
            time.sleep(2 ** attempt)
    raise PermissionError(f"could not rename {tmp.name} (OneDrive sync?)")


def make_units(metals, levels, n_replicates, models=None, n_blocks=N_BLOCKS) -> list[dict]:
    """Units for the two MCMC models only. Kriging is deliberately never in `models`'s default:
    a caller that forgets to restrict `models` here must not be able to silently sweep up the
    already-computed censored/L2-GP checkpoints (see `make_kriging_units`)."""
    models = models or list(MODELS)
    return [{"metal": m, "level": c, "replicate": r, "block": r % n_blocks, "model": mo}
            for m in metals for c in levels for r in range(n_replicates) for mo in models]


def make_kriging_units(metals, levels, n_replicates, n_blocks=N_BLOCKS) -> list[dict]:
    """Units for the classical baseline only — hito A1. Kept structurally separate from
    `make_units` so nothing generic ever constructs a kriging unit by accident, and so nothing
    generic constructs a censored-GP/L2-GP unit that a kriging-only driver might then run."""
    return [{"metal": m, "level": c, "replicate": r, "block": r % n_blocks, "model": KRIGING_MODEL}
            for m in metals for c in levels for r in range(n_replicates)]


def run_deep(u: dict, **override) -> dict:
    """Fit one (metal, level, replicate, block, model) unit with the deep sampler settings."""
    ck = ckpt_path(u)
    cached = _load(ck)
    if cached is not None:
        cached["_skipped"] = True
        return cached

    kw = {**DEEP, **override}
    ctx = mw._context()
    sb = ctx["sb"]
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    metal, level, rep, block, model_name = (u["metal"], u["level"], u["replicate"],
                                            u["block"], u["model"])
    seed = int(SEED0 + 1000 * rep + 97 * int(level * 100) + 13 * (hash(metal) % 1000))

    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    y_true_log = np.log(value)

    limit = float(np.quantile(value, level))
    censored = value < limit
    labels = sb.spatial_blocks(xy, k=N_BLOCKS, seed=20260906)        # same partition as before
    test = labels == block
    train = ~test
    min_det = float(value[~censored].min()) if (~censored).any() else None

    row = {**u, "run_id": RUN_ID, "limit": limit, "n_train": int(train.sum()),
           "n_test": int(test.sum()), "n_test_censored": int(censored[test].sum()),
           "deep": True}
    try:
        if model_name == KRIGING_MODEL:
            # A1: the classical baseline. Closed-form fit of the variogram (PyKrige), not MCMC —
            # tune/draws/target_accept do not apply. Same L/2 substitution and the same 120
            # train/test splits as the other two models, so the three-way comparison is clean.
            y_sub = np.where(censored[train], limit / 2, value[train])
            draws, kparams = sb.ordinary_kriging(xy[train], np.log(y_sub), xy[test], seed=seed)
            met = sb.metrics(y_true_log[test], draws, censored=censored[test],
                             log_limit=np.log(limit))
            row.update(met)
            # No MCMC diagnostics exist for a closed-form fit. Recorded explicitly as N/A rather
            # than a number that would look like a real R-hat/ESS/divergence count.
            row.update(rhat_max=None, ess_min=None, ess_bulk_min=None, ess_tail_min=None,
                       divergences=0, draws=int(draws.shape[0]), tune=0, chains=1,
                       diagnostics_na="closed-form kriging fit, no MCMC diagnostics apply",
                       kriging_params=kparams)
        else:
            model = sb.build_censored_gp(
                xy[train], cov[train], np.where(censored[train], np.nan, value[train]),
                censored[train], resolution, limit, ell_params=ctx["ell_params"],
                is_background=is_bg[train], is_composite=is_comp[train], anisotropic=True,
                min_detected=min_det, **MODELS[model_name])
            idata = sb.fit_model(model, seed=seed, **kw)
            pred = sb.predict_field(idata, xy[train], xy[test], cov[test],
                                    is_background_new=is_bg[test], n_samples=N_PRED, seed=seed,
                                    anisotropic=True)
            met = sb.metrics(y_true_log[test], pred["y"], censored=censored[test],
                             log_limit=np.log(limit))
            diag = sb.diagnose(idata)
            row.update(met)
            row.update(rhat_max=diag["rhat_max"],
                       ess_min=min(diag["ess_bulk_min"], diag["ess_tail_min"]),
                       ess_bulk_min=diag["ess_bulk_min"], ess_tail_min=diag["ess_tail_min"],
                       divergences=diag["divergences"],
                       draws=kw["draws"], tune=kw["tune"], chains=kw["chains"])
    except Exception as exc:                                        # noqa: BLE001
        row["error"] = repr(exc)[:300]

    _save(ck, row)
    return row


if __name__ == "__main__":                                          # pragma: no cover
    import time
    u = {"metal": "Hg", "level": 0.8, "replicate": 0, "block": 0, "model": "GP with L/2"}
    t0 = time.time()
    r = run_deep(u, draws=200, tune=200, chains=2, cores=2)
    print(f"{time.time() - t0:.1f} s", json.dumps({k: v for k, v in r.items()
                                                   if k != "_skipped"}, indent=1, default=str)[:600])
