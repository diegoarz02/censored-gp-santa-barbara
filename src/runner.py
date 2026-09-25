"""H1.1 + H1.3 + H1.7 — one parallel, checkpointed, resumable runner for every expensive loop.

Three properties, and all three matter:

**Parallel.** The previous run put 7 main models through in series on 4 of 20 cores. Every loop here
is a list of independent units, so it goes through `joblib` with a worker count measured on this
machine rather than assumed.

**Checkpointed per unit of work.** One file per unit, named deterministically from the unit's own
parameters, so the runner can tell what is missing without keeping a register.

**Resumable after a hard kill.** A half-written `.nc` exists and looks complete; a resume would
accept it. Every write goes through a temporary file and an atomic rename, every checkpoint is
*opened* rather than merely stat-ed before being trusted, and its `run_id` must match.

Windows spawns rather than forks, so the work function must live in an importable module and BLAS
threads must be pinned before numpy is imported anywhere in the child.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT, RUN_ID, atomic_write, observe, state_update  # noqa: E402

N_CPU = os.cpu_count() or 4


def workers(cores_per_fit: int = 2, cap: int = 16) -> int:
    """How many processes to run at once.

    The plan asks for 16 rather than 10 on a 20-core machine, with the caveat that memory bandwidth
    usually takes over somewhere around there. The number is capped by the cores each fit claims,
    so that `workers * cores_per_fit` never oversubscribes badly.
    """
    return max(1, min(cap, N_CPU // max(1, cores_per_fit)))


# ------------------------------------------------------------------------------- checkpoint layer
def ckpt_path(kind: str, key: str, suffix: str = ".npz") -> Path:
    """Deterministic name from the unit's parameters, never from a counter."""
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
    return d / f"{safe}{suffix}"


def ckpt_load(path: Path, expect_run_id: str = RUN_ID):
    """Open it — existing is not enough. A truncated file exists and looks complete.

    Returns the loaded object, or None if the file is missing, unreadable or from another run, in
    which case it is deleted so the unit is recomputed.
    """
    if not path.exists():
        return None
    try:
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=False)
            got = str(z["run_id"]) if "run_id" in z else None
            if expect_run_id and got != expect_run_id:
                path.unlink(missing_ok=True)
                return None
            return {k: z[k] for k in z.files}
        if path.suffix == ".json":
            o = json.loads(path.read_text(encoding="utf-8"))
            if expect_run_id and o.get("run_id") != expect_run_id:
                path.unlink(missing_ok=True)
                return None
            return o
    except Exception:                                              # noqa: BLE001
        path.unlink(missing_ok=True)
        return None
    return None


def ckpt_save(path: Path, obj: dict) -> Path:
    """Atomic: write to `.tmp`, then rename. A crash mid-write leaves no usable file."""
    obj = dict(obj)
    obj["run_id"] = RUN_ID
    if path.suffix == ".npz":
        return atomic_write(path, lambda p: np.savez_compressed(p, **obj))
    return atomic_write(path, lambda p: p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ------------------------------------------------------------------------------------ the runner
def run_units(milestone: str, units: list, fn, *, cores_per_fit: int = 2, cap: int = 16,
              verbose: int = 5, describe=None) -> list:
    """Run `fn(unit)` over `units` in parallel, skipping those already checkpointed.

    `fn` must be a top-level function in an importable module (Windows spawn re-imports it) and must
    do its own checkpoint load/save, so a unit finished in a previous process is never recomputed.
    """
    n_jobs = workers(cores_per_fit, cap)
    t0 = time.time()
    state_update(milestone, done=0, total=len(units), note=f"launching on {n_jobs} processes")
    print(f"[{milestone}] {len(units)} units on {n_jobs} processes "
          f"({cores_per_fit} cores each, {N_CPU} available)")
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(fn)(u) for u in units)
    dt = time.time() - t0
    n_skip = sum(1 for r in out if isinstance(r, dict) and r.get("_skipped"))
    state_update(milestone, done=len(units), total=len(units),
                 note=f"{dt / 60:.1f} min, {n_skip} skipped from checkpoint")
    print(f"[{milestone}] done in {dt / 60:.1f} min "
          f"({n_skip} skipped from checkpoint, {len(units) - n_skip} computed)")
    observe(f"{milestone}: {len(units)} unidades en {n_jobs} procesos, {dt / 60:.1f} min "
            f"({n_skip} saltadas por checkpoint, {len(units) - n_skip} calculadas).",
            "Bloque 1 - optimización")
    return out


def timed_probe(label: str, fn, n_small: int, n_total: int) -> float:
    """Rule 3: measure a representative fraction and extrapolate, never reason about the cost.

    Returns the projected total in minutes and prints both numbers, so the projection is on record
    before the long loop starts.
    """
    t0 = time.time()
    fn()
    dt = time.time() - t0
    proj = dt / n_small * n_total / 60
    print(f"[probe] {label}: {dt:.1f} s for {n_small} -> {proj:.1f} min for {n_total}")
    observe(f"Sonda de coste — {label}: {dt:.1f} s por {n_small} unidades, "
            f"proyección {proj:.1f} min para {n_total}.", "Bloque 1 - optimización")
    return proj
