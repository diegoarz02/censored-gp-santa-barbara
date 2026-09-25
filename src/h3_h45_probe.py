"""Cost probe for H3.1 + H4.5 before launching 240 deep fits.

Rule 3: time a representative fraction and extrapolate. In the previous run a reasoned estimate was
wrong by a factor of 12 and cost four hours.

Two units are timed at full depth, one per model, on the hardest level (80 % censoring, which is
where the headline lives and where the sampler struggles most).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

from config import observe  # noqa: E402
from runner import workers  # noqa: E402
import worker_v4 as w4  # noqa: E402

N_METALS, N_LEVELS, N_REPS, N_MODELS = 3, 4, 10, 2
TOTAL = N_METALS * N_LEVELS * N_REPS * N_MODELS

if __name__ == "__main__":
    times = {}
    for model in ("GP with L/2", "censored GP"):
        u = {"metal": "Hg", "level": 0.8, "replicate": 99, "block": 0, "model": model}
        p = w4.ckpt_path(u)
        p.unlink(missing_ok=True)                                   # probe unit, never reused
        t0 = time.time()
        r = w4.run_deep(u)
        dt = time.time() - t0
        p.unlink(missing_ok=True)
        times[model] = dt
        err = r.get("error")
        print(f"{model:16s}: {dt / 60:5.2f} min | rhat {r.get('rhat_max', float('nan')):.4f} "
              f"| ess {r.get('ess_min', float('nan')):7.1f} | div {r.get('divergences', '?')}"
              + (f" | ERROR {err[:90]}" if err else ""))

    per_unit = sum(times.values()) / len(times)
    n_jobs = workers(cores_per_fit=4, cap=16)
    serial_h = per_unit * TOTAL / 3600
    wall_h = serial_h / n_jobs
    print(f"\nmean per unit      : {per_unit / 60:.2f} min")
    print(f"units              : {TOTAL} (3 metals x 4 levels x 10 replicates x 2 models)")
    print(f"processes           : {n_jobs} at 4 cores each")
    print(f"PROJECTED serial    : {serial_h:.1f} h")
    print(f"PROJECTED wall time : {wall_h:.1f} h")
    observe(f"Sonda H3.1+H4.5: {per_unit / 60:.2f} min por ajuste profundo "
            f"(tune 2000, draws 1000, 4 cadenas, target_accept 0.99). {TOTAL} unidades en "
            f"{n_jobs} procesos -> **proyección {wall_h:.1f} h de reloj** "
            f"({serial_h:.1f} h en serie). Medido, no razonado.", "Bloque 1 - optimización")
