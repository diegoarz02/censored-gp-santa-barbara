"""H3.1 + H4.5 — 240 deep fits, parallel, checkpointed, resumable.

Launched as the real thing rather than after a throwaway probe: every unit is checkpointed, so the
first completions measure the rate *and* count towards the result. A probe that is thrown away
measures the same thing and produces nothing.

Progress is readable from disk at any moment (`outputs/h6_deep/*.json`), so the rate can be checked
without touching the running processes, and a kill at any point loses at most one fit per worker.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

from config import observe, state_update  # noqa: E402
from runner import run_units, workers  # noqa: E402
import worker_v4 as w4  # noqa: E402

METALS = ["Hg", "Pb", "As"]
LEVELS = [0.20, 0.40, 0.60, 0.80]
N_REPS = 10


# Worker geometry is read from the environment so the two configurations can be compared without
# editing code. 5 processes x 4 chains uses all 20 cores with one chain each; 10 x 2 runs twice as
# many fits with two chains sharing a core. Which wins is a memory-bandwidth question, not a
# reasoning question, so it is measured.
import os
CORES = int(os.environ.get("F3_CORES", "4"))
CAP = int(os.environ.get("F3_CAP", "5"))


def work(u):
    return w4.run_deep(u, cores=CORES)


if __name__ == "__main__":
    units = w4.make_units(METALS, LEVELS, N_REPS)
    # Order by scientific value, not by the order the grid happens to enumerate.
    #
    # The first launch ran metal by metal, so nine hours in Hg was nearly complete and As had not
    # started. If the run had to stop there we would have lost a whole analyte instead of the
    # levels that carry least weight. The claim lives at 80 % censoring — that is where the
    # coverage collapse is and where H3.1's objection bites — and at 20 % nothing separates.
    #
    # So: highest censoring first, and interleave the three analytes inside each level so all three
    # advance together. Whatever is unfinished is then the least informative part of the design.
    order_metal = {m: i for i, m in enumerate(METALS)}
    units.sort(key=lambda u: (-u["level"], u["replicate"], order_metal[u["metal"]], u["model"]))
    done_before = sum(1 for u in units if w4.ckpt_path(u).exists())
    print(f"H3.1+H4.5: {len(units)} units, {done_before} already checkpointed")
    print(f"settings: {w4.DEEP}")
    observe(f"H3.1+H4.5 lanzado: {len(units)} ajustes profundos "
            f"(tune {w4.DEEP['tune']}, draws {w4.DEEP['draws']}, {w4.DEEP['chains']} cadenas, "
            f"target_accept {w4.DEEP['target_accept']}), {done_before} ya en checkpoint. "
            f"Los dos modelos con los MISMOS ajustes, para que nadie diga que el baseline salió "
            f"handicapado.", "Bloque 3 - auditoría")
    t0 = time.time()
    rows = run_units("H3.1+H4.5", units, work, cores_per_fit=CORES, cap=CAP)
    ok = [r for r in rows if isinstance(r, dict) and "error" not in r]
    print(f"\n{len(ok)} of {len(units)} succeeded in {(time.time() - t0) / 60:.1f} min")
    state_update("H3.1+H4.5", done=len(ok), total=len(units), note="complete")
