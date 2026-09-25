"""H4.1 — el gradiente de censura REAL. La mejora de mayor retorno del proyecto.

**La crítica que cierra.** El experimento principal impone la censura por cuantiles sobre campos que
no la tienen. El argumento de validez es correcto —un límite de laboratorio fijo que produce un c %
de no detectados *es* un cuantil— pero un revisor lo va a señalar, y con razón.

**La respuesta está en los datos y no cuesta recolectar nada.** Hay seis metales con censura
**real** en las mismas 114 localizaciones, el mismo día y el mismo laboratorio:

| metal | censura real | LOD |
|---|---|---|
| Cobalto | 37.7 % | 0.8 |
| Antimonio | 76.3 % | 2.5 |
| Plata | 78.1 % | 0.6 |
| Cadmio | 79.8 % | 0.5 |
| Bismuto | 84.2 % | 1.5 |
| Níquel | 93.0 % | 1.0 |

**La pregunta que responde, y que la literatura no responde:** ¿la degradación medida con censura
impuesta predice la que se observa con censura real? Si las curvas coinciden, el diseño de
enmascaramiento queda **validado empíricamente** y la crítica desaparece. Si divergen, eso también
es un hallazgo y se publica.

Reutiliza `masking_worker.run_cv_fold`, que ya hace exactamente esta validación cruzada por bloques
espaciales y ya tiene checkpoint por pliegue.
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from config import OUT, RUN_ID, observe, state_update  # noqa: E402
import masking_worker as mw  # noqa: E402

# Los cuatro que ya están en cv_real_data.csv más los cinco que faltan del gradiente real.
GRADIENT = ["Co", "Sb", "Ag", "Bi", "Ni"]
MODELS = ["censored GP", "GP with L/2"]
N_BLOCKS = 5
CORES = 2
N_JOBS = 10


def work(job):
    return mw.run_cv_fold(job, k_blocks=N_BLOCKS, cores=CORES)


if __name__ == "__main__":
    jobs = [{"metal": m, "block": b, "model": mo}
            for m in GRADIENT for b in range(N_BLOCKS) for mo in MODELS]
    done = sum(1 for j in jobs if mw._cv_checkpoint_path(j).exists())
    print(f"H4.1: {len(jobs)} pliegues ({len(GRADIENT)} metales x {N_BLOCKS} bloques x "
          f"{len(MODELS)} modelos), {done} ya en checkpoint")
    observe(f"H4.1 lanzado: validación cruzada espacial sobre los {len(GRADIENT)} metales con "
            f"censura REAL que faltaban (Co 37.7 %, Sb 76.3 %, Ag 78.1 %, Bi 84.2 %, Ni 93.0 %). "
            f"Con Cd (79.8 %) ya hecho, el gradiente real cubre de 37.7 % a 93 % en las mismas 114 "
            f"localizaciones. Es lo que convierte la crítica 'tu censura es impuesta' en el mejor "
            f"argumento del artículo.", "Bloque 4 - artículo")
    state_update("H4.1", done=done, total=len(jobs), note="lanzado")
    t0 = time.time()
    rows = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(work)(j) for j in jobs)
    dt = (time.time() - t0) / 60
    ok = [r for r in rows if r and "samples" in r]
    print(f"\n{len(ok)} de {len(jobs)} pliegues en {dt:.1f} min")
    state_update("H4.1", done=len(ok), total=len(jobs), note=f"completo en {dt:.1f} min")
