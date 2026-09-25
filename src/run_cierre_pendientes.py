"""Lanza los tres pendientes, uno a uno y con checkpoint, para que un corte no obligue a repetir.

Cada bloque se salta si ya hay resultado en `outputs/cierre_pendientes.json`. Los fallos se
registran como resultado —con su excepción— en vez de detener la corrida: un intento fallido
documentado vale más que un hueco silencioso.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import OUT, log_attempt  # noqa: E402
import h_cierre_pendientes as H  # noqa: E402

CK = OUT / "cierre_pendientes.json"


def load():
    return json.loads(CK.read_text(encoding="utf-8")) if CK.exists() else {}


def save(res):
    CK.write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False), encoding="utf-8")


def step(res, key, titulo, fn, extra=None):
    if key in res:
        print(f"[skip] {titulo} — ya en checkpoint")
        return
    print(f"--- {titulo} ---", flush=True)
    try:
        out = fn()
        if extra:
            out = extra(out)
        res[key] = out
        print(out if not isinstance(out, list) else pd.DataFrame(out).round(3).to_string(index=False))
    except Exception as exc:                                        # noqa: BLE001
        res[key] = {"error": repr(exc)[:300]}
        log_attempt(titulo, repr(exc)[:150], "registrado como intento fallido, no bloquea")
        print("  falló:", repr(exc)[:220])
    save(res)


def _mixture():
    r, p, lab = H.mixture("Hg")
    np.savez_compressed(OUT / "mixture_assignment.npz", p_affected=p, oefa_label=lab)
    return r


if __name__ == "__main__":
    res = load()
    step(res, "mixture", "H8.5b mezcla bayesiana fondo/contaminado (Hg)", _mixture)
    step(res, "boxcox", "H4.3 Box-Cox con lambda estimado (Pb)", lambda: H.boxcox("Pb"))
    step(res, "preferential", "H4.4 muestreo preferencial, modelo conjunto (Hg)",
         lambda: H.preferential("Hg"))
    print("\nresultados en outputs/cierre_pendientes.json")
