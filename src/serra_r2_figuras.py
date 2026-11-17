"""SERRA-R2 T3 -- regenera Fig9 (SERRA, mapa de Hg) y FIG17_Hg (version en espanol) con los productos de
mapa de Hg del ajuste reforzado, sin reajustar ni re-predecir nada. Misma ruta de dibujo que antes
(figserra._mean_and_uncertainty_serra y mapas_f3.mean_and_uncertainty): solo cambian las claves Hg_*
de D["mp"]. Las figuras se sobrescriben (sin copias v1); la comparacion antes/despues se hace contra una
copia temporal y solo si hg_figuras_comparacion.csv aun no existe.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figserra as F  # noqa: E402
import mapas_f3 as M  # noqa: E402
from config import FIG, OUT  # noqa: E402

KEYS = ["mean", "median", "sd_f", "sd_y", "p_agri", "p_resid"]
SERRA = FIG / "serra"


def render(pdf, dpi=110):
    d = fitz.open(pdf)
    pm = d[0].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3]
    d.close()
    return arr


def main():
    new = dict(np.load(OUT / "serra_r2" / "posterior_maps_Hg_refuerzo.npz"))
    D = M.load()
    D_new = dict(D)
    mp = dict(D["mp"])
    for k in KEYS:
        mp[f"Hg_{k}"] = new[f"Hg_{k}"]
    D_new["mp"] = mp

    f_cmp = OUT / "serra_r2" / "hg_figuras_comparacion.csv"
    tmp = Path(tempfile.mkdtemp())
    pairs = [("Fig9 (SERRA)", SERRA / "Fig9.pdf", tmp / "Fig9_antes.pdf"),
             ("FIG17_Hg (espanol)", FIG / "FIG17_Hg_mean_and_uncertainty.pdf", tmp / "FIG17_Hg_antes.pdf")]
    for _, cur, copy in pairs:
        shutil.copy2(cur, copy)
    print("=== Fig9 (SERRA) ===")
    print(F._mean_and_uncertainty_serra(D_new, "Hg", "Fig9"))
    print("=== FIG17_Hg (espanol) ===")
    print(M.mean_and_uncertainty(D_new, "Hg"))

    if not f_cmp.exists():
        rows = []
        for lab, cur, copy in pairs:
            a, b = render(copy), render(cur)
            d = np.abs(a.astype(int) - b.astype(int)).max(-1)
            rows.append({"figura": lab, "pixeles": int(d.size),
                         "pct_pixeles_que_difieren_>8_niveles": float(100 * (d > 8).mean()),
                         "pct_pixeles_que_difieren_>32_niveles": float(100 * (d > 32).mean()),
                         "dif_maxima_nivel_0_255": int(d.max())})
        pd.DataFrame(rows).to_csv(f_cmp, index=False, encoding="utf-8")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
