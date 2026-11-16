"""SERRA-R1 T4 -- posible fuga de informacion en el umbral del enmascaramiento: el umbral se
calcula con el cuantil empirico de las 114 concentraciones (entrenamiento + prueba juntos) o solo
con el entrenamiento? Puro calculo, SIN reajustar ningun modelo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402
from config import OUT  # noqa: E402

OUT_DIR = OUT / "serra_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METALS = ["As", "Hg", "Pb"]
LEVELS = [0.2, 0.4, 0.6, 0.8]
N_BLOCKS = 5


def main():
    ctx = mw._context()
    df, loc, xy = ctx["df"], ctx["loc"], ctx["xy"]
    sb = ctx["sb"]
    labels = sb.spatial_blocks(xy, k=N_BLOCKS, seed=20260906)

    # ------------------------------------------------------------------------------------- paso 1
    print("[1] Codigo citado -- worker_v4.py::run_deep(), L.135-139:")
    print('    limit = float(np.quantile(value, level))   # value = las 114 concentraciones')
    print('    censored = value < limit                    # sin restringir a train')
    print('    labels = sb.spatial_blocks(xy, k=N_BLOCKS, seed=20260906)')
    print('    test = labels == block; train = ~test')
    print("    Confirmado leyendo el codigo: el umbral se calcula ANTES del split train/test, "
         "sobre las 114 concentraciones completas -- se aplica igual a ambos por construccion, "
         "no hay dos calculos distintos.")

    # ------------------------------------------------------------------------------------- paso 2
    rows = []
    for metal in METALS:
        g = df[df.analyte == metal].set_index("location_id").loc[loc.index]
        value = g["value"].to_numpy(float)
        for level in LEVELS:
            limit_global = float(np.quantile(value, level))
            for block in range(N_BLOCKS):
                test = labels == block
                train = ~test
                limit_train_only = float(np.quantile(value[train], level))
                diff_abs = limit_train_only - limit_global
                diff_rel = diff_abs / limit_global if limit_global != 0 else np.nan

                cens_global_test = value[test] < limit_global
                cens_trainonly_test = value[test] < limit_train_only
                n_flip = int((cens_global_test != cens_trainonly_test).sum())

                rows.append({"metal": metal, "level": level, "block": block,
                            "n_train": int(train.sum()), "n_test": int(test.sum()),
                            "limit_global": limit_global, "limit_train_only": limit_train_only,
                            "diff_abs": diff_abs, "diff_rel_pct": 100 * diff_rel,
                            "n_test_cambia_estado": n_flip,
                            "pct_test_cambia_estado": 100 * n_flip / int(test.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "umbral_fuga.csv", index=False, encoding="utf-8")

    print(f"\n[2] {len(out)} combinaciones metal x nivel x bloque (esperado "
         f"{len(METALS)}x{len(LEVELS)}x{N_BLOCKS}={len(METALS)*len(LEVELS)*N_BLOCKS}).")
    print(f"    diferencia relativa de umbral: min={out.diff_rel_pct.min():.2f}% "
         f"max={out.diff_rel_pct.max():.2f}% media={out.diff_rel_pct.mean():.2f}% "
         f"|media|={out.diff_rel_pct.abs().mean():.2f}%")
    print(f"    puntos de test que cambiarian de estado: "
         f"total={out.n_test_cambia_estado.sum()} de {out.n_test.sum()} "
         f"(media {out.pct_test_cambia_estado.mean():.2f}% por celda, "
         f"max {out.pct_test_cambia_estado.max():.2f}%)")
    print("\n    Por nivel:")
    print(out.groupby("level")[["diff_rel_pct", "pct_test_cambia_estado"]].agg(
        ["mean", "max"]).round(3).to_string())

    print("\nescrito: outputs/serra_r1/umbral_fuga.csv")
    return out


if __name__ == "__main__":
    main()
