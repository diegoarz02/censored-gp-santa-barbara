"""Flip L=LD para Cd -- F6: reencuadre de la sensibilidad con L=LD como referencia principal.

Reutiliza: outputs/flip_ld_cd/maps_50m.npz (L=LD, generado por flip_ld_cd_f2_f4.py) y los
checkpoints YA EXISTENTES de h_umbral_anclado_cd.py (oficial_r_libre, anchor_LQ) -- ningun
reajuste nuevo, solo aritmetica posterior sobre arrays de 7381 celdas ya calculados.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"


def main():
    ld = np.load(OUT / "flip_ld_cd" / "maps_50m.npz")
    rlibre = np.load(OUT / "umbral_anclado_cd" / "Cd_oficial_r_libre.npz")
    lq = np.load(OUT / "umbral_anclado_cd" / "Cd_anchor_LQ.npz")

    p_agri_ld = ld["p_agri"]
    width95_ld = ld["width95"]

    rows = []
    for name, d in [("r_libre", rlibre), ("LQ", lq)]:
        p_agri_other = d["p_agri"]
        width95_other = d["width95"]
        cls_ld = p_agri_ld > 0.5
        cls_other = p_agri_other > 0.5
        agree = float((cls_ld == cls_other).mean())
        width_ratio = float(width95_other.mean() / width95_ld.mean())
        rows.append({
            "comparacion": f"LD_vs_{name}",
            "pct_domain_agri_gt_0.5_LD": float(100 * cls_ld.mean()),
            "pct_domain_agri_gt_0.5_otro": float(100 * cls_other.mean()),
            "concordancia_espacial_pct": 100 * agree,
            "width95_medio_LD_mgkg": float(width95_ld.mean()),
            "width95_medio_otro_mgkg": float(width95_other.mean()),
            "razon_ancho_otro_sobre_LD": width_ratio,
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_LD_reencuadre_sensibilidad.csv", index=False, encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    print("\nescrito: figuras/TABLE_LD_reencuadre_sensibilidad.csv")


if __name__ == "__main__":
    main()
