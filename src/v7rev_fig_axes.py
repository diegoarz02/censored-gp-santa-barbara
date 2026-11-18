"""V7REV feedback de revision -- V1/V2: corrige las etiquetas del eje x
de FIG18_Cd_exceedance y FIG18_Cd_exceedance_residential (duplicadas por redondeo: "502 503 504
504 504 505") y regenera las otras 3 figuras que comparten la misma ruta de codigo
(mapas_f3.py::base_axes) para confirmar que el fix (estilo_figuras.py::utm_km_ticks,
explicit_ticks=True) no les cambia nada.

NO reajusta el modelo ni re-predice nada: reutiliza outputs/flip_ld_cd/maps_50m.npz (median, sd_y,
p_agri, p_resid ya calculados por src/flip_ld_cd_f2_f4.py) para el Cd bajo L=LD, y M.load() para
Hg (sin cambios, nunca dependio del flip). Reconstruye D_ld exactamente como
flip_ld_cd_f2_f4.py::main() lo hacia.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import mapas_f3 as M  # noqa: E402
from config import OUT  # noqa: E402

CK_DIR = OUT / "flip_ld_cd"


def main():
    r50 = dict(np.load(CK_DIR / "maps_50m.npz"))

    D = M.load()
    D_ld = dict(D)
    mp_ld = dict(D["mp"])
    mp_ld["Cd_median"] = r50["median"]
    mp_ld["Cd_mean"] = r50["median"]
    mp_ld["Cd_sd_y"] = r50["sd_y"]
    mp_ld["Cd_sd_f"] = r50["sd_y"]
    mp_ld["Cd_p_agri"] = r50["p_agri"]
    mp_ld["Cd_p_resid"] = r50["p_resid"]
    D_ld["mp"] = mp_ld

    print("=== regenerando FIG17_Cd, FIG18_Cd (agri/resid) bajo L=LD -- solo redibujo ===")
    info17cd = M.mean_and_uncertainty(D_ld, "Cd")
    info18a = M.exceedance(D_ld, "Cd", "agricultural")
    info18r = M.exceedance(D_ld, "Cd", "residential")
    print("FIG17_Cd_mean_and_uncertainty:", info17cd)
    print("FIG18_Cd_exceedance:", info18a)
    print("FIG18_Cd_exceedance_residential:", info18r)

    print("\n=== regenerando FIG17_Hg (sin cambios desde el flip) -- confirma que el fix no mueve nada ===")
    info17hg = M.mean_and_uncertainty(D, "Hg")
    print("FIG17_Hg_mean_and_uncertainty:", info17hg)

    print("\nescrito: figuras/FIG17_Cd_mean_and_uncertainty.pdf, FIG18_Cd_exceedance.pdf, "
          "FIG18_Cd_exceedance_residential.pdf, FIG17_Hg_mean_and_uncertainty.pdf")


if __name__ == "__main__":
    main()
