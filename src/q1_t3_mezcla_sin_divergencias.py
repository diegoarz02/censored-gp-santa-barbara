"""Q1-T3 (ronda Q1): intenta eliminar/reducir las 41 divergencias de la mezcla k=2 del Cd.

main.tex cita (tab:mix) el modelo de 2 componentes como el que domina en elpd-LOO, pero tiene 41
divergencias de 4000 (ver outputs/RESULTADOS_VERIFICADOS.md, V7 parte 2) -- no declaradas en el
texto. h_cierre_pendientes.mixture_censored ya documenta en su propio docstring que un intento
anterior de subir solo target_accept (sin tune) de 72->48 divergencias EMPEORO el R-hat (1.01->
1.53); la version actual (target_accept=0.99, tune=3000) ya es la segunda iteracion, llegando a
41. Aqui se prueba la tercera: target_accept=0.999 con tune mas alto todavia, reutilizando
exactamente la misma funcion de modelo (notacion ordenada en mu_comp ya esta en el codigo, no es
necesario reparametrizar mas alla de eso).

NO se borra la fila anterior (41 divergencias) -- se guarda con un run_id distinto y las dos
quedan documentadas, como pide el prompt.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import h_cierre_pendientes as H  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
CK_DIR = OUT / "q1_t3_mezcla_sin_div"
CK_DIR.mkdir(parents=True, exist_ok=True)

ATTEMPTS = [
    dict(tag="ta0999_tune3000", target_accept=0.999, tune=3000),
    dict(tag="ta0999_tune5000", target_accept=0.999, tune=5000),
]


def main():
    rows = []
    for att in ATTEMPTS:
        ck = CK_DIR / f"Cd_k2_{att['tag']}.json"
        if ck.exists():
            import json
            out = json.loads(ck.read_text(encoding="utf-8"))
        else:
            print(f"--- intento {att['tag']} ---", flush=True)
            t0 = time.time()
            out, p_aff = H.mixture_censored("Cd", k=2, target_accept=att["target_accept"],
                                            tune=att["tune"])
            out["seconds"] = time.time() - t0
            out["tag"] = att["tag"]
            import json
            ck.write_text(json.dumps(out, default=float, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"{att['tag']}: elpd={out['elpd_loo']:.2f} se={out['se_loo']:.2f} "
              f"R-hat={out['rhat_max']:.4f} div={out['divergences']} "
              f"sep={out.get('separacion_extremos', float('nan')):.3f} "
              f"acuerdo_oefa={out.get('acuerdo_con_etiqueta_oefa', float('nan')):.3f} "
              f"({out.get('seconds', 0):.0f}s)", flush=True)
        rows.append(out)

    # fila de referencia (la que cita main.tex hoy, 41 divergencias) -- NO se borra, se reutiliza
    old = pd.read_csv(FIG / "TABLE_B3_mixture_components.csv")
    old_k2 = old[(old.metal == "Cd") & (old.k == 2)].iloc[0].to_dict()
    old_k2["tag"] = "original_citado_en_main_tex"
    rows.insert(0, old_k2)

    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_Q1_T3_mezcla_k2_intentos.csv", index=False, encoding="utf-8")
    print("\n" + tab[[c for c in ["tag", "elpd_loo", "se_loo", "rhat_max", "divergences",
                                  "separacion_extremos", "acuerdo_con_etiqueta_oefa"]
                      if c in tab.columns]].round(4).to_string(index=False))
    print("\nescrito: figuras/TABLE_Q1_T3_mezcla_k2_intentos.csv")


if __name__ == "__main__":
    main()
