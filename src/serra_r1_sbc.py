"""SERRA-R1 T5 -- disponibilidad de draws del campo/eta_i en las replicas de SBC y de
recuperacion de parametros. Solo lectura de los checkpoints existentes; no se ejecuta ninguna
corrida nueva (la tarea lo pide explicitamente si no estan disponibles)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config import OUT  # noqa: E402

OUT_DIR = OUT / "serra_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    c1 = json.load(open(OUT / "c1_sbc" / "rep_0000.json", encoding="utf-8"))
    c2 = json.load(open(OUT / "c2_recovery" / "rep_0000.json", encoding="utf-8"))
    n_c1 = len(list((OUT / "c1_sbc").glob("rep_*.json")))
    n_c2 = len(list((OUT / "c2_recovery").glob("rep_*.json")))
    has_nc_c1 = len(list((OUT / "c1_sbc").glob("*.nc")))
    has_nc_c2 = len(list((OUT / "c2_recovery").glob("*.nc")))

    print(f"[1] outputs/c1_sbc/: {n_c1} replicas JSON, {has_nc_c1} archivos .nc")
    print(f"    claves guardadas por replica: {list(c1.keys())}")
    print(f"    'ranks' es un dict de rangos ESCALARES por parametro "
         f"({list(c1['ranks'].keys())}), no draws del campo ni de eta_i por ubicacion.")
    print(f"\n[2] outputs/c2_recovery/: {n_c2} replicas JSON, {has_nc_c2} archivos .nc")
    print(f"    claves guardadas por replica: {list(c2.keys())}")
    print("    Ningun .nc, ningun idata, ningun array de draws del campo latente -- solo "
         "metricas resumen (rangos SBC; conteos de censura, LOD y tiempos en recovery).")

    disponible = False
    nota = (
        "NO DISPONIBLE. Ni c1_sbc.py (200 replicas de SBC) ni c2_recovery_reps.py (50 replicas "
        "de recuperacion) guardan el idata completo ni arrays de draws del campo f o de eta_i(s) "
        "por ubicacion -- solo guardan metricas resumen por replica (rangos escalares de "
        "SBC: eta, ell_0, ell_1, sigma_n, beta0; conteos/tiempos en recovery). Verificado leyendo "
        "las claves reales de rep_0000.json en ambos directorios, no asumido por el nombre del "
        "script.\n\n"
        "Costo estimado de una corrida que SI los guarde (sin ejecutarla): habria que modificar "
        "run_replica()/simulate_one() (c1_sbc.py) y la funcion equivalente de "
        "c2_recovery_reps.py para persistir el idata completo (o al menos los arrays f/mu) por "
        "replica, y volver a correr las 250 replicas (200+50) porque el campo verdadero y el "
        "posterior se generan juntos en la misma llamada -- no se puede anadir el guardado sin "
        "repetir el ajuste. Medido en este proyecto: ~74 s/replica (bitacora de "
        "`c2_recovery_reps.py`, misma magnitud que la primera pasada de c1_sbc.py). "
        "250 replicas x 74 s ~= 5.1 horas de computo. Espacio: cada idata completo de este "
        "modelo pesa ~10-24 MB en outputs/idata/ (medido en los .nc ya existentes de este "
        "proyecto) -- guardar 250 (aunque sea solo f/mu, mas liviano que el idata completo) "
        "costaria varios GB en disco. No se ejecuta nada de esto aqui."
    )
    print("\n[3]", nota)

    out = pd.DataFrame([{
        "fuente": "c1_sbc (SBC, 200 replicas)", "n_replicas": n_c1,
        "guarda_campo_o_eta_i": False, "claves_guardadas": str(list(c1.keys())),
    }, {
        "fuente": "c2_recovery_reps (recuperacion, 50 replicas)", "n_replicas": n_c2,
        "guarda_campo_o_eta_i": False, "claves_guardadas": str(list(c2.keys())),
    }])
    out.to_csv(OUT_DIR / "sbc_derivadas.csv", index=False, encoding="utf-8")
    (OUT_DIR / "sbc_derivadas_nota.md").write_text(
        "# T5 -- SBC y cantidades derivadas\n\n" + nota + "\n", encoding="utf-8")
    print("\nescrito: outputs/serra_r1/sbc_derivadas.csv, sbc_derivadas_nota.md")
    return disponible


if __name__ == "__main__":
    main()
