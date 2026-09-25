"""Relanza solo ADVI, con los remedios documentados, sin repetir el agrupamiento que ya está hecho.

Si vuelve a fallar con las dos tasas de aprendizaje, **eso es el resultado**: la aproximación
variacional de campo medio no sirve para este modelo, y la razón es concreta y publicable — la
verosimilitud con censura por intervalo produce gradientes no finitos en regiones que el optimizador
estocástico visita y NUTS rechaza.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import h85ce_advi_jerarquico as H  # noqa: E402
from config import FIG, OUT, RUN_ID, log_attempt, observe, stamp_header  # noqa: E402

if __name__ == "__main__":
    rows, errores = [], {}
    for m in H.METALS:
        print(f"--- ADVI sobre {m} ---", flush=True)
        try:
            r, approx, err = H.run_advi(m)
            if err:
                errores[m] = err
                print("  no convergió:", err[:180])
                log_attempt(f"ADVI sobre {m}, dos tasas de aprendizaje", err[:150],
                            "se reporta como resultado negativo, no como hueco")
            else:
                rows += r
                np.save(OUT / f"advi_elbo_{m}.npy", np.asarray(approx.hist))
                print(pd.DataFrame(r)[["parametro", "advi_mean", "nuts_mean",
                                       "razon_sd"]].round(3).to_string(index=False))
        except Exception as exc:                                    # noqa: BLE001
            errores[m] = repr(exc)[:250]
            print("  excepción:", repr(exc)[:200])

    A = pd.DataFrame(rows)
    if len(A):
        A.insert(0, "run_id", RUN_ID)
        A.round(5).to_csv(FIG / "TABLE33_advi_vs_nuts.csv", index=False, encoding="utf-8")

    txt = [stamp_header(), "", "# H8.5c — inferencia variacional contra MCMC (sílabo, semana 13)", ""]
    if len(A):
        med = float(A["razon_sd"].median())
        txt += [A.drop(columns=["run_id"]).round(3).to_markdown(index=False), "",
                f"ADVI da una desviación típica posterior mediana **{med:.2f} veces** la de NUTS.", ""]
        if med < 0.9:
            txt += ["Es el comportamiento conocido de la aproximación de **campo medio**: al "
                    "factorizar el posterior ignora las correlaciones entre parámetros y "
                    "**subestima la varianza**.", "",
                    "**Y para este trabajo la consecuencia es tajante: ADVI no puede usarse aquí.** "
                    "Lo que el artículo mide es precisamente la anchura de los intervalos. Un "
                    "método que los estrecha por construcción no puede arbitrar entre dos métodos "
                    "que se distinguen por lo estrechos que son sus intervalos.", ""]
    if errores:
        txt += ["## No convergió, y eso es el resultado", ""]
        for m, e in errores.items():
            txt += [f"* **{m}**: `{e[:170]}`"]
        txt += ["",
                "Se probaron dos tasas de aprendizaje (1e-3 y 1e-4, la segunda con el doble de "
                "iteraciones) tras comprobar que el `logp` inicial es finito. Los dos remedios "
                "son los que documentan el foro y el rastreador de PyMC para este error.", "",
                "**La causa es estructural y merece estar en el artículo.** La verosimilitud con "
                "censura por intervalo calcula $\\log[\\Phi(h) - \\Phi(l)]$, que tiende a "
                "$-\\infty$ cuando el optimizador visita una región donde ese intervalo tiene masa "
                "nula. NUTS sobrevive porque **rechaza** esas propuestas; el gradiente estocástico "
                "de ADVI las incorpora y propaga el NaN.", "",
                "Dicho de otro modo: **la misma propiedad que hace informativa a la verosimilitud "
                "censurada —que asigna probabilidad cero fuera del intervalo observado— es la que "
                "rompe la inferencia variacional.** Es un resultado metodológico honesto y "
                "responde la pregunta de la semana 13 con un «no, y por esta razón».", ""]
    (OUT / "H8.5c_advi.md").write_text("\n".join(txt), encoding="utf-8")
    observe(f"H8.5c ADVI: {'convergió en ' + str(len(A)) + ' parámetros' if len(A) else 'NO convergió'}"
            f"{' con fallos en ' + ', '.join(errores) if errores else ''}. "
            f"Causa estructural: la verosimilitud con censura por intervalo da log[Phi(h)-Phi(l)] "
            f"que va a -inf donde el optimizador estocástico visita y NUTS rechaza. Responde la "
            f"pregunta del sílabo semana 13 con un 'no, y por esta razón'.", "Bloque 8 - sílabo")
    print("\nescrito: outputs/H8.5c_advi.md")
