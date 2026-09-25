"""El confundido fondo-minería como problema de IDENTIFICACIÓN, no de estimación.

Idea nueva, de la semana 7 del sílabo (modelos gráficos, DAG, d-separación) y no contemplada en
ningún plan previo. Cobertura teórica: Murphy, *Probabilistic Machine Learning: An Introduction*
(2022), §3.6 "Graphical models", p. 99 — `RECURSOS/book1.pdf`.

**El planteamiento.** Hasta ahora el intervalo del factor minero se ha tratado como una estimación
que salió ancha. Pero si el estrato es (casi) una función determinista de la posición, y la posición
entra además por el campo espacial y por las covariables, entonces **el efecto del estrato no está
identificado** y ningún estimador lo va a recuperar. Eso se decide *antes* de ajustar, con el grafo.

Lo que este script hace, y lo que no:

* **Sí**: mide, con los datos, cuánta variación del estrato sobrevive tras quitar la posición, las
  covariables y el campo espacial. Es la cantidad de la que depende la identificación.
* **No**: no ajusta un modelo causal ni afirma efectos causales. El DAG aquí es una herramienta de
  identificabilidad, no una afirmación sobre mecanismos.

El criterio se fija **antes** de mirar el resultado: si el R² de predecir el estrato a partir de la
posición supera 0.80, la variación residual es demasiado pequeña para identificar el efecto y el
contraste se declara **no identificable con este diseño**.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DATA, FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

R2_UMBRAL = 0.80          # fijado antes de ver el resultado


def main():
    df = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    loc = loc[loc["stratum"].isin(["background", "potential_interest"])].copy()

    y = (loc["stratum"] == "background").astype(float).to_numpy()
    xy = np.c_[loc["easting"], loc["northing"]] / 1000.0
    xy = xy - xy.mean(0)
    loc["ldd"] = np.log(loc["dist_waste_dump_m"].clip(lower=1.0))
    loc["ldc"] = np.log(loc["dist_city_m"])
    cov = loc[["elev_oefa", "slope_deg", "ldd", "ldc"]].to_numpy(float)
    cov = (cov - cov.mean(0)) / cov.std(0)

    def r2(X, y):
        """R² de regresión lineal de y sobre X con intercepto."""
        A = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        return float(1 - resid.var() / y.var()), resid

    rows = []
    # (a) posicion sola: lineal y con base polinomica de grado 3, que es lo que un campo suave puede
    #     reproducir sin ser un GP completo
    r2_lin, res_lin = r2(xy, y)
    poly = np.c_[xy, xy ** 2, xy ** 3, xy[:, 0:1] * xy[:, 1:2]]
    r2_poly, res_poly = r2(poly, y)
    # (b) covariables solas
    r2_cov, _ = r2(cov, y)
    # (c) todo junto
    r2_all, res_all = r2(np.c_[poly, cov], y)

    for nombre, val in [("posición (lineal)", r2_lin), ("posición (polinomio grado 3)", r2_poly),
                        ("covariables", r2_cov), ("posición + covariables", r2_all)]:
        rows.append({"predictores_del_estrato": nombre, "R2": val,
                     "variacion_residual_pct": 100 * (1 - val)})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))

    identificable = r2_all < R2_UMBRAL
    # distancia media entre estratos: cuanto de "otro sitio" es el fondo
    d_bg = xy[y == 1]
    d_pi = xy[y == 0]
    sep = float(np.linalg.norm(d_bg.mean(0) - d_pi.mean(0)))
    disp = float(np.sqrt(d_bg.var(0).sum() + d_pi.var(0).sum()))
    print(f"\nseparación entre centroides de estrato: {sep:.3f} km")
    print(f"dispersión combinada dentro de estratos : {disp:.3f} km")
    print(f"razón separación/dispersión             : {sep / disp:.3f}")
    print(f"\ncriterio fijado de antemano: identificable si R2(posición+covariables) < {R2_UMBRAL}")
    print(f"R2 obtenido = {r2_all:.4f}  ->  "
          f"{'IDENTIFICABLE' if identificable else 'NO IDENTIFICABLE con este diseño'}")

    t.insert(0, "run_id", RUN_ID)
    t.round(5).to_csv(FIG / "TABLE30_dag_identificabilidad.csv", index=False, encoding="utf-8")

    txt = f"""{stamp_header()}

# El contraste fondo-minería es un problema de identificación

Idea tomada de la semana 7 del sílabo (modelos gráficos, DAG, independencia condicional,
d-separación). Cobertura teórica en Murphy (2022), §3.6, p. 99.

## El grafo

```
        posición  ──────────────►  campo espacial f(s)
           │  │                          │
           │  └──────►  covariables ─────┤
           │              │              │
           ▼              ▼              ▼
        estrato ────────────────────►  log concentración
```

El efecto del estrato sobre la concentración es identificable **solo si queda variación en el
estrato que no explique la posición**, porque la posición entra en la respuesta por otras dos vías
que el modelo ya condiciona (el campo espacial y las covariables).

## La medición

Criterio **fijado antes de mirar el resultado**: identificable si el R² de predecir el estrato a
partir de posición y covariables es menor que {R2_UMBRAL}.

{t.drop(columns=['run_id']).round(3).to_markdown(index=False)}

Separación entre centroides de estrato: **{sep:.2f} km**. Dispersión combinada dentro de estratos:
{disp:.2f} km. Razón: **{sep / disp:.2f}**.

## Veredicto

**R² = {r2_all:.3f}** con posición y covariables, luego queda un
**{100 * (1 - r2_all):.1f} %** de variación del estrato sin explicar por la geografía.

{'**El contraste ES identificable en principio**, y el intervalo ancho refleja falta de potencia con n = 114, no aliasamiento total. Eso cambia la lectura: con más puntos se resolvería.' if identificable else '**El contraste NO es identificable con este diseño.** El estrato es esencialmente una función de la posición, y el modelo ya condiciona en la posición por dos vías. Ningún estimador va a recuperar el efecto: el intervalo ancho no es falta de potencia, es falta de identificación.'}

## Por qué esto importa para el artículo

Hasta ahora el intervalo que incluye el 1 se reportaba como un resultado decepcionante. Con el DAG
delante {'se puede decir que es cuestión de tamaño muestral y cuantificar cuántos puntos harían falta' if identificable else '**pasa a ser un resultado estructural del diseño de muestreo de OEFA**, enunciable antes de ajustar nada, y que ningún método estadístico puede arreglar'}.

Esto encaja con `LIMITES_METODOLOGICOS.md` §1, donde ya se descartó Restricted Spatial Regression
(Khan & Calder 2020) y se argumentó que Spatial+ probablemente tampoco separaría. {'El DAG matiza esa conclusión.' if identificable else 'El DAG explica **por qué** ninguno de los dos funciona: no es que el método sea malo, es que la cantidad no está identificada.'}
"""
    (OUT / "H_DAG_identificabilidad.md").write_text(txt, encoding="utf-8")
    observe(f"DAG del confundido (idea del sílabo semana 7): R2 de predecir el estrato desde "
            f"posición y covariables = {r2_all:.3f}, criterio previo {R2_UMBRAL}. Veredicto: "
            f"{'identificable, el intervalo ancho es falta de potencia' if identificable else 'NO identificable con este diseño'}. "
            f"Separación de centroides {sep:.2f} km sobre dispersión {disp:.2f} km.",
            "Bloque 4 - artículo")
    print("\nescrito: outputs/H_DAG_identificabilidad.md, TABLE30")


if __name__ == "__main__":
    main()
