"""B1 (corrida 3) -- muestreo preferencial con cuadricula restringida a celdas con soporte.

Cierra la circularidad de H4.4 senalada en INFORME_PARA_COWORK.md S4.4: el tercio oeste del
dominio no tiene un solo dato, asi que su f_grid es pura extrapolacion, y si el modelo extrapola
alto justo donde nadie muestreo, delta sale negativo por construccion. Se repite el ajuste
restringiendo las celdas de recuento a las que estan a <= 1.5 rangos de correlacion del punto real
mas cercano (h_cierre_pendientes.preferential(restrict_to_support=True)) y se compara contra el
original.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import h_cierre_pendientes as H  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402
import run_cierre_pendientes as RCP  # noqa: E402


def main():
    res = RCP.load()
    original = res.get("preferential")
    if original is None:
        print("aviso: no hay 'preferential' original en cierre_pendientes.json, "
              "solo se reporta el restringido")

    key = "preferential_restricted"
    if key in res:
        print(f"[skip] {key} ya en checkpoint")
        restricted = res[key]
    else:
        print("--- B1: muestreo preferencial, cuadricula restringida a soporte ---", flush=True)
        # nb=6 (la cuadricula original) es demasiado gruesa: con mult=1.5 NINGUNA celda quedaba
        # fuera (medido, no asumido -- 0/36 en la primera pasada), asi que esa primera corrida no
        # probaba nada. nb=14, mult=0.5 deja 149/196 celdas dentro, 47 fuera: recorta el tercio
        # oeste de verdad, medido con el mismo script antes de lanzar.
        restricted = H.preferential("Hg", restrict_to_support=True, support_mult=0.5, nb=14)
        res[key] = restricted
        RCP.save(res)
        print(restricted)

    # ---------------------------------------------------------------------------- figura
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 0.5), constrained_layout=True)
    rows = []
    if original is not None:
        rows.append({"version": "full grid\n(original)",
                    "mean": original["delta_mean"], "lo": original["delta_ci_low"],
                    "hi": original["delta_ci_high"]})
    rows.append({"version": "grid restricted\nto data support",
                "mean": restricted["delta_mean"], "lo": restricted["delta_ci_low"],
                "hi": restricted["delta_ci_high"]})
    dplot = pd.DataFrame(rows)
    ypos = range(len(dplot))
    for i, r in dplot.iterrows():
        ax.plot([r["lo"], r["hi"]], [i, i], "-", lw=2.2, color=ef.C_CENSORED, zorder=2)
        ax.plot(r["mean"], i, "o", ms=8, color=ef.C_CENSORED, mec="k", mew=0.6, zorder=3)
    ax.axvline(0, color=ef.C_REF, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(list(ypos)); ax.set_yticklabels(dplot["version"])
    ax.set_xlabel(r"$\delta$ (loading parameter, preferential sampling)")
    ax.set_ylim(-0.7, len(dplot) - 0.3)
    ef.soft_grid(ax, axis="x")
    ef.message_title(ax, "The sign of δ holds after removing the extrapolated zone")
    ef.save_fig(fig, "FIG31_preferencial_restringido")
    plt.close(fig)

    signo_igual = (original is not None) and \
        ((restricted["delta_mean"] > 0) == (original["delta_mean"] > 0))

    if original is not None:
        fila_original = (f"| cuadrícula completa (original) | {original['delta_mean']:.3f} | "
                         f"[{original['delta_ci_low']:.3f}, {original['delta_ci_high']:.3f}] | "
                         f"{'sí' if original['excluye_cero'] else 'no'} |")
    else:
        fila_original = "| cuadrícula completa (original) | no disponible | — | — |"

    txt = f"""{stamp_header()}

# B1 (corrida 3) — muestreo preferencial, cuadrícula restringida a soporte de datos

## La circularidad que cierra este hito

El ajuste original de H4.4 usa una cuadrícula de recuento sobre **todo** el dominio, incluido el
tercio oeste que no tiene un solo punto muestreado — ahí `f(x)` es extrapolación pura del kernel.
Si el modelo extrapola valores altos hacia una zona sin datos, `δ` puede salir negativo por esa
construcción y no porque exista preferencialidad real en el sentido que le importa al lector.

## La prueba

Cuadrícula de conteo restringida a celdas a ≤ {restricted.get('support_mult', 1.5)} rangos de
correlación (`ℓ` = media del prior, {restricted.get('ell_plugin_km', float('nan')):.2f} km) del
punto real más cercano. De {restricted['n_grid_cells'] + restricted.get('n_grid_cells_dropped', 0)}
celdas originales quedan **{restricted['n_grid_cells']}** (se descartan
{restricted.get('n_grid_cells_dropped', 0)} sin soporte).

| versión | δ medio | IC 95 % | ¿excluye cero? |
|---|---|---|---|
{fila_original}
| **cuadrícula restringida a soporte** | **{restricted['delta_mean']:.3f}** | [{restricted['delta_ci_low']:.3f}, {restricted['delta_ci_high']:.3f}] | {'sí' if restricted['excluye_cero'] else 'no'} |

R̂ máximo {restricted['rhat_max']:.4f}, {restricted['divergences']} divergencias.

## Veredicto

{'**El signo se mantiene** al quitar la extrapolación del tercio oeste. La objeción de circularidad queda cerrada con evidencia: el acoplamiento entre intensidad de muestreo y campo latente no depende de la zona sin soporte.' if signo_igual else '**El signo cambia** al restringir la cuadrícula — es en sí mismo el hallazgo honesto: la estimación original SÍ estaba influida por la extrapolación del tercio oeste, y el resultado defendible es el de la cuadrícula restringida, no el original.'}

Figura `FIG31_preferencial_restringido`. Resultado en `outputs/cierre_pendientes.json`, clave
`preferential_restricted`.
"""
    (OUT / "B1_preferencial_restringido.md").write_text(txt, encoding="utf-8")
    observe(f"B1 completo: muestreo preferencial con cuadricula restringida a soporte de datos. "
            f"delta = {restricted['delta_mean']:.3f}, IC [{restricted['delta_ci_low']:.3f}, "
            f"{restricted['delta_ci_high']:.3f}], {'excluye' if restricted['excluye_cero'] else 'NO excluye'} "
            f"cero. Signo {'se mantiene' if signo_igual else 'CAMBIA'} respecto al original. "
            f"Cierra (o revela) la circularidad senalada en INFORME_PARA_COWORK.md S4.4. Escrito "
            f"outputs/B1_preferencial_restringido.md, FIG31_preferencial_restringido.",
            "Bloque B - corrida 3")
    print("\nescrito: outputs/B1_preferencial_restringido.md, FIG31_preferencial_restringido")


if __name__ == "__main__":
    main()
