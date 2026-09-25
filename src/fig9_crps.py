"""FIG9 — CRPS contra el nivel de censura, regenerada desde los 240 ajustes profundos.

Por que se rehace
-----------------
La version que habia en `figuras/` venia del notebook y tenia **dos defectos medidos**, no opinados:

1. **La leyenda tapaba por completo el bloque de texto** con `sd_log`, `ell` y `rho`. La auditoria
   marcaba 12 colisiones por figura, 36 en total, y al renderizarla se ve que el bloque es
   ilegible. Era el unico sitio del entregable donde quedaba una colision real.
2. **Tipografia a 5.6 pt**, por debajo del minimo de 8 pt de la regla H2.0.

Ademas mostraba los tres modelos con los ajustes **poco profundos** de la fase 2, que tenian 488
divergencias en el baseline. Aqui se usan los **240 ajustes profundos** de F3, que son los que
sostienen la conclusion. El krigeado ordinario con L/2 no se reajusto a esa profundidad y **sale
peor que los dos GP en las doce celdas** (`TABLE15_masking_experiment.csv`), asi que queda en la
tabla y no en la figura: dibujar juntos ajustes de profundidad distinta seria enganoso.

Decision de diseno
------------------
El bloque de texto que se solapaba **no vuelve**. Repetia `ell` y `rho`, que ya estan en
`TABLE_H6_*`, y su sitio natural es el pie. Una figura tiene que ganarse cada objeto que se dibuja
encima de sus datos.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import FIG, OUT, RUN_ID, observe  # noqa: E402

METALS = ["As", "Hg", "Pb"]
MODELS = ["censored GP", "GP with L/2"]


def deep_table() -> pd.DataFrame:
    rows = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(OUT / "h6_deep" / "*.json"))]
    d = pd.DataFrame([r for r in rows if "error" not in r])
    return d


def main():
    d = deep_table()
    if d.empty:
        print("sin checkpoints profundos"); return

    ef.apply_style()
    made = []
    for metal in METALS:
        fig, ax = plt.subplots(figsize=(ef.W1, ef.W1 * 0.80), constrained_layout=True)
        for mdl in MODELS:
            s = d[(d.metal == metal) & (d.model == mdl)]
            if s.empty:
                continue
            g = s.groupby("level")["crps"]
            m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
            c = ef.MODEL_COLOR[mdl]
            ax.fill_between(m.index, lo.values, hi.values, color=c, alpha=0.15, lw=0)
            ax.plot(m.index, m.values, "-o", ms=3.4, lw=1.3, color=c, mec="k", mew=0.25,
                    label=ef.MODEL_SHORT[mdl])
        ax.set_xlabel("imposed censoring level")
        ax.set_xticks(sorted(d.level.unique()))
        # Etiqueta corta a proposito: la version larga desbordaba la pagina y el exponente se
        # perdia por arriba. "contra los valores verdaderos" es informacion de pie, no de eje.
        ax.set_ylabel(r"CRPS (log mg kg$^{-1}$)")
        ef.soft_grid(ax)
        # La leyenda va FUERA del area de datos. Era lo que tapaba el bloque de texto.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2,
                  fontsize=ef.FS_MIN)
        ax.text(0.99, 0.02, metal, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=ef.FS_MIN + 1, color=ef.C_REF, weight="bold")
        ef.save_fig(fig, f"FIG9_{metal}_crps")
        plt.close(fig)
        made.append(f"FIG9_{metal}_crps")

    # el dato que justifica dejar el krigeado fuera de la figura, recalculado y no citado de memoria
    t15 = pd.read_csv(FIG / "TABLE15_masking_experiment.csv")
    p = t15.pivot_table(index=["metal", "level"], columns="model", values="crps")
    peor = int((p["ordinary kriging L/2"] > p[["GP with L/2", "censored GP"]].max(axis=1)).sum())
    print(f"krigeado ordinario peor que los dos GP en {peor} de {len(p)} celdas")
    print("regeneradas:", ", ".join(made))

    observe(f"FIG9 regenerada desde los 240 ajustes profundos. Motivo medido: la version del "
            f"notebook tenia 36 colisiones (la leyenda tapaba entera la caja de sd_log/ell/rho, "
            f"ilegible al renderizar) y tipografia a 5.6 pt contra el minimo de 8. Se quita la caja "
            f"de texto en vez de moverla: repetia numeros que ya estan en TABLE_H6_*. El krigeado "
            f"ordinario sale de la figura porque no se reajusto a la profundidad de F3 y queda peor "
            f"que los dos GP en {peor} de {len(p)} celdas de TABLE15; mezclar profundidades en un "
            f"mismo eje seria enganoso.", "Bloque 2 - figuras")


if __name__ == "__main__":
    main()
