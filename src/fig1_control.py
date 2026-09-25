"""FIG1 — mapa de diseno muestral y mercurio, regenerada con el estilo F3.

Por que se rehace
------------------
`FIG1_sampling_design_and_mercury.pdf` es la figura mas vieja del proyecto: la genero
`notebooks/_src/nb01_dataset.py` el 5 de septiembre, **antes de que existiera `estilo_figuras.py`**.
Ese notebook trae su propio bloque `plt.rcParams` de fase 1 (`font.size` 7, ticks a 6 pt), y nadie
volvio a ejecutar esa celda despues de fijarse la regla H2.0 de minimo 8 pt. Medido: 37 bloques de
texto a 6.0 pt, 9 a 7.0 pt y algunos mas pequenos aun (detalle en `TABLE_S9_figure_text_issues.csv`
antes de este arreglo).

Por que un script aparte y no re-ejecutar el notebook
------------------------------------------------------
`nb01_dataset.py` construye el dataset desde los crudos de OEFA y **escribe en `data/`**, que esta
fuera de limites (regla explicita del cierre: nunca tocar `data/`). Volver a correr el notebook
entero para arreglar dos tamanos de fuente seria however arriesgar exactamente lo que la regla
prohibe. Este script en cambio **solo lee** `data/final/santa_barbara_soil.csv` y
`data/final/mine_workings.csv`, que ya estan exportados, y reconstruye la figura con el modulo de
estilo actual. No escribe nada en `data/`.

El contenido de los dos paneles es identico al original: (a) diseno muestral por estrato con las
desmonteras, (b) concentracion de mercurio en escala log10 con los puntos que superan el ECA
marcados. Solo cambia el sistema de estilo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import DATA, RUN_ID, observe  # noqa: E402

ECA_HG_AGRI = 6.6  # mg/kg, mismo valor que ECA_AGRI["Hg"] en mapas_f3.py


def scale_bar(ax, length_km=1.0, label=None, pos=(0.06, 0.06)):
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    dx, dy = np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]
    xa, ya = x0 + pos[0] * dx, y0 + pos[1] * dy
    ax.plot([xa, xa + length_km], [ya, ya], color="k", lw=1.4, solid_capstyle="butt", zorder=10)
    ax.text(xa + length_km / 2, ya + 0.015 * dy, label or f"{length_km:g} km",
            ha="center", va="bottom", fontsize=ef.FS_MIN, zorder=10)


def north_arrow(ax, pos=(0.93, 0.80), length=0.10):
    ax.annotate("N", xy=(pos[0], pos[1] + length), xytext=pos, xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="bottom", fontsize=ef.FS_MIN,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, mutation_scale=7))


def main():
    final = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    workings = pd.read_csv(DATA / "final" / "mine_workings.csv")

    loc1 = final.drop_duplicates("location_id")
    hg = final[final.analyte == "Hg"].set_index("location_id")

    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W15, ef.W15 * 1.05), constrained_layout=True)

    ax = axes[0]
    for st, mk, col, lab in [("potential_interest", "o", ef.C_CENSORED,
                              "Potential interest (simple)"),
                             ("background", "s", ef.C_KRIGING, "Background (composite)"),
                             ("unlabelled", "^", ef.C_REF, "Unlabelled")]:
        s = loc1[loc1.stratum == st]
        ax.scatter(s.easting / 1000, s.northing / 1000, marker=mk, s=11, facecolor=col,
                   edgecolor="k", linewidth=0.2, label=lab, zorder=3)
    w = workings[workings.feature_type == "waste_dump"]
    ax.scatter(w.easting / 1000, w.northing / 1000, marker="x", s=16, c="0.25", linewidth=0.8,
               label="Waste dump", zorder=4)
    ax.set_xlabel("Easting UTM 18S, EPSG:32718 (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=1,
              handletextpad=0.4, borderaxespad=0)
    scale_bar(ax, 1.0, pos=(0.06, 0.03))
    north_arrow(ax, pos=(0.88, 0.86), length=0.06)
    ax.text(0.97, 0.015, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

    ax = axes[1]
    v = hg.loc[loc1.location_id, "value"].to_numpy()
    over = v > ECA_HG_AGRI
    sc = ax.scatter(loc1.easting / 1000, loc1.northing / 1000, c=np.log10(v), cmap="viridis",
                    s=13, edgecolor="k", linewidth=0.2, zorder=3)
    ax.scatter(loc1.easting[over] / 1000, loc1.northing[over] / 1000, marker="o", s=38,
               facecolor="none", edgecolor="#D55E00", linewidth=0.5, zorder=4,
               label=f"Above ECA {ECA_HG_AGRI:g} mg kg$^{{-1}}$ (n = {int(over.sum())})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"log$_{10}$ Hg (mg kg$^{-1}$)")
    cb.ax.tick_params(labelsize=ef.FS_MIN)
    ax.set_xlabel("Easting UTM 18S, EPSG:32718 (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, borderaxespad=0)
    scale_bar(ax, 1.0, pos=(0.06, 0.03))
    north_arrow(ax, pos=(0.88, 0.86), length=0.06)
    ax.text(0.97, 0.015, "(b)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

    ef.save_fig(fig, "FIG1_sampling_design_and_mercury")
    plt.close(fig)

    observe("FIG1_sampling_design_and_mercury regenerada. Era la figura mas vieja del proyecto "
            "(5 de septiembre), de antes de estilo_figuras.py: su notebook trae su propio bloque "
            "rcParams de fase 1 con font.size 7 y ticks a 6 pt, y nunca se volvio a correr tras la "
            "regla de 8 pt minimo. 37 bloques a 6.0 pt y 9 a 7.0 pt. Arreglo: script nuevo "
            "src/fig1_control.py que lee solo data/final/ (ya exportado, sin tocar data/) y "
            "reconstruye la misma figura con el estilo actual, en vez de re-ejecutar el notebook "
            "entero, que escribe en data/ y esta fuera de limites. Tambien se subieron a 8 pt los "
            "fontsize de scale_bar/north_arrow en el propio nb01_dataset.py para que una futura "
            "corrida completa no reintroduzca el mismo problema.", "Bloque 2 - figuras")
    print("escrito: FIG1_sampling_design_and_mercury.pdf/.tif")


if __name__ == "__main__":
    main()
