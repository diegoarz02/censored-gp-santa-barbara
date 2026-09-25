"""Figura de la mezcla: las dos poblaciones, y dónde discrepan con la etiqueta oficial.

La mezcla (H8.5b) ajusta dos componentes sobre la concentración en escala log **sin usar la etiqueta
de estrato de OEFA**, con la proporción variando en el espacio a través de un campo gaussiano. Eso
permite dos cosas que la etiqueta no permite:

1. comprobar que las dos poblaciones existen en los datos, y con cuánta separación;
2. señalar **qué puntos concretos** clasifica distinto la concentración y la etiqueta oficial.

El segundo punto es el que vale para el artículo: la etiqueta de OEFA no es una descripción neutral
de la contaminación, es una decisión de diseño de muestreo, y aquí se ve dónde las dos cosas no
coinciden.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import mapas_f3 as M  # noqa: E402
from config import DATA, FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402


def main():
    z = np.load(OUT / "mixture_assignment.npz")
    p_aff, lab = z["p_affected"], z["oefa_label"].astype(bool)
    res = json.loads((OUT / "cierre_pendientes.json").read_text(encoding="utf-8"))["mixture"]

    df = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    hg = df[df.analyte == "Hg"].set_index("location_id").loc[loc.index]["value"].to_numpy(float)
    y = np.log(hg)

    asign = p_aff > 0.5
    acuerdo = asign == lab
    D = M.load()

    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.46), constrained_layout=True)

    # --- (a) the two populations over observed concentration
    ax = axes[0]
    bins = np.linspace(y.min() - 0.2, y.max() + 0.2, 26)
    ax.hist(y[~asign], bins=bins, color=ef.C_KRIGING, alpha=0.75, edgecolor="k",
            linewidth=0.3, label=f"background component (n={int((~asign).sum())})")
    ax.hist(y[asign], bins=bins, color=ef.C_SUB_GP, alpha=0.75, edgecolor="k",
            linewidth=0.3, label=f"affected component (n={int(asign.sum())})")
    ax.axvline(np.log(6.6), color="#d62728", lw=1.3, ls="--")
    ax.text(np.log(6.6), ax.get_ylim()[1] * 0.97, " ECA 6.6", color="#d62728", fontsize=8,
            va="top", ha="left")
    ax.set_xlabel(r"log Hg (mg kg$^{-1}$)")
    ax.set_ylabel("points")
    ef.soft_grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    ef.panel_label(ax, "a")

    # --- (b) where the mixture and the official label disagree
    ax = axes[1]
    e0, e1, n0, n1 = D["extent"]
    if D["hill"] is not None:
        ax.imshow(D["hill"], cmap="gray", extent=(e0, e1, n0, n1), origin="lower",
                  alpha=0.20, zorder=1, interpolation="bilinear")
    E, N = loc["easting"].to_numpy(), loc["northing"].to_numpy()
    ax.scatter(E[acuerdo], N[acuerdo], s=14, facecolor="#b9b5af", edgecolor="k",
               linewidth=0.25, zorder=4, label=f"agree ({int(acuerdo.sum())})")
    dis_hi = (~acuerdo) & asign
    dis_lo = (~acuerdo) & (~asign)
    ax.scatter(E[dis_hi], N[dis_hi], s=30, marker="^", facecolor=ef.C_SUB_GP, edgecolor="k",
               linewidth=0.35, zorder=5,
               label=f"mixture says affected, OEFA background ({int(dis_hi.sum())})")
    ax.scatter(E[dis_lo], N[dis_lo], s=30, marker="v", facecolor=ef.C_KRIGING, edgecolor="k",
               linewidth=0.35, zorder=5,
               label=f"mixture says background, OEFA of interest ({int(dis_lo.sum())})")
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting UTM 18S (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ef.utm_km_ticks(ax)
    ax.tick_params(labelsize=ef.FS_MIN)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=ef.FS_MIN)
    ef.panel_label(ax, "b")

    ef.message_title(fig, "The two populations exist in the data, but they are not what the "
                          "official label says")
    ef.save_fig(fig, "FIG25_mezcla_poblaciones")
    plt.close(fig)

    t = pd.DataFrame([{
        "run_id": RUN_ID, "separacion_log": res["separacion_medias"],
        "sep_ci_low": res["sep_ci_low"], "sep_ci_high": res["sep_ci_high"],
        "factor_concentracion": float(np.exp(res["separacion_medias"])),
        "acuerdo_pct": 100 * res["acuerdo_con_etiqueta_oefa"],
        "n_coinciden": int(acuerdo.sum()),
        "n_mezcla_afectado_oefa_fondo": int(dis_hi.sum()),
        "n_mezcla_fondo_oefa_interes": int(dis_lo.sum()),
        "rhat_max": res["rhat_max"], "divergences": res["divergences"]}])
    t.round(4).to_csv(FIG / "TABLE35_mezcla.csv", index=False, encoding="utf-8")
    print(t.round(3).to_string(index=False))

    txt = f"""{stamp_header()}

# H8.5b — mezcla bayesiana de dos poblaciones (sílabo, semana 10)

## Por qué se hizo así

El contraste fondo-minería no se resuelve con la etiqueta de OEFA porque esa etiqueta está aliasada
con la posición: el R² de predecirla desde las coordenadas y las covariables es **0.840**
(`H_DAG_identificabilidad.md`). Más modelo sobre la etiqueta no arregla eso.

La salida es **dejar de usar la etiqueta**. Se ajusta una mezcla de dos componentes sobre la
concentración en escala logarítmica, con la proporción variando en el espacio a través de un campo
gaussiano, y **sin que la etiqueta entre en el modelo**. Después se compara la asignación posterior
contra la etiqueta oficial.

## Resultado

| | |
|---|---|
| Separación entre las medias | **{res['separacion_medias']:.2f}** unidades log, IC 95 % [{res['sep_ci_low']:.2f}, {res['sep_ci_high']:.2f}] |
| En concentración | factor **{np.exp(res['separacion_medias']):.0f}×** |
| ¿Separan los componentes? | **sí**, el intervalo no toca el cero |
| Acuerdo con la etiqueta de OEFA | **{100 * res['acuerdo_con_etiqueta_oefa']:.1f} %** |
| R-hat máximo · divergencias | {res['rhat_max']:.4f} · **{int(res['divergences'])}** |

**Las dos poblaciones existen en los datos.** Separadas por un factor de
{np.exp(res['separacion_medias']):.0f} en concentración, con un intervalo que no se acerca al cero.
Eso no lo dice la etiqueta: lo dicen las concentraciones.

**Y coinciden con la etiqueta solo en {100 * res['acuerdo_con_etiqueta_oefa']:.0f} % de los puntos.**
En **{int(dis_hi.sum())}** localizaciones la mezcla dice «afectado» donde OEFA puso «fondo», y en
**{int(dis_lo.sum())}** al revés.

## Qué se puede decir con esto, y qué no

**Sí se puede decir** que la estructura de dos poblaciones está en los datos y no es un artefacto de
cómo se etiquetó el muestreo. Y que la etiqueta de OEFA **no es una descripción neutral de la
contaminación**: es una decisión de diseño, tomada antes de medir, que en tres de cada diez puntos
no coincide con lo que las concentraciones indican.

**No se puede decir** que la mezcla sustituya al informe 00341-2018-OEFA que falta. Ese informe
contiene profundidad de muestreo, identidad del laboratorio y las conclusiones de causalidad del
propio organismo. La mezcla no recupera nada de eso: reemplaza una **etiqueta** dudosa, no un
**documento** ausente.

**Marcado como exploratorio** en el artículo, por dos razones honestas: la asignación depende del
número de componentes fijado en dos, y el modelo de mezcla no incorpora la censura (usa LD/2 para
los no detectados, que en mercurio es irrelevante porque no hay censura, pero limita extenderlo a
cadmio sin trabajo adicional).

Figura `FIG25_mezcla_poblaciones`, tabla `TABLE35_mezcla.csv`.
"""
    (OUT / "H8.5b_mezcla.md").write_text(txt, encoding="utf-8")
    observe(f"H8.5b mezcla: los dos componentes SEPARAN ({res['separacion_medias']:.2f} unidades "
            f"log, IC excluye cero, factor {np.exp(res['separacion_medias']):.0f}x en "
            f"concentración) sin usar la etiqueta de OEFA. Acuerdo con la etiqueta "
            f"{100 * res['acuerdo_con_etiqueta_oefa']:.1f} %: discrepa en {int(dis_hi.sum())} + "
            f"{int(dis_lo.sum())} puntos. Ataca la amenaza 1 por la vía correcta. NO sustituye al "
            f"informe de OEFA que falta: reemplaza una etiqueta, no un documento.",
            "Bloque 8 - sílabo")
    print("\nescrito: outputs/H8.5b_mezcla.md, FIG25, TABLE35")


if __name__ == "__main__":
    main()
