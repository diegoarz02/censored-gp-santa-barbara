"""C2 (corrida 3) — diagrama de placas del modelo generativo.

Requisito 2 de la Guía del docente: "modelo generativo formal + diagrama de placas". El DAG de
`h_red_bayesiana.py` ya cubre el mecanismo causal (semana 7 del sílabo, d-separación); este es el
complemento de notación estándar de modelos gráficos (Bishop 2006 §8.1, Murphy PML1 §3.6): un
rectángulo ("placa") marca qué nodos se repiten por localización, y qué queda fuera porque es
compartido por las 114.

Mismos nodos y mismas aristas que el DAG (el modelo no cambia), presentados con la notación de placas
que pide la Guía: círculos pequeños, símbolo matemático puro (no la palabra descriptiva), observados
sombreados, deterministas con doble borde, la placa rotulada con su cardinalidad `N = 114`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import FIG, RUN_ID, observe, stamp_header  # noqa: E402

# ------------------------------------------------------------------------------- nodos y aristas
# Fuera de la placa: compartidos por las 114 localizaciones (un solo valor por ajuste del modelo).
GLOBAL = {
    "theta": (1.0, 5.4), "beta": (2.0, 5.4), "b_bg": (3.0, 5.4),
    "sigma_n": (4.0, 5.4), "r": (5.0, 5.4), "LOD": (6.0, 5.4),
}
GLOBAL_LABEL = {"theta": r"$\eta,\ell,\rho$", "beta": r"$\beta$", "b_bg": r"$b_{bg}$",
               "sigma_n": r"$\sigma_n$", "r": r"$r$", "LOD": r"LOD"}
GLOBAL_OBSERVED = {"LOD"}          # el límite reportado por el laboratorio; todo lo demás es latente

# Dentro de la placa "s = 1 .. N": un valor por localización.
PLATE = {
    "x": (1.2, 3.6), "g": (1.2, 2.6), "f": (2.4, 3.6), "mu": (3.6, 3.1),
    "L": (5.6, 4.1), "y": (4.8, 3.1), "d": (5.8, 2.5), "yobs": (5.8, 1.5),
}
PLATE_LABEL = {"x": r"$x(s)$", "g": r"$g(s)$", "f": r"$f(s)$", "mu": r"$\mu(s)$",
              "L": r"$L$", "y": r"$y(s)$", "d": r"$D(s)$", "yobs": r"$y^{obs}(s)$"}
PLATE_OBSERVED = {"x", "g", "d", "yobs"}
PLATE_DETERMINISTIC = {"mu", "L"}   # funcion determinista de sus padres, no un parametro libre

EDGES = [
    ("theta", "f"), ("x", "mu"), ("g", "mu"), ("beta", "mu"), ("b_bg", "mu"), ("f", "mu"),
    ("mu", "y"), ("sigma_n", "y"),
    ("y", "d"), ("L", "d"),
    ("y", "yobs"), ("d", "yobs"),
    ("LOD", "L"), ("r", "L"),
]


def draw_node(ax, xy, label, *, observed, deterministic=False, r=0.28):
    x, y = xy
    fc = "#d9d5d0" if observed else "white"
    ec = "#52514e" if observed else ef.C_CENSORED
    lw = 0.9 if observed else 1.1
    circ = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=5)
    ax.add_patch(circ)
    if deterministic:
        # doble borde: nodo determinista, no un parametro con su propio prior
        ax.add_patch(plt.Circle((x, y), r * 0.78, facecolor="none", edgecolor=ec,
                                linewidth=0.6, zorder=6))
    ax.text(x, y, label, ha="center", va="center", fontsize=ef.FS_MIN, zorder=7)


def draw_edge(ax, pos, a, b, r=0.28):
    x0, y0 = pos[a]; x1, y1 = pos[b]
    dx, dy = x1 - x0, y1 - y0
    d = (dx ** 2 + dy ** 2) ** 0.5
    if d < 1e-6:
        return
    ux, uy = dx / d, dy / d
    ax.annotate("", xy=(x1 - ux * r, y1 - uy * r), xytext=(x0 + ux * r, y0 + uy * r),
               arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#8a8681",
                               shrinkA=0, shrinkB=0), zorder=3)


def main():
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W2, ef.W2 * 0.60), constrained_layout=True)
    pos = {**GLOBAL, **PLATE}

    for a, b in EDGES:
        draw_edge(ax, pos, a, b)

    for k, xy in GLOBAL.items():
        draw_node(ax, xy, GLOBAL_LABEL[k], observed=(k in GLOBAL_OBSERVED))
    for k, xy in PLATE.items():
        draw_node(ax, xy, PLATE_LABEL[k], observed=(k in PLATE_OBSERVED),
                 deterministic=(k in PLATE_DETERMINISTIC))

    # la placa: un rectangulo alrededor de los nodos indexados por s, con N=114 en la esquina
    xs = [x for x, y in PLATE.values()]; ys = [y for x, y in PLATE.values()]
    pad = 0.55
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor="none", edgecolor="#3a3835", linewidth=1.1,
                                         zorder=1))
    ax.text(x0 + 0.12, y0 + 0.12, "$s = 1 \\ldots N$\n$N = 114$", ha="left", va="bottom",
           fontsize=ef.FS_MIN, color="#3a3835")

    # legend
    h = [
        plt.Line2D([], [], marker="o", ms=11, mfc="white", mec=ef.C_CENSORED, mew=1.1, lw=0,
                  label="latent (has a prior, estimated)"),
        plt.Line2D([], [], marker="o", ms=11, mfc="#d9d5d0", mec="#52514e", mew=0.9, lw=0,
                  label="observed"),
        plt.Line2D([], [], marker="o", ms=13, mfc="white", mec=ef.C_CENSORED, mew=1.6, lw=0,
                  label="deterministic (function of its parents)"),
    ]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
             ncol=3, fontsize=ef.FS_MIN)

    ax.set_xlim(0.2, 6.8); ax.set_ylim(0.7, 6.0)
    ax.set_aspect("equal"); ax.axis("off")
    ef.message_title(fig, "Plate diagram: six shared parameters, eight variables per each of the "
                          "114 locations")
    ef.save_fig(fig, "FIG30_plate_diagram")
    plt.close(fig)

    txt = f"""{stamp_header()}

# C2 (corrida 3) — diagrama de placas del modelo generativo

Requisito 2 de la Guía del docente ("modelo generativo formal + diagrama de placas"), la única
pieza de código que faltaba de los 7 requisitos obligatorios. Mismos nodos y aristas que el DAG de
`h_red_bayesiana.py` (el modelo no cambia); la notación es la estándar de modelos gráficos
(Bishop 2006 §8.1; Murphy, *Probabilistic Machine Learning: An Introduction*, §3.6, `RECURSOS/`).

**Fuera de la placa** (un valor compartido por las 114 localizaciones): `η, ℓ, ρ` (hiperparámetros
del kernel Matérn 5/2), `β` (coeficientes de covariables), `b_bg` (desplazamiento por estrato),
`σ_n` (desviación del ruido), `r` (razón LOQ/LOD, Currie 1968), `LOD` (límite reportado del
laboratorio).

**Dentro de la placa** `s = 1…N`, `N = 114`: `x(s)`, `g(s)` observados; `f(s)` el campo gaussiano
latente; `μ(s)` la media determinista; `y(s)` la concentración latente en escala log; `L` el límite
de cuantificación determinista (`L = LOD·r`); `D(s)` el indicador de detección observado; `y^obs(s)`
el valor reportado observado.

No hace falta la librería `daft`: dibujado a mano con `matplotlib.patches`, mismo patrón que
`h_red_bayesiana.py`, mismo módulo de estilo. Figura `FIG30_plate_diagram`.
"""
    from config import OUT
    (OUT / "C2_plate_diagram.md").write_text(txt, encoding="utf-8")
    observe("C2 completo: diagrama de placas del modelo generativo (requisito 2 de la Guia del "
            "docente, el ultimo de codigo que faltaba de los 7 obligatorios). Mismos nodos/aristas "
            "que el DAG de h_red_bayesiana.py, notacion estandar (Bishop 2006, Murphy PML1 S3.6). "
            "Escrito FIG30_plate_diagram, outputs/C2_plate_diagram.md.", "Bloque C - corrida 3")
    print("escrito: FIG30_plate_diagram, outputs/C2_plate_diagram.md")


if __name__ == "__main__":
    main()
