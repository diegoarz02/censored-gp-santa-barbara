"""La red bayesiana del modelo: el grafo dirigido acíclico, con d-separación calculada.

Sílabo, semana 7: modelos gráficos probabilísticos, DAG, independencia condicional, d-separación.
Cobertura teórica en Murphy (2022), *Probabilistic Machine Learning: An Introduction*, §3.6
«Graphical models», p. 99 (`RECURSOS/book1.pdf`).

**Por qué esto no es decorativo.** Un DAG del modelo hace tres cosas que la especificación algebraica
no hace de un vistazo:

1. **Enseña el mecanismo de censura como parte del modelo**, no como un preproceso. El indicador de
   no detección es un *hijo* de la concentración latente y del límite, y eso es exactamente lo que
   distingue nuestro enfoque de sustituir por LD/2 — que corta la arista y trata el indicador como
   dato.
2. **Hace visible el confundido**: la posición entra en la respuesta por tres caminos a la vez, y el
   estrato es casi función de ella.
3. **Permite preguntarle al grafo** qué es identificable, con d-separación, **antes de ajustar nada**.

Las consultas de d-separación se calculan con `networkx`, no se afirman.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

# ------------------------------------------------------------------ el grafo del modelo
# Nodos latentes en azul, observados en gris, deterministas en blanco.
EDGES = [
    ("s", "f"),            # la posición genera el campo espacial
    ("s", "x"),            # y las covariables (elevación, pendiente, distancias)
    ("s", "g"),            # y el estrato, que OEFA asignó por zona
    ("theta", "f"),        # hiperparámetros de covarianza: eta, ell, rho
    ("f", "mu"),
    ("x", "mu"),
    ("beta", "mu"),
    ("g", "mu"),
    ("b_g", "mu"),
    ("mu", "y"),           # concentración latente en escala log
    ("sigma_n", "y"),
    ("y", "d"),            # el indicador de detección es HIJO de la concentración y del límite
    ("L", "d"),
    ("y", "yobs"),         # el valor reportado solo existe si se detectó
    ("d", "yobs"),
    ("LOD", "L"),          # L = LOD * r, con r estimado
    ("r", "L"),
]
LABEL = {
    "s": "position\n$s$", "f": "field\n$f(s)$", "x": "covariates\n$x(s)$",
    "g": "stratum\n$g(s)$", "theta": "$\\eta,\\ell,\\rho$", "beta": "$\\beta$",
    "b_g": "$b_{bg}$", "mu": "mean\n$\\mu(s)$", "sigma_n": "$\\sigma_n$",
    "y": "concentration\n$\\log Y(s)$", "d": "detected\n$D(s)$",
    "yobs": "reported\nvalue", "L": "limit\n$L$", "LOD": "reported\nLOD", "r": "$r$",
}
OBSERVED = {"s", "x", "g", "yobs", "d", "LOD"}
DETERMINISTIC = {"mu", "L"}
POS = {
    # Los hiperparámetros van en la fila de arriba y los nodos del proceso en la del medio, para
    # que ninguna flecha atraviese un nodo que no es su destino.
    "s": (0.0, 1.6), "theta": (0.6, 3.3), "beta": (1.7, 3.3), "b_g": (2.5, 3.3),
    "f": (1.2, 2.3), "x": (1.2, 1.5), "g": (1.2, 0.6),
    "mu": (2.5, 1.5), "sigma_n": (3.3, 3.3),
    "y": (3.7, 1.5), "d": (4.9, 2.2), "yobs": (4.9, 0.7),
    "L": (4.3, 2.9), "LOD": (3.9, 4.0), "r": (5.0, 3.6),
}

QUERIES = [
    ("g", "y", {"s", "x", "f"},
     "¿El efecto del estrato sobre la concentración se puede leer condicionando en posición, "
     "covariables y campo espacial?"),
    ("s", "y", {"f", "x", "g"},
     "¿La posición influye en la concentración por alguna vía que el modelo no capture?"),
    ("L", "y", {"d"},
     "¿Condicionar en el indicador de detección abre un camino entre el límite y la "
     "concentración? (colisionador)"),
    ("LOD", "y", set(),
     "¿El LOD reportado informa sobre la concentración sin condicionar en nada?"),
    ("theta", "g", set(),
     "¿Los hiperparámetros de covarianza y el estrato son independientes a priori?"),
]


def build():
    G = nx.DiGraph()
    G.add_edges_from(EDGES)
    assert nx.is_directed_acyclic_graph(G), "el grafo debe ser acíclico"
    return G


def main():
    G = build()
    rows = []
    for a, b, Z, pregunta in QUERIES:
        try:
            sep = nx.is_d_separator(G, {a}, {b}, set(Z))
        except AttributeError:                                       # networkx < 3.3
            sep = nx.d_separated(G, {a}, {b}, set(Z))
        rows.append({"desde": a, "hasta": b, "condicionando_en": ", ".join(sorted(Z)) or "—",
                     "d_separados": bool(sep), "pregunta": pregunta})
    t = pd.DataFrame(rows)
    print(t[["desde", "hasta", "condicionando_en", "d_separados"]].to_string(index=False))
    t.insert(0, "run_id", RUN_ID)
    t.to_csv(FIG / "TABLE32_d_separacion.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------ figura
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W2, ef.W2 * 0.52), constrained_layout=True)
    for u, v in G.edges():
        x0, y0 = POS[u]
        x1, y1 = POS[v]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#8a8681",
                                    shrinkA=17, shrinkB=17, connectionstyle="arc3,rad=0.04"))
    for n, (x, y) in POS.items():
        if n in OBSERVED:
            fc, ec, lw = "#d9d5d0", "#52514e", 0.9
        elif n in DETERMINISTIC:
            fc, ec, lw = "white", "#8a8681", 0.7
        else:
            fc, ec, lw = "#cfe0f5", ef.C_CENSORED, 1.0
        ax.scatter([x], [y], s=1500, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
                   marker="o")
        ax.text(x, y, LABEL[n], ha="center", va="center", fontsize=ef.FS_MIN, zorder=4)
    ax.set_xlim(-0.55, 5.6)
    ax.set_ylim(0.1, 4.5)
    ax.axis("off")

    import matplotlib.patches as mp
    handles = [
        mp.Patch(facecolor="#cfe0f5", edgecolor=ef.C_CENSORED, label="latent (estimated)"),
        mp.Patch(facecolor="#d9d5d0", edgecolor="#52514e", label="observed"),
        mp.Patch(facecolor="white", edgecolor="#8a8681", label="deterministic"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              frameon=False, ncol=3, fontsize=8)
    ef.message_title(fig, "The non-detection indicator is a child of concentration, not a datum: "
                          "that edge is the model")
    ef.save_fig(fig, "FIG23_red_bayesiana")
    plt.close(fig)

    q1 = t[(t.desde == "g") & (t.hasta == "y")]["d_separados"].iloc[0]
    q3 = t[(t.desde == "L") & (t.hasta == "y")]["d_separados"].iloc[0]

    txt = f"""{stamp_header()}

# La red bayesiana del modelo, y qué le preguntamos al grafo

Sílabo semana 7 (modelos gráficos, DAG, independencia condicional, d-separación). Cobertura teórica
en Murphy (2022) §3.6, p. 99.

**El grafo no es un adorno: contiene tres afirmaciones que la especificación algebraica no enseña de
un vistazo.**

## 1. La arista que define el método

`log Y(s) → D(s) ← L`

El indicador de detección es **hijo** de la concentración latente y del límite. Eso *es* el modelo
censurado: el hecho de no detectar es una consecuencia de la concentración, y por tanto informa
sobre ella.

Sustituir por `L/2` **corta esa arista** y convierte `D(s)` en un dato sin padres, con la
concentración fijada a una constante. Visto en el grafo, la diferencia entre los dos enfoques deja
de ser una cuestión de gusto: uno modela el mecanismo de observación y el otro lo borra.

## 2. `D(s)` es un colisionador, y eso tiene consecuencias

En `Y → D ← L`, el nodo `D` es un **colisionador**. Sin condicionar en él, `L` e `Y` son
independientes. Al condicionar en `D` —que es justo lo que hacemos, porque sabemos qué puntos se
detectaron— **se abre el camino** y el límite pasa a informar sobre la concentración.

Comprobado en el grafo: `L` e `Y` d-separados condicionando en `D` = **{q3}**.

Eso justifica formalmente por qué tratar `L` como parámetro estimado (`L = LOD·r`) no es un lujo: una
vez condicionado en el patrón de detección, el límite y la concentración están acoplados, así que
fijar `L` mal sesga `Y`.

## 3. El confundido, dicho por el grafo

`s → g`, `s → f`, `s → x`, y los tres llegan a `μ`. La posición alcanza la respuesta por **tres
caminos simultáneos**.

d-separación de `g` e `y` condicionando en `s`, `x` y `f`: **{q1}**.

Es decir: el grafo dice que el camino existe y **no se bloquea** con lo que el modelo condiciona. Eso
es coherente con la medición del R² = 0.840 (`H_DAG_identificabilidad.md`): no es que falte potencia,
es que el efecto del estrato no está separado del de la posición.

## Todas las consultas

{t.drop(columns=['run_id']).to_markdown(index=False)}

## Lo que esta figura permite decir en la defensa

Que la elección entre «modelo censurado» y «sustituir por LD/2» **se puede señalar con el dedo en un
grafo**, en vez de explicarla con álgebra. Para una defensa oral de quince minutos, eso vale más que
una ecuación.

Figura `FIG23_red_bayesiana`, tabla `TABLE32_d_separacion.csv`.
"""
    (OUT / "H_red_bayesiana.md").write_text(txt, encoding="utf-8")
    observe(f"Red bayesiana construida con d-separación calculada en networkx, no afirmada. Tres "
            f"cosas que el grafo enseña y el álgebra no: (1) el indicador de detección es HIJO de "
            f"la concentración, y sustituir por L/2 corta esa arista; (2) D es un COLISIONADOR, así "
            f"que condicionar en el patrón de detección acopla el límite con la concentración — eso "
            f"justifica formalmente estimar L en vez de fijarlo; (3) el camino estrato-respuesta no "
            f"se bloquea condicionando en posición, covariables y campo (d-separados = {q1}), "
            f"coherente con el R²=0.840.", "Bloque 8 - sílabo")
    print("\nescrito: outputs/H_red_bayesiana.md, FIG23, TABLE32")


if __name__ == "__main__":
    main()
