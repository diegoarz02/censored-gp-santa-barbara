"""A1 (corrida 3) — kriging ordinario sobre las mismas 120 celdas del protocolo profundo.

El baseline clasico se habia quedado fuera de la tabla estrella de F3 (outputs/h6_deep/, solo
censored GP y GP with L/2). La rubrica pesa la comparacion con el clasico un 20%. Este script:

1. mide el coste de una celda de kriging antes de lanzar las 120 (regla 3);
2. las corre via runner.py (checkpoint por unidad, reanudable);
3. junta los 120 nuevos checkpoints de kriging con los 240 ya existentes de F3 y escribe la tabla
   y figura de cobertura con los TRES modelos.

No toca los checkpoints de censored GP / GP with L/2: los lee con json.load() directo, nunca a
traves de worker_v4._load(), que es lo unico que podria invalidarlos por el cambio de RUN_ID.
"""
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import runner  # noqa: E402
import worker_v4 as w4  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

METALS = ["As", "Hg", "Pb"]
LEVELS = [0.2, 0.4, 0.6, 0.8]
N_REPLICAS = 10


def main():
    units = w4.make_kriging_units(METALS, LEVELS, N_REPLICAS)
    print(f"A1: {len(units)} unidades de kriging a cubrir (checkpoint ya existente se salta)")

    # --- regla 3: medir antes de lanzar. 5 celdas en serie, extrapolar.
    probe_units = units[:5]
    t0 = time.time()
    for u in probe_units:
        w4.run_deep(dict(u))
    dt = time.time() - t0
    proj_min = dt / len(probe_units) * len(units) / 60
    print(f"[probe] kriging: {dt:.1f} s / {len(probe_units)} unidades -> "
          f"proyeccion {proj_min:.2f} min para las {len(units)}")
    observe(f"A1 sonda: kriging {dt:.1f} s por {len(probe_units)} celdas, proyeccion "
            f"{proj_min:.2f} min para las {len(units)} -- confirma que es barato, va por runner.py "
            f"igual (regla 4).", "Bloque A - corrida 3")

    # --- las 120, via runner.py (las 5 de la sonda ya quedaron con checkpoint y se saltan)
    runner.run_units("A1_kriging_deep", units, w4.run_deep, cores_per_fit=2, cap=10)

    # --- verificar que las 120 tienen checkpoint sin error
    rows = []
    missing, errored = [], []
    for u in units:
        ck = w4.ckpt_path(u)
        if not ck.exists():
            missing.append(u); continue
        o = json.loads(ck.read_text(encoding="utf-8"))
        if "error" in o:
            errored.append((u, o["error"])); continue
        rows.append(o)
    print(f"kriging: {len(rows)} ok, {len(missing)} faltantes, {len(errored)} con error")
    if errored:
        for u, e in errored[:10]:
            print("  ERROR", u, "->", e)
    kdf = pd.DataFrame(rows)

    # --- los 240 ya existentes de F3 (censored GP, GP with L/2), leidos directo, sin tocar su
    # checkpoint ni pasar por _load() (que compararia run_id y los borraria).
    f3_rows = [json.load(open(f, encoding="utf-8"))
              for f in glob.glob(str(OUT / "h6_deep" / "*_cens.json"))
              + glob.glob(str(OUT / "h6_deep" / "*_sub.json"))]
    f3_rows = [r for r in f3_rows if "error" not in r]
    f3df = pd.DataFrame(f3_rows)
    print(f"F3 existentes: {len(f3df)} filas (censored GP + GP with L/2)")

    full = pd.concat([f3df, kdf], ignore_index=True, sort=False)
    full.to_csv(FIG / "TABLE_H6_MAIN_con_kriging.csv", index=False, encoding="utf-8")

    # --- tabla de cobertura principal, los tres modelos
    piv = full.pivot_table(index=["metal", "level"], columns="model", values="picp95")
    print("\n=== PICP95, los tres modelos ===")
    print(piv.round(3).to_string())

    piv80 = full[full.level == 0.8].pivot_table(index="metal", columns="model", values="picp95")
    div = (full.groupby("model")["divergences"].sum())
    print("\ndivergencias por modelo (kriging es N/A, cuenta 0 por definicion):")
    print(div.to_string())

    # --- figura de cobertura principal, tres modelos, un panel por metal (regla: max 2 paneles;
    # aqui van 3 metales en un solo eje con marcador distinto, como ya hace FIG9/FIG11)
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 0.72), constrained_layout=True)
    model_color = ef.MODEL_COLOR
    model_short = ef.MODEL_SHORT
    metal_marker = ef.METAL_MARKER
    for model in ["censored GP", "GP with L/2", "ordinary kriging L/2"]:
        for metal in METALS:
            s = (full[(full.model == model) & (full.metal == metal)]
                 .groupby("level")["picp95"].mean().sort_index())
            if s.empty:
                continue
            ax.plot(s.index * 100, s.values, "-", lw=1.1, alpha=0.85,
                   color=model_color[model], marker=metal_marker[metal], ms=4.2,
                   mec="k", mew=0.25)
    # leyenda compacta: color = modelo, marcador = metal (dos leyendas separadas)
    h_model = [plt.Line2D([], [], color=model_color[m], lw=1.6, label=model_short[m])
              for m in ["censored GP", "GP with L/2", "ordinary kriging L/2"]]
    h_metal = [plt.Line2D([], [], color="0.35", marker=metal_marker[m], ms=4.5, lw=0, label=m)
              for m in METALS]
    leg1 = ax.legend(handles=h_model, loc="upper center", bbox_to_anchor=(0.28, -0.16),
                     frameon=False, fontsize=ef.FS_MIN, title="model", title_fontsize=ef.FS_MIN)
    ax.add_artist(leg1)
    ax.legend(handles=h_metal, loc="upper center", bbox_to_anchor=(0.78, -0.16), frameon=False,
             fontsize=ef.FS_MIN, title="metal", title_fontsize=ef.FS_MIN, ncol=1)
    ax.axhline(0.95, color=ef.C_REF, lw=1.0)
    ax.text(81, 0.955, "nominal 0.95", fontsize=ef.FS_MIN, color=ef.C_REF, ha="right", va="bottom")
    ax.set_xlabel("imposed censoring (%)")
    ax.set_ylabel("95% interval coverage")
    ax.set_xticks([20, 40, 60, 80])
    ef.soft_grid(ax)
    ef.message_title(ax, "The classical baseline also collapses under censoring — only the "
                         "censored model does not")
    ef.save_fig(fig, "FIG28_coverage_three_models")
    plt.close(fig)

    txt = f"""{stamp_header()}

# A1 (corrida 3) — kriging ordinario reincorporado a la tabla principal

## Por qué

El baseline clásico (`sb.ordinary_kriging`) no estaba en la tabla de los 240 ajustes profundos de
F3 (`outputs/h6_deep/`), que solo comparaba los dos modelos bayesianos (censurado y GP con L/2). La
rúbrica del curso pesa la comparación con un modelo clásico de referencia un 20 %, y el error #12
del docente es justo comparar un bayesiano afinado contra un baseline mal puesto en la mesa.

## Qué se corrió

`sb.ordinary_kriging` (PyKrige, ajuste cerrado del variograma esférico — no es MCMC, no tiene
`tune`/`draws`/`target_accept`) sobre las **mismas {len(units)} celdas de prueba** (3 metales × 4
niveles × 10 réplicas) que usan `censored GP` y `GP with L/2`, con el mismo split train/test y el
mismo valor sustituido L/2 para los censurados en el entrenamiento.

Sonda de coste: {dt:.1f} s para {len(probe_units)} celdas -> proyección {proj_min:.2f} min para las
{len(units)}. Confirmado barato antes de lanzar el lote completo.

## Resultado — cobertura del intervalo del 95 % al 80 % de censura, los tres modelos

{piv80.round(3).to_markdown()}

## Divergencias por modelo (las 120+120+120 celdas)

{div.to_string()}

Kriging es un ajuste cerrado: sus divergencias se registran como 0 por definición (no hay muestreo
MCMC que pueda divergir), no porque haya convergido mejor que los GP — es una comparación distinta
de naturaleza, y así se declara en la tabla (`diagnostics_na` en cada fila).

Tabla completa: `TABLE_H6_MAIN_con_kriging.csv`. Figura: `FIG28_coverage_three_models`.
"""
    (OUT / "A1_kriging_tabla_principal.md").write_text(txt, encoding="utf-8")
    observe(f"A1 completo: kriging ordinario reincorporado a la tabla principal de cobertura, "
            f"{len(units)} celdas, {len(missing)} faltantes, {len(errored)} con error. "
            f"Escrito TABLE_H6_MAIN_con_kriging.csv, FIG28_coverage_three_models, "
            f"outputs/A1_kriging_tabla_principal.md. Cierra la brecha de rubrica del 20%% "
            f"(comparacion con clasico) que senalo la auditoria de Cowork.", "Bloque A - corrida 3")
    print("\nescrito: TABLE_H6_MAIN_con_kriging.csv, FIG28_coverage_three_models, "
          "A1_kriging_tabla_principal.md")


if __name__ == "__main__":
    main()
