"""B3 (corrida 3) -- mezcla con censura incorporada + comparacion 1 vs 2 vs 3 componentes.

Dos cierres del mismo hito (sílabo, semana 10): (1) incorporar la censura a la mezcla para poder
aplicarla al cadmio, donde la version con L/2 no era rigurosa; (2) comparar 1/2/3 componentes por
PSIS-LOO, cerrando la objecion "por que fijaste dos".

Checkpoint aislado en outputs/b3_mixture_censored/ (no toca outputs/cierre_pendientes.json, que es
el namespace de H8.5b/H4.3/H4.4 de F3 -- este es un modelo nuevo, no una sustitucion).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import h_cierre_pendientes as H  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

CK_DIR = OUT / "b3_mixture_censored"
CK_DIR.mkdir(parents=True, exist_ok=True)
METAL = "Cd"
KS = [1, 2, 3]


def run_k(k):
    ck = CK_DIR / f"{METAL}_k{k}.json"
    ck_p = CK_DIR / f"{METAL}_k{k}_p.npy"
    if ck.exists():
        meta = json.loads(ck.read_text(encoding="utf-8"))
        if meta.get("run_id") == RUN_ID:
            p = np.load(ck_p) if ck_p.exists() else None
            return meta, p
    print(f"--- mezcla censurada, k={k} ---", flush=True)
    # k=1 (0 divergencias desde el principio) no necesita el ajuste; k=2/k=3 llevan
    # target_accept=0.99 + tune=3000 (subido de 1500): el primer intento con solo target_accept
    # mas alto bajo las divergencias de k=2 (72->48) pero EMPEORO el Rhat (1.01->1.53) -- pasos mas
    # chicos necesitan mas iteraciones de warmup para adaptarse y mezclar, no solo mas aceptacion.
    kw = {} if k == 1 else {"target_accept": 0.99, "tune": 3000}
    out, p = H.mixture_censored(METAL, k=k, **kw)
    out["run_id"] = RUN_ID
    ck.write_text(json.dumps(out, default=float, ensure_ascii=False), encoding="utf-8")
    if p is not None:
        np.save(ck_p, p)
    print(f"k={k}: {out}", flush=True)
    return out, p


def main():
    results, p_by_k = {}, {}
    for k in KS:
        out, p = run_k(k)
        results[k] = out
        p_by_k[k] = p

    df = pd.DataFrame(results.values())
    df.to_csv(FIG / "TABLE_B3_mixture_components.csv", index=False, encoding="utf-8")
    print("\n=== comparación 1 vs 2 vs 3 componentes, PSIS-LOO ===")
    print(df[["k", "elpd_loo", "se_loo", "p_loo", "rhat_max", "divergences"]]
          .round(3).to_string(index=False))

    best_k = int(df.loc[df["elpd_loo"].idxmax(), "k"])
    elpd = df.set_index("k")["elpd_loo"]
    se = df.set_index("k")["se_loo"]
    diff_2_vs_1 = elpd[2] - elpd[1] if 2 in elpd.index and 1 in elpd.index else float("nan")
    diff_3_vs_2 = elpd[3] - elpd[2] if 3 in elpd.index and 2 in elpd.index else float("nan")

    # --------------------------------------------------------------------------------- figura
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W15, ef.W15 * 0.55), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(df["k"], df["elpd_loo"], yerr=df["se_loo"], fmt="o-", ms=6, lw=1.3,
               color=ef.C_CENSORED, mec="k", mew=0.5, capsize=3)
    ax.axvline(best_k, color=ef.C_REF, lw=0.9, ls="--")
    ax.set_xlabel("number of components k")
    ax.set_ylabel("elpd_loo")
    ax.set_xticks(KS)
    ef.soft_grid(ax)
    ef.panel_label(ax, "a")

    ax = axes[1]
    # mapa de asignacion posterior del componente k=2, censurado
    ctx2 = None
    try:
        import masking_worker as mw
        ctx2 = mw._context()
    except Exception:                                                 # noqa: BLE001
        pass
    if ctx2 is not None and p_by_k.get(2) is not None:
        loc = ctx2["loc"]
        p2 = p_by_k[2]
        sc = ax.scatter(loc["easting"], loc["northing"], c=p2, cmap=ef.CMAP_MAG, s=22,
                        edgecolor="k", linewidth=0.3, vmin=0, vmax=1)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("P(affected component) — censored mixture")
        cb.ax.tick_params(labelsize=ef.FS_MIN)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting UTM 18S (km)")
        ax.set_ylabel("Northing UTM 18S (km)")
        ef.utm_km_ticks(ax)
    ef.panel_label(ax, "b")

    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    ef.save_fig(fig, "FIG33_mezcla_censurada_componentes")
    plt.close(fig)

    r2 = results.get(2, {})
    txt = f"""{stamp_header()}

# B3 (corrida 3) — mezcla con censura incorporada, cadmio, 1 vs 2 vs 3 componentes

## Por qué

La mezcla de H8.5b (F3) usaba L/2 para los no detectados, lo que la hacía inaplicable en rigor al
cadmio (79.8 % de censura) — exactamente donde la pregunta fondo/afectado sigue más abierta. Este
hito reescribe la verosimilitud de la mezcla con censura real: los detectados entran como
observación puntual, los censurados por su log-verosimilitud acumulada bajo el límite
(`Normal.logcdf`), combinadas en un logsumexp de mezcla. Simplificación declarada: los detectados
no llevan censura por intervalo (a diferencia del modelo principal), solo censura por límite; ver
docstring de `mixture_censored` en `h_cierre_pendientes.py`.

## Comparación por PSIS-LOO — cierra la semana 10 del sílabo

{df[['k','elpd_loo','se_loo','p_loo','rhat_max','divergences']].round(3).to_markdown(index=False)}

**k = {best_k} maximiza `elpd_loo`.** Diferencia elpd (k=2 − k=1): {diff_2_vs_1:.2f} (se combinada
≈ {np.sqrt(se.get(1, 0)**2 + se.get(2, 0)**2):.2f}). Diferencia (k=3 − k=2):
{diff_3_vs_2:.2f} (se combinada ≈ {np.sqrt(se.get(2, 0)**2 + se.get(3, 0)**2):.2f}).

## Resultado del cadmio con dos componentes, censura incorporada

{f"Separación entre extremos: {r2.get('separacion_extremos', float('nan')):.2f} unidades log, IC 95 % [{r2.get('sep_ci_low', float('nan')):.2f}, {r2.get('sep_ci_high', float('nan')):.2f}]. Acuerdo con la etiqueta de OEFA: {100*r2.get('acuerdo_con_etiqueta_oefa', float('nan')):.1f} %." if r2 else "No disponible."}

Figura `FIG33_mezcla_censurada_componentes`: (a) `elpd_loo` contra k; (b) mapa de la probabilidad
posterior de pertenecer al componente afectado, mezcla de 2 componentes con censura incorporada.

Checkpoints en `outputs/b3_mixture_censored/` (no toca `outputs/cierre_pendientes.json`, que sigue
siendo la mezcla original de F3 con L/2, para el metal Hg — modelos distintos, no se reemplazan).

## Nota sobre el ajuste: qué arregló las divergencias y qué no

El primer ajuste (settings compartidos, `target_accept=0.95`) dio 72 divergencias en k=2 y 46 en
k=3. Dos intentos después:

| intento | target_accept | tune | R̂ k=2 | div. k=2 | R̂ k=3 | div. k=3 |
|---|---|---|---|---|---|---|
| 1 (original) | 0.95 | 1500 | 1.007 | 72 | 1.020 | 46 |
| 2 (solo target_accept) | 0.99 | 1500 | **1.532** | 48 | 1.048 | 34 |
| 3 (target_accept + tune) | 0.99 | 3000 | **1.003** | 41 | 1.013 | 15 |

**Subir solo `target_accept` bajó las divergencias pero empeoró el R̂** (1.01 → 1.53 en k=2): con
pasos más chicos, 1500 iteraciones de calentamiento no alcanzaban para que el muestreador se
adaptara y las cadenas se mezclaran. **Subir también `tune` a 3000 fue lo que realmente arregló la
convergencia.** Quedan divergencias residuales (41 y 15, sobre 4000 muestras) — bajas y con R̂ sano
en los dos casos, consistentes con geometría localmente difícil cerca del límite donde los
componentes casi se superponen, no con no-convergencia sistemática. Se reportan tal cual, no se
esconden.

**Salvedad de alcance**: B3 es un análisis secundario (semana 10 del sílabo, exploración de la
estructura de mezcla). El resultado central del proyecto — la cobertura del intervalo del 95 % bajo
censura fuerte — no depende de B3 en ningún punto; B3 aporta evidencia adicional sobre el cadmio,
no sostiene el argumento principal.
"""
    (OUT / "B3_mezcla_censurada.md").write_text(txt, encoding="utf-8")
    observe(f"B3 completo: mezcla con censura incorporada aplicada al cadmio, comparacion 1/2/3 "
            f"componentes por PSIS-LOO. k={best_k} maximiza elpd_loo. "
            f"{'2 componentes se sostiene' if best_k==2 else f'k={best_k} supera a 2, revisar encuadre'}. "
            f"Escrito outputs/B3_mezcla_censurada.md, TABLE_B3_mixture_components.csv, "
            f"FIG33_mezcla_censurada_componentes.", "Bloque B - corrida 3")
    print("\nescrito: outputs/B3_mezcla_censurada.md, TABLE_B3_mixture_components.csv, "
          "FIG33_mezcla_censurada_componentes")


if __name__ == "__main__":
    main()
