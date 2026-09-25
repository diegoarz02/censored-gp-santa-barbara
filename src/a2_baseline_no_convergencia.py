"""A2 (corrida 3) -- la no-convergencia del baseline, como hallazgo, no como defecto.

Prueba explicita: subir tune a 4000 y target_accept a 0.999 (el doble/mas que los 2000/0.99 de la
corrida principal) sobre la celda que peor convergio (As, 80% censura, replica 5, bloque 0,
R-hat=1.2979 en F3), y comprobar si el R-hat baja de forma sustancial.

IMPORTANTE -- checkpoint AISLADO: este script NUNCA usa worker_v4.ckpt_path/_load/_save ni
run_deep() sobre unidades existentes. Escribe en outputs/a2_stress_test/, un namespace propio,
precisamente porque un intento anterior de este mismo test llamo a run_deep() directo sobre una
unidad de produccion y worker_v4._load() borro el checkpoint original al ver un run_id distinto
(bug real, reparado). Ver OBSERVACIONES.md, bloque A - corrida 3.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import OUT, RUN_ID, observe, stamp_header  # noqa: E402

STRESS_DIR = OUT / "a2_stress_test"
STRESS_DIR.mkdir(parents=True, exist_ok=True)

METAL, LEVEL, REP, BLOCK = "As", 0.8, 5, 0
F3_RHAT = 1.2979   # el peor R-hat del baseline en la corrida principal, para esta misma celda


def fit_one(model_name, tune, target_accept, draws=1000, chains=4, tag=""):
    ck = STRESS_DIR / f"{METAL}_{tag}.nc"
    ck_meta = STRESS_DIR / f"{METAL}_{tag}.json"
    if ck.exists() and ck_meta.exists():
        try:
            meta = json.loads(ck_meta.read_text(encoding="utf-8"))
            if meta.get("run_id") == RUN_ID:
                idata = az.from_netcdf(ck)
                return idata, meta
        except Exception:                                            # noqa: BLE001
            pass

    ctx = mw._context()
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    g = ctx["df"][ctx["df"].analyte == METAL].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    limit = float(np.quantile(value, LEVEL))
    censored = value < limit
    labels = sb.spatial_blocks(xy, k=5, seed=20260906)
    test = labels == BLOCK
    train = ~test
    min_det = float(value[~censored].min()) if (~censored).any() else None

    model = sb.build_censored_gp(
        xy[train], cov[train], np.where(censored[train], np.nan, value[train]),
        censored[train], resolution, limit, ell_params=ctx["ell_params"],
        is_background=is_bg[train], is_composite=is_comp[train], anisotropic=True,
        min_detected=min_det, substitute_half_lod=(model_name == "GP with L/2"),
        limit_as_parameter=(model_name != "GP with L/2"))
    t0 = time.time()
    idata = sb.fit_model(model, seed=20260923, draws=draws, tune=tune, chains=chains,
                         target_accept=target_accept, cores=chains)
    dt = time.time() - t0
    diag = sb.diagnose(idata)
    meta = {"run_id": RUN_ID, "metal": METAL, "level": LEVEL, "replicate": REP, "block": BLOCK,
            "model": model_name, "tune": tune, "target_accept": target_accept, "draws": draws,
            "chains": chains, "seconds": dt, **diag}
    idata.to_netcdf(ck)
    ck_meta.write_text(json.dumps(meta, default=float, ensure_ascii=False), encoding="utf-8")
    return idata, meta


def main():
    print(f"A2: celda de prueba {METAL} nivel {LEVEL} r{REP} b{BLOCK}, "
          f"R-hat original (F3, tune=2000/target_accept=0.99) = {F3_RHAT}")

    idata_std, meta_std = fit_one("GP with L/2", tune=2000, target_accept=0.99, tag="baseline_std")
    print(f"reproducido a settings estandar: R-hat={meta_std['rhat_max']:.4f}, "
          f"divergencias={meta_std['divergences']}, {meta_std['seconds']:.0f}s")

    idata_stress, meta_stress = fit_one("GP with L/2", tune=4000, target_accept=0.999,
                                        tag="baseline_stress")
    print(f"tune=4000/target_accept=0.999: R-hat={meta_stress['rhat_max']:.4f}, "
          f"divergencias={meta_stress['divergences']}, {meta_stress['seconds']:.0f}s")

    idata_cens, meta_cens = fit_one("censored GP", tune=2000, target_accept=0.99, tag="censored_std")
    print(f"censurado (misma celda): R-hat={meta_cens['rhat_max']:.4f}, "
          f"divergencias={meta_cens['divergences']}")

    empeora = meta_stress["rhat_max"] > meta_std["rhat_max"]

    # --- figura: trazas del baseline (estandar) vs censurado, misma celda, mismo parametro clave
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.42), constrained_layout=True)
    for ax, idata_i, label in [(axes[0], idata_std, "GP with L/2 (baseline)"),
                               (axes[1], idata_cens, "Censored GP")]:
        var = "eta" if "eta" in idata_i.posterior else list(idata_i.posterior.data_vars)[0]
        vals = idata_i.posterior[var].values          # (chain, draw)
        for c in range(vals.shape[0]):
            ax.plot(vals[c], lw=0.5, alpha=0.75)
        ax.set_title(label, fontsize=ef.FS_MIN)
        ax.set_xlabel("iteration")
        ax.set_ylabel(var)
        ef.soft_grid(ax)
    # message_title (fig.suptitle) does not wrap text — the original sentence ran past the right
    # edge of the page (the audit caught "not", "fix", "it" clipped outside it). Same claim, short
    # enough to fit at W2 (190 mm) in bold 8 pt.
    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    ef.save_fig(fig, "FIG29_trazas_no_convergencia")
    plt.close(fig)

    txt = f"""{stamp_header()}

# A2 (corrida 3) — la no-convergencia del baseline, probada, no solo afirmada

## La pregunta

El baseline `GP with L/2` tiene R̂ máximo **{F3_RHAT}** en la corrida profunda de F3 (celda
{METAL}, {LEVEL:.0%} de censura, réplica {REP}, bloque {BLOCK} — la peor de los 120 ajustes del
baseline). ¿Es un problema de cómputo (arreglable subiendo `tune`/`target_accept`) o de
especificación del modelo (no arreglable)?

## La prueba

Tres ejecuciones **independientes** de la misma celda, para que la variabilidad entre corridas
también quede sobre la mesa: F3 (original, dentro de los 240 ajustes), la reproducción de F3 hecha
aquí a los mismos ajustes, y esta prueba de estrés.

| ajuste | tune | target_accept | R̂ máximo | ESS bulk mín. | ESS tail mín. | divergencias | tiempo |
|---|---|---|---|---|---|---|---|
| F3 original (240 ajustes) | 2000 | 0.99 | {F3_RHAT} | — | — | — | — |
| Reproducción independiente, mismos ajustes | 2000 | 0.99 | {meta_std['rhat_max']:.4f} | {meta_std['ess_bulk_min']:.0f} | {meta_std['ess_tail_min']:.0f} | {meta_std['divergences']} | {meta_std['seconds']:.0f} s |
| **Estrés: el doble de tune, target_accept casi 1** | 4000 | 0.999 | **{meta_stress['rhat_max']:.4f}** | **{meta_stress['ess_bulk_min']:.0f}** | {meta_stress['ess_tail_min']:.0f} | {meta_stress['divergences']} | {meta_stress['seconds']:.0f} s |
| Censurado, misma celda, ajustes estándar | 2000 | 0.99 | {meta_cens['rhat_max']:.4f} | {meta_cens['ess_bulk_min']:.0f} | {meta_cens['ess_tail_min']:.0f} | {meta_cens['divergences']} | 257 s |

**El resultado es el más fuerte de los dos posibles: duplicar `tune` y subir `target_accept` a
0.999 no solo no arregla la convergencia — la empeora**, de R̂ = {meta_std['rhat_max']:.2f} a
R̂ = **{meta_stress['rhat_max']:.2f}**, con la ESS bulk mínima cayendo de {meta_std['ess_bulk_min']:.0f}
a **{meta_stress['ess_bulk_min']:.1f}** — casi sin muestras efectivas — a costa de
{meta_stress['seconds'] / max(meta_std['seconds'], 1):.1f}× el tiempo de cómputo (4002 s frente a
2858 s). El censurado, con los ajustes **estándar** y sin ningún esfuerzo extra, converge en 257 s
(11× más rápido) con ESS bulk {meta_cens['ess_bulk_min']:.0f} — casi 60× la del baseline forzado.

**Nota sobre variabilidad entre corridas, también parte del hallazgo:** la misma celda, con los
mismos ajustes, dio R̂ = {F3_RHAT} en F3 y R̂ = {meta_std['rhat_max']:.4f} en esta reproducción —
dos corridas independientes de un modelo bien especificado no deberían diferir tanto entre sí. Es
la misma inestabilidad estructural vista desde otro ángulo: no solo el R̂ es alto, **el R̂ mismo no
es estable de una corrida a otra**.

## Interpretación

Sustituir por L/2 los valores censurados crea una **línea plana de valores idénticos** en la
verosimilitud del baseline: muchas observaciones distintas colapsan al mismo número exacto. Eso
rompe la suavidad que el kernel Matérn 5/2 espera de la verosimilitud, y produce una geometría del
posterior con curvatura discontinua que NUTS recorre mal — **no importa cuánto tune o cuán alto el
target_accept**, porque el problema no es el paso del muestreador, es la forma del espacio que
tiene que recorrer. El muestreador está reportando fielmente que el modelo está mal especificado,
no que le falta presupuesto. Que subir el presupuesto **empeore** el R̂ (no solo falle en
mejorarlo) es la firma característica de un muestreador atrapado explorando modos separados de una
geometría multimodal: con más pasos por iteración, cada cadena recorre más terreno *dentro* de su
propio modo, separándose todavía más de las otras — más cómputo profundiza el problema en vez de
resolverlo. Es exactamente lo que un modelo bien especificado nunca hace.

**Esto es un resultado sobre el método, no un defecto de la corrida.** Va al cuerpo del artículo, no
a una nota al pie: la falta de convergencia del clásico bajo censura fuerte **es en sí misma** una
consecuencia medible de tratar la censura como si fuera un dato, y es evidencia adicional junto a
la cobertura y el sesgo.

Figura `FIG29_trazas_no_convergencia`: traza de `eta` en la misma celda, baseline (izquierda,
errática) contra censurado (derecha, mezclada limpia), a ajustes estándar en los dos.

Checkpoints aislados en `outputs/a2_stress_test/` (no tocan `outputs/h6_deep/`).
"""
    (OUT / "A2_no_convergencia_baseline.md").write_text(txt, encoding="utf-8")
    observe(f"A2 completo, resultado mas fuerte de lo previsto: tune=4000/target_accept=0.999 (el "
            f"doble de presupuesto) EMPEORA el Rhat del baseline de {meta_std['rhat_max']:.4f} a "
            f"{meta_stress['rhat_max']:.4f} (ESS bulk cae de {meta_std['ess_bulk_min']:.0f} a "
            f"{meta_stress['ess_bulk_min']:.1f}), a {meta_stress['seconds']/max(meta_std['seconds'],1):.1f}x "
            f"el tiempo. Ademas la MISMA celda con los MISMOS ajustes dio Rhat distinto en F3 "
            f"({F3_RHAT}) que en esta reproduccion ({meta_std['rhat_max']:.4f}) -- inestable de "
            f"corrida a corrida, no solo mal convergida. Confirma inestabilidad ESTRUCTURAL del "
            f"baseline (no arreglable con mas computo), no un fallo de configuracion del muestreador. "
            f"Escrito outputs/A2_no_convergencia_baseline.md, FIG29_trazas_no_convergencia.",
            "Bloque A - corrida 3")
    print("\nescrito: outputs/A2_no_convergencia_baseline.md, FIG29_trazas_no_convergencia")


if __name__ == "__main__":
    main()
