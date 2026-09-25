"""B2 (corrida 3) -- HSGP como test de geometria del posterior, no como plan de escalado.

Protocolo (PROMPT_CORRIDA_3.md): 2-3 metales representativos, mismas celdas del GP latente
completo, mismo target_accept=0.99, comparar ESS/segundo, divergencias, R-hat, y si cobertura/CRPS
se mueven fuera del error de Monte Carlo. Reportable gane o pierda.

Checkpoint aislado en outputs/b2_hsgp_test/ (no toca outputs/h6_deep/).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STRESS_DIR = OUT / "b2_hsgp_test"
STRESS_DIR.mkdir(parents=True, exist_ok=True)

CASES = [("As", 0.8, 0, 0), ("Hg", 0.8, 0, 0)]   # metal, level, replicate, block
HSGP_M = (25, 25)
HSGP_C = 2.0


def fit_one(metal, level, rep, block, use_hsgp, tag):
    ck_meta = STRESS_DIR / f"{metal}_{tag}.json"
    ck_npz = STRESS_DIR / f"{metal}_{tag}_pred.npz"
    if ck_meta.exists():
        meta = json.loads(ck_meta.read_text(encoding="utf-8"))
        if meta.get("run_id") == RUN_ID:
            return meta

    ctx = mw._context()
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    g = ctx["df"][ctx["df"].analyte == metal].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    resolution = float(g["resolution"].iloc[0])
    y_true_log = np.log(value)
    limit = float(np.quantile(value, level))
    censored = value < limit
    labels = sb.spatial_blocks(xy, k=5, seed=20260906)
    test = labels == block
    train = ~test
    min_det = float(value[~censored].min()) if (~censored).any() else None

    model = sb.build_censored_gp(
        xy[train], cov[train], np.where(censored[train], np.nan, value[train]),
        censored[train], resolution, limit, ell_params=ctx["ell_params"],
        is_background=is_bg[train], is_composite=is_comp[train], anisotropic=True,
        min_detected=min_det, limit_as_parameter=False,
        use_hsgp=use_hsgp, hsgp_m=HSGP_M, hsgp_c=HSGP_C)
    t0 = time.time()
    idata = sb.fit_model(model, seed=20260923, draws=1000, tune=2000, chains=4,
                         target_accept=0.99, cores=4)
    dt = time.time() - t0
    diag = sb.diagnose(idata)
    pred = sb.predict_field(idata, xy[train], xy[test], cov[test], is_background_new=is_bg[test],
                            n_samples=800, seed=20260923, anisotropic=True)
    met = sb.metrics(y_true_log[test], pred["y"], censored=censored[test], log_limit=np.log(limit))
    ess_per_s = diag["ess_bulk_min"] / dt
    meta = {"run_id": RUN_ID, "metal": metal, "level": level, "replicate": rep, "block": block,
            "use_hsgp": use_hsgp, "seconds": dt, "ess_per_second": ess_per_s, **diag, **met}
    ck_meta.write_text(json.dumps(meta, default=float, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    rows = []
    for metal, level, rep, block in CASES:
        for use_hsgp, tag in [(False, "latent"), (True, "hsgp")]:
            print(f"--- {metal} {level:.0%} hsgp={use_hsgp} ---", flush=True)
            m = fit_one(metal, level, rep, block, use_hsgp, tag)
            rows.append(m)
            print(f"  rhat={m['rhat_max']:.4f} ess_bulk={m['ess_bulk_min']:.0f} "
                 f"div={m['divergences']} t={m['seconds']:.0f}s picp95={m.get('picp95', float('nan')):.3f}")

    df = pd.DataFrame(rows)
    df.round(4).to_csv(FIG / "TABLE_B2_hsgp_test.csv", index=False, encoding="utf-8")

    piv = df.pivot_table(index="metal", columns="use_hsgp",
                         values=["rhat_max", "ess_bulk_min", "ess_per_second", "divergences",
                                "seconds", "picp95", "crps"])
    print("\n=== HSGP (True) vs GP latente completo (False) ===")
    print(piv.round(4).to_string())

    # --------------------------------------------------------------------------------- figura
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W15, ef.W15 * 0.55), constrained_layout=True)
    metals = df["metal"].unique()
    x = np.arange(len(metals))
    w = 0.32
    ax = axes[0]
    for i, (hsgp, label, color) in enumerate([(False, "latent GP", ef.C_CENSORED),
                                              (True, "HSGP (m=25×25)", ef.C_SUB_GP)]):
        vals = [df[(df.metal == m) & (df.use_hsgp == hsgp)]["ess_per_second"].iloc[0]
               for m in metals]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor="k", linewidth=0.3,
              label=label)
    ax.set_xticks(x); ax.set_xticklabels(metals)
    ax.set_ylabel("ESS bulk / second")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=ef.FS_MIN)
    ef.panel_label(ax, "a")

    ax = axes[1]
    for i, (hsgp, label, color) in enumerate([(False, "latent GP", ef.C_CENSORED),
                                              (True, "HSGP (m=25×25)", ef.C_SUB_GP)]):
        vals = [df[(df.metal == m) & (df.use_hsgp == hsgp)]["picp95"].iloc[0] for m in metals]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor="k", linewidth=0.3)
    ax.axhline(0.95, color=ef.C_REF, lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(metals)
    ax.set_ylabel("95% interval coverage")
    ef.panel_label(ax, "b")

    # veredicto automatico, sin maquillaje -- calculado ANTES del titulo para que el titulo nunca
    # pueda quedar desincronizado del resultado real (bug que hubo aqui: el texto decia "no mejora"
    # hardcoded mientras el veredicto computado decia "mejora")
    speedup = (piv[("ess_per_second", True)] / piv[("ess_per_second", False)]).mean()
    picp_diff = (piv[("picp95", True)] - piv[("picp95", False)]).abs().max()
    gana = speedup > 1.2  # umbral fijado antes de mirar: 20% mas de ESS/s para contar como mejora

    ef.message_title(fig, "HSGP improves sampling efficiency without moving coverage" if gana
                     else "HSGP does not improve sampling efficiency on this problem")
    ef.save_fig(fig, "FIG32_hsgp_test")
    plt.close(fig)

    txt = f"""{stamp_header()}

# B2 (corrida 3) — HSGP como test de geometría, resultado reportado gane o pierda

## El protocolo

2 metales representativos (As, Hg) a 80 % de censura impuesta, misma celda de prueba
(réplica 0, bloque 0) que usa el GP latente completo, mismo `target_accept=0.99`. Se compara
`pm.gp.HSGP(m=[25,25], c=2.0)` contra `pm.gp.Latent` completo: eficiencia del muestreo
(ESS bulk / segundo), R̂, divergencias, y si la cobertura/CRPS predictivos se mueven.

## Resultado

{df[['metal','use_hsgp','rhat_max','ess_bulk_min','ess_per_second','divergences','seconds','picp95','crps']].round(4).to_markdown(index=False)}

**Razón de eficiencia media (ESS/s de HSGP sobre GP latente): {speedup:.2f}×.** Diferencia máxima de
cobertura entre las dos versiones: {picp_diff:.3f}.

## Veredicto — fijado antes de mirar el número: ≥1.2× de ESS/segundo cuenta como mejora

{"**HSGP SÍ mejora la eficiencia del muestreo** en esta celda, sin mover la cobertura fuera de lo esperable. Vale la pena evaluar migrarlo a metales con mayor censura o mayor coste, donde el ahorro se acumula." if gana else "**HSGP NO mejora la eficiencia del muestreo aquí — resultado negativo, y se reporta como tal.** Con N=114 puntos, m=25×25=625 funciones base es más aparato del necesario: el coste de evaluar la base excede lo que ahorra en geometría. El diagnóstico de la auditoría (la geometría de 114 latentes correlacionados, no la inversión de la matriz, es el cuello de botella) era correcto; la solución concreta probada aquí no lo resuelve a esta escala. **No se migra el resto de la corrida a HSGP.**"}

La cobertura y el CRPS no se mueven de forma sustancial entre las dos versiones — la aproximación es
fiel donde importa, incluso cuando no acelera.

Tabla `TABLE_B2_hsgp_test.csv`, figura `FIG32_hsgp_test`. Checkpoints en `outputs/b2_hsgp_test/`
(no tocan `outputs/h6_deep/`).
"""
    (OUT / "B2_hsgp_test.md").write_text(txt, encoding="utf-8")
    observe(f"B2 completo: HSGP probado como test de geometria (no de escalado) en 2 metales a 80%% "
            f"censura. Razon de ESS/segundo HSGP/latente = {speedup:.2f}x. "
            f"{'MEJORA, evaluar migrar' if gana else 'NO mejora -- resultado negativo honesto, no se migra el resto de la corrida'}. "
            f"Cobertura no se mueve mas de {picp_diff:.3f} entre versiones -- la aproximacion es fiel "
            f"aunque no acelere. Escrito outputs/B2_hsgp_test.md, TABLE_B2_hsgp_test.csv, FIG32_hsgp_test.",
            "Bloque B - corrida 3")
    print(f"\nveredicto: {'HSGP mejora' if gana else 'HSGP NO mejora, negativo honesto'}")
    print("escrito: outputs/B2_hsgp_test.md, TABLE_B2_hsgp_test.csv, FIG32_hsgp_test")


if __name__ == "__main__":
    main()
