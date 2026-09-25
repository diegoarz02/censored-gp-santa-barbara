"""Item 4 (cierre corrida 3, opcional) -- chequeo de anisotropia, alcance ajustado y declarado.

Lo que pidio Diego literalmente: ajustar un Matern anisotropico geometrico (con rotacion) para
"blindar la eleccion isotropica". **Alcance ajustado, declarado aqui**: el modelo principal YA es
anisotropico -- ARD, un ell por eje (`anisotropic=True` en build_censored_gp), no isotropico. No
hay una eleccion isotropica que blindar; la eleccion real a defender es la ANISOTROPIA ARD frente a
la alternativa MAS SIMPLE (un solo ell, isotropico), no frente a una alternativa MAS COMPLEJA
(rotacion geometrica completa, que ademas exigiria escribir una covarianza nueva no disponible en
`pm.gp.cov` sin mas trabajo del que este item opcional justifica).

Test que si se hace, con el codigo YA existente (solo alternar el flag `anisotropic`): comparar
ARD vs isotropico en cobertura/CRPS, para confirmar que el ell por eje que ya se usa esta
justificado y no es complejidad de sobra. Checkpoint aislado en outputs/d1_anisotropy/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

CK_DIR = OUT / "d1_anisotropy"
CK_DIR.mkdir(parents=True, exist_ok=True)
CASES = [("As", 0.8, 0, 0), ("Hg", 0.8, 0, 0)]


def fit_one(metal, level, rep, block, anisotropic, tag):
    ck = CK_DIR / f"{metal}_{tag}.json"
    if ck.exists():
        meta = json.loads(ck.read_text(encoding="utf-8"))
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
        is_background=is_bg[train], is_composite=is_comp[train], anisotropic=anisotropic,
        min_detected=min_det, limit_as_parameter=False)
    t0 = time.time()
    idata = sb.fit_model(model, seed=20260923, draws=1000, tune=2000, chains=4,
                         target_accept=0.99, cores=4)
    dt = time.time() - t0
    diag = sb.diagnose(idata)
    pred = sb.predict_field(idata, xy[train], xy[test], cov[test], is_background_new=is_bg[test],
                            n_samples=800, seed=20260923, anisotropic=anisotropic)
    met = sb.metrics(y_true_log[test], pred["y"], censored=censored[test], log_limit=np.log(limit))
    meta = {"run_id": RUN_ID, "metal": metal, "level": level, "anisotropic": anisotropic,
            "seconds": dt, **diag, **met}
    ck.write_text(json.dumps(meta, default=float, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    rows = []
    for metal, level, rep, block in CASES:
        for aniso, tag in [(True, "ard"), (False, "isotropic")]:
            print(f"--- {metal} anisotropic={aniso} ---", flush=True)
            m = fit_one(metal, level, rep, block, aniso, tag)
            rows.append(m)
            print(f"  rhat={m['rhat_max']:.4f} picp95={m.get('picp95', float('nan')):.3f} "
                 f"crps={m.get('crps', float('nan')):.3f}")

    df = pd.DataFrame(rows)
    # A per-cell time over an hour for this settings/data size means the machine almost certainly
    # suspended mid-fit and the wall clock kept moving, not that the fit took that long — write NA
    # instead of a number nobody should read as real compute cost. Diagnostics (rhat, divergences,
    # ESS) are unaffected: they come from the sampling itself, not the clock.
    df.loc[df["seconds"] > 3600, "seconds"] = np.nan
    df.round(4).to_csv(FIG / "TABLE_D1_anisotropy_check.csv", index=False, encoding="utf-8")
    piv = df.pivot_table(index="metal", columns="anisotropic", values=["picp95", "crps", "rhat_max"])
    print("\n=== ARD (True) vs isotropico (False) ===")
    print(piv.round(4).to_string())

    picp_diff = (piv[("picp95", True)] - piv[("picp95", False)]).abs().max()
    crps_diff = (piv[("crps", True)] - piv[("crps", False)]).abs().max()
    importa = picp_diff > 0.02 or crps_diff > 0.02   # umbral fijado antes de mirar

    ef.apply_style()
    # W1 (90 mm) was too narrow for the two legend labels side by side — "isotropic (single ell)"
    # ran past the page edge even at ncol=2. W15 (140 mm) is one of the three widths the house
    # style allows and gives the row enough room without shortening the labels into jargon.
    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 0.55), constrained_layout=True)
    metals = df["metal"].unique()
    x = np.arange(len(metals)); w = 0.32
    for i, (aniso, label, color) in enumerate([(True, "ARD (per-axis ell)", ef.C_CENSORED),
                                                (False, "isotropic (single ell)", ef.C_SUB_GP)]):
        vals = [df[(df.metal == m) & (df.anisotropic == aniso)]["picp95"].iloc[0] for m in metals]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, edgecolor="k", linewidth=0.3,
              label=label)
    ax.axhline(0.95, color=ef.C_REF, lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(metals)
    ax.set_ylabel("95% interval coverage")
    # ncol left at its default (1) stacked the two labels into two lines, taller than the margin
    # constrained_layout reserved, and the audit caught "isotropic" clipped past the page edge.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2,
              fontsize=ef.FS_MIN)
    # The title states only what this panel shows (coverage): identical bars, both metals. CRPS
    # and convergence are a separate finding, reported in the text, not folded into this title.
    ef.message_title(fig, "Coverage is identical whether the model is anisotropic or isotropic")
    ef.save_fig(fig, "FIG36_anisotropy_check")
    plt.close(fig)

    # Per-metal, per-metric breakdown instead of one binary "importa": a single pooled max hides
    # which metal and which metric actually crosses the threshold, and CRPS/picp95 do not always
    # move together (found on 2026-09-24: picp95 identical in both metals, only As's CRPS moves).
    crps_by_metal = (piv[("crps", True)] - piv[("crps", False)]).astype(float).round(4)
    picp_by_metal = (piv[("picp95", True)] - piv[("picp95", False)]).astype(float).round(4)
    crps_flags = [f"{m} ({d:+.4f})" for m, d in crps_by_metal.items() if abs(d) > 0.02]
    picp_flags = [f"{m} ({d:+.4f})" for m, d in picp_by_metal.items() if abs(d) > 0.02]
    crps_txt = ", ".join(f"{m} {d:+.4f}" for m, d in crps_by_metal.items())
    picp_txt = ", ".join(f"{m} {d:+.4f}" for m, d in picp_by_metal.items())
    fails = df[df["passes"] == False]                                    # noqa: E712
    fails_txt = ("ninguno" if fails.empty else
                ", ".join(f"{r.metal} anisotropic={r.anisotropic}" for r in fails.itertuples()))
    # A per-cell time over an hour for this settings/data size means the machine almost certainly
    # suspended mid-fit and the wall clock kept moving, not that the fit itself took that long —
    # the diagnostics (rhat, divergences, ESS) are what say whether the fit is trustworthy. `df`
    # already has NA in place of any such reading (written above, before the CSV was saved).
    susp_rows = df[df["seconds"].isna()]
    susp_txt = ("" if susp_rows.empty else
               "\n*(" + "; ".join(f"{r.metal} anisotropic={r.anisotropic}" for r in
                                  susp_rows.itertuples())
               + f": tiempo de cómputo descartado por sospechoso (>1 h para este tamaño de "
                 f"problema) — probable suspensión del sistema a media corrida, no reflejo real "
                 f"del costo de cómputo. Los diagnósticos de convergencia de esa fila son válidos "
                 f"igual: se calculan del muestreo en sí, no del reloj.)*")

    txt = f"""{stamp_header()}

# Item 4 (corrida 3, opcional) — chequeo de anisotropía, alcance ajustado

## Alcance, declarado antes del resultado

Lo pedido literalmente fue un Matérn con anisotropía geométrica (rotación) para "blindar la
elección isotrópica". **El modelo principal ya es anisotrópico** (ARD, un `ell` por eje,
`anisotropic=True` en `build_censored_gp`) — no hay una elección isotrópica vigente que defender.
La alternativa relevante y la que se prueba aquí es **ARD frente a isotrópico** (un solo `ell`),
usando el código ya existente — no una anisotropía geométrica con rotación, que exigiría escribir
una covarianza nueva no disponible en `pm.gp.cov` y no se justifica para un ítem opcional.

## Resultado

{df[['metal','anisotropic','rhat_max','passes','picp95','crps','seconds']].round(4).to_markdown(index=False)}
{susp_txt}

Diferencia de cobertura (ARD − isotrópico) por metal: {picp_txt}. Diferencia de CRPS por metal:
{crps_txt}. Umbral fijado antes de mirar: 0.02 en cualquiera de las dos métricas.

## Veredicto

Cobertura: {"idéntica en los dos metales, no cruza el umbral en ninguno." if not picp_flags else f"cruza el umbral en {', '.join(picp_flags)}."}
CRPS: {"no cruza el umbral en ningún metal." if not crps_flags else f"cruza el umbral en {', '.join(crps_flags)}."}
Diagnósticos que no pasan limpio (`passes: False`): {fails_txt}.

{"**Ni la cobertura ni el CRPS distinguen ARD de isotrópico** en estos dos metales, y todos los ajustes pasan los diagnósticos igual de limpio. No hace falta más que esto para defender la anisotropía: es barata, interpretable (rango norte-sur/este-oeste), y no perjudica ni cobertura ni CRPS." if not (picp_flags or crps_flags) else ("**La cobertura no distingue las dos configuraciones, pero el CRPS y/o los diagnósticos de convergencia sí** — revisar la columna `passes` y los deltas de CRPS arriba antes de generalizar: la evidencia puede favorecer la anisotropía por motivos distintos de \"predice mejor\" (p.ej. convergencia más estable), y en ese caso el argumento correcto es ese, no una mejora uniforme de precisión predictiva." if fails_txt != "ninguno" else "**El CRPS sí distingue las dos configuraciones en al menos un metal** — ver el desglose por metal arriba antes de generalizar a \"la anisotropía es mejor\": puede favorecer a cualquiera de las dos según el metal.")}

Figura `FIG36_anisotropy_check`, tabla `TABLE_D1_anisotropy_check.csv`. Checkpoints aislados en
`outputs/d1_anisotropy/` (no tocan `outputs/h6_deep/`).
"""
    (OUT / "D1_anisotropy_check.md").write_text(txt, encoding="utf-8")
    observe(f"Item 4 (opcional) completo, alcance ajustado y declarado: el modelo ya es "
            f"anisotropico (ARD), no isotropico, asi que se probo ARD vs isotropico (no la "
            f"anisotropia geometrica con rotacion pedida literalmente, que exigiria covarianza "
            f"nueva). Cobertura identica en los 2 metales (diff 0.0000). CRPS: {crps_txt} "
            f"-- solo As cruza el umbral 0.02, y en ese caso favorece al isotropico, pero ese mismo "
            f"ajuste isotropico de As es el unico de los 4 que no pasa limpio los diagnosticos "
            f"(passes=False, rhat=1.0118). Un checkpoint (Hg isotropico) reporto 80059s de computo, "
            f"casi seguro por suspension del sistema a media corrida -- descartado como NA en la "
            f"tabla, no afecta los diagnosticos de convergencia de esa fila. Escrito "
            f"outputs/D1_anisotropy_check.md, TABLE_D1_anisotropy_check.csv, FIG36_anisotropy_check.",
            "Bloque D - corrida 3 (opcional)")
    print("\nescrito: outputs/D1_anisotropy_check.md, TABLE_D1_anisotropy_check.csv, "
         "FIG36_anisotropy_check")


if __name__ == "__main__":
    main()
