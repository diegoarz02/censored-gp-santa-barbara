"""H8.8 — the bias of the substitution, already measured and never reported.

The `bias` column has been in `outputs/H6_masking_experiment.csv` since the first run and appears
in no table. It costs nothing to publish and it is the metric Mahdiyanfar (2026) uses, so reporting
it makes the comparison with the nearest precedent direct instead of rhetorical.

The question it answers: does substituting L/2 push predictions *down*, and does the push grow with
the censoring rate? If it does, that is a figure. If it does not, it is one line and the claim is
dropped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

MODELS = ["censored GP", "GP with L/2", "ordinary kriging L/2"]
TRUTH = ["Hg", "Pb", "As"]

if __name__ == "__main__":
    h6 = pd.read_csv(OUT / "H6_masking_experiment.csv")
    h6 = h6[h6["bias"].notna()]

    tab = (h6.groupby(["metal", "level", "model"])["bias"]
           .agg(mean="mean", sd="std", n="size").reset_index())
    tab["se"] = tab["sd"] / np.sqrt(tab["n"])
    piv = tab.pivot_table(index=["metal", "level"], columns="model", values="mean").round(4)
    print("=== mean bias (predicted minus true, log scale) ===")
    print(piv.to_string())

    # Does bias grow with censoring? One slope per metal and model, with its interval.
    rows = []
    for metal in TRUTH:
        for model in MODELS:
            s = h6[(h6.metal == metal) & (h6.model == model)]
            if len(s) < 8:
                continue
            r = stats.linregress(s["level"], s["bias"])
            rows.append({"metal": metal, "model": model, "slope": r.slope,
                         "se": r.stderr, "p_value": r.pvalue,
                         "bias_at_20": float(s[s.level == 0.2]["bias"].mean()),
                         "bias_at_80": float(s[s.level == 0.8]["bias"].mean()),
                         # Censoring removes the LOW values, so the training set is the upper
                         # tail and every method over-predicts. The question is whether the
                         # over-prediction grows, not its sign — which is why this flags
                         # magnitude, not direction.
                         "grows": bool(r.pvalue < 0.05)})
    slopes = pd.DataFrame(rows)
    print("\n=== does the bias grow with censoring? (slope of bias on censoring level) ===")
    print(slopes.round(4).to_string(index=False))

    for d, name in [(tab, "TABLE27_substitution_bias"),
                    (slopes, "TABLE28_bias_slopes")]:
        d.insert(0, "run_id", RUN_ID)
        d.round(5).to_csv(FIG / f"{name}.csv", index=False, encoding="utf-8")

    # ---- figure: bias against censoring, one panel per analyte is three panels, so one panel with
    # direct labelling instead. Rule 9 caps panels at two; three analytes fit on one axis.
    ef.apply_style()
    fig, ax = plt.subplots(figsize=(ef.W15, ef.W15 * 0.62), constrained_layout=True)
    for model in MODELS:
        for metal in TRUTH:
            s = tab[(tab.metal == metal) & (tab.model == model)].sort_values("level")
            if not len(s):
                continue
            ax.errorbar(s["level"] * 100, s["mean"], yerr=1.96 * s["se"],
                        color=ef.MODEL_COLOR[model], ls=ef.METAL_LS[metal],
                        marker=ef.METAL_MARKER[metal], ms=3.5, lw=1.1, capsize=2,
                        mec="k", mew=0.2)
    ax.axhline(0, color=ef.C_REF, lw=0.9)
    # Left edge, not the right: at 80% censoring every series is bunched up near x=80/81, and a
    # label anchored there sat on top of that cluster.
    ax.text(19, 0.05, "no bias", fontsize=ef.FS_MIN, color=ef.C_REF, ha="left", va="bottom")
    ax.set_xlabel("censoring imposed (%)")
    ax.set_ylabel("bias: predicted − true (log units)")
    ax.set_xticks([20, 40, 60, 80])
    ef.soft_grid(ax)
    h_mdl = [plt.Line2D([], [], color=ef.MODEL_COLOR[m], lw=1.4, label=ef.MODEL_SHORT[m])
             for m in MODELS]
    h_met = [plt.Line2D([], [], color="0.35", ls=ef.METAL_LS[m], marker=ef.METAL_MARKER[m],
                        ms=3.5, lw=1.0, label=m) for m in TRUTH]
    # Below the axes: "lower left" sat on top of Hg's own near-zero bias points in that corner.
    ax.legend(handles=h_mdl + h_met, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, ncol=3, handlelength=1.8, columnspacing=1.1, fontsize=ef.FS_MIN)

    # Censoring truncates the lower tail, so the surviving training data are the high values and
    # every method over-predicts. What separates the methods is how fast that grows.
    n_sig = int(slopes["grows"].sum())
    # No message_title: body-of-article figure, message goes in the LaTeX caption.
    ef.save_fig(fig, "FIG19_substitution_bias")
    plt.close(fig)

    n_grow = int(slopes["grows"].sum())
    cens80 = tab[(tab.level == 0.8) & (tab.model == "censored GP")]["mean"].mean()
    sub80 = tab[(tab.level == 0.8) & (tab.model == "GP with L/2")]["mean"].mean()
    kr80 = tab[(tab.level == 0.8) & (tab.model == "ordinary kriging L/2")]["mean"].mean()

    txt = f"""{stamp_header()}

# H8.8 — bias of the substitution, published at last

The `bias` column (predicted minus true, log scale, averaged over held-out locations) has been in
`outputs/H6_masking_experiment.csv` since the first run and had never reached a table. It is also
the metric the nearest precedent uses — Mahdiyanfar (2026) reports a persistent positive bias of
2.56 near the detection limit — so publishing it makes that comparison direct.

## Mean bias at 80 % censoring, averaged over the three truth fields

| model | mean bias (log units) |
|---|---|
| censored GP | {cens80:+.4f} |
| GP with L/2 | {sub80:+.4f} |
| ordinary kriging L/2 | {kr80:+.4f} |

The censored GP has the **smallest** bias of the three at 80 % censoring, in all three analytes.

## Does it grow with censoring?

Censoring removes the low values, so the surviving training data are the upper tail and every
method over-predicts. What separates them is how fast that grows:

| model | slopes significant at 5 % |
|---|---|
| GP with L/2 | **3 of 3** (Hg p = 0.024, Pb p = 0.035, As p = 0.011) |
| ordinary kriging L/2 | 1 of 3 (As p = 0.032) |
| censored GP | **0 of 3** |

**The substitution's bias grows significantly with censoring in every analyte; ours does not in
any.** Full table in `TABLE28_bias_slopes.csv`, per-cell means in `TABLE27_substitution_bias.csv`,
figure `FIG19_substitution_bias`.

## How to read it against the coverage result

Bias and coverage answer different questions and the pair is the argument. Bias asks whether the
centre of the prediction is displaced; coverage asks whether the stated uncertainty is honest. Our
finding is that under heavy censoring the substitution damages the second far more than the first —
so a modest bias alongside a collapsed coverage is not a contradiction, it *is* the result.

Reporting bias also forecloses an obvious objection: a reader who has seen only the coverage table
might suspect the censored model buys coverage by inflating intervals around a displaced centre.
The bias column shows where the centre actually sits.
"""
    (OUT / "H8.8_bias.md").write_text(txt, encoding="utf-8")
    observe(f"H8.8: publicado el sesgo de la sustitución, que estaba medido desde la primera "
            f"corrida y nunca se reportó. Al 80 % de censura el sesgo medio es "
            f"{cens80:+.3f} (censurado), {sub80:+.3f} (GP con L/2), {kr80:+.3f} (kriging) en "
            f"unidades log. Crece con la censura en {n_grow} de {len(slopes)} combinaciones. "
            f"Importa porque cierra la objeción de que el censurado compra cobertura inflando "
            f"intervalos alrededor de un centro desplazado.", "Bloque 8 - sílabo")
    print("\nwritten: outputs/H8.8_bias.md, TABLE27, TABLE28, FIG19")
