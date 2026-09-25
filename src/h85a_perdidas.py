"""H8.5a — the exceedance decision as a formal Bayesian decision, not a cut at 0.5.

The maps produce P(concentration > ECA) and the previous run then thresholded at 0.5. That is the
optimal rule **only if the two errors cost the same**, and they do not: declaring clean a soil that
is contaminated exposes people; declaring contaminated a soil that is clean pays for a remediation
that was not needed.

With loss l(a, theta), cost c_FN for a missed exceedance and c_FP for a false alarm, the posterior
expected loss of declaring exceedance is c_FP * (1 - p) and of not declaring it is c_FN * p, so
declaring is optimal when

    p > c_FP / (c_FP + c_FN)  =  p*

A single map at p* = 0.5 hides that the answer depends on a value judgement. What the decision maker
needs is **the map of the threshold**: how the declared area moves as the cost ratio changes.

This is a syllabus unit ("Pérdidas y riesgos") and it is also the piece that connects the model to
the decision the court order actually requires.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import mapas_f3 as M  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

RATIOS = [1, 2, 5, 10, 20]          # c_FN : c_FP, how much worse a missed exceedance is


def p_star(ratio):
    """Optimal declaration threshold when a false negative costs `ratio` times a false positive."""
    return 1.0 / (1.0 + ratio)


def bayes_risk(p, thr, ratio):
    """Posterior expected loss per cell under the rule 'declare exceedance if p > thr'.

    Units are false-positive costs, so c_FP = 1 and c_FN = ratio. Averaged over the domain.
    """
    declare = p > thr
    return float(np.mean(np.where(declare, 1.0 * (1.0 - p), ratio * p)))


if __name__ == "__main__":
    D = M.load()
    rows = []
    for a in M.MAIN:
        p = D["mp"][f"{a}_p_agri"]
        for r in RATIOS:
            t = p_star(r)
            rows.append({"analyte": a, "cost_ratio_FN_to_FP": r, "threshold": t,
                         "pct_domain_declared": 100 * float((p > t).mean()),
                         "bayes_risk_optimal": bayes_risk(p, t, r),
                         "bayes_risk_at_0.5": bayes_risk(p, 0.5, r)})
    tab = pd.DataFrame(rows)
    tab["risk_penalty_of_using_0.5"] = tab["bayes_risk_at_0.5"] - tab["bayes_risk_optimal"]
    tab.insert(0, "run_id", RUN_ID)
    tab.round(5).to_csv(FIG / "TABLE29_decision_thresholds.csv", index=False, encoding="utf-8")
    print(tab.drop(columns=["run_id"]).round(4).to_string(index=False))

    # ---- figure: how the declared area moves with the cost ratio, and the risk penalty of 0.5
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.42), constrained_layout=True)
    cols = {"Hg": ef.C_CENSORED, "Pb": ef.C_SUB_GP, "As": ef.C_KRIGING,
            "Cd": "#8c6bb1", "Ba": "#b0aca6"}
    # Direct labelling fails here and a legend is the honest choice: Hg, Pb and As all saturate at
    # 100 % of the domain, so their end labels land on the same point and stack on top of one
    # another. Rule 5 prefers labelling at the line end *when the lines separate*; these do not.
    for a in M.MAIN:
        s = tab[tab.analyte == a]
        axes[0].plot(s["cost_ratio_FN_to_FP"], s["pct_domain_declared"], "-o", ms=3.5,
                     color=cols[a], mec="k", mew=0.2, label=a)
        axes[1].plot(s["cost_ratio_FN_to_FP"], s["risk_penalty_of_using_0.5"], "-o", ms=3.5,
                     color=cols[a], mec="k", mew=0.2, label=a)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(RATIOS)
        ax.set_xticklabels([f"1:{r}" for r in RATIOS])
        ax.set_xlabel("cost ratio, false alarm : missed exceedance")
        ef.soft_grid(ax)
    axes[0].set_ylabel("% of domain declared in exceedance")
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.88, edgecolor="#cfccc7",
                   ncol=2, columnspacing=1.0, handlelength=1.4)
    axes[1].set_ylabel("extra Bayes risk from cutting at 0.5")
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.88, edgecolor="#cfccc7",
                   ncol=2, columnspacing=1.0, handlelength=1.4)
    ef.panel_label(axes[0], "a")
    ef.panel_label(axes[1], "b")
    ef.message_title(fig, "Cutting the exceedance map at 0.5 assumes a false alarm costs as much "
                          "as a missed exposure")
    ef.save_fig(fig, "FIG20_decision_thresholds")
    plt.close(fig)

    cd = tab[tab.analyte == "Cd"]
    hg = tab[tab.analyte == "Hg"]
    cd1 = cd[cd.cost_ratio_FN_to_FP == 1]["pct_domain_declared"].iloc[0]
    cd10 = cd[cd.cost_ratio_FN_to_FP == 10]["pct_domain_declared"].iloc[0]
    hg1 = hg[hg.cost_ratio_FN_to_FP == 1]["pct_domain_declared"].iloc[0]
    hg10 = hg[hg.cost_ratio_FN_to_FP == 10]["pct_domain_declared"].iloc[0]

    txt = f"""{stamp_header()}

# H8.5a — the exceedance decision, formalised

## The rule

With action $a \\in \\{{\\text{{declare}}, \\text{{do not declare}}\\}}$, a false alarm costing
$c_{{FP}}$ and a missed exceedance costing $c_{{FN}}$, the posterior expected loss is
$c_{{FP}}(1-p)$ for declaring and $c_{{FN}}\\,p$ for not declaring. Declaring is optimal when

$$p > p^{{*}} = \\frac{{c_{{FP}}}}{{c_{{FP}} + c_{{FN}}}}$$

Cutting at 0.5 is the special case $c_{{FP}} = c_{{FN}}$: it asserts that a remediation ordered
unnecessarily costs exactly as much as an exposure missed. Nobody would defend that in writing,
and the previous run assumed it implicitly.

## The map of the threshold, not a map per threshold

| cost ratio (alarm : miss) | $p^{{*}}$ | Hg declared | Cd declared |
|---|---|---|---|
| 1 : 1 | 0.500 | {hg1:.1f} % | {cd1:.1f} % |
| 1 : 10 | 0.091 | {hg10:.1f} % | {cd10:.1f} % |

**Mercury barely moves** ({hg1:.1f} % to {hg10:.1f} %): the posterior is so far above the standard
that no defensible cost ratio changes the conclusion. That is itself a finding — for mercury the
decision is robust to the value judgement.

**Cadmium moves a great deal** ({cd1:.1f} % to {cd10:.1f} %), because its posterior sits near the
threshold over much of the domain. Cadmium is where the loss function actually decides, and it is
also the analyte where censoring is heaviest and where our model loses on the joint score. The
three facts are the same fact seen from three directions: **cadmium is the case where the answer is
genuinely uncertain**, and pretending otherwise by cutting at 0.5 hides it.

Full table `TABLE29_decision_thresholds.csv`, figure `FIG20_decision_thresholds`.

## Why this belongs in the paper and not only in the appendix

A probabilistic map whose only use is to be thresholded at 0.5 has thrown away what made it
probabilistic. The court order asks where the State must act; the answer depends on the relative
cost of acting where it was not needed and of not acting where it was. Publishing the threshold map
hands that judgement to whoever is entitled to make it, instead of burying a 1:1 assumption in the
method section.
"""
    (OUT / "H8.5a_perdidas_riesgos.md").write_text(txt, encoding="utf-8")
    observe(f"H8.5a: la decisión de excedencia formalizada. Umbral óptimo p*=c_FP/(c_FP+c_FN); "
            f"cortar en 0.5 afirma que una remediación innecesaria cuesta lo mismo que una "
            f"exposición no detectada. Hg apenas se mueve ({hg1:.1f}% a {hg10:.1f}% al pasar de "
            f"1:1 a 1:10): la conclusión es robusta al juicio de valor. **Cd se mueve mucho** "
            f"({cd1:.1f}% a {cd10:.1f}%), porque su posterior está pegado al umbral. Cd es donde "
            f"la función de pérdida realmente decide, y es el mismo analito con más censura y "
            f"donde perdemos el score conjunto: son tres caras del mismo hecho.",
            "Bloque 8 - sílabo")
    print("\nwritten: outputs/H8.5a_perdidas_riesgos.md, TABLE29, FIG20")
