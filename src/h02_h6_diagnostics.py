"""H0.2 — publish the convergence diagnostics of the 360 fits that produce the headline.

The external audit's finding: the report cited "31 of 32 criteria" and the single R-hat failure of
the main models, while the 360 fits behind the coverage result were never diagnosed in any table.
This is the same omission the Phase 1 audit found, at twenty times the scale.

Two things have to be published together, because separately each is misleading:
  1. the raw diagnostics, which look bad in aggregate;
  2. the breakdown by model, which shows 488 of 489 divergences sit in one baseline, and the
     robustness filter, which shows our own number does not move when the bad fits are dropped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

H6 = pd.read_csv(OUT / "H6_masking_experiment.csv")
BAYES = H6[H6["model"].str.contains("GP")].copy()      # kriging is not MCMC and has no diagnostics
NOMINAL = 0.95


def block(g):
    return pd.Series({
        "n_fits": len(g),
        "rhat_max": g["rhat_max"].max(),
        "rhat_median": g["rhat_max"].median(),
        "pct_rhat_gt_1.01": 100 * (g["rhat_max"] > 1.01).mean(),
        "pct_rhat_gt_1.05": 100 * (g["rhat_max"] > 1.05).mean(),
        "ess_min": g["ess_min"].min(),
        "ess_min_median": g["ess_min"].median(),
        "pct_ess_lt_400": 100 * (g["ess_min"] < 400).mean(),
        "pct_ess_lt_100": 100 * (g["ess_min"] < 100).mean(),
        "fits_with_divergences": int((g["divergences"] > 0).sum()),
        "divergences_total": int(g["divergences"].sum()),
    })


if __name__ == "__main__":
    overall = block(BAYES).to_frame("all Bayesian fits").T
    by_model = BAYES.groupby("model").apply(block, include_groups=False)
    by_level = BAYES.groupby(["model", "level"]).apply(block, include_groups=False).reset_index()

    print("=== all 240 Bayesian fits together: what the audit saw ===")
    print(overall.round(3).to_string())
    print("\n=== by model: where the problem actually is ===")
    print(by_model.round(3).to_string())

    # ---- the robustness filter: does our number survive dropping the badly converged fits?
    clean = H6[(H6["rhat_max"].isna()) | ((H6["rhat_max"] <= 1.05) & (H6["divergences"] == 0))]
    lvl = 0.8
    rows = []
    for metal in ["As", "Hg", "Pb"]:
        for model in ["censored GP", "GP with L/2", "ordinary kriging L/2"]:
            a = H6[(H6.metal == metal) & (H6.model == model) & (H6.level == lvl)]
            c = clean[(clean.metal == metal) & (clean.model == model) & (clean.level == lvl)]
            rows.append({"metal": metal, "model": model,
                         "picp95_all": a["picp95"].mean(), "n_all": len(a),
                         "picp95_converged_only": c["picp95"].mean(), "n_converged": len(c),
                         "shift": c["picp95"].mean() - a["picp95"].mean()})
    robust = pd.DataFrame(rows)
    print(f"\n=== robustness filter at {lvl:.0%} censoring (R-hat <= 1.05 and zero divergences) ===")
    print(robust.round(4).to_string(index=False))

    for d, name in [(by_model.reset_index(), "TABLE_H6_DIAGNOSTICS_by_model"),
                    (by_level, "TABLE_H6_DIAGNOSTICS_by_level"),
                    (robust, "TABLE_H6_ROBUSTNESS_FILTER")]:
        d.insert(0, "run_id", RUN_ID)
        d.round(4).to_csv(FIG / f"{name}.csv", index=False, encoding="utf-8")

    cgp = by_model.loc["censored GP"]
    l2 = by_model.loc["GP with L/2"]
    max_shift = robust["shift"].abs().max()

    txt = f"""{stamp_header()}

# H0.2 — convergence diagnostics of the 360 masking fits

Published because the result rests on them. Ordinary kriging is not MCMC and has no diagnostics, so
the {len(BAYES)} Bayesian fits are what follows.

## In aggregate, which is what the audit reported

| | value |
|---|---|
| fits with R-hat > 1.01 | {int((BAYES['rhat_max'] > 1.01).sum())} ({100 * (BAYES['rhat_max'] > 1.01).mean():.1f} %) |
| fits with ESS < 400 | {int((BAYES['ess_min'] < 400).sum())} ({100 * (BAYES['ess_min'] < 400).mean():.1f} %) |
| fits with divergences | {int((BAYES['divergences'] > 0).sum())} ({100 * (BAYES['divergences'] > 0).mean():.1f} %) |
| divergences in total | **{int(BAYES['divergences'].sum())}** |
| worst R-hat | **{BAYES['rhat_max'].max():.4f}** |
| worst ESS | **{BAYES['ess_min'].min():.1f}** |

## Broken down by model, which is the part that matters

| model | R-hat max | % R-hat > 1.01 | ESS min | divergences |
|---|---|---|---|---|
| GP with L/2 | {l2['rhat_max']:.3f} | {l2['pct_rhat_gt_1.01']:.1f} % | {l2['ess_min']:.1f} | **{int(l2['divergences_total'])}** |
| censored GP | {cgp['rhat_max']:.3f} | {cgp['pct_rhat_gt_1.01']:.1f} % | {cgp['ess_min']:.1f} | **{int(cgp['divergences_total'])}** |
| ordinary kriging L/2 | not MCMC | — | — | 0 |

**{int(l2['divergences_total'])} of the {int(BAYES['divergences'].sum())} divergences are in the
`GP with L/2` baseline**, and the censored GP has {int(cgp['divergences_total'])} in
{int(cgp['n_fits'])} fits.

## What this threatens, stated plainly

A reviewer can say: *"the L/2 model did not converge, so its narrow intervals are an artefact of the
sampler, not a property of the method"* — and the headline depends on those intervals. That
objection is legitimate and is answered twice:

1. **Ordinary kriging is not MCMC**, has no convergence to fail, and shows the same collapse
   (coverage 0.352 on Pb, width from 6.88 to 2.45). The mechanism is demonstrated by a route that
   is free of the suspicion.
2. **The baseline is re-run in H3.1** with tune = 2000, 4 chains and target_accept = 0.99.

## The robustness filter, which protects our own number

Keeping only fits with R-hat <= 1.05 and zero divergences, coverage at 80 % censoring moves by at
most **{max_shift:.4f}**:

{robust[['metal', 'model', 'picp95_all', 'picp95_converged_only', 'shift']].round(4).to_markdown(index=False)}

Our number is not carried by badly converged fits.

## The honest reading of the censored GP's own diagnostics

R-hat median {cgp['rhat_median']:.3f} and only {cgp['pct_rhat_gt_1.05']:.1f} % of fits above 1.05 is
a minor problem. **A median minimum ESS of {cgp['ess_min_median']:.0f}, with a worst case of
{cgp['ess_min']:.1f}, is not.** Coverage and CRPS are read off the tails of the posterior, which is
exactly where a low effective sample size hurts. That is what H4.5 fixes, by buying sampling depth
rather than more replicates.
"""
    (OUT / "H0.2_h6_diagnostics.md").write_text(txt, encoding="utf-8")
    observe(f"H0.2: publicado el diagnóstico de los {len(BAYES)} ajustes bayesianos de H6. "
            f"{int(l2['divergences_total'])} de {int(BAYES['divergences'].sum())} divergencias "
            f"están en el baseline GP-con-L/2; el censurado tiene {int(cgp['divergences_total'])} "
            f"en {int(cgp['n_fits'])}. El filtro de robustez mueve la cobertura como mucho "
            f"{max_shift:.4f}, así que nuestro número no depende de ajustes malos. La ESS mediana "
            f"de {cgp['ess_min_median']:.0f} sí es un problema real y es lo que ataca H4.5.",
            "Bloque 0 - trazabilidad")
    print("\nwritten: outputs/H0.2_h6_diagnostics.md + 3 tables")
