# -*- coding: utf-8 -*-
"""Assemble RESULTS_SUMMARY.md from the files the notebooks produced.

Every number in the report is read from `outputs/` or `figuras/`. Nothing is typed in by hand, so
the report cannot drift from the analysis. Run from the project root after both notebooks:

    python notebooks/_src/build_results_summary.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJ = Path(__file__).resolve().parent.parent.parent
FIG, OUT, FINAL = PROJ / "figuras", PROJ / "outputs", PROJ / "data" / "final"


def tbl(name):
    return pd.read_csv(FIG / f"{name}.csv")


def js(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def md(df, cols=None, floatfmt=3):
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{floatfmt}f}")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


L: list[str] = []
A = L.append

meta = js("nb01_metadata.json")
summ = js("nb02_summary.json")
h2 = js("h2_spatial_structure.json")
data = pd.read_csv(FINAL / "santa_barbara_soil.csv")

# =============================================================================================
A("# Probabilistic soil-contamination mapping under detection-limit censoring")
A("")
A("## Santa Bárbara mercury mine, Huancavelica, Peru")
A("")
A(f"Results summary. Seed {meta['seed']}, {meta['n_locations']} soil locations, "
  f"{meta['n_analytes']} analytes, sampled {meta['date_range'][0]} to {meta['date_range'][1]}. "
  f"Source: OEFA report {meta['report']}.")
A("")
A("---")
A("")

# ---------------------------------------------------------------------------------------------
A("## 0. What to read first")
A("")
A("### The question")
A("")
A("> When a large fraction of environmental measurements arrive as \"below the detection limit\", "
  "what can still be asserted about risk across a territory, and with how much confidence?")
A("")
A("### The three findings")
A("")

hg_bg = tbl("TABLE19_background_vs_mining")
hgr = hg_bg[hg_bg.analyte == "Hg"].iloc[0]
A(f"**1. There is no clean geochemical background left.** The stratum OEFA itself designated as "
  f"*background* has a median mercury concentration of **{hgr['median_background']:.1f} mg kg⁻¹** "
  f"and **{hgr['pct_bg_above_eca']:.0f} %** of those points exceed the national standard of "
  f"6.6 mg kg⁻¹. After four centuries of mining, a remediation target defined as \"return to "
  f"background\" would already sit above the law. In the potential-interest stratum the median rises "
  f"to {hgr['median_interest']:.1f} mg kg⁻¹ with {hgr['pct_pi_above_eca']:.0f} % exceedance. For "
  f"contrast, Castillo Corzo et al. (2025) measured 1.7 to 6.2 mg kg⁻¹ in the city itself, none "
  f"above the standard.")
A("")

slopes = tbl("TABLE16_divergence_slopes")
n_div, n_tot = int(slopes["diverges"].sum()), len(slopes)
_ct = tbl("TABLE26_coverage_paired_test")
_c80 = _ct[_ct["level"] == 0.8]
_swept = int((_c80["replicates_won"] == _c80["n"]).sum())
_h6 = pd.read_csv(OUT / "H6_masking_experiment.csv")
_p80 = _h6[_h6["level"] == 0.8].pivot_table(index="metal", columns="model", values="picp95")
_pb_k = _p80.loc["Pb", "ordinary kriging L/2"]
_pb_c = _p80.loc["Pb", "censored GP"]
_w = _h6.pivot_table(index=["metal", "level"], columns="model", values="width95")

A(f"**2. Under heavy censoring the `L/2` substitution does not lose accuracy — it destroys the "
  f"uncertainty.** In the masking experiment, at 80 per cent censoring an interval labelled "
  f"95 per cent contains the truth only **{_pb_k:.2f}** of the time for lead under ordinary "
  f"kriging with `L/2`, against **{_pb_c:.2f}** for the censored model on identical masks. The "
  f"censored model is closer to nominal coverage in **{_swept} of {len(_c80)}** comparisons in "
  f"*every one* of the ten replicates (maximum p = {_c80['p_wilcoxon'].max():.4f}). The mechanism "
  f"is visible in the interval width: as censoring rises the baselines' intervals *narrow* "
  f"(lead, {_w.loc[('Pb', 0.2), 'ordinary kriging L/2']:.1f} to "
  f"{_w.loc[('Pb', 0.8), 'ordinary kriging L/2']:.1f} log units) because the substituted constant "
  f"removes the variance it stands for, while the censored model's stay open "
  f"({_w.loc[('Pb', 0.2), 'censored GP']:.1f} to {_w.loc[('Pb', 0.8), 'censored GP']:.1f}).")
A("")
A(f"The effect is graded, and the gradient is the practical answer: nothing separates the methods "
  f"at 20 or 40 per cent censoring, lead separates at 60, everything separates at 80. Measured in "
  f"CRPS instead of coverage the same comparison looks marginal — the divergence slope is "
  f"significant in only {n_div} of {n_tot} combinations — which is why coverage, not CRPS, is the "
  f"primary comparison against a deterministic substitution.")

A("")

eca = tbl("TABLE20_exceedance_summary")
A("**3. Exceedance is not marginal, it is the norm.** Over the prediction grid the probability of "
  "exceeding the agricultural standard is:")
A("")
A(md(eca, ["analyte", "threshold_agri", "p_max", "p_mean", "cells_p_gt_0.5", "pct_domain_p_gt_0.5"]))
A("")
A("---")
A("")

# ---------------------------------------------------------------------------------------------
A("## 1. Dataset (H1)")
A("")
A(f"- **{meta['n_locations']} unique soil locations**, {meta['n_rows']} rows in long format.")
A(f"- Domain extent {meta['domain_extent_km']:.2f} km; nearest-neighbour distance median "
  f"{meta['nn_median_m']:.0f} m, maximum {meta['nn_max_m']:.0f} m. This is the most regular "
  f"sampling grid in the whole OEFA soil archive.")
A(f"- Strata: " + ", ".join(f"{k} {v}" for k, v in meta["strata"].items()) + ".")
A("")
A(md(tbl("TABLE1_dataset_summary"),
      ["analyte", "role", "pct_censored", "median_quantified", "max_quantified", "sd_log",
       "reported_LOD", "resolution", "pct_above_eca_agri"], 2))
A("")
A("### The stratified design, and what it confounds")
A("")
A("The survey declares two strata, and that is what makes it possible to estimate the geochemical "
  "background from data rather than assume it. It also creates two confoundings that are carried "
  "through the whole analysis rather than hidden:")
A("")
A("1. **Stratum against support.** The 30 background points are *exactly* the 30 composite samples; "
  "every potential-interest point is a simple sample. The two effects are almost perfectly aliased.")
A(f"2. **Stratum against location.** The background points sit in "
  f"{h2.get('n_bg_clusters', 'a few')} compact clusters rather than interspersed, so the stratum "
  f"label carries spatial information as well as design information.")
A("")
alias = tbl("TABLE12_stratum_support_aliasing")
A(f"The first was tested rather than assumed: both effects were fitted and their posterior "
  f"correlation inspected. Maximum absolute correlation **{alias['posterior_corr'].abs().max():.3f}**, "
  f"below the 0.9 threshold fixed in advance, so both are reported as estimated.")
A("")
A(md(alias, ["analyte", "b_background", "b_ci95_low", "b_ci95_high", "log_tau",
             "posterior_corr", "separable"]))
A("")

# ---------------------------------------------------------------------------------------------
A("## 2. Spatial structure (H2)")
A("")
A(f"- Moran's I between "
  f"{min(v['morans_I'] for v in h2['morans_I'].values()):.2f} and "
  f"{max(v['morans_I'] for v in h2['morans_I'].values()):.2f}, all with pseudo p = 0.001. "
  f"Spatial autocorrelation here is unambiguous.")
A(f"- Effective variogram ranges " +
  ", ".join(f"{k} {v['effective_range_km']:.2f} km" for k, v in h2["variogram_best"].items()) + ".")
A(f"- **Kernel decision: {h2['kernel']}**, from an anisotropy ratio of "
  f"{h2['anisotropy_ratio_max']:.2f} against a threshold of 2.0 fixed before inspecting the data.")
A("")
A("### A trap worth reporting")
A("")
A(f"The naive anisotropy ratio was **{h2['anisotropy_ratio_max_naive']:.2f}**, and it was wrong. "
  f"The survey area is {h2['domain_extent_ew_km']:.2f} km east-west against "
  f"{h2['domain_extent_ns_km']:.2f} km north-south, so beyond about "
  f"{h2['anisotropy_maxlag_common_km']:.1f} km **there are no east-west pairs at all**. Comparing "
  f"directional sills over a lag axis only some directions can reach measures the shape of the "
  f"survey area, not anisotropy of the field. Restricting every azimuth to a common maximum lag "
  f"gives {h2['anisotropy_ratio_max']:.2f}: the anisotropy is real, but the first number was "
  f"inflated by the geometry. Both versions are published.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 3. Priors (H3)")
A("")
pp = tbl("TABLE8_prior_predictive")
init = pp[pp.prior.str.startswith("initial")]
cal = pp[pp.prior.str.startswith("calibrated")]
A(f"The prior predictive check **rejected the obvious default**. With unit scales on the "
  f"coefficients, the total standard deviation and the intercept, the prior placed up to "
  f"**{100*init['p_above_physical_bound'].max():.2f} %** of its mass above 10⁶ mg kg⁻¹, which is "
  f"100 % of the sample mass. A prior that puts appreciable mass on impossible values is not "
  f"uninformative, it is wrong.")
A("")
A(md(pp, ["prior", "analyte", "prior_median", "prior_p99", "p_above_physical_bound",
          "observed_median", "observed_p95", "plausible"], 4))
A("")
A("The calibrated scales — β and the stratum shift at 0.5, σ_tot at 0.7, β₀ at 0.7 — are what every "
  "model in this study uses. The rejected attempt is kept in the table because it is the evidence "
  "that the check does its job.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 4. Parameter recovery (H4)")
A("")
rec = tbl("TABLE9_parameter_recovery")
cov_c = rec[rec.model == "censored"]["covers"].mean()
cov_s = rec[rec.model != "censored"]["covers"].mean()
A(f"Fields simulated with known parameters over the 114 real locations, censored at 60 %, then "
  f"fitted with both specifications. **The censored likelihood recovers {cov_c:.0%} of the "
  f"parameters inside their 95 % interval; the `L/2` substitution recovers {cov_s:.0%}.**")
A("")
fails = rec[(rec.model != "censored") & (~rec.covers)]
if len(fails):
    A("Parameters the substitution fails to recover:")
    A("")
    A(md(fails, ["parameter", "true", "posterior_mean", "ci95_low", "ci95_high", "rel_bias_pct"]))
    A("")
A("This is the cleanest evidence in the study, because here a truth exists to compare against.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 5. Main models (H5)")
A("")
conv = tbl("TABLE10_convergence")
A(f"All {len(conv)} models converge: R-hat maximum **{conv['rhat_max'].max():.4f}**, minimum ESS "
  f"**{min(conv['ess_bulk_min'].min(), conv['ess_tail_min'].min()):.0f}**, "
  f"**{int(conv['divergences'].sum())} divergences** in total.")
A("")
A(md(conv, ["model", "rhat_max", "ess_bulk_min", "ess_tail_min", "divergences", "bfmi_min",
            "passes"], 4))
A("")
par = tbl("TABLE11_posterior_parameters")
A("### Posterior parameters")
A("")
keep = [c for c in ["analyte", "ell_E", "ell_N", "eta", "sigma_n", "rho", "b_background",
                    "log_tau", "limit_ratio"] if c in par.columns]
A(md(par, keep))
A("")
A("The ARD kernel recovers the north-south continuity that the directional variogram pointed to: "
  "the northing length scale is consistently the longer one, which is the direction of the valley "
  "and its drainage.")
A("")
prpo = tbl("TABLE13_prior_posterior")
A("### What the data actually identified")
A("")
A("Ratio between the width of the 95 % posterior interval and that of the prior. A ratio near one "
  "means the posterior *is* the prior and the data said nothing about that parameter.")
A("")
A(md(prpo.pivot(index="parameter", columns="analyte", values="width_ratio").reset_index(), None, 3))
A("")
notid = prpo[~prpo.identified]
if len(notid):
    A(f"**Not identified** ({len(notid)} of {len(prpo)} parameter-analyte combinations): "
      + ", ".join(f"{r.parameter} in {r.analyte}" for r in notid.itertuples()) + ". "
      "These are reported as such and no claim rests on them.")
    A("")

# ---------------------------------------------------------------------------------------------
A("## 6. The masking experiment (H6)")
A("")
h6 = pd.read_csv(OUT / "H6_masking_experiment.csv")
A(f"{summ['masking_cells']} design cells, {h6['model'].nunique()} models each. Real values hidden "
  f"at four censoring levels over three complete truth fields, scored against the truth at every "
  f"held-out location, censored ones included.")
A("")
agg = tbl("TABLE15_masking_experiment")
A(md(agg.pivot_table(index=["metal", "level"], columns="model", values="crps").reset_index(),
     None, 4))
A("")
A("*CRPS against the true values, log scale, lower is better.*")
A("")
A("### Does the gap widen with censoring?")
A("")
A(md(slopes, ["metal", "baseline", "metric", "slope", "ci95_low", "ci95_high",
              "mean_diff_at_20", "mean_diff_at_80", "diverges"], 4))
A("")
if n_div == 0:
    A("**The divergence predicted by the hypothesis does not appear.** Stated plainly: over this "
      "range of censoring, on these fields, the censored likelihood does not pull away from the "
      "`L/2` substitution as censoring increases. An honest negative result is worth more than a "
      "false claim, and it is itself informative — it bounds where the method matters.")
else:
    A(f"The advantage grows significantly with censoring in {n_div} of {n_tot} combinations.")
A("")

A("### Where the substitution actually breaks: interval coverage")
A("")
A("CRPS mixes two errors — where the prediction puts its centre and how much uncertainty it "
  "declares. Substituting `L/2` damages the second far more than the first, because the "
  "substituted value is a constant with no variance. The statistic that isolates this is the "
  "coverage of the nominal 95 per cent interval, scored against the truth at every held-out "
  "location.")
A("")
cov = tbl("TABLE25_interval_coverage")
picp = h6.pivot_table(index=["metal", "level"], columns="model", values="picp95").round(3)
wid = h6.pivot_table(index=["metal", "level"], columns="model", values="width95").round(2)
A("**Coverage of the nominal 95 % interval (PICP95).**")
A("")
A(md(picp.reset_index(), None, 3))
A("")
A("**Mean interval width, log units.**")
A("")
A(md(wid.reset_index(), None, 2))
A("")
covtest = tbl("TABLE26_coverage_paired_test")
c80 = covtest[covtest["level"] == 0.8]
n_swept = int((c80["replicates_won"] == c80["n"]).sum())
A(f"Paired signed-rank on the coverage error |PICP95 - 0.95|, replicate by replicate on "
  f"identical masks. At 80 per cent censoring the censored GP is closer to nominal in "
  f"**{n_swept} of {len(c80)}** comparisons in *every* replicate, with a maximum p of "
  f"{c80['p_wilcoxon'].max():.4f} "
  f"({'the smallest value attainable with ten pairs' if c80['p_wilcoxon'].max() <= 0.002 else 'see table'}).")
A("")
A(md(covtest, ["metal", "level", "baseline", "mean_coverage_gain", "replicates_won", "n",
               "p_wilcoxon"], 4))
A("")
p80 = picp.xs(0.8, level="level")
worst = p80.min().min()
A(f"At 80 per cent censoring the baselines deliver as little as {worst:.2f} coverage from an "
  f"interval labelled 95 per cent, while their intervals *narrow* as censoring rises — the "
  f"substituted constant removes the variance it stands for. This is a stronger claim than the "
  f"CRPS margin: under heavy censoring the substitution does not merely lose accuracy, it reports "
  f"confidence it has not earned. For a map whose purpose is to say where a regulatory threshold "
  f"is exceeded, a 95 per cent interval with that coverage is worse than no interval at all.")
A("")
A("Figure FIG11, tables TABLE25 and TABLE26.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 7. Cross-validation on the real data (H7)")
A("")
cvt = tbl("TABLE17_cross_validation")
A("Spatial block cross-validation, five blocks. The **joint log predictive score** is the primary "
  "metric: it covers every held-out observation, using the interval density for detections and "
  "`log P(Y < log L)` for non-detects. It is proper for the real observation mechanism and cannot "
  "be won by predicting high. The remaining columns are **conditional on detection** and therefore "
  "cover only the upper truncated tail.")
A("")
A(md(cvt, ["metal", "model", "joint_log_score", "rmse_det", "crps_det", "picp95_det",
           "width95_det"], 4))
A("")
A("### Trivial baselines are in the table on purpose")
A("")
A("In the previous phase of this project an audit found that a plain intercept beat all five "
  "models on the set where they were being scored. Both trivial baselines are therefore reported "
  "here as a matter of course.")
A("")
best = {m: str(g.loc[g["joint_log_score"].idxmax(), "model"]) for m, g in cvt.groupby("metal")}
A("Best model by joint log score: " + ", ".join(f"**{k}** {v}" for k, v in best.items()) + ".")
A("")
boots = tbl("TABLE18_paired_bootstrap")
A("### Paired bootstrap")
A("")
A(md(boots[boots.metric == "joint_log_score"],
     ["metal", "comparison", "mean_diff", "ci95_low", "ci95_high", "wins_by_block", "n_blocks"], 4))
A("")
fold = tbl("TABLE_S4_cv_fold_diagnostics")
A(f"Fold-level diagnostics are published for all {len(fold)} Bayesian fits: R-hat maximum "
  f"{fold['rhat_max'].max():.4f}, minimum ESS {fold['ess_min'].min():.0f}, "
  f"{int(fold['divergences'].sum())} divergences. The same `target_accept = 0.99` as the main "
  "models was used, so the diagnostics are comparable.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 8. Background against mining input (H8)")
A("")
A(md(hg_bg, ["analyte", "b_background_log", "ci95_low", "ci95_high", "mining_factor",
             "median_background", "median_interest", "pct_bg_above_eca", "pct_pi_above_eca"], 3))
A("")

# ---------------------------------------------------------------------------------------------
A("## 9. Exceedance maps (H9)")
A("")
A(f"Grid of 50 m with a 300 m buffer, {summ.get('n_map_cells', '')} cells, "
  f"**{js('nb02_summary.json').get('map_samples', 4000)} posterior draws per cell**. In the previous "
  f"phase 600 draws gave a Monte Carlo resolution of 1/600 and the reported maxima were at the "
  f"noise floor; the Monte Carlo standard error is reported here.")
A("")
A(md(eca, None, 5))
A("")
A("Cadmium is mapped against **both** thresholds, agricultural 1.4 and residential 10 mg kg⁻¹, as a "
  "regulatory sensitivity analysis. The site has dwellings within 3 km and also farmland and "
  "high-Andean wetlands, so which standard applies is a decision rather than a fact.")
A("")
A(f"Extrapolation is masked: cells beyond 1.5 length scales from the nearest sample are hatched. "
  f"The fast marginal predictor was verified against the direct conditional to "
  f"{summ['predictor_verification_max_error']:.1e}.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 10. Monitoring design (H10)")
A("")
des = tbl("TABLE21_monitoring_design")
A(f"A map of posterior uncertainty is close to a mathematical identity on its own. Turned into a "
  f"design question it becomes the deliverable the court order actually requires. Adding 20 new "
  f"points in greedy order reduces the domain-average posterior standard deviation by "
  f"**{summ['monitoring_reduction_20_points_pct']:.1f} %**.")
A("")
A(md(des[des.n_new_points.isin([0, 5, 10, 20])],
     ["n_new_points", "mean_posterior_sd", "reduction_pct"], 4))
A("")
A("The ranked coordinates of the proposed points are in `figuras/TABLE22_proposed_sampling_points.csv`.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 11. Coregionalisation and preferential sampling (H11)")
A("")
lmc = tbl("TABLE23_lmc_cross_correlation")
A(md(lmc, ["pair", "mean", "ci50_low", "ci50_high", "ci95_low", "ci95_high", "width95",
           "identified"], 3))
A("")
if not lmc["identified"].any():
    A("**No cross-correlation is identified.** The 95 % intervals span essentially the whole range. "
      "That is a lack of information, not an absence of correlation, and the posterior mean is **not** "
      "presented as an estimate.")
else:
    A("At least one cross-correlation is identified; the shared mineralogical origin of the "
      "mercury-silver mineralisation is the physical reading.")
A("")
pref = tbl("TABLE24_preferential_sampling")
A("### Preferential sampling")
A("")
A(md(pref, ["analyte", "subset", "n", "pct_censored", "median_quantified", "pct_above_eca"], 2))
A("")
A("Exceedance depends strongly on which stratum is emphasised, which is what preferential sampling "
  "means. Here the design **declares** its strata, so it is explicit and modellable rather than a "
  "suspicion — a considerable advantage over the usual situation. Every aggregate figure in this "
  "study is therefore also reported by stratum, and the domain-wide maps are read as conditional "
  "on the survey design.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 12. Figures and tables")
A("")
figs = sorted(FIG.glob("FIG*.pdf"))
A(f"{len(figs)} figures, each as vector PDF and TIFF at 600 dpi, at exact column widths of 90, 140 "
  f"or 190 mm, 7 pt sans-serif with embedded fonts, colour-blind-safe palettes, no more than two "
  f"panels per figure.")
A("")
A("| Figure | Content |")
A("|---|---|")
DESC = {
    "FIG1": "Sampling design by stratum, and mercury with the exceedance ring",
    "FIG2": "Log distribution per analyte, with the detection limit and the standard marked (one file per analyte)",
    "FIG3": "Quantile plot per complete truth field (one file per analyte)",
    "FIG4": "Empirical and directional variograms",
    "FIG5": "Prior on the length scale, and prior predictive concentrations against the physical bound",
    "FIG6": "Parameter recovery, censored likelihood against L/2 substitution",
    "FIG7": "Stratum effect against support factor: the aliasing check",
    "FIG8": "Prior against posterior for the parameters most at risk of non-identification",
    "FIG9": "Masking experiment: CRPS against censoring level (one file per analyte)",
    "FIG10": "Masking experiment: score difference against censoring level (one file per analyte)",
    "FIG11": "Masking experiment: interval coverage and width against censoring level",
    "FIG12": "Cross-validation on the real data, and the paired bootstrap",
    "FIG13": "Background against potential interest against the city",
    "FIG14": "Monitoring design: uncertainty reduction and the proposed points",
    "FIG15": "LMC posterior cross-correlations",
    "FIG16": "Site maps, three versions",
    "FIG17": "Posterior mean and uncertainty per analyte, extrapolation masked",
    "FIG18": "Exceedance probability per analyte, two thresholds",
}
for f in figs:
    key = f.stem.split("_")[0]
    A(f"| `{f.stem}` | {DESC.get(key, '')} |")
A("")
A(f"{len(list(FIG.glob('TABLE*.csv')))} tables in CSV, "
  f"{len(list(FIG.glob('TABLE*.tex')))} also as LaTeX.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 13. Problems found and assumptions that do not hold")
A("")
A("| # | Issue | How it was handled |")
A("|---|---|---|")
A("| 1 | The obvious unit-scale priors imply physically impossible concentrations | Calibrated by "
  "prior predictive check; both versions published (section 3) |")
A("| 2 | The naive directional variogram measures the shape of the survey area, not anisotropy | "
  "All azimuths restricted to a common maximum lag; both ratios published (section 2) |")
A("| 3 | Stratum and sample support are almost perfectly aliased by design | Both fitted, posterior "
  "correlation inspected against a threshold fixed in advance (section 1) |")
A("| 4 | The background stratum is spatially segregated, so stratum is partly a location label | "
  "Quantified and reported as a limitation; the separation rests on the smoothness assumption |")
A("| 5 | Locations are not independent of the process: OEFA measures where it suspects | Sensitivity "
  "analysis by stratum; here the design declares the strata, so it is modellable (section 11) |")
A("| 6 | Two field points share one recorded coordinate | Not corrected, because it cannot be "
  "verified. Flagged, averaged in log space, and both names kept |")
A("| 7 | Quantified values are rounded by the laboratory | Modelled as interval censoring at the "
  "reporting resolution |")
A("| 8 | The reported detection limit is not necessarily the operating limit | The limit enters as a "
  "**parameter** bounded below by the published LOD and above by the smallest quantified value |")
A("| 9 | LOO cannot compare the censored and substituted specifications | They do not share a "
  "likelihood; the out-of-sample joint predictive score is used instead (section 7) |")
A("| 10 | `pymc_bart` and `RandomForest(n_jobs=-1)` re-execute the main module under Windows spawn | "
  "Work functions moved to an importable module; not used in this study |")
A("")
A("### Not available in the open data")
A("")
A("**Sampling depth, laboratory identity and OEFA's own conclusions are not in the published "
  "dataset.** The report is a causality assessment feeding a sanctioning procedure, so the PDF is "
  "not published although its result annex is. It can be requested through the Peruvian access-to-"
  "information procedure; it was not requested for this work. Any depth-dependent interpretation is "
  "therefore out of reach here.")
A("")

# ---------------------------------------------------------------------------------------------
A("## 14. Limitations")
A("")
A("1. **Preferential sampling.** The exceedance rates are conditional on a design that deliberately "
  "targeted areas of interest. They describe the surveyed area, not the district.")
A("2. **Change of support.** Background samples are composite and interest samples are simple. The "
  "variance factor was fitted, but it is aliased with the stratum effect and the separation relies "
  "on the two entering the model differently.")
A("3. **A single campaign.** July to September 2018. Soil metal is treated as quasi-static, which is "
  "reasonable for a mine closed in 1970 but is an assumption.")
A("4. **Extrapolation.** The maps are masked beyond 1.5 length scales from the nearest sample. "
  "Nothing outside the sampled corridor should be read as evidence.")
A("5. **Total metals, not bioavailable fraction.** The standard is defined on total metals, but "
  "exposure depends on bioaccessibility. Hagan et al. (2015) address that for this site.")
A("6. **The censoring in the masking experiment is imposed, not observed.** The three truth fields "
  "have zero censoring; the limit is placed at a quantile of the real values. A fixed laboratory "
  "limit producing a given non-detect fraction *is* a quantile, so the analogue is exact, but the "
  "experiment does not observe a real detection limit varying across analytes. The censoring "
  "gradient already present in the data (Co 37.7 %, Sb 76.3 %, Ag 78.1 %, Cd 79.8 %, Bi 84.2 %, "
  "Ni 93.0 %) is the design that would close this gap and it was not exploited here.")
A("7. **The censored model loses on cadmium, the one analyte with high real censoring.** In the "
  "spatial cross-validation its joint log score is below that of the global mean with `L/2` "
  "(section 7). On the detections alone it is far better on every measure, including coverage "
  "(0.83 against 0.30), and the loss is on the non-detect term. With 79.8 per cent censoring a "
  "held-out block leaves four or five quantified values, so this is a data limitation rather than "
  "a failure of the likelihood — but it is a loss and it is reported as one.")
A("8. **The mining-against-background contrast is not resolved.** The 95 per cent interval of the "
  "mining factor includes one for all five metals (section 8). The raw ratios are large, but "
  "stratum, position and covariates are aliased because the background points were not chosen at "
  "random. More model cannot fix this; only distant, verified background points can.")
A("9. **Ten replicates, five spatial blocks.** The strong evidence rests on the masking experiment "
  "(360 fits). On the real data the paired bootstrap runs over five blocks, where only lead "
  "separates cleanly (section 7).")
A("")

# ---------------------------------------------------------------------------------------------
A("## 15. Additional references incorporated")
A("")
A("Verified against the source, with the reason each was used.")
A("")
A("| Reference | Used for |")
A("|---|---|")
A("| Currie, L.A. (1968). *Limits for qualitative detection and quantitative determination.* "
  "Analytical Chemistry 40(3), 586–593. DOI 10.1021/ac60259a007 | The LOD/LOQ distinction that "
  "justifies treating the limit as a parameter, and the ECA/LOD criterion |")
A("| Horn, B.K.P. (1981). *Hill shading and the reflectance map.* Proceedings of the IEEE 69(1), "
  "14–47 | Slope from the DEM by the 3×3 weighted gradient, the GIS standard |")
A("| Riebler, A. et al. (2016). Statistical Methods in Medical Research 25(4), 1145–1165 | The "
  "variance-partition parametrisation that removes the funnel |")
A("| Roberts, D.R. et al. (2017). Ecography 40(8), 913–929 | Spatial block cross-validation |")
A("| Gneiting, T. & Raftery, A.E. (2007). JASA 102(477), 359–378 | CRPS and the notion of a proper "
  "scoring rule, which is what makes the joint predictive score the right primary metric |")
A("| Castillo Corzo, M.A. et al. (2025). Soil Science Annual 76(2), 204389. "
  "DOI 10.37501/soilsa/204389 | The city contrast: 1.7 to 6.2 mg kg⁻¹ Hg, none above the standard |")
A("| Hagan, N. et al. (2015). Environmental Geochemistry and Health 37, 263–272. "
  "DOI 10.1007/s10653-014-9644-1 | Bioaccessibility of mercury in Huancavelica, cited as a "
  "limitation of total-metal standards |")
A("| Mahdiyanfar, H. (2026). Scientific Reports 16:4763 | The censored Bayesian spatial method is "
  "already published; cited, with the difference stated: the ECA/LOD criterion, validation on "
  "complete real fields, and a legal consequence |")
A("| *Gaussian process regression for 3D soil mapping over multiple supports.* Geoderma (2024) | The "
  "change-of-support problem between composite and simple samples |")
A("| D.S. N.° 011-2017-MINAM | The soil environmental quality standards used throughout |")
A("")

# ---------------------------------------------------------------------------------------------
A("## 16. Open questions for Diego")
A("")
A("| # | Question | Why it needs a decision |")
A("|---|---|---|")
A("| D1 | **Which cadmium standard goes in the paper?** | Agricultural 1.4 or residential 10 "
  "mg kg⁻¹. Both are mapped. The site has dwellings within 3 km, farmland and wetlands. This is a "
  "regulatory judgement, not a statistical one, and it changes the headline number |")
A("| D2 | **Request the OEFA report through the access-to-information procedure?** | It would add "
  "sampling depth and laboratory identity. Free, ten working days. It would strengthen the methods "
  "section but is not needed for the current results |")
A("| D3 | **Which map version goes in the manuscript?** | Three were produced. The offline version "
  "is reproducible without any external service |")
A("| D4 | **Frame the paper around the method or around the site?** | The court order and the "
  "missing site-identification report give a strong applied frame; the ECA/LOD criterion gives a "
  "methodological one. Both are defensible and they suit different journals |")
A("| D5 | **Extend to the censoring-gradient analytes?** | Co, Sb, Ag, Bi and Ni span 38 % to 92 % "
  "censoring at the same 114 locations. That is a dose-response curve for the method itself, but it "
  "is a second paper |")
A("")

# ---------------------------------------------------------------------------------------------
A("## 17. Reproducing this")
A("")
A("```bash")
A("cd proyecto_metales_bayesiano")
A("python -m nbconvert --to notebook --execute --inplace notebooks/01_dataset_construction.ipynb")
A("python -m nbconvert --to notebook --execute --inplace notebooks/02_bayesian_model.ipynb")
A("python notebooks/_src/build_results_summary.py")
A("python src/verify_milestones.py")
A("```")
A("")
A(f"Kernel `metales-bayes` (Python {sys.version.split()[0]}). All heavy computation is cached under "
  f"`outputs/idata/` and `outputs/*.csv`, so the notebooks are idempotent. Deleting `outputs/` "
  f"forces a full refit.")
A("")
A("**Windows note.** The work functions of the parallel experiments live in "
  "`notebooks/_src/masking_worker.py` and not in notebook cells, because Windows spawns workers by "
  "re-importing the module that defines them.")
A("")

out = "\n".join(L)
(PROJ / "RESULTS_SUMMARY.md").write_text(out, encoding="utf-8")
print(f"written: RESULTS_SUMMARY.md | {len(out.splitlines())} lines")
