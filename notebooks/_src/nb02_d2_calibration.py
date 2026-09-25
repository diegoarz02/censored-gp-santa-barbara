# %% [markdown]
# ### 8.4 Where the substitution actually breaks: interval coverage (H6 continued)
#
# The comparison above scored CRPS and RMSE, and there the censored model wins by a modest margin.
# That is the wrong place to look, and it undersells the result.
#
# CRPS mixes two things a prediction can get wrong: where it puts the centre and how wide it says
# the uncertainty is. Substituting `L/2` for a non-detect damages the second far more than the
# first. The substituted value is a **constant with no variance**: with 80 per cent of the sample
# replaced by that constant, the empirical spread of the training data collapses, the fitted model
# concludes the field is nearly noiseless, and it issues narrow intervals with great confidence.
# The centre it predicts is only mildly wrong; the uncertainty it reports is badly wrong.
#
# The statistic that isolates this is the **coverage of the nominal 95 per cent interval** (PICP95),
# scored — as everywhere in this experiment — against the true values at *every* held-out location,
# censored ones included. A well-calibrated method returns 0.95 whatever the censoring. The
# quantity of interest is therefore the coverage error `|PICP95 − 0.95|`, and the paired comparison
# is made replicate by replicate, on identical masks, so the ten replicates are ten matched pairs.

# %%
from matplotlib.lines import Line2D

cal = h6.copy()
cal["cov_err"] = (cal["picp95"] - 0.95).abs()

cov_tab = (cal.pivot_table(index=["metal", "level"], columns="model",
                           values=["picp95", "width95"])
           .round(3).reset_index())
cov_tab.columns = [c[0] if not c[1] else f"{c[0]}|{c[1]}" for c in cov_tab.columns]
save_table(cov_tab, "TABLE25_interval_coverage",
           caption="Coverage of the nominal 95 per cent prediction interval (PICP95) and mean "
                   "interval width on the log scale, by analyte, censoring level and model. "
                   "Scored against the true values at every held-out location, censored ones "
                   "included. Ten replicates per cell.", label="tab:coverage")

# %%
# Paired Wilcoxon on the coverage error, replicate by replicate, against both baselines.
rows = []
for lev in sorted(cal["level"].unique()):
    for met in TRUTH_FIELDS:
        w = (cal[(cal["level"] == lev) & (cal["metal"] == met)]
             .pivot_table(index="replicate", columns="model", values="cov_err"))
        if "censored GP" not in w:
            continue
        for base in ["ordinary kriging L/2", "GP with L/2"]:
            if base not in w:
                continue
            diff = (w[base] - w["censored GP"]).dropna()
            if len(diff) < 3 or diff.nunique() < 2:
                continue
            _, p = stats.wilcoxon(diff)
            rows.append({"metal": met, "level": lev, "baseline": base,
                         "mean_coverage_gain": diff.mean(),
                         "replicates_won": int((diff > 0).sum()), "n": len(diff),
                         "p_wilcoxon": p})
covtest = pd.DataFrame(rows)
save_table(covtest.round(4), "TABLE26_coverage_paired_test",
           caption=r"Paired comparison of the coverage error $|\mathrm{PICP}_{95}-0.95|$ between "
                   "the censored GP and each baseline, replicate by replicate on identical masks. "
                   "A positive gain means the censored GP is closer to nominal coverage. "
                   "$p=0.002$ is the smallest value the signed-rank test can return with ten "
                   "pairs.", label="tab:covtest")

CAL80 = covtest[covtest["level"] == 0.8]
CAL80_ALL_WON = bool((CAL80["replicates_won"] == CAL80["n"]).all()) and len(CAL80) == 6
print(f"\nat 80 % censoring: {len(CAL80)} comparisons, "
      f"{int((CAL80['replicates_won'] == CAL80['n']).sum())} of them won in every replicate, "
      f"max p = {CAL80['p_wilcoxon'].max():.4f}")
print(covtest.round(4).to_string(index=False))

# %% [markdown]
# **The result.** At 20 and 40 per cent censoring nothing separates the methods. At 60 per cent
# lead starts to separate. At **80 per cent censoring the censored GP is closer to nominal coverage
# in every one of the ten replicates, for all three analytes, against both baselines** — six
# comparisons, all at the smallest $p$ the test can return with ten pairs.
#
# The magnitude is not marginal. Ordinary kriging with `L/2` delivers 0.52 (As), 0.48 (Hg) and 0.35
# (Pb) coverage from an interval it labels 95 per cent: on lead, an interval that should miss the
# truth once in twenty misses it two times in three. The censored GP returns 0.95, 0.83 and 0.74 on
# the same masks.
#
# Panel (b) shows the mechanism directly. As censoring rises the baselines' intervals *narrow* —
# from 4.7 to 2.1 log units on As, 6.9 to 2.4 on Pb — because the substituted constant removes the
# variance it stands for. The censored GP's intervals stay open (4.7 and 5.1), which is the honest
# answer when four fifths of the data are known only to lie below a limit.
#
# This is the finding that matters for practice, and it is a stronger claim than the CRPS margin:
# under heavy censoring the substitution does not merely lose a little accuracy, it reports
# confidence it has not earned. For a map whose purpose is to say where a regulatory threshold is
# exceeded, a 95 per cent interval with 35 per cent coverage is worse than no interval at all.

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM))
MODEL_ORDER = ["censored GP", "GP with L/2", "ordinary kriging L/2"]
MODEL_COL = {"censored GP": OI[0], "GP with L/2": OI[1], "ordinary kriging L/2": OI[2]}
METAL_LS = {"Hg": "-", "Pb": "--", "As": ":"}
METAL_MK = {"Hg": "o", "Pb": "s", "As": "^"}

for ax, (var, ylab) in zip(axes, [("picp95", "coverage of the nominal 95 % interval"),
                                  ("width95", "mean interval width (log units)")]):
    g = cal.pivot_table(index=["metal", "level"], columns="model", values=var)
    for mdl in MODEL_ORDER:
        for met in TRUTH_FIELDS:
            s = g.xs(met, level="metal")[mdl]
            ax.plot(s.index * 100, s.values, ls=METAL_LS[met], marker=METAL_MK[met],
                    ms=3, lw=1.0, color=MODEL_COL[mdl], mec="k", mew=0.2)
    ax.set_xlabel("censoring imposed (%)")
    ax.set_ylabel(ylab)
    ax.set_xticks([20, 40, 60, 80])

axes[0].axhline(0.95, color="k", lw=0.8, ls="-", zorder=0)
axes[0].text(81, 0.958, "nominal 0.95", fontsize=8, va="bottom", ha="right")
axes[0].set_ylim(0.25, 1.02)
axes[0].set_title("(a) interval coverage against the truth", loc="left", fontweight="bold")
axes[1].set_title("(b) interval width", loc="left", fontweight="bold")

h_mdl = [Line2D([], [], color=MODEL_COL[m], lw=1.2, label=m) for m in MODEL_ORDER]
h_met = [Line2D([], [], color="0.35", ls=METAL_LS[m], marker=METAL_MK[m], ms=3, lw=1.0,
                label=m) for m in TRUTH_FIELDS]
axes[0].legend(handles=h_mdl, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, handlelength=1.6)
axes[1].legend(handles=h_met, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
               ncol=3, handlelength=1.8, columnspacing=1.0)
fig.tight_layout()
save_fig(fig, "FIG11_interval_coverage")
