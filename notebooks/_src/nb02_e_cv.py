
# %% [markdown]
# ## 9. Spatial cross-validation on the real data (H7)
#
# The masking experiment of section 8 measures skill against a known truth. This section measures
# it on the data as they actually arrive, censoring included, and it is the honest complement: here
# there is no truth at the censored locations, so the design has to be chosen so that the scores
# still mean something.
#
# **Why only one classical baseline.** A reviewer will ask why there is no comparison against
# machine learning. The answer is deliberate: the claim of this paper is about the **treatment of
# censoring**, not about the model class. Ordinary kriging with `L/2` substitution is exactly the
# practice being criticised, and the GP with `L/2` isolates the censoring treatment as the only
# difference — same model class, same priors, same sampler. Adding a random forest would vary two
# things at once and answer a question nobody asked. The previous phase of this project ran five
# models and the audit showed the comparison measured predicted level rather than skill.
#
# **What the audit forces into this section**
#
# * **Trivial baselines in the table.** An intercept fitted to the training detections, and the
#   global mean of the `L/2`-substituted values. Without these the table misleads: in the previous
#   phase the intercept beat all five models on the set where they were scored.
# * **A proper joint predictive score as the primary metric**, over every held-out observation:
#   the interval density for detections and `log P(Y < log L)` for non-detects. It is proper for the
#   real observation mechanism and cannot be won by predicting high.
# * RMSE, MAE, CRPS and PICP over detections are reported but **labelled as conditional on
#   detection**, because that subset is the upper truncated tail.
# * `target_accept = 0.99` in the cross-validation fits too, with a supplementary table of R-hat,
#   ESS and divergences for **every fold**.
# * **Paired bootstrap** on the held-out points for every metric difference, plus the per-block win
#   count.
# * **LOO** between the Bayesian specifications.

# %%
from verde import BlockKFold

CV_METALS = ["Hg", "Pb", "As", "Cd"]
K_BLOCKS = 5
block_labels = sb.spatial_blocks(XY, k=K_BLOCKS, seed=SEED)
print("block sizes:", np.bincount(block_labels))
print("Blocks come from k-means on the coordinates. `verde.BlockKFold` was also available; k-means\n"
      "was kept because it is the partition used by the masking experiment, so the two sections\n"
      "share one design.")

for b in range(K_BLOCKS):
    m = block_labels == b
    print(f"  block {b}: n={m.sum():3d}, background={int(IS_BACKGROUND[m].sum()):2d}, "
          f"extent {np.ptp(XY[m,0]):.2f} x {np.ptp(XY[m,1]):.2f} km")

# %%
def trivial_predictions(y_train_log, n_test, n_samples=800, seed=0):
    """Intercept baseline: the mean and spread of the training log-detections."""
    r = np.random.default_rng(seed)
    m, s = float(np.nanmean(y_train_log)), float(np.nanstd(y_train_log))
    return m + r.standard_normal((n_samples, n_test)) * s


def score_fold(samples_log, value, cens, te, lod):
    """Scores for one held-out block.

    Two families, deliberately kept apart:

    * the **joint predictive score** over every held-out observation, which is proper for the real
      observation mechanism;
    * the point and interval scores over the **detections only**, which are reported for continuity
      with the literature but are conditional on detection and therefore restricted to the upper
      truncated tail.
    """
    from scipy import stats as sps
    y_true = np.where(cens[te], np.nan, np.log(np.where(cens[te], 1.0, value[te])))
    mu, sd = samples_log.mean(0), samples_log.std(0)
    det = ~cens[te]

    ll = np.empty(int(te.sum()))
    ll[det] = sps.norm.logpdf(y_true[det], mu[det], sd[det])
    if (~det).any():
        ll[~det] = sps.norm.logcdf(np.log(lod), mu[~det], sd[~det])
    out = {"n_test": int(te.sum()), "n_detected": int(det.sum()),
           "joint_log_score": float(ll.mean())}

    if det.any():
        s = samples_log[:, det]
        yd = y_true[det]
        out.update({
            "rmse_det": float(np.sqrt(np.mean((s.mean(0) - yd) ** 2))),
            "mae_det": float(np.mean(np.abs(s.mean(0) - yd))),
            "crps_det": float(sb.crps_samples(yd, s).mean()),
        })
        for lvl in (0.90, 0.95):
            lo = np.quantile(s, (1 - lvl) / 2, axis=0)
            hi = np.quantile(s, 1 - (1 - lvl) / 2, axis=0)
            out[f"picp{int(lvl*100)}_det"] = float(np.mean((yd >= lo) & (yd <= hi)))
            out[f"width{int(lvl*100)}_det"] = float(np.mean(hi - lo))
    return out


# %% [markdown]
# The 40 Bayesian fits of this section run at the same sampler settings as the main models
# (`target_accept = 0.99`, 4 chains, 2 500 warm-up, 2 000 draws), which in series would take some
# three hours. The folds are independent, so the same process-based parallelism as the masking
# experiment applies, through the same worker module.

# %%
CV_PATH = DIR_OUT / "cv_real_data.csv"

if CV_PATH.exists():
    cv = pd.read_csv(CV_PATH)
    print(f"[cache] loaded {len(cv)} cross-validation rows")
else:
    from joblib import Parallel, delayed

    jobs = [{"metal": m, "block": b, "model": mdl}
            for m in CV_METALS for b in range(K_BLOCKS)
            for mdl in ["censored GP", "GP with L/2"]]
    n_jobs_cv = max(1, N_CPU // 4)          # 4 chains inside each fit -> N_CPU processes
    print(f"{len(jobs)} Bayesian fits, {n_jobs_cv} in parallel with 4 chains each")
    t0 = time.time()
    fits = Parallel(n_jobs=n_jobs_cv, backend="loky", verbose=5)(
        delayed(mw.run_cv_fold)(j, k_blocks=K_BLOCKS, seed0=SEED, cores=4) for j in jobs)
    print(f"[run] cross-validation fits in {(time.time()-t0)/60:.1f} min")

    rows = []
    for f in fits:
        m = metal_arrays(f["metal"])
        te = block_labels == f["block"]
        rows.append({"metal": f["metal"], "block": f["block"], "model": f["model"],
                     **score_fold(f["samples"], m["value"], m["censored"], te, f["lod"]),
                     "rhat_max": f["rhat_max"], "ess_min": f["ess_min"],
                     "divergences": f["divergences"]})

    # classical and trivial baselines: no MCMC, so they run here directly
    for metal in CV_METALS:
        m = metal_arrays(metal)
        value, cens = m["value"], m["censored"]
        lod = m["lod"] if np.isfinite(m["lod"]) else float(np.nanmin(value))
        y_log = np.where(cens, np.nan, np.log(np.where(cens, 1.0, value)))
        y_sub = np.where(cens, np.log(lod / 2.0), np.log(np.where(cens, 1.0, value)))
        for b in range(K_BLOCKS):
            te = block_labels == b
            tr = ~te
            d_, _ = sb.ordinary_kriging(XY[tr], y_sub[tr], XY[te], n_samples=1000, seed=SEED + b)
            rows.append({"metal": metal, "block": b, "model": "ordinary kriging L/2",
                         **score_fold(d_, value, cens, te, lod),
                         "rhat_max": np.nan, "ess_min": np.nan, "divergences": np.nan})
            d_ = trivial_predictions(y_log[tr], int(te.sum()), seed=SEED + b)
            rows.append({"metal": metal, "block": b, "model": "intercept (detections)",
                         **score_fold(d_, value, cens, te, lod),
                         "rhat_max": np.nan, "ess_min": np.nan, "divergences": np.nan})
            d_ = trivial_predictions(y_sub[tr], int(te.sum()), seed=SEED + b)
            rows.append({"metal": metal, "block": b, "model": "global mean with L/2",
                         **score_fold(d_, value, cens, te, lod),
                         "rhat_max": np.nan, "ess_min": np.nan, "divergences": np.nan})

    cv = pd.DataFrame(rows)
    cv.to_csv(CV_PATH, index=False, encoding="utf-8")

print("\nCV rows:", len(cv))

# %% [markdown]
# ### 9.1 Fold-level convergence
#
# The audit of the previous phase found 39 undiagnosed divergences across the cross-validation
# fits while the report claimed "zero divergences". Every fold is published here.

# %%
cvdiag = cv[cv.model.str.contains("GP")][["metal", "block", "model", "rhat_max", "ess_min",
                                          "divergences"]].round(4)
print(cvdiag.to_string(index=False))
save_table(cvdiag, "TABLE_S4_cv_fold_diagnostics")
print(f"\nR-hat max across folds {cvdiag['rhat_max'].max():.4f} | "
      f"ESS min {cvdiag['ess_min'].min():.0f} | total divergences {int(cvdiag['divergences'].sum())}")

# %% [markdown]
# ### 9.2 Results

# %%
CV_MODELS = ["intercept (detections)", "global mean with L/2", "ordinary kriging L/2",
             "GP with L/2", "censored GP"]


def weighted(g, col, w):
    v, wt = g[col].to_numpy(float), g[w].to_numpy(float)
    ok = np.isfinite(v)
    return float(np.sum(v[ok] * wt[ok]) / np.sum(wt[ok])) if ok.any() else np.nan


rows = []
for (metal, model), g in cv.groupby(["metal", "model"]):
    rows.append({
        "metal": metal, "model": model,
        "joint_log_score": weighted(g, "joint_log_score", "n_test"),
        "rmse_det": float(np.sqrt(np.sum(g["n_detected"] * g["rmse_det"] ** 2) / g["n_detected"].sum())),
        "mae_det": weighted(g, "mae_det", "n_detected"),
        "crps_det": weighted(g, "crps_det", "n_detected"),
        "picp95_det": weighted(g, "picp95_det", "n_detected"),
        "width95_det": weighted(g, "width95_det", "n_detected"),
    })
cvsum = pd.DataFrame(rows)
cvsum["model"] = pd.Categorical(cvsum["model"], CV_MODELS, ordered=True)
cvsum = cvsum.sort_values(["metal", "model"]).round(4)
print(cvsum.to_string(index=False))
save_table(cvsum, "TABLE17_cross_validation",
           caption="Spatial block cross-validation on the real data. The joint log score covers "
                   "every held-out observation and is the primary metric; the remaining columns are "
                   "conditional on detection and cover only the upper truncated tail. Higher is "
                   "better for the joint score, lower for the rest.",
           label="tab:cv")

print("\nbest model by joint log score (higher is better):")
for metal, g in cvsum.groupby("metal"):
    b = g.loc[g["joint_log_score"].idxmax()]
    print(f"  {metal}: {b['model']} ({b['joint_log_score']:.4f})")

# %% [markdown]
# ### 9.3 Paired bootstrap and per-block wins
#
# With five blocks of 15 to 35 points, a difference of means without an interval says nothing.

# %%
def paired_bootstrap(cv, metal, model_a, model_b, metric, n_boot=5000, seed=SEED):
    a = cv[(cv.metal == metal) & (cv.model == model_a)].set_index("block")[metric]
    b = cv[(cv.metal == metal) & (cv.model == model_b)].set_index("block")[metric]
    n = cv[(cv.metal == metal) & (cv.model == model_a)].set_index("block")["n_test"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    w = n.loc[common].to_numpy(float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(n_boot, len(d)))
    boot = np.array([np.average(d[i], weights=w[i]) for i in idx])
    return {"metal": metal, "metric": metric, "comparison": f"{model_a} - {model_b}",
            "mean_diff": float(np.average(d, weights=w)),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "p_positive": float((boot > 0).mean()),
            "wins_by_block": int((d > 0).sum()), "n_blocks": len(d)}


boot_rows = []
for metal in CV_METALS:
    for other in ["ordinary kriging L/2", "GP with L/2", "intercept (detections)"]:
        boot_rows.append(paired_bootstrap(cv, metal, "censored GP", other, "joint_log_score"))
        boot_rows.append(paired_bootstrap(cv, metal, other, "censored GP", "crps_det"))
boots = pd.DataFrame(boot_rows).round(4)
print(boots.to_string(index=False))
save_table(boots, "TABLE18_paired_bootstrap",
           caption="Paired bootstrap over held-out blocks. For the joint log score a positive "
                   "difference favours the censored GP; for CRPS the comparison is written so that "
                   "a positive difference also favours it.", label="tab:bootstrap")

# %% [markdown]
# ### 9.4 LOO between the Bayesian specifications

# %%
# LOO is computed where it is well defined: among the main models of section 7, which share the
# censored likelihood and therefore share the observations their pointwise log-likelihood refers to.
loo_rows = []
for metal in MAIN_METALS:
    try:
        idt = idata_main[metal]
        if "log_likelihood" in idt.groups():
            l = az.loo(idt)
            loo_rows.append({"metal": metal, "elpd_loo": float(l.elpd_loo),
                             "se": float(l.se), "p_loo": float(l.p_loo)})
        else:
            loo_rows.append({"metal": metal, "elpd_loo": np.nan, "se": np.nan, "p_loo": np.nan})
    except Exception as exc:                                            # noqa: BLE001
        loo_rows.append({"metal": metal, "note": f"{type(exc).__name__}: {exc}"[:110]})
loo = pd.DataFrame(loo_rows)
print(loo.to_string(index=False))
save_table(loo.round(4), "TABLE_S5_loo")

print("\nWhy LOO does not settle the comparison of this paper.\n"
      "The censored GP and the L/2 substitution **do not share a likelihood**: one integrates the\n"
      "censored observations, the other replaces them with a number and treats it as measured. Their\n"
      "pointwise log-likelihoods are therefore defined on different data, and `az.compare` would be\n"
      "ranking models fitted to different datasets. That is a category error, not a technicality.\n"
      "\n"
      "The primary comparison is the out-of-sample **joint predictive score** of section 9.2, which\n"
      "is well defined for both because it scores the same held-out observations under the same\n"
      "observation mechanism. LOO is reported above only among the censored specifications, where\n"
      "it is meaningful, and the models were fitted without pointwise log-likelihood storage to keep\n"
      "the traces small, so the column is empty unless that is switched on.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.36), constrained_layout=True)

ax = axes[0]
w = 0.15
for k, mdl in enumerate(CV_MODELS):
    s = cvsum[cvsum.model == mdl].set_index("metal").loc[CV_METALS]
    ax.bar(np.arange(len(CV_METALS)) + (k - 2) * w, s["joint_log_score"], w,
           color=OI[k], edgecolor="k", linewidth=0.3, label=mdl)
ax.set_xticks(np.arange(len(CV_METALS)))
ax.set_xticklabels(CV_METALS)
ax.set_ylabel("Joint log predictive score (higher is better)")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", fontsize=8, ncol=1)
ax.text(0.97, 0.05, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

ax = axes[1]
s = boots[(boots.metric == "joint_log_score")]
ypos = np.arange(len(s))[::-1]
ax.errorbar(s["mean_diff"], ypos,
            xerr=[s["mean_diff"] - s["ci95_low"], s["ci95_high"] - s["mean_diff"]],
            fmt="o", ms=3, lw=0.9, capsize=2, color=OI[0], mec="k", mew=0.25)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.metal}: vs {r.comparison.split(' - ')[1][:18]}" for r in s.itertuples()],
                   fontsize=8)
ax.set_xlabel("Joint log score difference, censored GP minus baseline")
ax.text(0.97, 0.05, "(b)", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
save_fig(fig, "FIG12_cross_validation")
plt.show()
