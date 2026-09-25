
# %% [markdown]
# ## 8. The masking experiment (H6)
#
# This is the experiment the paper rests on.
#
# **The problem it solves.** On real censored data there is no truth to compare against at the
# censored locations, so any comparison of methods has to be made on the detections alone — and
# that subset is the upper truncated tail, where whichever model predicts higher wins
# automatically. The audit of the previous phase showed exactly this: five models were all about
# three times worse than a plain intercept on the set where they were being scored.
#
# **The design.** Three analytes measured here with **zero censoring** are used as complete truth
# fields. Real values are *hidden*, not simulated:
#
# 1. take the real measurements at the 114 locations;
# 2. left-censor at level `c` by putting the limit at the `c` quantile of those real values;
# 3. hold out one whole spatial block and fit on the rest;
# 4. score against the true values at **every** held-out location, censored ones included.
#
# Censoring at a quantile is the correct analogue of the real mechanism: the quantile *is* a fixed
# number, and a fixed laboratory limit producing `c` per cent non-detects is precisely what it
# represents. This is masking, not fabrication — the same logic as cross-validation, which also
# hides real observations to see whether a model recovers them.
#
# **Why three analytes and not two or four.** The analytes are not replicates: each is a field with
# its own spatial structure, and that is exactly the axis that has to vary to know whether a
# conclusion is general or an artefact of one field. With two, a pattern cannot be told from a
# coincidence; the third turns a pair into a trend. A fourth (Ba) would have cost about a quarter of
# the compute while adding little: its log standard deviation falls between the others, its
# exceedance is low and its threshold of 750 mg kg⁻¹ sits so far from any detection limit that it
# says nothing about the claim. Ba stays in the main models and in the maps.
#
# The three chosen span the space of structures: **Hg** (sd log 1.81, the mine's defining metal),
# **Pb** (sd log 1.80 but far heavier tails — maximum 58 000 against a median of 383, a factor of
# 150, which stresses robustness to extremes) and **As** (sd log 1.11, the weakest signal).

# %%
import masking_worker as mw

LEVELS = [0.20, 0.40, 0.60, 0.80]
N_REPLICATES = 10          # reduced from 20 after timing the machine; see the note below
N_BLOCKS = 5

design = mw.make_design(TRUTH_FIELDS, LEVELS, N_REPLICATES, n_blocks=N_BLOCKS)
print(f"design: {len(TRUTH_FIELDS)} analytes x {len(LEVELS)} levels x {N_REPLICATES} replicates "
      f"= {len(design)} cells")
print(f"each cell fits 3 models (ordinary kriging, censored GP, GP with L/2) "
      f"-> {len(design)*3} model fits")
print("\nReplicates were cut from 20 to 10 after measuring this machine: a single censored "
      "GP fit over ~90 training points takes about 27 s with two chains, and the full "
      "design at 20 replicates would not finish in a working session. The study plan fixed "
      "replicates as the first thing to cut for exactly this reason, ahead of touching the "
      "analytes, the levels or the validation of the cheap inference.")

# %% [markdown]
# ### 8.1 Parallelism
#
# The loop is embarrassingly parallel: every (analyte, level, replicate) cell is independent. Three
# constraints shape how it is run.
#
# 1. **`asyncio` is the wrong tool.** This is CPU-bound work, not I/O-bound, and the GIL blocks it.
#    It has to be processes, not coroutines or threads.
# 2. **Windows uses *spawn*.** Every worker re-imports the module holding the work function, so that
#    function lives in `masking_worker.py` and not in a notebook cell, and the launch is guarded.
#    The previous phase of this project hit this exact failure with `RandomForest(n_jobs=-1)`.
# 3. **Cores against chains.** NUTS already parallelises *within* a fit, one chain per process.
#    Launching `n` fits in parallel with 4 chains each asks for `4n` processes and the machine
#    thrashes. The configuration used is **2 chains per fit**, `cores=1` inside the fit, and
#    `n_jobs = cpu_count // 2` fits in parallel. Each worker also sets the BLAS thread count to 1,
#    without which every process spawns its own thread pool.

# %%
N_JOBS = max(1, N_CPU // 2)
print(f"detected cores: {N_CPU} -> {N_JOBS} cells in parallel, 2 chains each, "
      f"so {N_JOBS*2} processes on {N_CPU} cores")
print("Measured on this machine: one fit takes 52 s with sequential chains and 27 s with "
      "two, so the two-chain configuration is what makes the experiment affordable.")

MASK_KW = dict(draws=500, tune=500, chains=2, cores=2, target_accept=0.95,
               n_pred=800, n_blocks=N_BLOCKS)
print("inference inside the experiment:", MASK_KW)
print("\nThe experiment compares *methods against each other* under an identical budget, so a\n"
      "consistent cheaper approximation is legitimate. Section 8.3 validates that shortcut against\n"
      "the full sampler on a subset of cells; without that validation the shortcut would not be\n"
      "defensible.")

# %%
H6_PATH = DIR_OUT / "H6_masking_experiment.csv"

if H6_PATH.exists():
    h6 = pd.read_csv(H6_PATH)
    print(f"[cache] loaded {len(h6)} rows from {H6_PATH.name}")
else:
    from joblib import Parallel, delayed
    # Each worker writes its cell to outputs/h6_cells/ as soon as it finishes, so an interrupted
    # run resumes rather than repeating the experiment. Count what is already there before starting.
    done_before = len(list((DIR_OUT / "h6_cells").glob("*.json"))) if (DIR_OUT / "h6_cells").exists() else 0
    if done_before:
        print(f"[resume] {done_before} of {len(design)} cells already checkpointed")
    t0 = time.time()
    chunks = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(mw.run_cell)(c, **MASK_KW) for c in design)
    h6 = pd.DataFrame([r for ch in chunks for r in ch])
    h6.to_csv(H6_PATH, index=False, encoding="utf-8")
    print(f"[run] {len(design)} cells ({len(design)-done_before} newly fitted) in "
          f"{(time.time()-t0)/60:.1f} min -> {len(h6)} rows")

n_err = int(h6["error"].notna().sum()) if "error" in h6.columns else 0
print(f"rows: {len(h6)} | failed fits: {n_err}")
if n_err:
    print(h6[h6["error"].notna()][["metal", "level", "replicate", "model", "error"]].head().to_string(index=False))
    h6 = h6[h6["error"].isna()].copy()

print("\nconvergence inside the experiment (Bayesian models only):")
gp = h6[h6.model.str.contains("GP")]
print(f"  R-hat max {gp['rhat_max'].max():.4f} | ESS min {gp['ess_min'].min():.0f} "
      f"| total divergences {int(gp['divergences'].sum())} | "
      f"cells with divergences {int((gp['divergences'] > 0).sum())} of {len(gp)}")

# %% [markdown]
# ### 8.2 Results

# %%
MODELS = ["ordinary kriging L/2", "GP with L/2", "censored GP"]
agg = (h6.groupby(["metal", "level", "model"])
       .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
            mae=("mae", "mean"), crps=("crps", "mean"), crps_sd=("crps", "std"),
            picp95=("picp95", "mean"), width95=("width95", "mean"),
            joint=("joint_log_score", "mean"), n=("rmse", "size"))
       .reset_index())
piv = agg.pivot_table(index=["metal", "level"], columns="model", values=["rmse", "crps"])
print(piv.round(4).to_string())
save_table(agg.round(5), "TABLE15_masking_experiment",
           caption="Masking experiment. Predictive scores against the true values at every held-out "
                   "location, censored ones included, averaged over the replicates. Log scale.",
           label="tab:masking")

# %% [markdown]
# ### The key quantity: does the gap widen with censoring?
#
# The hypothesis is not "the censored model is better"; it is that **the advantage grows with the
# censoring rate**. That is a statement about a slope, so a slope is what gets fitted, with its
# interval, for each analyte and each competing model.

# %%
from scipy import stats as sps

slope_rows = []
for metal in TRUTH_FIELDS:
    base = h6[(h6.metal == metal) & (h6.model == "censored GP")].set_index(["level", "replicate"])
    for other in ["ordinary kriging L/2", "GP with L/2"]:
        comp = h6[(h6.metal == metal) & (h6.model == other)].set_index(["level", "replicate"])
        common = base.index.intersection(comp.index)
        for metric in ["rmse", "crps"]:
            diff = (comp.loc[common, metric] - base.loc[common, metric]).to_numpy(float)
            lev = np.array([i[0] for i in common], float)
            lr = sps.linregress(lev, diff)
            slope_rows.append({
                "metal": metal, "baseline": other, "metric": metric,
                "slope": lr.slope, "slope_se": lr.stderr,
                "ci95_low": lr.slope - 1.96 * lr.stderr, "ci95_high": lr.slope + 1.96 * lr.stderr,
                "p_value": lr.pvalue,
                "mean_diff_at_20": float(diff[lev == 0.20].mean()),
                "mean_diff_at_80": float(diff[lev == 0.80].mean()),
                "diverges": bool(lr.slope > 0 and lr.slope - 1.96 * lr.stderr > 0)})
slopes = pd.DataFrame(slope_rows).round(5)
print(slopes.to_string(index=False))
save_table(slopes, "TABLE16_divergence_slopes",
           caption="Slope of the score difference (baseline minus censored GP) against the "
                   "censoring level. A positive slope whose interval excludes zero means the "
                   "advantage of the censored model grows with censoring.",
           label="tab:slopes")

n_div = int(slopes["diverges"].sum())
print(f"\ncells where the advantage grows significantly with censoring: {n_div} of {len(slopes)}")
if n_div == 0:
    print("\nThe divergence predicted by the hypothesis DOES NOT APPEAR. This is stated plainly and\n"
          "the conclusion is rewritten accordingly: an honest negative result is worth more than a\n"
          "false claim.")
else:
    print(f"\nThe advantage of the censored model grows with the censoring rate in {n_div} of the\n"
          f"{len(slopes)} analyte-baseline-metric combinations tested.")

# %%
# One analyte per file: at most two panels per figure, so each field gets its own.
for metal in TRUTH_FIELDS:
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.85), constrained_layout=True)
    for k, mdl in enumerate(MODELS):
        s = h6[(h6.metal == metal) & (h6.model == mdl)]
        g = s.groupby("level")["crps"]
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        ax.plot(m.index, m.values, "-o", ms=3, lw=1.0, color=OI[k], mec="k", mew=0.2, label=mdl)
        ax.fill_between(m.index, lo.values, hi.values, color=OI[k], alpha=0.15, lw=0)
    ax.set_xlabel("Artificial censoring level")
    ax.set_xticks(LEVELS)
    ax.set_ylabel("CRPS against the true values (log mg kg$^{-1}$)")
    sd_log = float(np.std(np.log(df[df.analyte == metal]["value"].dropna())))
    p = idata_main[metal].posterior
    ell_m = p["ell"].values.reshape(-1, 2).mean(0)
    rho_m = float(p["rho"].mean())
    ax.text(0.03, 0.97, f"{metal}\n"
                        f"sd$_{{\\log}}$ = {sd_log:.2f}\n"
                        f"$\\ell$ = {ell_m[0]:.2f}, {ell_m[1]:.2f} km\n"
                        f"$\\rho$ = {rho_m:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, fontsize=8)
    save_fig(fig, f"FIG9_{metal}_crps")
plt.show()

# %%
for metal in TRUTH_FIELDS:
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.85), constrained_layout=True)
    for k, other in enumerate(["ordinary kriging L/2", "GP with L/2"]):
        base = h6[(h6.metal == metal) & (h6.model == "censored GP")].set_index(["level", "replicate"])
        comp = h6[(h6.metal == metal) & (h6.model == other)].set_index(["level", "replicate"])
        common = base.index.intersection(comp.index)
        d_ = (comp.loc[common, "crps"] - base.loc[common, "crps"])
        lev = np.array([i[0] for i in common], float)
        mm = pd.Series(d_.to_numpy(), index=lev).groupby(level=0)
        ax.errorbar(mm.mean().index, mm.mean().values,
                    yerr=1.96 * mm.std().values / np.sqrt(mm.count().values),
                    fmt="-o", ms=3, lw=1.0, capsize=2, color=OI[k], mec="k", mew=0.2,
                    label=f"{other} minus censored GP")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Artificial censoring level")
    ax.set_xticks(LEVELS)
    ax.set_ylabel("CRPS difference (positive favours the censored GP)")
    ax.text(0.03, 0.97, metal, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, fontsize=8)
    save_fig(fig, f"FIG10_{metal}_difference")
plt.show()

# %% [markdown]
# ### 8.3 Validating the cheap inference
#
# The experiment runs 2 chains with 500 warm-up and 500 sampling draws instead of the 4 chains and
# 2 500/2 000 of the main models. That shortcut has to be checked, not assumed: a subset of cells is
# refitted with the full sampler and the conclusions compared. Without this the shortcut is not
# defensible.

# %%
H6_FULL_PATH = DIR_OUT / "H6_full_nuts_validation.csv"
N_VALIDATE = 12

if H6_FULL_PATH.exists():
    h6_full = pd.read_csv(H6_FULL_PATH)
    print(f"[cache] loaded {len(h6_full)} rows")
else:
    from joblib import Parallel, delayed
    r_ = np.random.default_rng(SEED)
    # stratified over analyte and level so the validation covers the whole design
    sub = []
    for metal in TRUTH_FIELDS:
        for lv in LEVELS:
            pool = [c for c in design if c["metal"] == metal and c["level"] == lv]
            sub += [pool[i] for i in r_.choice(len(pool), 1, replace=False)]
    t0 = time.time()
    chunks = Parallel(n_jobs=max(1, N_CPU // 4), backend="loky", verbose=5)(
        delayed(mw.run_cell)(c, full_nuts=True, n_pred=800, n_blocks=N_BLOCKS) for c in sub)
    h6_full = pd.DataFrame([r for ch in chunks for r in ch])
    h6_full.to_csv(H6_FULL_PATH, index=False, encoding="utf-8")
    print(f"[run] {len(sub)} cells with the full sampler in {(time.time()-t0)/60:.1f} min")

if "error" in h6_full.columns:
    h6_full = h6_full[h6_full["error"].isna()].copy()

key = ["metal", "level", "replicate", "block", "model"]
cmp_ = (h6_full.set_index(key)[["rmse", "crps"]]
        .join(h6.set_index(key)[["rmse", "crps"]], rsuffix="_cheap", how="inner"))
cmp_ = cmp_.dropna()
print(f"\ncells compared: {len(cmp_)}")
for m_ in ["rmse", "crps"]:
    r_full, r_cheap = cmp_[m_], cmp_[f"{m_}_cheap"]
    print(f"  {m_}: correlation {np.corrcoef(r_full, r_cheap)[0,1]:.4f} | "
          f"mean absolute difference {np.abs(r_full - r_cheap).mean():.4f} | "
          f"mean relative difference {100*np.mean((r_cheap - r_full)/r_full):+.2f}%")

# does the ranking of models survive?
rank_full = (h6_full.groupby(["metal", "level", "model"])["crps"].mean()
             .groupby(level=[0, 1]).rank())
rank_cheap = (h6[h6.set_index(key).index.isin(h6_full.set_index(key).index)]
              .groupby(["metal", "level", "model"])["crps"].mean().groupby(level=[0, 1]).rank())
common_r = rank_full.index.intersection(rank_cheap.index)
agree = float((rank_full.loc[common_r] == rank_cheap.loc[common_r]).mean())
print(f"  model ranking preserved in {agree:.0%} of the validated analyte-level combinations")
save_table(cmp_.reset_index().round(5), "TABLE_S3_cheap_vs_full_nuts")
print("\nThe cheap inference is used only inside the experiment, where every method receives the\n"
      "same budget. The main models, the cross-validation and the maps all use the full sampler.")
