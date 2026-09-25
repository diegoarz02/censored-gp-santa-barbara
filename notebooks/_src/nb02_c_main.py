
# %% [markdown]
# ## 7. Main hierarchical censored GP models (H5)
#
# One model per analyte for **Hg, Pb, As, Ba and Cd**, with:
#
# * the **stratum in the mean**, which is what separates geochemical background from mining input;
# * an **ARD Matérn 5/2** kernel, following the anisotropy found in section 4;
# * **interval censoring** on the detections at the laboratory reporting resolution, and the
#   cumulative mass below the limit for the non-detects;
# * the **limit as a parameter** for the censored analyte;
# * a **composite-support variance factor**, fitted jointly with the stratum effect so that their
#   posterior correlation can be inspected before anything is claimed about either.
#
# NUTS through nutpie, 4 chains, 2 500 warm-up and 2 000 sampling draws, `target_accept = 0.99`.

# %%
def fit_main(analyte, force=False):
    m = metal_arrays(analyte)
    min_det = np.nanmin(m["value"]) if np.isfinite(m["value"]).any() else None
    return cache_idata(
        f"main_{analyte}",
        lambda: sb.fit_model(
            sb.build_censored_gp(
                XY, COV, m["value"], m["censored"], m["resolution"], m["lod"],
                ell_params=ELL_PARAMS, is_background=IS_BACKGROUND, is_composite=IS_COMPOSITE,
                anisotropic=USE_ANISOTROPIC, limit_as_parameter=True, min_detected=min_det),
            seed=SEED, **KW_MAIN),
        force=force)


idata_main = {a: fit_main(a) for a in MAIN_METALS}

# %% [markdown]
# ### 7.1 Convergence diagnostics
#
# R-hat and ESS are computed with `az.rhat` and `az.ess` directly. `az.summary` rounds to two
# decimals, which hides an R-hat of 1.014 as "1.01" and would let a failure through.

# %%
diag_rows = []
for a in MAIN_METALS:
    d = sb.diagnose(idata_main[a])
    d["model"] = f"main {a}"
    diag_rows.append(d)
for nm, idt in [("simulation, censored", idata_sim_cens),
                ("simulation, L/2", idata_sim_sub)]:
    d = sb.diagnose(idt)
    d["model"] = nm
    diag_rows.append(d)
diag = pd.DataFrame(diag_rows)[["model", "rhat_max", "ess_bulk_min", "ess_tail_min",
                                "divergences", "bfmi_min", "n_draws", "passes"]].round(4)
print(diag.to_string(index=False))
save_table(diag, "TABLE10_convergence",
           caption="Convergence diagnostics of the main models, computed without the two-decimal "
                   "rounding of \\texttt{az.summary}. BFMI is the Bayesian fraction of missing "
                   "information from the energy transitions.", label="tab:conv")
print(f"\nall main models pass: {bool(diag['passes'].all())}")

# %% [markdown]
# ### 7.2 Posterior parameters

# %%
par_rows = []
for a in MAIN_METALS:
    p = idata_main[a].posterior
    row = {"analyte": a}
    for nm in ["ell", "eta", "sigma_n", "sigma_tot", "rho", "beta0", "b_background",
               "log_tau", "limit_ratio"]:
        if nm not in p:
            continue
        v = p[nm].values.reshape(-1, *p[nm].shape[2:])
        if v.ndim == 2 and v.shape[1] == 2:
            for j, ax_ in enumerate(["E", "N"]):
                row[f"{nm}_{ax_}"] = v[:, j].mean()
        else:
            row[nm] = v.ravel().mean()
    for j, cname in enumerate(COVARS):
        b = p["beta"].values.reshape(-1, COV.shape[1])[:, j]
        lo, hi = np.quantile(b, [0.025, 0.975])
        row[f"beta_{cname}"] = b.mean()
        row[f"beta_{cname}_sig"] = bool(lo > 0 or hi < 0)
    par_rows.append(row)
params = pd.DataFrame(par_rows).round(4)
print(params.to_string(index=False))
save_table(params, "TABLE11_posterior_parameters",
           caption="Posterior means of the main model parameters. Length scales in km, variances "
                   "in log mg kg$^{-1}$. $\\rho$ is the spatially structured proportion of the "
                   "total variance.", label="tab:params")

print("\nanisotropy recovered by the model (northing / easting length scale):")
for a in MAIN_METALS:
    e = idata_main[a].posterior["ell"].values.reshape(-1, 2)
    print(f"  {a}: {e[:, 1].mean()/e[:, 0].mean():.2f}x  "
          f"(E {e[:, 0].mean():.2f} km, N {e[:, 1].mean():.2f} km)")

# %% [markdown]
# ### 7.3 The identifiability check the design forces: stratum against support
#
# The 30 background points are exactly the 30 composite samples. The two effects are therefore
# almost perfectly aliased. They are not fully aliased — the stratum shifts the **mean** and the
# support scales the **variance**, which leave different fingerprints — but identification is weak.
# The procedure fixed in advance: fit both, inspect the posterior correlation, and if it exceeds
# 0.9 in absolute value declare that they cannot be separated, keep only the mean effect and report
# support as a limitation.

# %%
alias_rows = []
for a in MAIN_METALS:
    p = idata_main[a].posterior
    if "log_tau" not in p or "b_background" not in p:
        continue
    b = p["b_background"].values.ravel()
    t = p["log_tau"].values.ravel()
    r_ = float(np.corrcoef(b, t)[0, 1])
    lo_b, hi_b = np.quantile(b, [0.025, 0.975])
    lo_t, hi_t = np.quantile(t, [0.025, 0.975])
    alias_rows.append({"analyte": a, "b_background": b.mean(), "b_ci95_low": lo_b,
                       "b_ci95_high": hi_b, "log_tau": t.mean(), "tau_ci95_low": lo_t,
                       "tau_ci95_high": hi_t, "posterior_corr": r_,
                       "separable": bool(abs(r_) < 0.9)})
alias = pd.DataFrame(alias_rows).round(4)
print(alias.to_string(index=False))
save_table(alias, "TABLE12_stratum_support_aliasing",
           caption="Posterior correlation between the stratum mean effect and the composite-support "
                   "variance factor. A correlation above 0.9 in absolute value would mean the two "
                   "cannot be separated by these data.", label="tab:alias")

SEPARABLE = bool(alias["separable"].all())
print(f"\nmaximum absolute posterior correlation: {alias['posterior_corr'].abs().max():.3f}")
if SEPARABLE:
    print("Below the 0.9 threshold for every analyte, so both effects are reported as estimated.\n"
          "The stratum effect is nonetheless partly confounded with location (section 3.2) and that\n"
          "remains a limitation.")
else:
    print("At or above the threshold: the two effects cannot be separated by these data. The\n"
          "support factor is dropped and reported as a limitation, keeping only the stratum mean.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.36), constrained_layout=True)
ax = axes[0]
for k, a in enumerate(MAIN_METALS):
    p = idata_main[a].posterior
    ax.scatter(p["b_background"].values.ravel()[::20], p["log_tau"].values.ravel()[::20],
               s=1.2, alpha=0.25, color=OI[k], label=f"{a} (r = {alias.set_index('analyte').loc[a,'posterior_corr']:+.2f})")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.axvline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("Stratum effect $b_{\\mathrm{background}}$ (log mg kg$^{-1}$)")
ax.set_ylabel("Support factor $\\log\\tau$")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", markerscale=6, fontsize=8)
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")

ax = axes[1]
ypos = np.arange(len(MAIN_METALS))[::-1]
s = alias.set_index("analyte").loc[MAIN_METALS]
ax.errorbar(s["b_background"], ypos + 0.12,
            xerr=[s["b_background"] - s["b_ci95_low"], s["b_ci95_high"] - s["b_background"]],
            fmt="o", ms=3.2, lw=0.9, capsize=2, color=OI[0], mec="k", mew=0.25,
            label="Stratum effect")
ax.errorbar(s["log_tau"], ypos - 0.12,
            xerr=[s["log_tau"] - s["tau_ci95_low"], s["tau_ci95_high"] - s["log_tau"]],
            fmt="s", ms=3.0, lw=0.9, capsize=2, color=OI[1], mec="k", mew=0.25,
            label="Support factor")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels(MAIN_METALS)
ax.set_xlabel("Posterior estimate")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7")
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
save_fig(fig, "FIG7_stratum_support")
plt.show()

# %% [markdown]
# ### 7.4 Prior against posterior
#
# The only honest way to show what the data identified and what they did not. In the previous phase
# of this project this figure revealed that $\rho$ had a posterior 91 % as wide as its prior, i.e.
# it was not identified at all, and a claim resting on it had to be withdrawn.

# %%
def prior_draws(name, n=20000, seed=SEED):
    r = np.random.default_rng(seed)
    if name == "ell":
        return ig.rvs(n, random_state=int(seed))
    if name == "sigma_tot":
        return np.abs(r.normal(0, 1, n))
    if name == "rho":
        return r.beta(2, 2, n)
    if name == "b_background":
        return r.normal(0, 1, n)
    if name == "log_tau":
        return r.normal(0, 0.5, n)
    raise KeyError(name)


PP_PARAMS = ["ell", "sigma_tot", "rho", "b_background", "log_tau"]
pp_rows = []
for a in MAIN_METALS:
    p = idata_main[a].posterior
    for nm in PP_PARAMS:
        if nm not in p:
            continue
        v = p[nm].values
        v = v.reshape(-1, *v.shape[2:])
        v = v[:, 0] if v.ndim == 2 else v.ravel()
        pri = prior_draws(nm)
        w_post = float(np.diff(np.quantile(v, [0.025, 0.975]))[0])
        w_pri = float(np.diff(np.quantile(pri, [0.025, 0.975]))[0])
        pp_rows.append({"analyte": a, "parameter": nm, "prior_width95": w_pri,
                        "posterior_width95": w_post, "width_ratio": w_post / w_pri,
                        "identified": bool(w_post / w_pri < 0.6)})
prpo = pd.DataFrame(pp_rows).round(4)
print(prpo.pivot(index="parameter", columns="analyte", values="width_ratio").round(3).to_string())
save_table(prpo, "TABLE13_prior_posterior",
           caption="Ratio between the width of the 95\\% posterior interval and that of the prior. "
                   "A ratio near one means the data were uninformative about that parameter.",
           label="tab:priorpost")
print("\nA ratio near 1 means the posterior is the prior: the data said nothing about that\n"
      "parameter. The threshold of 0.6 is a convention declared here, not a test.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.34), constrained_layout=True)
for ax, nm, lab in zip(axes, ["rho", "ell"],
                       ["$\\rho$ (spatial fraction of variance)", "$\\ell_E$ (km)"]):
    pri = prior_draws(nm)
    ax.hist(pri, bins=60, density=True, histtype="stepfilled", color="0.85", edgecolor="0.4",
            lw=0.6, label="Prior", range=(0, 1) if nm == "rho" else (0, 4))
    for k, a in enumerate(MAIN_METALS):
        v = idata_main[a].posterior[nm].values
        v = v.reshape(-1, *v.shape[2:])
        v = v[:, 0] if v.ndim == 2 else v.ravel()
        ax.hist(v, bins=60, density=True, histtype="step", color=OI[k], lw=1.0, label=a,
                range=(0, 1) if nm == "rho" else (0, 4))
    ax.set_xlabel(lab)
    ax.set_ylabel("Density")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=2, fontsize=8)
axes[0].text(0.97, 0.95, "(a)", transform=axes[0].transAxes, ha="right", va="top", fontweight="bold")
axes[1].text(0.97, 0.95, "(b)", transform=axes[1].transAxes, ha="right", va="top", fontweight="bold")
save_fig(fig, "FIG8_prior_posterior")
plt.show()

# %% [markdown]
# ### 7.5 Posterior predictive checks

# %%
def posterior_predictive_stats(analyte, idata, n_rep=600, seed=SEED):
    """Replicate the observation process, censoring included, and compare summary statistics."""
    m = metal_arrays(analyte)
    r = np.random.default_rng(seed)
    p = idata.posterior
    flat = p.sizes["chain"] * p.sizes["draw"]
    idx = r.choice(flat, n_rep, replace=n_rep > flat)
    mu = p["mu"].values.reshape(flat, -1)[idx]
    sn = p["sigma_n"].values.reshape(flat)[idx]
    tau = np.exp(p["log_tau"].values.reshape(flat)[idx]) if "log_tau" in p else np.ones(n_rep)
    sd = sn[:, None] * np.where(IS_COMPOSITE[None, :], tau[:, None], 1.0)
    yrep = mu + r.standard_normal(mu.shape) * sd

    if "log_limit" in p:
        loglim = p["log_limit"].values.reshape(flat)[idx]
    elif np.isfinite(m["lod"]):
        loglim = np.full(n_rep, np.log(m["lod"]))
    else:
        loglim = np.full(n_rep, -np.inf)
    cens_rep = yrep < loglim[:, None]

    obs_det = np.log(m["value"][~m["censored"]])
    stats_obs = {"median_log": np.median(obs_det), "sd_log": np.std(obs_det),
                 "max_log": obs_det.max(), "pct_censored": 100 * m["censored"].mean(),
                 "pct_above_eca": 100 * np.mean(m["value"][~m["censored"]] > m["eca"][0])}
    out = []
    for k, fn in [("median_log", np.median), ("sd_log", np.std), ("max_log", np.max)]:
        rep = np.array([fn(yrep[s][~cens_rep[s]]) if (~cens_rep[s]).any() else np.nan
                        for s in range(n_rep)])
        p_bayes = float(np.nanmean(rep >= stats_obs[k]))
        out.append({"analyte": analyte, "statistic": k, "observed": stats_obs[k],
                    "replicated_mean": np.nanmean(rep),
                    "rep_ci95_low": np.nanquantile(rep, 0.025),
                    "rep_ci95_high": np.nanquantile(rep, 0.975),
                    "p_bayes": p_bayes, "compatible": bool(0.025 < p_bayes < 0.975)})
    rep_c = 100 * cens_rep.mean(1)
    p_b = float(np.mean(rep_c >= stats_obs["pct_censored"]))
    out.append({"analyte": analyte, "statistic": "pct_censored",
                "observed": stats_obs["pct_censored"], "replicated_mean": rep_c.mean(),
                "rep_ci95_low": np.quantile(rep_c, 0.025), "rep_ci95_high": np.quantile(rep_c, 0.975),
                "p_bayes": p_b, "compatible": bool(0.025 < p_b < 0.975)})
    return out


ppc = pd.DataFrame([r_ for a in MAIN_METALS
                    for r_ in posterior_predictive_stats(a, idata_main[a])]).round(4)
print(ppc.to_string(index=False))
save_table(ppc, "TABLE14_posterior_predictive",
           caption="Posterior predictive checks. Each statistic is recomputed on replicated "
                   "datasets that reproduce the censoring mechanism, and compared with the observed "
                   "value.", label="tab:ppc")
bad = ppc[~ppc.compatible]
print(f"\nincompatible statistics: {len(bad)} of {len(ppc)}")
if len(bad):
    print(bad[["analyte", "statistic", "observed", "replicated_mean", "p_bayes"]].to_string(index=False))
    print("\nThese are reported rather than explained away; the discussion returns to them.")
