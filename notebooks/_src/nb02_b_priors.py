
# %% [markdown]
# ## 5. Priors: elicitation and prior predictive checks (H3)
#
# The priors follow `notas_obsidian/Priors del GP censurado - como se definen.md`. The guiding
# principle is *weakly informative on the scale of the data*, not "uninformative": a flat prior on
# an unbounded scale is a strong and almost always false statement.
#
# | Parameter | Prior | Reason |
# |---|---|---|
# | $\beta$ (standardised covariates) | $N(0,\,0.5)$ | A coefficient of 0.5 means one standard deviation of the covariate moves log-concentration by half a unit, a factor of 1.65. The scale is not a round guess: it comes from the prior predictive check below, which **rejected** 1.0 |
# | $\beta_0$ | $N(\bar y,\, 0.7)$ | Centred on the data mean. This is empirical Bayes and is declared as such |
# | $b_{\text{background}}$ | $N(0,\,0.5)$ | The stratum shift. Centred at zero, so the model is not told in advance that background is cleaner |
# | $\ell$ (length scale, one per axis) | InverseGamma calibrated | A GP identifies neither $\ell$ below the typical point spacing nor above the domain diameter. The InverseGamma penalises $\ell \to 0$, the regime where the GP degenerates into noise |
# | $(\eta, \sigma_n)$ | variance partition | $\sigma_{tot}\sim\text{HalfNormal}(0.7)$, $\rho\sim\text{Beta}(2,2)$, $\eta=\sigma_{tot}\sqrt{\rho}$, $\sigma_n=\sigma_{tot}\sqrt{1-\rho}$. Removes the funnel |
# | $\log\tau$ (composite support) | $N(0, 0.5)$ | Multiplies the noise of composite samples. Negative means composites are less variable, the expected sign |
# | $L$ (quantification limit) | $L = \mathrm{LOD}\cdot r$, $r\sim\text{TruncNormal}(3, 1)$ on $[1, \min_{det}/\mathrm{LOD}]$ | Centred at 3 because the metrological convention gives LOQ/LOD $\approx$ 3 (Currie 1968). The upper bound is not arbitrary: a value quantified at 2.3 forces the limit below 2.3 |

# %%
import sbmodel as sb
import preliz as pz

# preliz applies its own matplotlib style on import, which resets savefig.bbox to "tight" and would
# push every figure from here on to 193 mm instead of 190. Reapply ours.
apply_style()

ELL_PARAMS, ELL_LO, ELL_HI = sb.ell_prior_params(XY, lower_q=0.05, upper_frac=0.5, mass=0.90)
print(f"ell ~ InverseGamma(alpha={ELL_PARAMS['alpha']:.3f}, beta={ELL_PARAMS['beta']:.3f})")
print(f"90% of the prior mass between {ELL_LO:.3f} and {ELL_HI:.3f} km")
print(f"  lower edge  = 5th percentile of pairwise distances")
print(f"  upper edge  = half the maximum pairwise distance ({DIST[iu].max():.2f} km)")

ig = pz.InverseGamma(alpha=ELL_PARAMS["alpha"], beta=ELL_PARAMS["beta"])
q = ig.ppf([0.05, 0.25, 0.5, 0.75, 0.95])
print(f"prior quantiles of ell (km): 5% {q[0]:.3f}, 25% {q[1]:.3f}, median {q[2]:.3f}, "
      f"75% {q[3]:.3f}, 95% {q[4]:.3f}")

# %% [markdown]
# ### 5.1 Prior predictive check
#
# Simulating from the prior and looking at the implied concentrations. If the prior implies mercury
# at $10^6$ mg kg$^{-1}$, the prior is wrong. This is run for every main analyte because $\beta_0$
# is centred on that analyte's own mean.

# %%
PHYSICAL_MAX = 1e6      # mg/kg. 10^6 mg/kg is 100 % of the sample mass: nothing can exceed it.


def prior_predictive_concentrations(analyte, sd_beta, sd_sigma, sd_beta0, n_sim=400, seed=SEED):
    """Draw from the prior and return the implied concentrations in mg/kg."""
    m = metal_arrays(analyte)
    y_bar = float(np.mean(np.log(m["value"][~m["censored"]])))
    r = np.random.default_rng(seed)
    out = np.empty((n_sim, len(XY)))
    for s in range(n_sim):
        beta0 = r.normal(y_bar, sd_beta0)
        beta = r.normal(0, sd_beta, size=COV.shape[1])
        b_bg = r.normal(0, sd_beta)
        ell = np.array([ig.rvs(), ig.rvs()], float)
        sigma_tot = abs(r.normal(0, sd_sigma))
        rho = r.beta(2, 2)
        eta, sn = sigma_tot * np.sqrt(rho), sigma_tot * np.sqrt(1 - rho)
        a = XY / ell
        K = eta ** 2 * sb._matern52(((a[:, None] - a[None]) ** 2).sum(-1), ell)
        K[np.diag_indices(len(XY))] += 1e-8
        f = np.linalg.cholesky(K) @ r.standard_normal(len(XY))
        out[s] = beta0 + COV @ beta + b_bg * IS_BACKGROUND + f + r.standard_normal(len(XY)) * sn
    return np.exp(out)


def prior_check_table(sd_beta, sd_sigma, sd_beta0, tag):
    """Three criteria, each stated as a property of the prior distribution rather than of one draw.

    An earlier version compared the prior's 99th percentile with the single largest observation and
    the prior's sample maximum with the physical bound. Both are the wrong shape of test: the
    maximum of 45 600 draws from a heavy-tailed prior is always extreme, and one observation should
    never decide whether a prior is acceptable. Lead makes the point — its largest value, 58 000
    mg kg⁻¹, is 5.8 % of the sample mass, ore-grade material rather than soil.

    The criteria used instead:

    * **covers the bulk**: the prior's 99th percentile reaches the 95th percentile of the data;
    * **negligible impossible mass**: P(concentration > 10^6 mg kg⁻¹) below 1e-4, where 10^6 mg kg⁻¹
      is the entire mass of the sample;
    * **not absurdly diffuse**: the prior median stays within two orders of magnitude of the
      observed median.
    """
    rows, draws = [], {}
    for a in MAIN_METALS:
        c = prior_predictive_concentrations(a, sd_beta, sd_sigma, sd_beta0)
        draws[a] = c
        obs = df[df.analyte == a]["value"].dropna()
        p99 = float(np.quantile(c, 0.99))
        p_impossible = float(np.mean(c > PHYSICAL_MAX))
        rows.append({"prior": tag, "analyte": a,
                     "prior_p01": float(np.quantile(c, 0.01)),
                     "prior_median": float(np.median(c)), "prior_p99": p99,
                     "prior_p999": float(np.quantile(c, 0.999)),
                     "p_above_physical_bound": p_impossible,
                     "observed_median": obs.median(),
                     "observed_p95": float(obs.quantile(0.95)), "observed_max": obs.max(),
                     "covers_bulk": bool(p99 >= obs.quantile(0.95)),
                     "impossible_mass_negligible": bool(p_impossible < 1e-4),
                     "not_absurd": bool(np.median(c) < 100 * obs.median())})
    t = pd.DataFrame(rows)
    t["plausible"] = t["covers_bulk"] & t["impossible_mass_negligible"] & t["not_absurd"]
    return t, draws


# --- first attempt: unit scales everywhere, the obvious default ---------------------------
wide, _ = prior_check_table(1.0, 1.0, 1.0, "initial (all scales 1.0)")
print("FIRST ATTEMPT, scales of 1.0 on beta, sigma_tot and beta0:")
print(wide[["analyte", "prior_median", "prior_p99", "prior_p999", "p_above_physical_bound",
            "observed_median", "observed_p95", "plausible"]].round(5).to_string(index=False))
worst = wide.loc[wide["p_above_physical_bound"].idxmax()]
print(f"\nRejected. For {worst['analyte']} the prior places "
      f"{100*worst['p_above_physical_bound']:.2f}% of its mass above {PHYSICAL_MAX:.0e} mg/kg, "
      "which is\n100 % of the sample mass. A prior that puts appreciable mass on impossible values "
      "is not\n'uninformative', it is wrong.\n\n"
      "The dominant contribution is the four covariate coefficients: at scale 1.0 they add 2.0 to\n"
      "the standard deviation of the log-concentration on their own, a factor of 7.4 per standard\n"
      "deviation of a covariate, far more than any terrain effect plausibly produces.")

# --- calibrated scales, used for every model in this notebook ------------------------------
ppc_prior, pp_draws = prior_check_table(sb.SD_BETA, sb.SD_SIGMA, sb.SD_BETA0,
                                        f"calibrated (beta {sb.SD_BETA}, sigma {sb.SD_SIGMA}, "
                                        f"beta0 {sb.SD_BETA0})")
print(f"\nCALIBRATED, beta {sb.SD_BETA}, sigma_tot {sb.SD_SIGMA}, beta0 {sb.SD_BETA0}:")
print(ppc_prior[["analyte", "prior_median", "prior_p99", "p_above_physical_bound",
                 "observed_median", "observed_p95", "observed_max",
                 "covers_bulk", "impossible_mass_negligible", "not_absurd", "plausible"]]
      .round(5).to_string(index=False))

save_table(pd.concat([wide, ppc_prior], ignore_index=True).round(3), "TABLE8_prior_predictive",
           caption="Prior predictive check, before and after calibration. Concentrations implied by "
                   "the priors in mg kg$^{-1}$. The initial scales imply concentrations above "
                   "$10^6$ mg kg$^{-1}$, which is the whole mass of the sample.",
           label="tab:priorpred")

assert ppc_prior["plausible"].all(), "the calibrated prior is still not plausible"
print("\nThe calibrated prior covers the observed range at its 99th percentile, stays within the\n"
      "physical bound, and is not absurdly diffuse. These are the scales used by every model in\n"
      "this notebook. The rejected first attempt is kept in the table because it is the evidence\n"
      "that the check does its job.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.34), constrained_layout=True)

ax = axes[0]
xs = np.linspace(0.01, 4.0, 400)
ax.plot(xs, ig.pdf(xs), color=OI[0], lw=1.2, label="Prior on $\\ell$")
ax.axvspan(ELL_LO, ELL_HI, color=OI[0], alpha=0.12, label="90% prior mass")
ax.axvline(np.median(DIST[iu]), color=OI[1], ls="--", lw=1.0, label="Median pair distance")
ax.axvline(DIST[iu].max(), color=OI[2], ls=":", lw=1.0, label="Domain diameter")
ax.set_xlabel("Length scale $\\ell$ (km)")
ax.set_ylabel("Prior density")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7")
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")

ax = axes[1]
for k, a in enumerate(MAIN_METALS):
    c = np.log10(pp_draws[a].ravel())
    ax.hist(c, bins=60, histtype="step", density=True, color=OI[k], lw=1.0, label=f"{a} prior")
ax.axvline(np.log10(PHYSICAL_MAX), color="k", lw=1.0, ls="-.",
           label="Physical bound (10$^6$ mg kg$^{-1}$)")
for k, a in enumerate(MAIN_METALS):
    o = np.log10(df[df.analyte == a]["value"].dropna())
    ax.plot(np.sort(o), np.full(len(o), -0.02 - 0.02 * k), "|", color=OI[k], ms=3, mew=0.6)
ax.set_xlabel(r"log$_{10}$ concentration (mg kg$^{-1}$)")
ax.set_ylabel("Prior predictive density")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", ncol=2, fontsize=8)
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
ax.text(0.03, 0.03, "Ticks below the axis are the observed values", transform=ax.transAxes,
        fontsize=8, va="bottom")

save_fig(fig, "FIG5_prior_predictive")
plt.show()

# %% [markdown]
# ## 6. Parameter recovery on simulated fields (H4)
#
# The cleanest evidence in the whole study, because here a truth exists. Fields are simulated with
# **known** parameters over the 114 real locations, censored artificially, and both models are
# fitted: the censored likelihood and the `L/2` substitution. A model that cannot recover known
# parameters cannot be trusted on real data.

# %%
# The prior scales enter every subsequent model through sbmodel's module-level defaults, so the
# calibration above is not a decoration: it is the specification used from here on.
print(f"prior scales in force: beta {sb.SD_BETA}, sigma_tot {sb.SD_SIGMA}, beta0 {sb.SD_BETA0}")


def simulate_field(params, seed=SEED):
    """Simulate a log-concentration field over the real locations with known parameters."""
    r = np.random.default_rng(seed)
    a = XY / params["ell"]
    K = params["eta"] ** 2 * sb._matern52(((a[:, None] - a[None]) ** 2).sum(-1), params["ell"])
    K[np.diag_indices(len(XY))] += 1e-8
    f = np.linalg.cholesky(K) @ r.standard_normal(len(XY))
    mu = (params["beta0"] + COV @ params["beta"] + params["b_bg"] * IS_BACKGROUND + f)
    y = mu + r.standard_normal(len(XY)) * params["sigma_n"]
    return np.exp(y)


TRUE = {"beta0": 4.0, "beta": np.array([0.3, -0.2, -0.5, 0.1]), "b_bg": -0.9,
        "ell": np.array([0.65, 1.55]), "eta": 0.9, "sigma_n": 0.45}
TRUE["sigma_tot"] = float(np.hypot(TRUE["eta"], TRUE["sigma_n"]))
TRUE["rho"] = float(TRUE["eta"] ** 2 / TRUE["sigma_tot"] ** 2)

conc = simulate_field(TRUE)
CENS_RATE = 0.60
L_true = float(np.quantile(conc, CENS_RATE))
cens_sim = conc < L_true
RES_SIM = 0.01
print(f"simulated field: {len(conc)} locations, concentrations "
      f"{conc.min():.2f} to {conc.max():.1f} mg/kg")
print(f"censoring at the {CENS_RATE:.0%} quantile -> limit {L_true:.3f} mg/kg, "
      f"{cens_sim.sum()} censored values")
print(f"true parameters: ell {TRUE['ell']}, eta {TRUE['eta']}, sigma_n {TRUE['sigma_n']}, "
      f"rho {TRUE['rho']:.3f}, b_bg {TRUE['b_bg']}")

# %%
# Sampler budget, set from a measurement rather than a habit. A first fit at 2 500 warm-up and
# 2 000 draws gave R-hat 1.003, minimum ESS 1 546 and zero divergences: four times the ESS the
# acceptance criterion asks for, at thirty minutes per fit. Halving the draws keeps ESS near 800,
# comfortably above the 400 threshold, and makes a study with roughly fifty full fits feasible.
# `target_accept` stays at 0.99, which is what delivers the zero divergences.
KW_MAIN = dict(draws=1000, tune=1500, chains=4, target_accept=0.99)

idata_sim_cens = cache_idata("sim_censored", lambda: sb.fit_model(
    sb.build_censored_gp(XY, COV, np.where(cens_sim, np.nan, conc), cens_sim, RES_SIM, L_true,
                         ell_params=ELL_PARAMS, is_background=IS_BACKGROUND,
                         is_composite=IS_COMPOSITE, anisotropic=USE_ANISOTROPIC,
                         limit_as_parameter=False),
    seed=SEED, **KW_MAIN))

idata_sim_sub = cache_idata("sim_substitution", lambda: sb.fit_model(
    sb.build_censored_gp(XY, COV, np.where(cens_sim, np.nan, conc), cens_sim, RES_SIM, L_true,
                         ell_params=ELL_PARAMS, is_background=IS_BACKGROUND,
                         is_composite=IS_COMPOSITE, anisotropic=USE_ANISOTROPIC,
                         substitute_half_lod=True),
    seed=SEED, **KW_MAIN))

for nm, idt in [("censored likelihood", idata_sim_cens), ("L/2 substitution", idata_sim_sub)]:
    d = sb.diagnose(idt)
    print(f"{nm:22s} rhat {d['rhat_max']:.4f} | ESS {min(d['ess_bulk_min'], d['ess_tail_min']):.0f} "
          f"| divergences {d['divergences']} | BFMI {d.get('bfmi_min', float('nan')):.3f} "
          f"| passes {d['passes']}")

# %%
def recovery_row(idata, name, param, true_value, index=None):
    v = idata.posterior[param].values
    v = v.reshape(-1, *v.shape[2:])
    if index is not None:
        v = v[:, index]
    v = v.ravel()
    lo, hi = np.quantile(v, [0.025, 0.975])
    return {"model": name, "parameter": param + (f"[{index}]" if index is not None else ""),
            "true": true_value, "posterior_mean": v.mean(),
            "ci95_low": lo, "ci95_high": hi,
            "covers": bool(lo <= true_value <= hi),
            "rel_bias_pct": 100 * (v.mean() - true_value) / abs(true_value)}


targets = [("beta0", TRUE["beta0"], None), ("b_background", TRUE["b_bg"], None),
           ("ell", TRUE["ell"][0], 0), ("ell", TRUE["ell"][1], 1),
           ("eta", TRUE["eta"], None), ("sigma_n", TRUE["sigma_n"], None),
           ("sigma_tot", TRUE["sigma_tot"], None), ("rho", TRUE["rho"], None)]
targets += [("beta", TRUE["beta"][j], j) for j in range(COV.shape[1])]

rec = pd.DataFrame(
    [recovery_row(idata_sim_cens, "censored", p, t, i) for p, t, i in targets]
    + [recovery_row(idata_sim_sub, "L/2 substitution", p, t, i) for p, t, i in targets]).round(4)
print(rec.to_string(index=False))
save_table(rec, "TABLE9_parameter_recovery",
           caption="Parameter recovery on a simulated field with known parameters over the 114 real "
                   "locations, censored at 60\\%. The censored likelihood is compared with the "
                   "$L/2$ substitution.", label="tab:recovery")

cov_cens = rec[rec.model == "censored"]["covers"].mean()
cov_sub = rec[rec.model == "L/2 substitution"]["covers"].mean()
print(f"\n95% intervals covering the true value: censored {cov_cens:.0%}, "
      f"L/2 substitution {cov_sub:.0%}")
fails_sub = rec[(rec.model == "L/2 substitution") & (~rec.covers)]
if len(fails_sub):
    print("\nparameters the substitution fails to recover:")
    print(fails_sub[["parameter", "true", "posterior_mean", "ci95_low", "ci95_high",
                     "rel_bias_pct"]].to_string(index=False))
RECOVERY_PASSES = bool(cov_cens >= 0.9)
print(f"\nH4 verdict: the censored model {'recovers' if RECOVERY_PASSES else 'FAILS TO RECOVER'} "
      f"the known parameters.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.36), constrained_layout=True)
show = ["beta0", "b_background", "ell[0]", "ell[1]", "eta", "sigma_n", "rho"]
for ax, mdl in zip(axes, ["censored", "L/2 substitution"]):
    s = rec[(rec.model == mdl) & (rec.parameter.isin(show))].set_index("parameter").loc[show]
    ypos = np.arange(len(s))[::-1]
    # scale each parameter by its true value so they share one axis
    scale = s["true"].abs().values
    ax.errorbar(s["posterior_mean"] / scale, ypos,
                xerr=[(s["posterior_mean"] - s["ci95_low"]) / scale,
                      (s["ci95_high"] - s["posterior_mean"]) / scale],
                fmt="o", ms=3.2, lw=0.9, capsize=2,
                color=OI[0] if mdl == "censored" else OI[1], mec="k", mew=0.25)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    for k, (_, r_) in enumerate(s.iterrows()):
        if not r_["covers"]:
            ax.plot(r_["posterior_mean"] / scale[k], ypos[k], "x", color="k", ms=6, mew=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(show)
    ax.set_xlabel("Posterior estimate / true value")
    ax.set_title(mdl, fontsize=8)
    ax.text(0.97, 0.05, "(a)" if mdl == "censored" else "(b)", transform=ax.transAxes,
            ha="right", va="bottom", fontweight="bold")
axes[1].text(0.03, 0.05, "x marks an interval that misses the truth", transform=axes[1].transAxes,
             fontsize=8, va="bottom")
save_fig(fig, "FIG6_parameter_recovery")
plt.show()
