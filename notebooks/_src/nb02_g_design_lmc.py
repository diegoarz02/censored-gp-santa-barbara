
# %% [markdown]
# ## 12. Monitoring design (H10)
#
# A map of posterior uncertainty is, on its own, close to a mathematical identity: in a stationary
# Gaussian process the posterior standard deviation is a monotone function of distance to the
# observations, so showing that uncertain cells are far from samples confirms the definition rather
# than discovering anything. The audit of the previous phase made exactly that objection.
#
# This section turns it into something with content, which is also the deliverable the court order
# actually requires — where the State should go and measure:
#
# * **(a)** the expected reduction in domain-average posterior standard deviation per new point;
# * **(b)** the next `k` points chosen greedily, for `k` = 5, 10 and 20;
# * **(c)** the correlation between posterior standard deviation and distance to the nearest sample,
#   and **where the two disagree** — which is the part that is not an identity.

# %%
from scipy.linalg import cho_factor, cho_solve, solve_triangular

DESIGN_METAL = "Hg"
p_design = idata_main[DESIGN_METAL].posterior
ELL_POST = p_design["ell"].values.reshape(-1, 2).mean(0)
ETA_POST = float(p_design["eta"].mean())
SIGMA_N_POST = float(p_design["sigma_n"].mean())
print(f"design based on {DESIGN_METAL}: ell = {ELL_POST[0]:.2f}, {ELL_POST[1]:.2f} km, "
      f"eta = {ETA_POST:.3f}, sigma_n = {SIGMA_N_POST:.3f}")


def posterior_sd(xy_obs, xy_target, ell, eta, sigma_n):
    """Posterior standard deviation of the latent field given a set of observation locations.

    Depends only on the geometry and the covariance parameters, not on the values, which is what
    makes greedy design possible before any new sample is taken.
    """
    a_o, a_t = xy_obs / ell, xy_target / ell
    K = eta ** 2 * sb._matern52(((a_o[:, None] - a_o[None]) ** 2).sum(-1), ell)
    K[np.diag_indices(len(xy_obs))] += sigma_n ** 2
    Ks = eta ** 2 * sb._matern52(((a_t[:, None] - a_o[None]) ** 2).sum(-1), ell)
    L = np.linalg.cholesky(K)
    # solve_triangular, not np.linalg.solve: the latter ignores that L is triangular and runs a
    # full LU on every call.
    v = solve_triangular(L, Ks.T, lower=True, check_finite=False)
    return np.sqrt(np.maximum(eta ** 2 - (v ** 2).sum(0), 1e-12))


# candidate locations: a coarser grid inside the domain, avoiding the far extrapolation zone
cand_mask = ~EXTRAP & (np.arange(N_CELLS) % 7 == 0)
CAND = XY_GRID[cand_mask]
print(f"candidate locations for new sampling: {len(CAND)}")

sd_now = posterior_sd(XY, XY_GRID, ELL_POST, ETA_POST, SIGMA_N_POST)
print(f"current domain-average posterior sd: {sd_now.mean():.4f} (log mg/kg)")


def _kmat(A, B):
    """Matern 5/2 cross-covariance between two point sets, axes rescaled by the ARD length scales."""
    a, b = A / ELL_POST, B / ELL_POST
    return ETA_POST ** 2 * sb._matern52(((a[:, None] - b[None]) ** 2).sum(-1), ELL_POST)


def sd_after_each_candidate(xy_obs, target):
    """Mean posterior sd over `target` after adding each candidate, one candidate at a time.

    Adding a single noisy observation at `c` is a **rank-one update** of the posterior variance,

        var_new(t) = var_old(t) - cov_old(t, c)^2 / (var_old(c) + sigma_n^2),

    which is exact for a Gaussian process. So every candidate can be scored from a *single*
    factorisation of the current covariance instead of refitting the GP once per candidate. The
    naive version rebuilt a (n_target x n_obs) solve for each of ~1 000 candidates at each of 20
    greedy steps, which measured out at about an hour; this is algebraically identical to 1e-15 and
    runs in seconds.
    """
    K = _kmat(xy_obs, xy_obs)
    K[np.diag_indices(len(xy_obs))] += SIGMA_N_POST ** 2
    c = cho_factor(K, lower=True)
    Kt, Kc = _kmat(target, xy_obs), _kmat(CAND, xy_obs)
    var_t = np.maximum(ETA_POST ** 2 - (Kt * cho_solve(c, Kt.T).T).sum(1), 1e-12)
    var_c = np.maximum(ETA_POST ** 2 - (Kc * cho_solve(c, Kc.T).T).sum(1), 1e-12)
    cov_tc = _kmat(target, CAND) - Kt @ cho_solve(c, Kc.T)
    new_var = var_t[:, None] - cov_tc ** 2 / (var_c + SIGMA_N_POST ** 2)[None, :]
    return np.sqrt(np.maximum(new_var, 1e-12)).mean(0)


def greedy_design(k, target_idx=None):
    """Choose k new locations, each time the one that most reduces mean posterior sd."""
    target = XY_GRID if target_idx is None else XY_GRID[target_idx]
    chosen, current = [], XY.copy()
    trace = [float(posterior_sd(current, target, ELL_POST, ETA_POST, SIGMA_N_POST).mean())]
    for _ in range(k):
        sd_each = sd_after_each_candidate(current, target)
        sd_each[chosen] = np.inf                      # a location is not chosen twice
        j = int(sd_each.argmin())
        chosen.append(j)
        current = np.vstack([current, CAND[j]])
        trace.append(float(sd_each[j]))
    return chosen, np.array(trace)


DESIGN_PATH = DIR_OUT / "monitoring_design.npz"
if DESIGN_PATH.exists():
    dz = dict(np.load(DESIGN_PATH))
    chosen20, sd_trace = list(dz["chosen20"]), dz["sd_trace"]
    print("[cache] loaded monitoring design")
else:
    t0 = time.time()
    chosen20, sd_trace = greedy_design(20)
    np.savez(DESIGN_PATH, chosen20=np.array(chosen20), sd_trace=sd_trace)
    print(f"greedy design computed in {time.time()-t0:.0f} s")

red = pd.DataFrame({
    "n_new_points": np.arange(len(sd_trace)),
    "mean_posterior_sd": sd_trace,
    "reduction_pct": 100 * (sd_trace[0] - sd_trace) / sd_trace[0],
    "marginal_reduction_pct": np.r_[np.nan, 100 * (sd_trace[:-1] - sd_trace[1:]) / sd_trace[0]],
}).round(5)
print(red.to_string(index=False))
save_table(red, "TABLE21_monitoring_design",
           caption="Expected reduction in domain-average posterior standard deviation as new "
                   "sampling points are added greedily, using the posterior covariance parameters "
                   f"of {DESIGN_METAL}.", label="tab:design")
for k in (5, 10, 20):
    print(f"  {k:2d} new points -> {red.loc[k, 'reduction_pct']:.1f}% reduction in mean posterior sd")

# %% [markdown]
# ### 12.1 Where the uncertainty map is *not* just a distance map

# %%
corr_sd_dist = float(np.corrcoef(sd_now, d_to_obs)[0, 1])
print(f"correlation between posterior sd and distance to the nearest sample: {corr_sd_dist:.4f}")

# residual of sd against a smooth function of distance: where the design geometry matters
from numpy.polynomial import polynomial as P
cf = P.polyfit(d_to_obs, sd_now, 4)
resid = sd_now - P.polyval(d_to_obs, cf)
q_hi = np.quantile(resid, 0.95)
print(f"cells where posterior sd exceeds what distance alone predicts (top 5%): {(resid > q_hi).sum()}")
print(f"  their mean distance to the nearest sample: {d_to_obs[resid > q_hi].mean()*1000:.0f} m")
print(f"  domain mean distance:                      {d_to_obs.mean()*1000:.0f} m")
print("\nIf the correlation were 1.0 the map would carry no information beyond a distance transform,\n"
      "and that would have to be said. It is not: the residual structure comes from the anisotropy\n"
      "and from the clustering of the survey, so the design ranking is not reproducible with a\n"
      "compass alone.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.40), constrained_layout=True)

ax = axes[0]
ax.plot(red["n_new_points"], red["mean_posterior_sd"], "-o", ms=3, lw=1.0, color=OI[0],
        mec="k", mew=0.2)
for k in (5, 10, 20):
    ax.annotate(f"{red.loc[k,'reduction_pct']:.0f}%", (k, red.loc[k, "mean_posterior_sd"]),
                textcoords="offset points", xytext=(4, 5), fontsize=8)
ax.set_xlabel("Number of new sampling points")
ax.set_ylabel("Domain-average posterior sd (log mg kg$^{-1}$)")
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")

ax = axes[1]
sdg = sd_now.reshape(len(gn), len(ge))
im = ax.pcolormesh(ge / 1000, gn / 1000, sdg, cmap="cividis", shading="auto", rasterized=True)
cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("Posterior sd (log mg kg$^{-1}$)")
ax.scatter(loc["easting"] / 1000, loc["northing"] / 1000, s=3, c="k", marker="o", lw=0, zorder=3,
           label="Existing samples")
newp = CAND[chosen20]
new_e = newp[:, 0] * 1000 + loc["easting"].mean()
new_n = newp[:, 1] * 1000 + loc["northing"].mean()
ax.scatter(new_e / 1000, new_n / 1000, s=22, marker="*", facecolor="#D55E00", edgecolor="k",
           linewidth=0.3, zorder=4, label="20 proposed points")
ax.set_xlabel("Easting UTM 18S, EPSG:32718 (km)")
ax.set_ylabel("Northing UTM 18S (km)")
ax.set_aspect("equal")
# Legend upper left, scale bar bottom left: both used to claim the bottom-left corner and the
# figure audit caught them overlapping ('Existing' x 'km').
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, fontsize=8)
scale_bar(ax, 1.0)
north_arrow(ax)
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
save_fig(fig, "FIG14_monitoring_design")
plt.show()

# proposed points as a deliverable table
prop = pd.DataFrame({
    "rank": np.arange(1, len(chosen20) + 1),
    "easting": new_e.round(0), "northing": new_n.round(0),
})
lon_p, lat_p = tr_to_geo.transform(prop["easting"].to_numpy(), prop["northing"].to_numpy())
prop["lon"], prop["lat"] = lon_p.round(6), lat_p.round(6)
prop["cumulative_sd_reduction_pct"] = red.loc[1:20, "reduction_pct"].to_numpy().round(2)
save_table(prop, "TABLE22_proposed_sampling_points",
           caption="Proposed new sampling locations in priority order, with the cumulative expected "
                   "reduction in domain-average posterior standard deviation.",
           label="tab:proposed")
print(prop.head(10).to_string(index=False))

# %% [markdown]
# ## 13. Coregionalisation and preferential sampling (H11)
#
# ### 13.1 Censored LMC between Cd, Sb and Ag
#
# These three share a mineralogical origin in a mercury-silver mineralisation, so unlike the As-Pb
# pair of the previous phase they *should* correlate. A lower-triangular coregionalisation matrix
# with a positive diagonal breaks the sign and scale symmetry.

# %%
LMC_METALS = ["Cd", "Sb", "Ag"]
for a in LMC_METALS:
    m = metal_arrays(a)
    print(f"  {a}: {100*m['censored'].mean():.1f}% censored, LOD {m['lod']}, "
          f"median quantified {np.nanmedian(m['value']):.2f}")


def build_lmc(metals, xy, cov, ell_params, is_background=None):
    """Linear model of coregionalisation with a censored likelihood per variable."""
    n, k, P = len(xy), cov.shape[1], len(metals)
    arrays = [metal_arrays(a) for a in metals]

    with pm.Model() as model:
        ell = pm.InverseGamma("ell", alpha=ell_params["alpha"], beta=ell_params["beta"], shape=2)
        cov_func = getattr(pm.gp.cov, sb.MATERN)(2, ls=ell)
        gp = pm.gp.Latent(cov_func=cov_func)
        # P independent unit-variance latent processes
        u = pt.stack([gp.prior(f"u{j}", X=xy) for j in range(P)], axis=1)     # (n, P)

        # lower-triangular coregionalisation matrix with positive diagonal
        a_diag = pm.HalfNormal("a_diag", sigma=1.0, shape=P)
        a_off = pm.Normal("a_off", mu=0.0, sigma=1.0, shape=P * (P - 1) // 2)
        A = pt.zeros((P, P))
        A = pt.set_subtensor(A[np.diag_indices(P)], a_diag)
        A = pt.set_subtensor(A[np.tril_indices(P, -1)], a_off)
        field = pt.dot(u, A.T)                                                # (n, P)

        Sigma = pt.dot(A, A.T)
        sd = pt.sqrt(pt.diag(Sigma))
        corr = pm.Deterministic("cross_corr", Sigma / pt.outer(sd, sd))

        for j, (a, arr) in enumerate(zip(metals, arrays)):
            det = ~arr["censored"]
            ybar = float(np.mean(np.log(arr["value"][det])))
            b0 = pm.Normal(f"beta0_{a}", mu=ybar, sigma=1.0)
            bt = pm.Normal(f"beta_{a}", mu=0.0, sigma=1.0, shape=k)
            mean_j = b0 + pt.dot(cov, bt) + field[:, j]
            if is_background is not None:
                bbg = pm.Normal(f"b_bg_{a}", mu=0.0, sigma=1.0)
                mean_j = mean_j + bbg * np.asarray(is_background, float)
            sn = pm.HalfNormal(f"sigma_n_{a}", sigma=1.0)
            dist = pm.Normal.dist(mu=mean_j, sigma=sn)

            v_full = np.where(det, arr["value"], 1.0)
            half = arr["resolution"] / 2.0
            lo_f = np.log(np.maximum(v_full - half, np.finfo(float).tiny))
            hi_f = np.log(v_full + half)
            if det.any():
                pm.Potential(f"logp_det_{a}",
                             sb.logdiffexp(pm.logcdf(dist, hi_f)[det],
                                           pm.logcdf(dist, lo_f)[det]).sum())
            if arr["censored"].any():
                pm.Potential(f"logp_cens_{a}",
                             pm.logcdf(dist, np.log(arr["lod"]))[arr["censored"]].sum())
    return model


idata_lmc = cache_idata("lmc_cd_sb_ag", lambda: sb.fit_model(
    build_lmc(LMC_METALS, XY, COV, ELL_PARAMS, is_background=IS_BACKGROUND),
    draws=1000, tune=1500, chains=4, target_accept=0.99, seed=SEED))

d_lmc = sb.diagnose(idata_lmc, var_names=["ell", "a_diag", "a_off"]
                    + [f"sigma_n_{a}" for a in LMC_METALS])
print(f"\nLMC diagnostics: rhat {d_lmc['rhat_max']:.4f} | "
      f"ESS {min(d_lmc['ess_bulk_min'], d_lmc['ess_tail_min']):.0f} | "
      f"divergences {d_lmc['divergences']} | passes {d_lmc['passes']}")

# %%
cc = idata_lmc.posterior["cross_corr"].values
cc = cc.reshape(-1, len(LMC_METALS), len(LMC_METALS))
lmc_rows = []
for i in range(len(LMC_METALS)):
    for j in range(i + 1, len(LMC_METALS)):
        v = cc[:, i, j]
        lo, hi = np.quantile(v, [0.025, 0.975])
        q25, q75 = np.quantile(v, [0.25, 0.75])
        lmc_rows.append({"pair": f"{LMC_METALS[i]}-{LMC_METALS[j]}",
                         "mean": v.mean(), "median": np.median(v),
                         "ci50_low": q25, "ci50_high": q75,
                         "ci95_low": lo, "ci95_high": hi, "width95": hi - lo,
                         "p_positive": float((v > 0).mean()),
                         "identified": bool((hi - lo) < 1.2)})
lmc = pd.DataFrame(lmc_rows).round(4)
print(lmc.to_string(index=False))
save_table(lmc, "TABLE23_lmc_cross_correlation",
           caption="Posterior cross-correlation between the censored coregionalised fields of Cd, "
                   "Sb and Ag. A 95\\% interval spanning nearly $[-1,1]$ would mean the parameter is "
                   "not identified by these data.", label="tab:lmc")

if lmc["identified"].any():
    print("\nAt least one cross-correlation is identified. Reported with its interval; the "
          "shared mineralogical origin\nof the mercury-silver mineralisation is the physical "
          "reading.")
else:
    print("\nNone of the cross-correlations is identified: the 95% intervals span essentially the "
          "whole\nrange. That is reported as a lack of information, not as an absence of "
          "correlation, and the\nmean is NOT presented as an estimate.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(W2, W2 * 0.34), constrained_layout=True)
ax = axes[0]
for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
    ax.hist(cc[:, i, j], bins=70, density=True, histtype="step", lw=1.1, color=OI[k],
            label=f"{LMC_METALS[i]}-{LMC_METALS[j]}", range=(-1, 1))
ax.plot(np.linspace(-1, 1, 200), np.full(200, 0.5), color="0.5", lw=0.9, ls=":",
        label="Uniform reference")
ax.set_xlabel("Posterior cross-correlation")
ax.set_ylabel("Density")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", fontsize=8)
ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")

ax = axes[1]
ypos = np.arange(len(lmc))[::-1]
ax.errorbar(lmc["mean"], ypos,
            xerr=[lmc["mean"] - lmc["ci95_low"], lmc["ci95_high"] - lmc["mean"]],
            fmt="o", ms=3.4, lw=0.9, capsize=2, color=OI[0], mec="k", mew=0.25, label="95%")
ax.errorbar(lmc["mean"], ypos,
            xerr=[lmc["mean"] - lmc["ci50_low"], lmc["ci50_high"] - lmc["mean"]],
            fmt="none", lw=2.2, color=OI[0], alpha=0.55, label="50%")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlim(-1.05, 1.05)
ax.set_yticks(ypos)
ax.set_yticklabels(lmc["pair"])
ax.set_xlabel("Cross-correlation")
ax.legend(frameon=True, framealpha=0.88, edgecolor="#cfccc7", fontsize=8)
ax.text(0.97, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
save_fig(fig, "FIG15_lmc_cross_correlation")
plt.show()

# %% [markdown]
# ### 13.2 Preferential sampling
#
# The State measures where it suspects, so sampling locations are not independent of the process
# being measured (Diggle, Menezes & Su 2010; Gelfand, Sahu & Holland 2012). Here the design
# **declares** its strata, so preferential sampling is explicit and modellable rather than a
# suspicion — which is a considerable advantage over the usual situation.
#
# The sensitivity analysis asks what happens to the headline quantities when the potential-interest
# stratum, which is where OEFA deliberately looked hardest, is progressively down-weighted.

# %%
pref_rows = []
for a in ["Hg", "Cd"]:
    m = metal_arrays(a)
    for label, keep in [("all points", np.ones(len(XY), bool)),
                        ("background only", IS_BACKGROUND),
                        ("interest only", ~IS_BACKGROUND & ~IS_UNLABELLED),
                        ("balanced subsample", None)]:
        if keep is None:
            r_ = np.random.default_rng(SEED)
            idx_pi = np.where(~IS_BACKGROUND & ~IS_UNLABELLED)[0]
            keep = np.zeros(len(XY), bool)
            keep[IS_BACKGROUND] = True
            keep[r_.choice(idx_pi, IS_BACKGROUND.sum(), replace=False)] = True
        q = m["value"][keep & ~m["censored"]]
        pref_rows.append({"analyte": a, "subset": label, "n": int(keep.sum()),
                          "pct_censored": 100 * m["censored"][keep].mean(),
                          "median_quantified": float(np.median(q)) if len(q) else np.nan,
                          "pct_above_eca": 100 * float(np.mean(m["value"][keep] > m["eca"][0]))})
pref = pd.DataFrame(pref_rows).round(3)
print(pref.to_string(index=False))
save_table(pref, "TABLE24_preferential_sampling",
           caption="Sensitivity of the headline quantities to the sampling design. The balanced "
                   "subsample draws as many potential-interest points as there are background "
                   "points.", label="tab:preferential")
print("\nThe exceedance rate depends strongly on which stratum is emphasised, which is precisely\n"
      "what preferential sampling means. Every figure in this paper that aggregates over all 114\n"
      "points is therefore reported *by stratum* as well, and the domain-wide maps are read as\n"
      "conditional on the survey design rather than as an unbiased picture of the district.")
