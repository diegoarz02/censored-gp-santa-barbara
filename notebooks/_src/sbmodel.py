# -*- coding: utf-8 -*-
"""Shared model code for the Santa Bárbara censored spatial analysis.

This lives in an importable module rather than in a notebook cell on purpose. The masking
experiment of H6 runs across processes, and on Windows `multiprocessing` uses *spawn*, so every
worker re-imports the module that defines the work function. A notebook cell cannot be re-imported;
a module can.

Contents
--------
`logdiffexp`               numerically stable log(exp(a) - exp(b))
`ell_prior_params`         InverseGamma parameters calibrated to the point configuration
`build_censored_gp`        the hierarchical censored Gaussian-process model
`fit_model`                NUTS wrapper (nutpie) with a consistent diagnostic summary
`predict_field`            exact marginal posterior prediction at new locations
`crps_samples`, `metrics`  scoring, including the proper joint predictive score
`spatial_blocks`           k-means spatial blocks for cross-validation
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ["SD_BETA", "SD_SIGMA", "SD_BETA0", "logdiffexp", "ell_prior_params", "build_censored_gp", "fit_model", "predict_field",
           "crps_samples", "metrics", "spatial_blocks", "ordinary_kriging", "MATERN"]

MATERN = "Matern52"


# --------------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------------
def logdiffexp(a, b):
    """log(exp(a) - exp(b)) for a > b, computed stably.

    Verified against a high-precision reference over interval widths from 1e-5 to 0.5 standard
    deviations: maximum absolute error 2e-8. The plain difference of exponentials underflows in
    this range; the density-times-width approximation is off by 1e-2 at the wide end.
    """
    return a + pt.log1p(-pt.exp(b - a))


def ell_prior_params(xy, lower_q=0.05, upper_frac=0.5, mass=0.90):
    """InverseGamma parameters putting `mass` of the prior between two data-driven distances.

    The lower edge is the `lower_q` quantile of pairwise distances and the upper edge is
    `upper_frac` of the maximum distance. A Gaussian process cannot identify a length scale below
    the typical spacing of the points nor above the diameter of the domain: the likelihood is flat
    in both regimes. The InverseGamma is chosen because it penalises `ell -> 0`, which is the
    pathological regime where the GP degenerates into pure noise.
    """
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(xy), 1)
    lo = float(np.quantile(d[iu], lower_q))
    hi = float(upper_frac * d[iu].max())
    params = pm.find_constrained_prior(
        pm.InverseGamma, lower=lo, upper=hi, mass=mass, init_guess={"alpha": 3.0, "beta": 2.0})
    return params, lo, hi


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
# Prior scales, calibrated by prior predictive check (see notebook section 5.1). The first attempt
# used 1.0 everywhere and implied mercury concentrations of 1.8e7 mg/kg, above the physical bound of
# 1e6 mg/kg (100 % of the sample mass). These scales cover the observed range at the 99th percentile
# while putting essentially no mass on impossible values.
SD_BETA = 0.5      # coefficients on standardised covariates and the stratum shift
SD_SIGMA = 0.7     # HalfNormal scale of the total standard deviation
SD_BETA0 = 0.7     # spread of the intercept around the empirical mean


def build_censored_gp(xy, cov, y_value, censored, resolution, lod, *, ell_params,
                      is_background=None, is_composite=None, anisotropic=True,
                      limit_as_parameter=True, interval_censoring=True,
                      substitute_half_lod=False, min_detected=None, y_bar=None,
                      sd_beta=SD_BETA, sd_sigma=SD_SIGMA, sd_beta0=SD_BETA0,
                      prior_family="default", pc_lambda=None,
                      use_hsgp=False, hsgp_m=(25, 25), hsgp_c=2.0):
    """Hierarchical Gaussian-process model with a censored likelihood.

    Model
    -----
        y(s) = beta0 + b_bg * background(s) + beta' x(s) + f(s) + eps(s)
        f ~ GP(0, Matern 5/2),   eps ~ N(0, sigma_n^2)

    Priors follow `notas_obsidian/Priors del GP censurado - como se definen.md`:

    * `beta ~ N(0, sd_beta)` on standardised covariates, `sd_beta = 0.5` by default.
    * `beta0 ~ N(mean(y), sd_beta0)` — empirical Bayes, declared as such.
      The scales are not round numbers by accident: they come from the prior predictive check of
      section 5.1, which rejected the initial choice of 1.0 everywhere.
    * `ell ~ InverseGamma` calibrated by `ell_prior_params`. With `anisotropic=True` there is one
      length scale per axis (ARD), which represents the north-south continuity of the valley found
      in the directional variogram.
    * Variance partition instead of independent priors on the two variances:
      `sigma_tot ~ HalfNormal(sd_sigma)`, `rho ~ Beta(2, 2)`, `eta = sigma_tot*sqrt(rho)`,
      `sigma_n = sigma_tot*sqrt(1-rho)`. This removes the funnel that produced dozens of divergences
      in the previous phase of the project (same idea as BYM2, Riebler et al. 2016).
    * The quantification limit is a **parameter**: `L = LOD * r`, `r ~ LogNormal(log 3, 0.25)`
      truncated to `[1, min_detected/LOD]`. Centred at 3 because the metrological convention is
      LOD = 3.3 sigma and LOQ = 10 sigma, so LOQ/LOD is about 3 (Currie 1968). The upper bound is
      not arbitrary: if a value was quantified at 2.3 then the limit is necessarily below 2.3.

    Likelihood
    ----------
    Detected values are **interval censored** at the laboratory reporting resolution: a result
    printed as 383 with resolution 1 means the true value lies in [382.5, 383.5). Censored values
    contribute the cumulative mass below the limit.

    Parameters
    ----------
    substitute_half_lod : bool
        Baseline behaviour. Replaces every censored value by LOD/2 and treats it as an exact
        observation, which is the practice this study is testing.
    """
    n, k = len(xy), cov.shape[1]
    censored = np.asarray(censored, bool)
    y_value = np.asarray(y_value, float)

    if substitute_half_lod:
        # The substituted value has to replace the concentration itself, not only the log used for
        # centring: the likelihood below reads `y_value`, and leaving NaN there makes the log
        # density undefined and the sampler fail at initialisation.
        y_value = np.where(censored, lod / 2.0, y_value)
        censored = np.zeros(n, bool)

    obs = np.where(censored, np.nan, np.log(np.where(censored, 1.0, y_value)))
    y_bar = float(np.nanmean(obs)) if y_bar is None else y_bar

    with pm.Model() as model:
        # ---- mean -------------------------------------------------------------------------
        beta0 = pm.Normal("beta0", mu=y_bar, sigma=sd_beta0)
        beta = pm.Normal("beta", mu=0.0, sigma=sd_beta, shape=k)
        mean_terms = beta0 + pt.dot(cov, beta)
        if is_background is not None:
            b_bg = pm.Normal("b_background", mu=0.0, sigma=sd_beta)
            mean_terms = mean_terms + b_bg * np.asarray(is_background, float)

        # ---- spatial field ----------------------------------------------------------------
        if anisotropic:
            ell = pm.InverseGamma("ell", alpha=ell_params["alpha"], beta=ell_params["beta"],
                                  shape=2)
        else:
            ell = pm.InverseGamma("ell", alpha=ell_params["alpha"], beta=ell_params["beta"])
        if prior_family == "pc":
            # Penalised-complexity prior (Riebler et al. 2016, following Simpson et al.): the scale
            # gets an Exponential, whose single rate is fixed by one interpretable statement,
            # Prob(sigma_tot > U) = alpha  =>  lambda = -ln(alpha)/U.
            # U is not chosen: `pc_rate_from_bound` solves it so the prior predictive keeps
            # essentially no mass above the physical bound of 1e6 mg/kg. The prior predictive check
            # stops being an exam applied after choosing and becomes the equation that chooses.
            sigma_tot = pm.Exponential("sigma_tot", lam=pc_lambda)
            # Shrinks towards rho = 0, i.e. towards no spatial field: the model has to earn the
            # spatial structure. Labelled PC-motivated rather than "the PC prior", because the
            # exact BYM2 mixing prior has no closed form outside that structure.
            rho = pm.Beta("rho", alpha=1.0, beta=2.0)
        elif prior_family == "vague":
            # Banerjee, Carlin & Gelfand (2015) p. 149 recommend a relatively vague prior on the
            # variance and an informative one on the range. This variant tests that advice.
            sigma_tot = pm.HalfNormal("sigma_tot", sigma=2.0)
            rho = pm.Beta("rho", alpha=1.0, beta=1.0)
        else:
            sigma_tot = pm.HalfNormal("sigma_tot", sigma=sd_sigma)
            rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        eta = pm.Deterministic("eta", sigma_tot * pt.sqrt(rho))
        sigma_n = pm.Deterministic("sigma_n", sigma_tot * pt.sqrt(1.0 - rho))

        cov_func = eta ** 2 * getattr(pm.gp.cov, MATERN)(2, ls=ell)
        if use_hsgp:
            # B2, corrida 3: no es un plan de escalado (con N=114 la inversion de K es barata) --
            # es un test de GEOMETRIA. pm.gp.Latent pone N=114 parametros latentes correlacionados
            # que NUTS tiene que recorrer bajo un target_accept alto; HSGP los sustituye por
            # hsgp_m[0]*hsgp_m[1] coeficientes de base casi independientes (Riutort-Mayol et al.,
            # RECURSOS/), que es una geometria mas facil de muestrear aunque N sea chico.
            gp = pm.gp.HSGP(m=list(hsgp_m), c=hsgp_c, cov_func=cov_func)
        else:
            gp = pm.gp.Latent(cov_func=cov_func)
        f = gp.prior("f", X=xy)
        mu = pm.Deterministic("mu", mean_terms + f)

        # ---- observation noise, with the composite-support factor -------------------------
        if is_composite is not None and is_composite.any():
            # A composite sample averages several sub-samples, so its measurement variance is
            # smaller. log_tau < 0 means composites are less noisy, which is the expected sign.
            log_tau = pm.Normal("log_tau", mu=0.0, sigma=0.5)
            sd_i = sigma_n * pt.exp(log_tau * np.asarray(is_composite, float))
        else:
            sd_i = sigma_n * pt.ones(n)

        # ---- likelihood -------------------------------------------------------------------
        dist = pm.Normal.dist(mu=mu, sigma=sd_i)

        det = ~censored
        if det.any():
            # The observation vectors have to be full length before indexing: `dist` is defined
            # over all n locations, so evaluating logcdf on a shorter array would try to broadcast
            # (n_detected,) against (n,). Censored positions carry a placeholder that is dropped
            # immediately afterwards by the boolean index.
            v_full = np.where(det, y_value, 1.0)
            if interval_censoring:
                half = resolution / 2.0
                lo_full = np.log(np.maximum(v_full - half, np.finfo(float).tiny))
                hi_full = np.log(v_full + half)
                logp_det = logdiffexp(pm.logcdf(dist, hi_full)[det],
                                      pm.logcdf(dist, lo_full)[det])
            else:
                logp_det = pm.logp(dist, np.log(v_full))[det]
            pm.Potential("logp_detected", logp_det.sum())

        if censored.any():
            if limit_as_parameter and min_detected is not None and np.isfinite(lod):
                r_max = float(min_detected / lod)
                if r_max <= 1.0:
                    log_limit = np.log(lod)
                else:
                    r = pm.TruncatedNormal("limit_ratio", mu=3.0, sigma=1.0, lower=1.0,
                                           upper=r_max, initval=min(3.0, 0.5 * (1.0 + r_max)))
                    log_limit = pm.Deterministic("log_limit", pt.log(lod * r))
            else:
                log_limit = np.log(lod)
            pm.Potential("logp_censored", pm.logcdf(dist, log_limit)[censored].sum())

    return model



def pc_rate_from_bound(y_bar, *, bound=1e6, max_mass=1e-4, alpha=0.01, grid=None):
    """Solve for the Exponential rate of a PC prior on the total standard deviation.

    The statement is Prob(sigma_tot > U) = alpha, so lambda = -ln(alpha)/U. What fixes U is not
    taste: it is the largest U whose implied prior predictive keeps less than `max_mass` above the
    physical bound (1e6 mg/kg is 100 % of the sample mass, so anything above it is impossible).

    This inverts the usual order. Instead of choosing a scale and then checking the prior
    predictive, the prior predictive requirement *is* the equation that determines the scale.

    Returns (lambda, U, achieved_mass).
    """
    rng = np.random.default_rng(0)
    grid = grid if grid is not None else np.linspace(0.5, 8.0, 61)
    log_bound = np.log(bound)
    best = None
    for U in grid:
        lam = -np.log(alpha) / U
        s = rng.exponential(1.0 / lam, 40000)
        # one draw of the log concentration: intercept plus a field excursion of size sigma_tot
        y = y_bar + s * rng.standard_normal(40000)
        mass = float((y > log_bound).mean())
        if mass < max_mass:
            best = (float(lam), float(U), mass)
        else:
            break
    if best is None:
        U = float(grid[0])
        best = (float(-np.log(alpha) / U), U, np.nan)
    return best

def fit_model(model, *, draws=2000, tune=2500, chains=4, target_accept=0.99, seed=0,
              progressbar=False, cores=None):
    """Sample with NUTS through nutpie and the numba backend.

    numba is used because this machine has no C compiler available for PyTensor; nutpie compiles
    the log-density through numba instead, which is both available and faster here.
    """
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          cores=cores if cores is not None else chains,
                          nuts_sampler="nutpie", backend="numba",
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar,
                          idata_kwargs={"log_likelihood": False})
    return idata


def diagnose(idata, var_names=None):
    """R-hat, ESS and divergences computed without the two-decimal rounding of `az.summary`."""
    import arviz as az
    if var_names is None:
        var_names = [v for v in ["beta0", "beta", "b_background", "ell", "sigma_tot", "rho",
                                 "log_tau", "limit_ratio"] if v in idata.posterior]

    def worst(ds, fn):
        return float(fn([fn(np.asarray(v.values)) for v in ds.data_vars.values()]))

    rhat = worst(az.rhat(idata, var_names=var_names), np.nanmax)
    ess_b = worst(az.ess(idata, var_names=var_names, method="bulk"), np.nanmin)
    ess_t = worst(az.ess(idata, var_names=var_names, method="tail"), np.nanmin)
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    out = {"rhat_max": rhat, "ess_bulk_min": ess_b, "ess_tail_min": ess_t,
           "divergences": div, "n_draws": n_draws,
           "passes": bool(rhat < 1.01 and min(ess_b, ess_t) > 400 and div == 0)}
    if "energy" in idata.sample_stats:
        e = idata.sample_stats["energy"].values
        de = np.diff(e, axis=1)
        out["bfmi_min"] = float(np.min(np.var(de, axis=1) / np.var(e, axis=1)))
    return out


# --------------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------------
def _matern52(d2, ell=1.0):
    """Matérn 5/2 from an already-scaled squared distance. `ell` is accepted for call
    compatibility and unused: the scaling is applied to the distance before calling."""
    r = np.sqrt(np.maximum(d2, 0.0))
    s5 = np.sqrt(5.0) * r
    return (1.0 + s5 + 5.0 * r ** 2 / 3.0) * np.exp(-s5)


def predict_field(idata, xy_train, xy_new, cov_new, *, is_background_new=None,
                  n_samples=1000, seed=0, anisotropic=True, include_noise=True,
                  block_size=None, rng=None, return_mean=False):
    """Exact marginal posterior prediction of the latent field and of a new observation.

    Computed directly rather than through `gp.conditional`, which rebuilds the PyTensor graph for
    every draw and is prohibitively slow over a prediction grid. Verified against `gp.conditional`
    to 1e-12 in the notebook.

    Returns
    -------
    dict with `f` (latent field draws) and, if `include_noise`, `y` (predictive draws), each of
    shape (n_samples, n_new) in log space. With `return_mean=True` it also returns `mu`, the
    posterior **conditional mean** of the latent field per draw, with no realisation added.

    `mu` exists so the correctness check can have a threshold fixed in advance. Comparing a sampled
    field against the analytic conditional mean leaves Monte Carlo noise of order sd/sqrt(n), and a
    real algebraic error would produce a difference of the same size: a check whose acceptance
    threshold cannot be written before seeing the number is not a check.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    post = idata.posterior
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    flat = n_chain * n_draw
    idx = rng.choice(flat, size=min(n_samples, flat), replace=n_samples > flat)

    def gv(name):
        return post[name].values.reshape(flat, *post[name].shape[2:])[idx]

    ell = gv("ell")
    eta = gv("eta")
    sigma_n = gv("sigma_n")
    beta0 = gv("beta0")
    beta = gv("beta")
    f_tr = gv("f")
    b_bg = gv("b_background") if "b_background" in post else None

    n_tr, n_new = len(xy_train), len(xy_new)
    S = len(idx)
    bs = block_size or n_new
    F = np.empty((S, n_new))
    Y = np.empty((S, n_new)) if include_noise else None
    MU = np.empty((S, n_new)) if return_mean else None

    # mean of the new locations, without the spatial field
    m_new = cov_new @ beta.T + beta0[None, :]
    if b_bg is not None and is_background_new is not None:
        m_new = m_new + np.asarray(is_background_new, float)[:, None] * b_bg[None, :]
    m_new = m_new.T                                       # (S, n_new)

    from scipy.linalg import cho_solve, solve_triangular

    # Squared distances are computed once, outside the loop over posterior draws. Anisotropy only
    # rescales the axes, so a per-draw length scale is applied to the *components* of the squared
    # distance rather than recomputing the whole pairwise difference tensor 4 000 times. Building
    # that (n_new, n_train, 2) tensor inside the loop was costing 290 ms per draw, which put a
    # five-analyte map at eight hours.
    dx_tt = (xy_train[:, None, :] - xy_train[None, :, :]) ** 2      # (n_tr, n_tr, 2)
    dx_nt = (xy_new[:, None, :] - xy_train[None, :, :]) ** 2        # (n_new, n_tr, 2)

    for s in range(S):
        ls = np.atleast_1d(ell[s]).astype(float)
        if ls.size == 1:
            ls = np.repeat(ls, 2)
        inv2 = 1.0 / ls ** 2

        Ktt = eta[s] ** 2 * _matern52(dx_tt @ inv2, 1.0)
        Ktt[np.diag_indices(n_tr)] += 1e-8
        Lc = np.linalg.cholesky(Ktt)
        # cho_solve applies both triangular solves; np.linalg.solve would run a full LU each time
        alpha = cho_solve((Lc, True), f_tr[s])

        for b0 in range(0, n_new, bs):
            b1 = min(b0 + bs, n_new)
            Knt = eta[s] ** 2 * _matern52(dx_nt[b0:b1] @ inv2, 1.0)
            mu_f = Knt @ alpha
            v = solve_triangular(Lc, Knt.T, lower=True, check_finite=False)
            var_f = np.maximum(eta[s] ** 2 - (v ** 2).sum(0), 1e-12)
            if return_mean:
                MU[s, b0:b1] = m_new[s, b0:b1] + mu_f
            F[s, b0:b1] = m_new[s, b0:b1] + mu_f + rng.standard_normal(b1 - b0) * np.sqrt(var_f)
            if include_noise:
                Y[s, b0:b1] = F[s, b0:b1] + rng.standard_normal(b1 - b0) * sigma_n[s]

    out = {"f": F}
    if include_noise:
        out["y"] = Y
    if return_mean:
        out["mu"] = MU
    return out


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------
def crps_samples(y_true, samples):
    """CRPS estimated from predictive draws, per observation.

    Uses the identity CRPS = E|X - y| - 0.5 E|X - X'|, evaluated on sorted draws so the second term
    costs O(n log n) rather than O(n^2).
    """
    y_true = np.asarray(y_true, float)
    out = np.empty(len(y_true))
    for i, y in enumerate(y_true):
        x = np.sort(samples[:, i])
        n = len(x)
        term1 = np.abs(x - y).mean()
        w = 2.0 * np.arange(1, n + 1) - n - 1.0
        term2 = 2.0 * (w * x).sum() / (n * n)
        out[i] = term1 - 0.5 * term2
    return out


def metrics(y_true_log, samples_log, *, censored=None, log_limit=None, prefix=""):
    """Predictive scores.

    `y_true_log` must be the true log concentration at every held-out location, including the ones
    that were censored in training. That is the whole point of the masking experiment: there *is* a
    truth to compare against, so the scores are not restricted to the upper truncated tail.

    When `censored` and `log_limit` are supplied the proper **joint predictive score** is also
    returned: the log density for the detected observations and log P(Y < log L) for the censored
    ones. That score is proper for the actual observation mechanism and, unlike a score computed on
    detections only, cannot be won by predicting high.
    """
    from scipy import stats
    y = np.asarray(y_true_log, float)
    m = samples_log.mean(0)
    out = {
        f"{prefix}rmse": float(np.sqrt(np.mean((m - y) ** 2))),
        f"{prefix}mae": float(np.mean(np.abs(m - y))),
        f"{prefix}bias": float(np.mean(m - y)),
        f"{prefix}crps": float(crps_samples(y, samples_log).mean()),
    }
    for lvl in (0.90, 0.95):
        lo = np.quantile(samples_log, (1 - lvl) / 2, axis=0)
        hi = np.quantile(samples_log, 1 - (1 - lvl) / 2, axis=0)
        out[f"{prefix}picp{int(lvl*100)}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"{prefix}width{int(lvl*100)}"] = float(np.mean(hi - lo))

    if censored is not None and log_limit is not None:
        censored = np.asarray(censored, bool)
        sd = samples_log.std(0)
        mu = samples_log.mean(0)
        ll = np.empty(len(y))
        ll[~censored] = stats.norm.logpdf(y[~censored], mu[~censored], sd[~censored])
        if censored.any():
            ll[censored] = stats.norm.logcdf(log_limit, mu[censored], sd[censored])
        out[f"{prefix}joint_log_score"] = float(ll.mean())
    return out


def spatial_blocks(xy, k=5, seed=0):
    """k-means spatial blocks. Whole blocks are held out so autocorrelation cannot leak."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(xy)


# --------------------------------------------------------------------------------------------
# classical baseline
# --------------------------------------------------------------------------------------------
def ordinary_kriging(xy_train, y_train_log, xy_new, *, n_samples=800, seed=0,
                     variogram_model="spherical"):
    """Ordinary kriging with the censored values already substituted by L/2.

    This is the literal classical baseline: the practice the study is testing. `PyKrige` returns a
    kriging mean and variance, which are turned into Gaussian predictive draws so that the same
    probabilistic scores can be applied to it as to the Bayesian models.

    Ordinary kriging assumes an unknown constant mean, so no covariates enter. That is a fair
    representation of the standard practice rather than a handicap: the comparison of interest is
    the treatment of censoring, and both models see the same substituted data.
    """
    from pykrige.ok import OrdinaryKriging
    rng = np.random.default_rng(seed)
    ok = OrdinaryKriging(xy_train[:, 0], xy_train[:, 1], y_train_log,
                         variogram_model=variogram_model, enable_plotting=False,
                         coordinates_type="euclidean", nlags=12)
    mu, var = ok.execute("points", xy_new[:, 0], xy_new[:, 1])
    mu = np.asarray(mu, float).ravel()
    sd = np.sqrt(np.maximum(np.asarray(var, float).ravel(), 1e-9))
    draws = mu[None, :] + rng.standard_normal((n_samples, len(mu))) * sd[None, :]
    return draws, {"variogram_model": variogram_model,
                   "params": {k: float(v) for k, v in
                              zip(["sill_or_psill", "range", "nugget"], ok.variogram_model_parameters)}}


def verify_predictor(idata, xy_train, xy_new, cov_new, *, is_background_new=None,
                     n_samples=60, seed=0, anisotropic=True, safety=10.0):
    """Check `predict_field` against the conditional mean written out independently.

    The acceptance threshold is fixed **before** running, and it is not a round number pulled from
    the air: for a linear solve the attainable accuracy is bounded by

        |error| <~ eps * cond(K) * |f|

    and the covariance matrices here have condition numbers of order 1e8 to 1e9, so a threshold of
    1e-8 is below what double precision can deliver and would fail on correct code. `safety` is the
    slack over that bound.

    The comparison uses `return_mean=True`, so no realisation is added and the difference is pure
    algebra plus rounding — a sampled comparison would hide an algebraic error under Monte Carlo
    noise of order sd/sqrt(n).

    Returns
    -------
    dict with `max_abs_error`, `threshold`, `cond_median` and `passes`.
    """
    post = idata.posterior
    flat = post.sizes["chain"] * post.sizes["draw"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat, size=min(n_samples, flat), replace=False)

    fast = predict_field(idata, xy_train, xy_new, cov_new,
                         is_background_new=is_background_new, n_samples=n_samples, seed=seed,
                         anisotropic=anisotropic, include_noise=False, return_mean=True)["mu"]

    ell = post["ell"].values.reshape(flat, -1)[idx]
    eta = post["eta"].values.reshape(flat)[idx]
    f_tr = post["f"].values.reshape(flat, -1)[idx]
    beta = post["beta"].values.reshape(flat, -1)[idx]
    beta0 = post["beta0"].values.reshape(flat)[idx]
    b_bg = (post["b_background"].values.reshape(flat)[idx]
            if "b_background" in post else None)

    ref = np.empty((len(idx), len(xy_new)))
    conds = np.empty(len(idx))
    for s in range(len(idx)):
        ls = np.atleast_1d(ell[s]).astype(float)
        a_tr, a_ne = xy_train / ls, xy_new / ls
        K = eta[s] ** 2 * _matern52(((a_tr[:, None] - a_tr[None]) ** 2).sum(-1), ls)
        K[np.diag_indices(len(xy_train))] += 1e-8
        Ks = eta[s] ** 2 * _matern52(((a_ne[:, None] - a_tr[None]) ** 2).sum(-1), ls)
        conds[s] = np.linalg.cond(K)
        ref[s] = Ks @ np.linalg.solve(K, f_tr[s]) + cov_new @ beta[s] + beta0[s]
        if b_bg is not None and is_background_new is not None:
            ref[s] = ref[s] + np.asarray(is_background_new, float) * b_bg[s]

    err = float(np.abs(fast - ref).max())
    thr = float(safety * np.finfo(float).eps * np.median(conds) * np.abs(f_tr).max())
    return {"max_abs_error": err, "threshold": thr, "cond_median": float(np.median(conds)),
            "passes": bool(err < thr)}
