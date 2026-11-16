# T3c -- diff de implementacion GP-L/2 vs censurado

sbmodel.py build_censored_gp(), L.122-127: `if substitute_half_lod: y_value = np.where(censored, lod/2.0, y_value); censored = np.zeros(n, bool)` -- unica rama que distingue los dos modelos.

mismo bloque de priors (beta0~N(ybar,sd_beta0), beta~N(0,sd_beta), b_bg~N(0,sd_beta), ell~InvGamma(ell_params), sigma_tot~HalfNormal(sd_sigma), rho~Beta(2,2), eta=sigma_tot*sqrt(rho), sigma_n=sigma_tot*sqrt(1-rho) -- la parametrizacion de varianza particionada que evita el funnel, L.132-190), mismo nucleo Matern 5/2, mismo gp.Latent, mismo ajuste de ruido por soporte compuesto (log_tau), y mismo fit_model() (draws/tune/target_accept/chains/seed identicos, nutpie+numba, L.260-274) -- NO hay construccion ni llamada de ajuste separada para el baseline, es la MISMA funcion con un solo flag.

## Codigo completo de build_censored_gp (fuente unica, ambos modelos)

```python
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

```
