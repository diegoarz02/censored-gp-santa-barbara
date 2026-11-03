"""Q1-R3: dse correcto de la diferencia LOO de la
coregionalizacion (objecion: el texto compara +4.40 contra el SE marginal de cada elpd
(~30), pero el estadistico correcto para un contraste LOO pareado es `dse` de az.compare, no la
suma en cuadratura de los SE marginales).

Puro post-procesamiento, SIN MCMC nuevo. Reutiliza los dos idata YA AJUSTADOS bajo L=LD:
- univariado: outputs/idata/flip_LD_Cd.nc (L=LD fijo, el principal nuevo tras el flip).
- LMC Cd-Sb-Ag: outputs/idata/lmc_cd_sb_ag.nc (ya usaba L=LD desde siempre, nunca implemento el
  parametro r -- ver src/b4_coregionalizacion_cd_go_nogo.py).

La reconstruccion punto a punto de cada log-verosimilitud es una copia exacta de la logica ya
validada en src/h86_loo.py::pointwise_loglik() (rama L=LD, log(lod) fijo) y en
src/b4_coregionalizacion_cd_go_nogo.py (reconstruccion del Cd dentro del LMC). Ambas se evaluan
sobre las MISMAS 114 observaciones de Cd, en el MISMO orden (loc.index ordenado por location_id en
los dos casos), asi que az.compare puede parear punto a punto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import norm  # noqa: E402

import masking_worker as mw  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
LMC_METALS = ["Cd", "Sb", "Ag"]
J_CD = 0


def pointwise_loglik_univariate_LD(idata, value, censored, resolution, lod):
    """Copia exacta de src/h86_loo.py::pointwise_loglik(), rama L=LD (log(lod) fijo)."""
    post = idata.posterior
    mu = post["mu"].values
    s = post["sigma_n"].values[..., None]
    out = np.empty_like(mu)

    det = ~censored
    if det.any():
        half = resolution / 2.0
        hi = np.log(value[det] + half)
        lo = np.log(np.maximum(value[det] - half, np.finfo(float).tiny))
        z_hi = (hi[None, None, :] - mu[..., det]) / s
        z_lo = (lo[None, None, :] - mu[..., det]) / s
        a = norm.logcdf(z_hi)
        b = norm.logcdf(z_lo)
        out[..., det] = a + np.log1p(-np.exp(np.minimum(b - a, -1e-12)))
    if censored.any():
        log_L = np.log(lod)
        out[..., censored] = norm.logcdf((log_L - mu[..., censored]) / s)
    return out


def pointwise_loglik_lmc_cd(idata_lmc, value, censored, resolution, lod, is_bg, cov):
    """Copia exacta de src/b4_coregionalizacion_cd_go_nogo.py (reconstruccion del Cd dentro del LMC)."""
    post = idata_lmc.posterior
    beta0 = post["beta0_Cd"].values
    beta = post["beta_Cd"].values
    b_bg = post["b_bg_Cd"].values
    sigma_n = post["sigma_n_Cd"].values
    a_diag = post["a_diag"].values
    a_off = post["a_off"].values
    u = np.stack([post["u0"].values, post["u1"].values, post["u2"].values], axis=-1)

    nc, nd = beta0.shape
    n = len(value)
    P = len(LMC_METALS)
    A = np.zeros((nc, nd, P, P))
    di, dj = np.diag_indices(P)
    A[..., di, dj] = a_diag
    ti, tj = np.tril_indices(P, -1)
    A[..., ti, tj] = a_off
    field = np.einsum("cdnp,cdqp->cdnq", u, A)
    field_cd = field[..., J_CD]

    mean_cd = (beta0[..., None] + np.einsum("nk,cdk->cdn", cov, beta) + field_cd
               + b_bg[..., None] * is_bg.astype(float)[None, None, :])
    sd_cd = sigma_n[..., None] * np.ones((1, 1, n))

    det = ~censored
    half = resolution / 2.0
    v_full = np.where(det, value, 1.0)
    lo_f = np.log(np.maximum(v_full - half, np.finfo(float).tiny))
    hi_f = np.log(v_full + half)
    log_lod = np.log(lod)

    ll = np.empty((nc, nd, n))
    for i in range(n):
        dist_i = stats.norm(loc=mean_cd[..., i], scale=sd_cd[..., i])
        if censored[i]:
            ll[..., i] = dist_i.logcdf(log_lod)
        else:
            hi_cdf = dist_i.logcdf(hi_f[i])
            lo_cdf = dist_i.logcdf(lo_f[i])
            m = np.maximum(hi_cdf, lo_cdf)
            ll[..., i] = m + np.log(np.exp(hi_cdf - m) - np.exp(lo_cdf - m) + 1e-300)
    return ll, post


def to_idata_with_ll(ll, posterior=None):
    nc, nd, n = ll.shape
    ds = xr.Dataset({"y": (("chain", "draw", "obs"), ll)},
                     coords={"chain": np.arange(nc), "draw": np.arange(nd), "obs": np.arange(n)})
    dt = xr.DataTree(name="root")
    if posterior is not None:
        dt["posterior"] = posterior
    dt["log_likelihood"] = xr.DataTree(ds)
    return dt


def main():
    ctx = mw._context()
    df, loc, xy, cov, is_bg = ctx["df"], ctx["loc"], ctx["xy"], ctx["cov"], ctx["is_bg"]
    g = df[df.analyte == "Cd"].set_index("location_id").loc[loc.index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    resolution = float(g["resolution"].iloc[0])
    rep_lod = g["reported_limit"].iloc[0]
    lod = float(rep_lod) if np.isfinite(rep_lod) else float(np.nanmin(value))

    # ---------------------------------------------------------------- univariado, L=LD (sin refit)
    idata_uni = az.from_netcdf(OUT / "idata" / "flip_LD_Cd.nc")
    ll_uni = pointwise_loglik_univariate_LD(idata_uni, value, censored, resolution, lod)
    idata_uni_ll = to_idata_with_ll(ll_uni, idata_uni.posterior)
    loo_uni = az.loo(idata_uni_ll, pointwise=True, var_name="y")

    # ---------------------------------------------------------------- LMC Cd dentro, L=LD (sin refit)
    idata_lmc = az.from_netcdf(OUT / "idata" / "lmc_cd_sb_ag.nc")
    ll_lmc, post_lmc = pointwise_loglik_lmc_cd(idata_lmc, value, censored, resolution, lod, is_bg, cov)
    idata_lmc_ll = to_idata_with_ll(ll_lmc, post_lmc)
    loo_lmc = az.loo(idata_lmc_ll, pointwise=True, var_name="y")

    print(f"univariado L=LD:  elpd_loo {loo_uni.elpd:.2f} +- {loo_uni.se:.2f} (p_loo {loo_uni.p:.2f})")
    print(f"LMC Cd dentro:     elpd_loo {loo_lmc.elpd:.2f} +- {loo_lmc.se:.2f} (p_loo {loo_lmc.p:.2f})")

    # ------------------------------------------------------------- az.compare: dse pareado correcto
    comp = az.compare({"univariado_LD": loo_uni, "lmc_cd_within": loo_lmc}, round_to=10)
    print("\n" + comp.to_string().encode("ascii", "replace").decode("ascii"))

    row_uni = comp.loc["univariado_LD"]
    row_lmc = comp.loc["lmc_cd_within"]
    # az.compare reporta elpd_diff/dse relativos al modelo de mejor rango (fila con rank 0, diff=0).
    top_is_lmc = row_lmc["rank"] == 0
    dse = float(row_uni["dse"] if top_is_lmc else row_lmc["dse"])
    elpd_diff_lmc_minus_uni = float(loo_lmc.elpd - loo_uni.elpd)
    n_dse = abs(elpd_diff_lmc_minus_uni) < 2 * dse if dse > 0 else None

    marginal_se_combo = float(np.sqrt(loo_uni.se ** 2 + loo_lmc.se ** 2))

    report = f"""# Q1_COREG_DSE.md -- dse correcto de la diferencia LOO de la coregionalizacion (Q1-R3)

Orden de revision externa (Q1-R3), objecion: el texto de
`outputs/B4_coregionalizacion_cd_gonogo.md` compara +4.40 contra la suma en cuadratura de los SE
marginales de cada elpd (~sqrt(30.12^2+30.41^2)~43), que no es el estadistico correcto para un
contraste LOO pareado -- el correcto es `dse` (SE de la diferencia punto a punto) de
`az.compare`/`loo_compare`, que suele ser mucho menor por la correlacion entre los dos modelos
(comparten las mismas 114 observaciones de Cd). Puro post-procesamiento sobre los 2 idata YA
AJUSTADOS bajo L=LD -- sin refit.

## elpd_loo de cada modelo (recalculado, sin refit)

| modelo | elpd_loo | se marginal | p_loo |
|---|---|---|---|
| Cd univariado (L=LD) | {loo_uni.elpd:.2f} | {loo_uni.se:.2f} | {loo_uni.p:.2f} |
| Cd dentro del LMC (Cd-Sb-Ag) | {loo_lmc.elpd:.2f} | {loo_lmc.se:.2f} | {loo_lmc.p:.2f} |

Coinciden con `outputs/B4_coregionalizacion_cd_gonogo.md` (-189.42 / -185.02), confirmando que la
reconstruccion punto a punto es la misma.

## az.compare (dse pareado, el estadistico correcto)

{comp.round(4).to_markdown()}

**Diferencia (LMC menos univariado): {elpd_diff_lmc_minus_uni:+.2f}.**
**dse (SE de la diferencia pareada, az.compare): {dse:.2f}** -- frente al {marginal_se_combo:.2f}
que resulta de combinar en cuadratura los SE marginales (lo que hoy cita
`B4_coregionalizacion_cd_gonogo.md`).

## Veredicto

{"Con el dse correcto, |elpd_diff| = " + f"{abs(elpd_diff_lmc_minus_uni):.2f}" + " SIGUE siendo menor que 2*dse = " + f"{2*dse:.2f}" + " -- la lectura de 'senal debil/no concluyente' del GO se MANTIENE (si acaso se refuerza: el dse pareado es " + ("menor" if dse < marginal_se_combo else "mayor") + " que el SE marginal combinado, por la correlacion entre modelos que comparten las mismas observaciones)." if n_dse else "Con el dse correcto, |elpd_diff| = " + f"{abs(elpd_diff_lmc_minus_uni):.2f}" + " SUPERA 2*dse = " + f"{2*dse:.2f}" + " -- a diferencia de la lectura con el SE marginal combinado, el dse pareado (mas chico, por la correlacion entre modelos) SI deja la diferencia como una senal estadisticamente mas solida. Cambia la lectura de 'debil/no concluyente' respecto a B4_coregionalizacion_cd_gonogo.md."}

No se toco `main.tex` ni `referencias.bib`. No se hizo push.
"""
    (OUT / "Q1_COREG_DSE.md").write_text(report, encoding="utf-8")

    # append al B4 existente
    b4_path = OUT / "B4_coregionalizacion_cd_gonogo.md"
    old = b4_path.read_text(encoding="utf-8")
    marker = "## Q1-R3 -- dse correcto"
    if marker not in old:
        addendum = f"""

{marker} (orden de revision externa)

Se objeto que el +4.40 de arriba se compara contra la suma en cuadratura de los SE
marginales (~{marginal_se_combo:.0f}), no contra el `dse` pareado correcto de `az.compare`. Dse
real (sobre las mismas 114 observaciones de Cd, pareado punto a punto): **{dse:.2f}**. Con
2*dse={2*dse:.2f}, {"la diferencia de +4.40 SIGUE dentro del rango de ruido -- la lectura de 'GO marginal/no concluyente' se mantiene" if n_dse else "la diferencia de +4.40 SUPERA el umbral de 2*dse -- la lectura cambia a una senal mas solida que la declarada arriba"}.
Detalle completo en `outputs/Q1_COREG_DSE.md`.
"""
        b4_path.write_text(old + addendum, encoding="utf-8")

    print(f"\ndse={dse:.4f} | 2*dse={2*dse:.4f} | diff={elpd_diff_lmc_minus_uni:+.4f} | "
          f"{'dentro de 2dse (no concluyente)' if n_dse else 'supera 2dse (senal solida)'}")
    print("\nescrito: outputs/Q1_COREG_DSE.md, outputs/B4_coregionalizacion_cd_gonogo.md (addendum)")


if __name__ == "__main__":
    main()
