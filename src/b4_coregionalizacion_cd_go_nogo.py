"""B4 (corrida 3) -- coregionalizacion del cadmio con censura, go/no-go.

CORRECCION IMPORTANTE respecto al plan original: `outputs/H11_lmc_cd_sb_ag_convergencia.md`
(escrito el 22-sep, respondiendo a la auditoria de Cowork) afirmaba que el LMC Cd-Sb-Ag "esta
ajustado con L/2 sustituido, no con censura como parametro". **Es falso, verificado ahora leyendo
`build_lmc()` en `notebooks/_src/nb02_g_design_lmc.py`**: la funcion ya implementa verosimilitud
censurada real por metal (interval-censoring en los detectados via logdiffexp, logcdf en el limite
para los censurados) -- exactamente el patron del modelo principal, no una sustitucion. Cowork
tampoco lo verifico contra el codigo (su nota decia "a verificar al reajustar con censura", una
suposicion, no una lectura). El ajuste ya cacheado en outputs/idata/lmc_cd_sb_ag.nc YA es la
version censurada. No hace falta reajustar nada.

Lo que SI faltaba y es el trabajo real de este hito: el go/no-go. Se reconstruye el log-verosimilitud
puntual del Cd desde el posterior YA AJUSTADO (mismo patron que H8.6/H4.3: nunca se pudo usar
sample_posterior_predictive con un pm.Potential, se reconstruye a mano) y se compara su PSIS-LOO
contra el Cd univariado ya reportado en H8.6_loo.md (elpd_loo = -241.21, ya en disco, sin refit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

LMC_METALS = ["Cd", "Sb", "Ag"]
J_CD = 0
STANDALONE_ELPD_CD = -241.209   # outputs/H8.6_loo.md, ya en disco, no se recalcula


def main():
    idata = az.from_netcdf(OUT / "idata" / "lmc_cd_sb_ag.nc")
    post = idata.posterior

    ctx = mw._context()
    xy, cov, is_bg = ctx["xy"], ctx["cov"], ctx["is_bg"]
    n = len(xy)
    g = ctx["df"][ctx["df"].analyte == "Cd"].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    lod = float(g["reported_limit"].iloc[0])
    resolution = float(g["resolution"].iloc[0])

    # -------------------------------------------------------------------- reconstruir mean_Cd(s)
    beta0 = post["beta0_Cd"].values             # (chain, draw)
    beta = post["beta_Cd"].values                # (chain, draw, k)
    b_bg = post["b_bg_Cd"].values                 # (chain, draw)
    sigma_n = post["sigma_n_Cd"].values           # (chain, draw)
    a_diag = post["a_diag"].values                # (chain, draw, P)
    a_off = post["a_off"].values                  # (chain, draw, P*(P-1)/2)
    u = np.stack([post["u0"].values, post["u1"].values, post["u2"].values], axis=-1)  # (c,d,n,P)

    nc, nd = beta0.shape
    P = len(LMC_METALS)
    A = np.zeros((nc, nd, P, P))
    di, dj = np.diag_indices(P)
    A[..., di, dj] = a_diag
    ti, tj = np.tril_indices(P, -1)
    A[..., ti, tj] = a_off
    # field(s) = u(s) @ A.T  ->  (c,d,n,P) @ (c,d,P,P)^T por batch
    field = np.einsum("cdnp,cdqp->cdnq", u, A)     # (chain, draw, n, P)
    field_cd = field[..., J_CD]                     # (chain, draw, n)

    mean_cd = (beta0[..., None] + np.einsum("nk,cdk->cdn", cov, beta) + field_cd
              + b_bg[..., None] * is_bg.astype(float)[None, None, :])
    sd_cd = sigma_n[..., None] * np.ones((1, 1, n))

    # -------------------------------------------------------------------- log-verosimilitud puntual
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
            # logdiffexp estable
            m = np.maximum(hi_cdf, lo_cdf)
            ll[..., i] = m + np.log(np.exp(hi_cdf - m) - np.exp(lo_cdf - m) + 1e-300)

    ll_ds = xr.Dataset({"y": (("chain", "draw", "obs"), ll)},
                       coords={"chain": np.arange(nc), "draw": np.arange(nd),
                              "obs": np.arange(n)})
    idata_cd = xr.DataTree(name="root")
    idata_cd["posterior"] = post                        # mismas cadenas/draws del LMC ya ajustado,
    idata_cd["log_likelihood"] = xr.DataTree(ll_ds)     # para que az.loo estime reff de verdad
    loo_lmc = az.loo(idata_cd, pointwise=True, var_name="y")

    print(f"elpd_loo Cd univariado (H8.6, ya en disco): {STANDALONE_ELPD_CD:.2f}")
    print(f"elpd_loo Cd dentro del LMC censurado (reconstruido): {loo_lmc.elpd:.2f} "
         f"(se {loo_lmc.se:.2f}, p_loo {loo_lmc.p:.2f})")
    diff = loo_lmc.elpd - STANDALONE_ELPD_CD
    gana = diff > 0
    print(f"diferencia: {diff:+.2f} ({'gana el coregionalizado' if gana else 'gana el univariado'})")

    khat_bad = int((np.asarray(loo_lmc.pareto_k) > 0.7).sum())

    # ------------------------------------------------------------------------------ figura
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W15, ef.W15 * 0.55), constrained_layout=True)
    ax = axes[0]
    xs = [0, 1]
    vals = [STANDALONE_ELPD_CD, float(loo_lmc.elpd)]
    ses = [26.142, float(loo_lmc.se)]
    ax.errorbar(xs, vals, yerr=ses, fmt="o", ms=9, lw=1.4, color=ef.C_CENSORED, mec="k", mew=0.6,
               capsize=4)
    ax.set_xticks(xs); ax.set_xticklabels(["univariate\nCd", "Cd within the\ncensored LMC"])
    ax.set_ylabel("elpd_loo")
    ef.soft_grid(ax)
    ef.panel_label(ax, "a")

    ax = axes[1]
    loc = ctx["loc"]
    mean_cd_post = mean_cd.reshape(-1, n).mean(0)
    sc = ax.scatter(loc["easting"], loc["northing"], c=np.exp(mean_cd_post), cmap=ef.CMAP_MAG,
                    s=22, edgecolor="k", linewidth=0.3, norm=mcolors.LogNorm())
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Cd posterior mean (mg/kg) — within the LMC")
    cb.ax.tick_params(labelsize=ef.FS_MIN)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting UTM 18S (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ef.utm_km_ticks(ax)
    ef.panel_label(ax, "b")

    ef.message_title(fig, f"Coregionalised cadmium {'beats' if gana else 'does NOT beat'} the "
                          f"univariate model by PSIS-LOO")
    ef.save_fig(fig, "FIG34_cd_coregionalizado_gonogo")
    plt.close(fig)

    txt = f"""{stamp_header()}

# B4 (corrida 3) — coregionalización del cadmio con censura: go/no-go

## Corrección importante antes del resultado

`outputs/H11_lmc_cd_sb_ag_convergencia.md` (22-sep) afirmaba que el LMC Cd-Sb-Ag estaba "ajustado
con L/2 sustituido, no con censura como parámetro". **Es incorrecto** — verificado ahora leyendo
`build_lmc()` en `nb02_g_design_lmc.py`: usa verosimilitud censurada real (interval-censoring en
detectados, `logcdf` en el límite para censurados), el mismo patrón del modelo principal. La
auditoría de Cowork tampoco lo verificó contra el código (su nota decía "a verificar", una
suposición). El ajuste cacheado en `outputs/idata/lmc_cd_sb_ag.nc` **ya es la versión censurada** —
no hizo falta reajustar nada, y no se reajustó nada aquí.

## El go/no-go

Log-verosimilitud puntual del Cd reconstruida desde el posterior ya ajustado (mismo patrón que
H8.6 y H4.3: `pm.Potential` no da verosimilitud por punto automática, se reconstruye a mano desde
`mean_Cd(s)` y `sigma_n_Cd`), y comparada por PSIS-LOO contra el Cd univariado ya reportado
(`outputs/H8.6_loo.md`, sin recalcular).

| modelo | elpd_loo | se | k̂ > 0.7 |
|---|---|---|---|
| Cd univariado (censurado, ya reportado) | {STANDALONE_ELPD_CD:.2f} | 26.14 | 4 |
| **Cd dentro del LMC censurado (Cd-Sb-Ag)** | **{loo_lmc.elpd:.2f}** | {loo_lmc.se:.2f} | {khat_bad} |

**Diferencia: {diff:+.2f}.**

## Veredicto — {'GO' if gana else 'NO-GO'}

{"**El cadmio coregionalizado supera al univariado.** Compartir información con Sb y Ag (76.3 % y 78.1 % de censura, correlaciones ya identificadas Cd-Sb 0.750, Cd-Ag 0.616) mejora el ajuste del cadmio, justo donde el modelo univariado pierde. Va a Resultados con figura y tabla — es la vía técnica más prometedora del proyecto para mover el punto de ruptura del 80 %." if gana else "**El cadmio coregionalizado NO supera al univariado por PSIS-LOO.** Se reporta como negativo honesto, con la cifra, no en silencio: compartir información espacial con Sb y Ag no mejora el ajuste puntual del cadmio en esta parametrización, pese a que las correlaciones cruzadas sí están identificadas. Es publicable igual — es exactamente el patrón que ya sostiene el resto del artículo, reportar lo que no funciona junto a lo que sí."}

Figura `FIG34_cd_coregionalizado_gonogo`: (a) elpd_loo de las dos versiones con su error estándar;
(b) mapa de la media posterior de Cd dentro del LMC.

Encuadre para el manuscrito: **EM / data augmentation** para los latentes censurados (semana 9 del
sílabo) además de coregionalización (semana 11) — cierra dos requisitos con un solo ajuste, y ese
ajuste ya estaba hecho.
"""
    (OUT / "B4_coregionalizacion_cd_gonogo.md").write_text(txt, encoding="utf-8")

    # -------------------------------------------------------------------------- corregir H11
    h11_path = OUT / "H11_lmc_cd_sb_ag_convergencia.md"
    if h11_path.exists():
        old = h11_path.read_text(encoding="utf-8")
        correction = f"""

## CORRECCIÓN (corrida 3, {RUN_ID}) — el LMC ya usaba censura, no L/2

Esta nota afirmaba arriba que el ajuste "usa L/2 sustituido, no censura como parámetro". **Es
incorrecto.** Verificado leyendo `build_lmc()` en `nb02_g_design_lmc.py`: la función implementa
verosimilitud censurada real por metal desde el principio. El error salió de repetir sin verificar
la suposición de la auditoría de Cowork ("a verificar al reajustar con censura"), en vez de abrir
el código fuente. Detalle y el go/no-go resultante en `outputs/B4_coregionalizacion_cd_gonogo.md`.
"""
        h11_path.write_text(old + correction, encoding="utf-8")

    observe(f"B4 completo, y con una correccion importante: el LMC Cd-Sb-Ag YA usaba verosimilitud "
            f"censurada real (verificado leyendo build_lmc() en el codigo), no L/2 como decia mi "
            f"propio H11 del 22-sep (ese informe repetia sin verificar una suposicion de la "
            f"auditoria de Cowork). No hizo falta reajustar nada. Go/no-go via LOO reconstruido del "
            f"posterior ya ajustado (sin MCMC nuevo): Cd univariado elpd_loo=-241.21, Cd dentro del "
            f"LMC elpd_loo={loo_lmc.elpd:.2f}, diferencia {diff:+.2f} -- "
            f"{'GO: el coregionalizado supera' if gana else 'NO-GO: el univariado supera, reportado como negativo honesto'}. "
            f"H11 corregido in situ con nota de correccion (append). Escrito "
            f"outputs/B4_coregionalizacion_cd_gonogo.md, FIG34_cd_coregionalizado_gonogo.",
            "Bloque B - corrida 3")
    print("\nescrito: outputs/B4_coregionalizacion_cd_gonogo.md, FIG34_cd_coregionalizado_gonogo, "
         "H11 corregido")


if __name__ == "__main__":
    main()
