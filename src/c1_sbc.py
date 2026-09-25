"""C1 (corrida 3) -- Simulation-Based Calibration del modelo principal (Talts et al. 2018).

Convierte "el modelo es correcto" en una figura verificable: se dibuja theta del prior, se simula
y con el mismo generador (incluida la censura), se reajusta el modelo a la simulacion con el
protocolo BARATO ya validado en F3 (500 tune / 500 draws / 2 cadenas, correlacion 0.9994 contra
NUTS completo -- medido, no asumido, TIEMPOS_MEDIDOS.md), y se guarda el rango del valor
verdadero dentro de la posterior para cada parametro clave. Con muchas replicas, los rangos deben
ser uniformes si el modelo y la inferencia son correctos.

Regla 3 (medir, no calcular): se mide 1 replica antes de comprometerse a las 200, y el numero real
de replicas ejecutadas queda documentado con el porque.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CK_DIR = OUT / "c1_sbc"
CK_DIR.mkdir(parents=True, exist_ok=True)
CHEAP = dict(draws=500, tune=500, chains=2, target_accept=0.95, cores=2)
PARAMS = ["eta", "ell_0", "ell_1", "sigma_n", "beta0"]
N_TARGET = 200


def simulate_one(ctx, rng):
    """Dibuja theta del prior y simula y con el mismo generador que build_censored_gp."""
    xy, cov, is_bg = ctx["xy"], ctx["cov"], ctx["is_bg"]
    ep = ctx["ell_params"]
    n = len(xy)

    ell = np.array([stats_invgamma_rvs(ep["alpha"], ep["beta"], rng),
                    stats_invgamma_rvs(ep["alpha"], ep["beta"], rng)])
    sigma_tot = abs(rng.normal(0, 0.7))
    rho = rng.beta(2, 2)
    eta = sigma_tot * np.sqrt(rho)
    sigma_n = sigma_tot * np.sqrt(1 - rho)
    beta0 = rng.normal(3.0, 0.7)          # media log-concentracion tipica del sitio
    beta = rng.normal(0, 0.5, size=cov.shape[1])
    b_bg = rng.normal(0, 0.5)

    a = xy / ell
    d2 = ((a[:, None] - a[None]) ** 2).sum(-1)
    K = eta ** 2 * sb._matern52(d2, ell)
    K[np.diag_indices(n)] += 1e-8
    f = rng.multivariate_normal(np.zeros(n), K)
    mu = beta0 + cov @ beta + f + b_bg * is_bg.astype(float)
    y_true_log = mu + rng.normal(0, sigma_n, n)
    value = np.exp(y_true_log)

    # censura con un LOD y resolucion reales (As), para que el mecanismo sea el mismo que en los
    # datos de verdad, no un umbral inventado
    g_as = ctx["df"][ctx["df"].analyte == "As"].set_index("location_id").loc[ctx["loc"].index]
    lod = float(g_as["reported_limit"].iloc[0])
    resolution = float(g_as["resolution"].iloc[0])
    censored = value < lod

    theta_true = {"eta": eta, "ell_0": ell[0], "ell_1": ell[1], "sigma_n": sigma_n,
                 "beta0": beta0}
    return value, censored, lod, resolution, theta_true


def stats_invgamma_rvs(alpha, beta, rng):
    return beta / rng.gamma(alpha, 1.0)   # X~Gamma(a,1) => beta/X ~ InvGamma(a,beta)


def run_replica(i, ctx):
    ck = CK_DIR / f"rep_{i:04d}.json"
    if ck.exists():
        try:
            o = json.loads(ck.read_text(encoding="utf-8"))
            if o.get("run_id") == RUN_ID:
                return o
        except Exception:                                             # noqa: BLE001
            pass

    rng = np.random.default_rng(20260923 + i)
    xy, cov, is_bg, is_comp = ctx["xy"], ctx["cov"], ctx["is_bg"], ctx["is_comp"]
    value, censored, lod, resolution, theta_true = simulate_one(ctx, rng)
    min_det = float(value[~censored].min()) if (~censored).any() else None

    t0 = time.time()
    model = sb.build_censored_gp(
        xy, cov, np.where(censored, np.nan, value), censored, resolution, lod,
        ell_params=ctx["ell_params"], is_background=is_bg, is_composite=is_comp,
        anisotropic=True, limit_as_parameter=False, min_detected=min_det)
    try:
        idata = sb.fit_model(model, seed=20260923 + i, **CHEAP)
        dt = time.time() - t0
        diag = sb.diagnose(idata)
        post = idata.posterior
        n_draws_total = post.sizes["chain"] * post.sizes["draw"]
        ranks = {}
        for p in PARAMS:
            if p.startswith("ell_"):
                j = int(p[-1])
                draws_p = post["ell"].values[..., j].ravel()
            else:
                draws_p = post[p].values.ravel()
            ranks[p] = int((draws_p < theta_true[p]).sum())
        out = {"run_id": RUN_ID, "rep": i, "seconds": dt, "n_draws_total": n_draws_total,
              "rhat_max": diag["rhat_max"], "divergences": diag["divergences"],
              "ranks": ranks, "theta_true": theta_true, "ok": True}
    except Exception as exc:                                          # noqa: BLE001
        out = {"run_id": RUN_ID, "rep": i, "ok": False, "error": repr(exc)[:300]}
    ck.write_text(json.dumps(out, default=float, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ctx = mw._context()

    # --------------------------------------------------------------------------- regla 3: medir
    # Continuacion (2026-09-23, a peticion explicita de Diego: "ejecuta todo lo pendiente, no
    # dejes nada en stand by"). La primera pasada corrio 73/200 bajo un techo de tiempo de 90 min;
    # esta corrida completa las 200 sin techo -- ya no es una estimacion, es lo que falta medido
    # directamente: 127 replicas nuevas x el costo ya medido por replica en la primera pasada.
    already = len(list(CK_DIR.glob("rep_*.json")))
    t0 = time.time()
    r0 = run_replica(0, ctx)                       # cacheada si already>0: sirve de verificacion,
    dt0 = time.time() - t0                         # no de sonda de costo en ese caso
    if already >= N_TARGET:
        print(f"[C1] ya hay {already} replicas -- nada pendiente")
    else:
        per_rep_s = 74.0 if already > 0 else dt0    # medido en la pasada anterior: ~90 min/73 rep
        proj_min = per_rep_s * (N_TARGET - already) / 60
        print(f"[C1] {already} réplicas ya calculadas, faltan {N_TARGET - already} -> "
             f"proyección {proj_min:.0f} min, sin techo: se corre hasta las {N_TARGET}")
        observe(f"C1 continuacion: {already} replicas ya calculadas, corriendo las "
                f"{N_TARGET - already} restantes hasta completar {N_TARGET}, sin techo de tiempo "
                f"(peticion explicita de completar todo lo pendiente).", "Bloque C - corrida 3")
    n_reps = N_TARGET

    rows = [r0]
    for i in range(1, n_reps):
        rows.append(run_replica(i, ctx))
        if i % 20 == 0:
            print(f"  {i}/{n_reps} réplicas", flush=True)

    ok_rows = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok_rows)} de {len(rows)} réplicas completadas sin error")

    rank_df = pd.DataFrame([{"rep": r["rep"], **r["ranks"], "n_draws_total": r["n_draws_total"]}
                            for r in ok_rows])
    rank_df.insert(0, "run_id", RUN_ID)
    rank_df.to_csv(FIG / "TABLE_C1_sbc_ranks.csv", index=False, encoding="utf-8")

    n_total = int(ok_rows[0]["n_draws_total"]) if ok_rows else 1000
    from scipy import stats as sps
    chi2_rows = []
    for p in PARAMS:
        ranks = rank_df[p].to_numpy()
        # ranks deberian ser ~ Uniforme(0, n_total): chi-cuadrado sobre 10 bins
        bins = np.linspace(0, n_total, 11)
        obs, _ = np.histogram(ranks, bins=bins)
        exp = len(ranks) / 10
        chi2, pval = sps.chisquare(obs, f_exp=np.full(10, exp))
        chi2_rows.append({"param": p, "chi2": chi2, "p_value": pval,
                          "uniforme": bool(pval > 0.05)})
    chi2_df = pd.DataFrame(chi2_rows)
    print("\n=== uniformidad de los rangos (chi-cuadrado, H0: uniforme) ===")
    print(chi2_df.round(4).to_string(index=False))

    # ------------------------------------------------------------------------------ figura
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W15, ef.W15 * 0.5), constrained_layout=True)
    for ax, p in zip(axes, ["eta", "sigma_n"]):
        ranks = rank_df[p].to_numpy() / n_total
        ax.hist(ranks, bins=10, range=(0, 1), color=ef.C_CENSORED, edgecolor="k", linewidth=0.4,
               alpha=0.85)
        ax.axhline(len(ranks) / 10, color=ef.C_REF, lw=1.0, ls="--")
        ax.set_xlabel(f"normalised rank, {p}")
        ax.set_ylabel("replicates")
        ef.soft_grid(ax)
    # No message_title: this is a body-of-article figure, the message (and the uniform/deviates
    # verdict) goes in the LaTeX caption and in outputs/C1_sbc.md, not printed on the image.
    ef.save_fig(fig, "FIG35_sbc_ranks")
    plt.close(fig)

    fail_rate = 1 - len(ok_rows) / len(rows) if rows else 0
    txt = f"""{stamp_header()}

# C1 (corrida 3) — Simulation-Based Calibration (Talts et al. 2018)

## Protocolo

Por réplica: dibujar θ del prior del modelo principal, simular `y` con el mismo generador
(incluida la censura, con LOD y resolución reales de As para que el mecanismo sea realista),
reajustar `build_censored_gp` a la simulación con el protocolo barato ya validado en F3
(500 tune / 500 draws / 2 cadenas — correlación 0.9994 contra NUTS completo, medido, no asumido),
y guardar el rango del valor verdadero dentro de las muestras posteriores de cada parámetro.

## Cuántas réplicas, y por qué

Primera pasada: 73 de 200 réplicas, bajo un techo de tiempo de 90 min fijado para no bloquear el
resto de la corrida 3. **Completadas las {len(rows)} de las {N_TARGET} objetivo en una segunda
pasada**, a petición explícita de no dejar nada pendiente — **{len(ok_rows)} completadas sin
error** ({100*fail_rate:.1f} % de fallo, casos donde el ajuste no convergió o lanzó una excepción).

## Resultado — uniformidad de los rangos

{chi2_df.round(4).to_markdown(index=False)}

{"**Los rangos son compatibles con uniforme en todos los parámetros probados** (p > 0.05): no hay evidencia de que el modelo o la inferencia estén sesgados, con la potencia completa de las {} réplicas objetivo.".format(len(ok_rows)) if chi2_df['uniforme'].all() else "**Al menos un parámetro se desvía de uniforme** — revisar el histograma de rangos (`FIG35_sbc_ranks`) para el patrón (forma de U = subestima la incertidumbre; forma de campana = la sobreestima) antes de confiar en la calibración del modelo para ese parámetro."}

Figura `FIG35_sbc_ranks` (histogramas de rango para η y σ_n). Tabla completa de rangos
`TABLE_C1_sbc_ranks.csv`. Checkpoints por réplica en `outputs/c1_sbc/` (reanudable: una réplica ya
calculada no se repite).

## Estado

**Las {N_TARGET} réplicas objetivo están completas**, sin techo de tiempo pendiente. No queda
ninguna réplica de SBC por correr.
"""
    (OUT / "C1_sbc.md").write_text(txt, encoding="utf-8")
    observe(f"C1 COMPLETADO a las {N_TARGET} replicas objetivo (venia de una primera pasada de 73 "
            f"bajo techo de tiempo; completado a peticion explicita de no dejar nada pendiente). "
            f"{len(ok_rows)} de {len(rows)} sin error ({100*fail_rate:.1f}%% de fallo). "
            f"Uniformidad de rangos: {'todos los parametros compatibles con uniforme' if chi2_df['uniforme'].all() else 'al menos un parametro se desvia'}. "
            f"Escrito outputs/C1_sbc.md, TABLE_C1_sbc_ranks.csv, FIG35_sbc_ranks. Ya no queda SBC "
            f"pendiente.", "Bloque C - corrida 3")
    print("\nescrito: outputs/C1_sbc.md, TABLE_C1_sbc_ranks.csv, FIG35_sbc_ranks")


if __name__ == "__main__":
    main()
