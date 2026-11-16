"""SERRA-R1 T3 -- estabilidad del baseline GP con L/2: diagnosticos, cobertura restringida a
ajustes sanos, diff de implementacion contra el censurado, e investigacion (sin reajustar, salvo
decision explicita) del mecanismo detras de los Rhat/divergencias altos.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402
from config import FIG, OUT  # noqa: E402

OUT_DIR = OUT / "serra_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def t3a(full):
    rows = []
    for (model, level), g in full.groupby(["model", "level"]):
        rows.append({"model": model, "level": level, "n": len(g),
                    "rhat_max_min": g.rhat_max.min(), "rhat_max_median": g.rhat_max.median(),
                    "rhat_max_max": g.rhat_max.max(),
                    "divergences_sum": int(g.divergences.sum()),
                    "divergences_max": int(g.divergences.max()),
                    "ess_bulk_min_min": g.ess_bulk_min.min(),
                    "ess_bulk_min_median": g.ess_bulk_min.median()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_diag.csv", index=False, encoding="utf-8")

    gp2 = full[full.model == "GP with L/2"]
    cens = full[full.model == "censored GP"]
    checks = {
        "gp_l2_divergences_total": (int(gp2.divergences.sum()), 52),
        "gp_l2_rhat_max": (float(gp2.rhat_max.max()), 1.19),
        "censored_rhat_max": (float(cens.rhat_max.max()), 1.024),
    }
    for k, (got, exp) in checks.items():
        print(f"  {k}: {got:.4g} (esperado {exp})")
    return out


def t3b(full):
    rows = []
    for thr_rhat, thr_name in [(1.01, "rhat_lt_1.01"), (1.05, "rhat_lt_1.05")]:
        piv_keys = ["metal", "level", "replicate"]
        wide = full.pivot_table(index=piv_keys, columns="model",
                                values=["rhat_max", "divergences"])
        ok = ((wide[("rhat_max", "censored GP")] < thr_rhat)
              & (wide[("divergences", "censored GP")] == 0)
              & (wide[("rhat_max", "GP with L/2")] < thr_rhat)
              & (wide[("divergences", "GP with L/2")] == 0))
        healthy_idx = wide[ok].index
        sub = full.set_index(piv_keys).loc[full.set_index(piv_keys).index.isin(healthy_idx)]
        for (metal, level), g in sub.groupby(["metal", "level"]):
            for model in ["censored GP", "GP with L/2", "ordinary kriging L/2"]:
                gm = g[g.model == model]
                if len(gm) == 0:
                    continue
                rows.append({"criterio": thr_name, "metal": metal, "level": level,
                            "model": model, "n": len(gm),
                            "picp95_mean": float(gm.picp95.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_cobertura_restringida.csv", index=False, encoding="utf-8")

    chk = out[(out.criterio == "rhat_lt_1.05") & (out.metal == "As") & (out.level == 0.8)]
    print("  As, 80%, Rhat<1.05 y 0 divergencias en ambos GP (esperado: censurado 0.681, "
         "GP-L/2 0.949, n=3):")
    print(chk[["model", "n", "picp95_mean"]].to_string(index=False))
    return out


def t3c():
    import inspect

    import sbmodel as sb
    src = inspect.getsource(sb.build_censored_gp)
    return {
        "cita_sustitucion": "sbmodel.py build_censored_gp(), L.122-127: "
            "`if substitute_half_lod: y_value = np.where(censored, lod/2.0, y_value); "
            "censored = np.zeros(n, bool)` -- unica rama que distingue los dos modelos.",
        "resto_compartido": "mismo bloque de priors (beta0~N(ybar,sd_beta0), beta~N(0,sd_beta), "
            "b_bg~N(0,sd_beta), ell~InvGamma(ell_params), sigma_tot~HalfNormal(sd_sigma), "
            "rho~Beta(2,2), eta=sigma_tot*sqrt(rho), sigma_n=sigma_tot*sqrt(1-rho) -- la "
            "parametrizacion de varianza particionada que evita el funnel, L.132-190), mismo "
            "nucleo Matern 5/2, mismo gp.Latent, mismo ajuste de ruido por soporte compuesto "
            "(log_tau), y mismo fit_model() (draws/tune/target_accept/chains/seed identicos, "
            "nutpie+numba, L.260-274) -- NO hay construccion ni llamada de ajuste separada para "
            "el baseline, es la MISMA funcion con un solo flag.",
        "codigo_completo_build_censored_gp": src,
    }


def t3d_mechanism(full):
    """Investigacion del mecanismo (sin reajustar nada): en las celdas de GP-L/2 con Rhat>1.05 o
    divergencias, que parametro concentra el problema, y cuantos empates en L/2 hay en el
    entrenamiento."""
    ctx = mw._context()
    df, loc, xy = ctx["df"], ctx["loc"], ctx["xy"]
    sb = ctx["sb"]

    gp2 = full[full.model == "GP with L/2"]
    bad = gp2[(gp2.rhat_max > 1.05) | (gp2.divergences > 0)]
    print(f"  celdas de GP-L/2 con Rhat>1.05 o divergencias: {len(bad)} de {len(gp2)}")

    rows = []
    for _, row in bad.iterrows():
        metal, level, rep, block = row["metal"], row["level"], row["replicate"], row["block"]
        g = df[df.analyte == metal].set_index("location_id").loc[loc.index]
        value = g["value"].to_numpy(float)
        limit = float(np.quantile(value, level))
        censored = value < limit
        labels = sb.spatial_blocks(xy, k=5, seed=20260906)
        train = labels != block
        n_ties = int(np.isclose(value[train & censored], limit / 2.0, rtol=0, atol=1e-9).sum())
        # equivalente: todo punto censurado en train queda exactamente en limit/2 por construccion
        n_censored_train = int(censored[train].sum())
        rows.append({"metal": metal, "level": level, "replicate": rep, "block": block,
                    "rhat_max": row["rhat_max"], "divergences": row["divergences"],
                    "n_train": int(train.sum()), "n_censored_train": n_censored_train,
                    "pct_tied_at_half_lod": 100 * n_censored_train / max(int(train.sum()), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_mecanismo_ties.csv", index=False, encoding="utf-8")
    print(f"  % de entrenamiento empatado en L/2 (GP-L/2, celdas problematicas): "
         f"min={out.pct_tied_at_half_lod.min():.1f}% max={out.pct_tied_at_half_lod.max():.1f}% "
         f"media={out.pct_tied_at_half_lod.mean():.1f}%")
    return out


def main():
    full = pd.read_csv(FIG / "TABLE_H6_MAIN_con_kriging.csv")
    print("=== T3a: diagnosticos por modelo x nivel ===")
    t3a(full)

    print("\n=== T3b: cobertura restringida a ajustes sanos ===")
    t3b(full)

    print("\n=== T3c: diff de implementacion GP-L/2 vs censurado ===")
    diff = t3c()
    print(" ", diff["cita_sustitucion"])
    print(" ", diff["resto_compartido"])

    print("\n=== T3d: mecanismo (investigacion, SIN reajustar nada) ===")
    mech = t3d_mechanism(full)

    with open(OUT_DIR / "baseline_diff_implementacion.md", "w", encoding="utf-8") as f:
        f.write("# T3c -- diff de implementacion GP-L/2 vs censurado\n\n")
        f.write(diff["cita_sustitucion"] + "\n\n")
        f.write(diff["resto_compartido"] + "\n\n")
        f.write("## Codigo completo de build_censored_gp (fuente unica, ambos modelos)\n\n")
        f.write("```python\n" + diff["codigo_completo_build_censored_gp"] + "\n```\n")

    print("\nescrito: outputs/serra_r1/baseline_diag.csv, baseline_cobertura_restringida.csv, "
         "baseline_diff_implementacion.md, baseline_mecanismo_ties.csv")


if __name__ == "__main__":
    main()
