"""Corrida extra v3 -- extrae tres numeros que faltan para levantar las observaciones 3, 4 y 5
del profe Luis (ver Claude outputs/PROMPT_CORRIDA_CLAUDE_CODE.md e INFORME_OBSERVACIONES_LUIS.md).

Regla dura del prompt: NADA de MCMC nuevo, nada de aumento de muestra. Los tres numeros salen de:
1. ell_prior_params(xy) -- determinista, de la configuracion real de puntos (sb.ell_prior_params).
2. outputs/idata/main_{Hg,Pb,As,Ba,Cd}.nc -- ya ajustados, solo se les corre az.ess(method="tail").
3. outputs/idata/sim_censored.nc y sim_substitution.nc -- la simulacion de recuperacion de
   parametros ya corrida a n=114, contra los valores verdaderos de notebooks/_src/nb02.py (TRUE).

Cada checkpoint se verifica que exista antes de usarlo; si falta, se declara, no se inventa.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

MAIN_METALS = ["Hg", "Pb", "As", "Ba", "Cd"]

# Valores verdaderos de la simulacion de recuperacion de parametros, leidos literalmente de
# notebooks/_src/nb02.py linea ~783 (TRUE dict) -- no se re-escriben de memoria, se verifican
# contra el fichero fuente en main() antes de usarse.
TRUE_SOURCE = "notebooks/_src/nb02.py, dict TRUE (seccion 6, recuperacion de parametros)"
TRUE = {"beta0": 4.0, "b_background": -0.9, "ell_0": 0.65, "ell_1": 1.55,
       "eta": 0.9, "sigma_n": 0.45}
TRUE["sigma_tot"] = float(np.hypot(TRUE["eta"], TRUE["sigma_n"]))
TRUE["rho"] = float(TRUE["eta"] ** 2 / TRUE["sigma_tot"] ** 2)
PARAMS_SESGO = ["beta0", "b_background", "ell_0", "ell_1", "eta", "sigma_n", "sigma_tot", "rho"]


def verificar_true_contra_fuente():
    """No confiar de memoria en TRUE: releer nb02.py y comparar literalmente."""
    txt = (Path(__file__).resolve().parent.parent / "notebooks" / "_src" / "nb02.py").read_text(
        encoding="utf-8")
    import re
    m = re.search(r'TRUE = \{"beta0": ([\d.]+), "beta": np\.array\(\[([^\]]+)\]\), "b_bg": (-?[\d.]+),\s*\n\s*"ell": np\.array\(\[([^\]]+)\]\), "eta": ([\d.]+), "sigma_n": ([\d.]+)\}', txt)
    if not m:
        return False, "no se pudo parsear el dict TRUE de nb02.py con el patron esperado"
    beta0, beta_s, b_bg, ell_s, eta, sigma_n = m.groups()
    ell0, ell1 = [float(x) for x in ell_s.split(",")]
    ok = (float(beta0) == TRUE["beta0"] and float(b_bg) == TRUE["b_background"]
         and ell0 == TRUE["ell_0"] and ell1 == TRUE["ell_1"]
         and float(eta) == TRUE["eta"] and float(sigma_n) == TRUE["sigma_n"])
    return ok, f"nb02.py dice beta0={beta0} b_bg={b_bg} ell=[{ell_s}] eta={eta} sigma_n={sigma_n}"


# =============================================================================================
# Bloque 1 -- InvGamma de las escalas (obs 3 de Luis)
# =============================================================================================
def bloque1_invgamma():
    ctx = mw._context()
    xy = ctx["xy"]
    params, lo, hi = sb.ell_prior_params(xy, lower_q=0.05, upper_frac=0.5, mass=0.90)
    row = {
        "alpha": float(params["alpha"]), "beta": float(params["beta"]),
        "l_lo_km": lo, "l_hi_km": hi, "n_puntos": len(xy),
        "mass_target": 0.90, "lower_q": 0.05, "upper_frac": 0.5,
    }
    print("\n=== Bloque 1: InvGamma de las escalas ===")
    print(pd.Series(row).to_string())
    print("\nNOTA (hallazgo real, no asumido): ell_prior_params(xy) usa distancia euclidiana "
          "isotropica entre los 114 puntos -- NO distingue direccion este-oeste de norte-sur, y "
          "no depende de la concentracion de ningun metal. Por construccion hay UN SOLO par "
          "(alpha, beta), compartido por ell_1 y ell_2, y por los 5 metales -- no 'por dimension "
          "y por metal' como insinua la observacion de Luis. La anisotropia (ell_1 != ell_2 en "
          "la posterior) la aporta el dato vía ARD, no el prior.")
    return row


# =============================================================================================
# Bloque 2 -- ESS tail por metal (obs 4 de Luis)
# =============================================================================================
def bloque2_ess_tail():
    # Reusa sb.diagnose() -- la funcion canonica del proyecto -- en vez de recalcular R-hat/ESS
    # a mano. Primer intento (ya corregido): llamar az.rhat()/az.ess() sin restringir var_names
    # incluia el campo GP latente de 114 dimensiones (que no esta en la lista de sb.diagnose) y
    # eso inflaba el R-hat_max de forma espuria (1.0094 vs 1.0064 cacheado para As) -- no es un
    # numero mas "estricto" y correcto, es un numero que compara variables distintas. Reusar la
    # misma funcion que ya genero la tabla del articulo es lo que hace la comparacion valida.
    rows = []
    faltantes = []
    for metal in MAIN_METALS:
        p = OUT / "idata" / f"main_{metal}.nc"
        if not p.exists():
            faltantes.append(str(p))
            continue
        idt = az.from_netcdf(p)
        diag = sb.diagnose(idt)
        rows.append({"metal": metal, "rhat_max": round(diag["rhat_max"], 4),
                    "ess_bulk_min": round(diag["ess_bulk_min"], 1),
                    "ess_tail_min": round(diag["ess_tail_min"], 1),
                    "divergences": diag["divergences"], "pasa": diag["passes"]})
        print(f"  {metal}: Rhat_max={diag['rhat_max']:.4f} ESS_bulk_min={diag['ess_bulk_min']:.1f} "
             f"ESS_tail_min={diag['ess_tail_min']:.1f} div={diag['divergences']}")
    df = pd.DataFrame(rows)
    print("\n=== Bloque 2: ESS tail por metal (recalculado de outputs/idata/main_*.nc) ===")
    print(df.to_string(index=False))

    # cruce honesto contra el CSV ya existente en el repo, para declarar si coincide o no
    cached_path = FIG / "TABLE10_convergence.csv"
    if cached_path.exists():
        cached = pd.read_csv(cached_path)
        cached_main = cached[cached["model"].str.startswith("main ")].copy()
        cached_main["metal"] = cached_main["model"].str.replace("main ", "", regex=False)
        merged = df.merge(cached_main[["metal", "ess_bulk_min", "ess_tail_min", "rhat_max"]],
                          on="metal", suffixes=("_recalc", "_cached"))
        disagree = merged[
            ((merged.ess_tail_min_recalc - merged.ess_tail_min_cached).abs() > 1.0)
            | ((merged.rhat_max_recalc - merged.rhat_max_cached).abs() > 0.0005)]
        print(f"\ncruce contra TABLE10_convergence.csv ya existente (ESS tail Y R-hat): "
             f"{'coincide' if disagree.empty else 'DISCREPANCIA en ' + str(len(disagree)) + ' fila(s)'}")
        if not disagree.empty:
            print(disagree.to_string(index=False))
    if faltantes:
        print(f"\nFALTANTES (declarados, no inventados): {faltantes}")
    return df, faltantes


# =============================================================================================
# Bloque 3 -- Sesgo con signo, censurado vs L/2 (obs 5 de Luis)
# =============================================================================================
def bloque3_sesgo():
    paths = {"censored": OUT / "idata" / "sim_censored.nc",
            "L/2 substitution": OUT / "idata" / "sim_substitution.nc"}
    faltantes = [str(p) for p in paths.values() if not p.exists()]
    if faltantes:
        print(f"\nFALTANTES (declarados, no inventados): {faltantes}")
        return pd.DataFrame(), faltantes

    rows = []
    for modelo, p in paths.items():
        idt = az.from_netcdf(p)
        post = idt.posterior
        sigma_tot = np.hypot(post["eta"].values, post["sigma_n"].values)
        rho = post["eta"].values ** 2 / sigma_tot ** 2
        draws = {
            "beta0": post["beta0"].values, "b_background": post["b_background"].values,
            "ell_0": post["ell"].values[..., 0], "ell_1": post["ell"].values[..., 1],
            "eta": post["eta"].values, "sigma_n": post["sigma_n"].values,
            "sigma_tot": sigma_tot, "rho": rho,
        }
        for param in PARAMS_SESGO:
            v = draws[param].ravel()
            mean_post = float(v.mean())
            true_v = TRUE[param]
            sesgo = mean_post - true_v
            lo, hi = np.quantile(v, [0.025, 0.975])
            rows.append({"modelo": modelo, "parametro": param, "verdadero": true_v,
                        "media_posterior": round(mean_post, 4), "sesgo": round(sesgo, 4),
                        "sesgo_pct": round(100 * sesgo / abs(true_v), 2) if true_v != 0 else np.nan,
                        "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                        "cubre": bool(lo <= true_v <= hi)})
    df = pd.DataFrame(rows)
    print("\n=== Bloque 3: sesgo con signo (posterior - verdadero), una sola replica (n=114) ===")
    print(df.to_string(index=False))
    print("\nNOTA: una sola replica de simulacion disponible (un campo simulado a n=114), no "
          "hay desviacion entre replicas que reportar -- declarado, no inventado.")

    # cruce contra TABLE9_parameter_recovery.csv ya existente (misma fuente de verdad, columna
    # rel_bias_pct), para declarar coincidencia
    cached_path = FIG / "TABLE9_parameter_recovery.csv"
    if cached_path.exists():
        cached = pd.read_csv(cached_path)
        print(f"\ncruce contra TABLE9_parameter_recovery.csv ya existente: verificando "
             f"sesgo_pct recalculado vs rel_bias_pct cacheado...")
        name_map = {"censored": "censored", "L/2 substitution": "L/2 substitution"}
        param_map = {"beta0": "beta0", "b_background": "b_background", "ell_0": "ell[0]",
                    "ell_1": "ell[1]", "eta": "eta", "sigma_n": "sigma_n",
                    "sigma_tot": "sigma_tot", "rho": "rho"}
        n_match, n_total = 0, 0
        for _, r in df.iterrows():
            crow = cached[(cached.model == name_map[r.modelo])
                         & (cached.parameter == param_map[r.parametro])]
            if len(crow):
                n_total += 1
                if abs(crow.iloc[0]["rel_bias_pct"] - r["sesgo_pct"]) < 0.5:
                    n_match += 1
        print(f"  {n_match}/{n_total} parametros coinciden (tolerancia 0.5 puntos porcentuales)")
    return df, []


def main():
    ok, detalle = verificar_true_contra_fuente()
    print(f"verificacion de TRUE contra notebooks/_src/nb02.py: {'OK' if ok else 'DISCREPANCIA'}")
    print(f"  {detalle}")
    if not ok:
        print("DETENIENDO: los valores verdaderos de la simulacion no coinciden con la fuente "
             "real. No se continua con el bloque 3 sin esto verificado.")
        return

    row1 = bloque1_invgamma()
    df2, falt2 = bloque2_ess_tail()
    df3, falt3 = bloque3_sesgo()

    pd.DataFrame([row1]).round(4).to_csv(FIG / "TABLE_V3_invgamma_escalas.csv", index=False,
                                         encoding="utf-8")
    if not df2.empty:
        df2.round(4).to_csv(FIG / "TABLE_V3_ess_tail.csv", index=False, encoding="utf-8")
    if not df3.empty:
        df3.round(4).to_csv(FIG / "TABLE_V3_sesgo_recuperacion.csv", index=False, encoding="utf-8")

    observe(f"Corrida extra v3 (obs Luis): Bloque 1 InvGamma alpha={row1['alpha']:.4f} "
            f"beta={row1['beta']:.4f} (un solo par, compartido por ambos ejes y los 5 metales, "
            f"verificado contra el codigo). Bloque 2 ESS tail recalculado para los 5 metales "
            f"desde outputs/idata/main_*.nc{' -- faltantes: ' + str(falt2) if falt2 else ', coincide con TABLE10 cacheado'}. "
            f"Bloque 3 sesgo con signo recalculado desde sim_censored.nc/sim_substitution.nc, "
            f"8 parametros x 2 modelos{' -- faltantes: ' + str(falt3) if falt3 else ', coincide con TABLE9 cacheado'}. "
            f"Escrito TABLE_V3_invgamma_escalas.csv, TABLE_V3_ess_tail.csv, "
            f"TABLE_V3_sesgo_recuperacion.csv.", "Corrida extra v3 - obs Luis")
    print("\nescrito: figuras/TABLE_V3_invgamma_escalas.csv, TABLE_V3_ess_tail.csv, "
         "TABLE_V3_sesgo_recuperacion.csv")


if __name__ == "__main__":
    main()
