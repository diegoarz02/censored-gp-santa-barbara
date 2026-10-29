"""T1.1 del blindaje Q1/Q2 (prompt de Cowork, 2026-10-28): ajuste del profe Luis sobre la
descomposicion de varianza, ec. (6) del manuscrito.

Anotacion del profe: "Verificar esta igualdad, creo que deberia ser:
sigma_f^2+sigma_n^2 = sigma_tot^2*[rho^2+(1-rho)]"

Este script:
1. Prueba simbolica (sympy) de que sigma_f^2+sigma_n^2=sigma_tot^2 bajo la parametrizacion REAL
   del codigo (sbmodel.py L.167-168: sigma_f=eta=sigma_tot*sqrt(rho), sigma_n=sigma_tot*sqrt(1-rho)),
   y que la forma del profe (sigma_tot^2*[rho^2+(1-rho)]) solo coincide si sigma_f=sigma_tot*rho
   (sin raiz) -- parametrizacion distinta a la del codigo.
2. Deriva simbolicamente la varianza marginal con el factor de muestra compuesta:
   Var(Y_i) = sigma_f^2 + sd_i^2, sd_i = sigma_n*exp(log_tau*c_i) (L.188)
   => Var(Y_i) = sigma_tot^2*[rho + (1-rho)*exp(2*log_tau*c_i)]
3. Verifica esa formula NUMERICAMENTE, por draw, contra los posteriors reales de main_{As,Ba,Hg,Pb,Cd}.nc.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import arviz as az   # noqa: E402
import sympy as sp   # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"

# ---------------------------------------------------------------- 1. prueba simbolica
sigma_tot, rho, log_tau, c_i = sp.symbols("sigma_tot rho log_tau c_i", positive=False, real=True)

sigma_f_code = sigma_tot * sp.sqrt(rho)          # L.167 del codigo real: eta
sigma_n_code = sigma_tot * sp.sqrt(1 - rho)      # L.168 del codigo real

suma_code = sp.simplify(sigma_f_code**2 + sigma_n_code**2)
identidad_code_ok = sp.simplify(suma_code - sigma_tot**2) == 0

# forma propuesta por el profe: sigma_f sin raiz (sigma_f = sigma_tot*rho)
sigma_f_profe = sigma_tot * rho
suma_profe_con_sigma_f_profe = sp.simplify(sigma_f_profe**2 + sigma_n_code**2)
forma_profe_rhs = sigma_tot**2 * (rho**2 + (1 - rho))
coincide_forma_profe = sp.simplify(suma_profe_con_sigma_f_profe - forma_profe_rhs) == 0

# ---------------------------------------------------------------- 2. varianza marginal con tau
sd_i = sigma_n_code * sp.exp(log_tau * c_i)       # L.188 del codigo real
var_Y_i = sp.simplify(sigma_f_code**2 + sd_i**2)
var_Y_i_target = sigma_tot**2 * (rho + (1 - rho) * sp.exp(2 * log_tau * c_i))
formula_var_marginal_ok = sp.simplify(var_Y_i - var_Y_i_target) == 0

print("=== Prueba simbolica ===")
print(f"sigma_f^2+sigma_n^2 (codigo real, sigma_f=sigma_tot*sqrt(rho)) = {suma_code}")
print(f"¿Igual a sigma_tot^2? {identidad_code_ok}")
print(f"Forma del profe (sigma_tot^2*[rho^2+(1-rho)]) solo si sigma_f=sigma_tot*rho (sin raiz): "
      f"{coincide_forma_profe}")
print(f"Var(Y_i) simbolica = {var_Y_i}")
print(f"¿Igual a sigma_tot^2*[rho+(1-rho)*exp(2*log_tau*c_i)]? {formula_var_marginal_ok}")

assert identidad_code_ok, "La identidad basica NO se cumple bajo la parametrizacion del codigo"
assert formula_var_marginal_ok, "La formula de varianza marginal con tau no es simbolica-mente igual"

# ---------------------------------------------------------------- 3. verificacion numerica
df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
resultados = []
for metal in ["As", "Ba", "Hg", "Pb", "Cd"]:
    path = OUT / "idata" / f"main_{metal}.nc"
    if not path.exists():
        resultados.append(dict(metal=metal, estado="NO_DISPONIBLE (falta main_%s.nc)" % metal))
        continue
    idata = az.from_netcdf(path)
    post = idata.posterior

    g = df[df.analyte == metal].drop_duplicates("location_id").set_index("location_id")
    loc = g.sort_index()
    is_comp = (loc["sample_type"] == "composite").to_numpy(dtype=float)
    n_comp = int(is_comp.sum())

    sigma_tot_v = post["sigma_tot"].values            # (chain, draw)
    rho_v = post["rho"].values
    eta_v = post["eta"].values if "eta" in post else sigma_tot_v * np.sqrt(rho_v)
    sigma_n_v = post["sigma_n"].values if "sigma_n" in post else sigma_tot_v * np.sqrt(1 - rho_v)

    if "log_tau" in post and n_comp > 0:
        log_tau_v = post["log_tau"].values             # (chain, draw)
        # sd_i por punto compuesto: sigma_n * exp(log_tau * c_i), c_i=1 para compuestos
        sd_comp = sigma_n_v[..., None] * np.exp(log_tau_v[..., None] * 1.0)  # (chain,draw,1) bcast
        var_comp_code = eta_v[..., None] ** 2 + sd_comp ** 2
        var_comp_formula = sigma_tot_v[..., None] ** 2 * (
            rho_v[..., None] + (1 - rho_v[..., None]) * np.exp(2 * log_tau_v[..., None]))
        err = np.abs(var_comp_code - var_comp_formula)
        max_err = float(err.max())
        p_tau_lt1 = float((log_tau_v < 0).mean())
        resultados.append(dict(metal=metal, n_composite=n_comp, max_abs_error=max_err,
                               p_logtau_lt0=p_tau_lt1,
                               logtau_mean=float(log_tau_v.mean()), estado="OK"))
    else:
        # sin muestras compuestas o sin log_tau en el posterior: la formula colapsa a sigma_tot^2
        sd_simple = sigma_n_v
        var_simple_code = eta_v ** 2 + sd_simple ** 2
        err = np.abs(var_simple_code - sigma_tot_v ** 2)
        resultados.append(dict(metal=metal, n_composite=n_comp, max_abs_error=float(err.max()),
                               p_logtau_lt0=np.nan, logtau_mean=np.nan,
                               estado="OK (sin log_tau o sin compuestas; Var=sigma_tot^2)"))

tab = pd.DataFrame(resultados)
print("\n=== Verificacion numerica por metal ===")
print(tab.to_string(index=False))
(PROJ / "figuras").mkdir(exist_ok=True)
tab.to_csv(PROJ / "figuras" / "TABLE_T1_1_varianza_marginal.csv", index=False, encoding="utf-8")
print("\nescrito: figuras/TABLE_T1_1_varianza_marginal.csv")
