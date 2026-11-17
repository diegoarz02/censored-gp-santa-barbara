"""SERRA-R2 T1 -- inventario de todo lo que depende de main_Hg.nc (ajuste estandar de Hg) o de
posterior_maps.npz, y de si el articulo lo usa. Barrido automatico del codigo + tabla curada.

"En el articulo" se contrasta con main_FINAL_v15.tex (version en espanol, fuera del repo); el manuscrito
SERRA en ingles no esta en disco, asi que lo que solo pudiera aparecer alli figura como "no verificable".
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config import OUT, PROJ  # noqa: E402

OUT_DIR = OUT / "serra_r2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"main_Hg|main_\{|main_%|f\"main_|idata_main|posterior_maps|probe_Hg_strict")


def scan():
    rows = []
    for base in [PROJ / "src", PROJ / "notebooks" / "_src"]:
        for f in sorted(base.glob("*.py")):
            if f.name.startswith("serra_r2_"):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                m = PAT.search(line)
                if m:
                    rows.append({"archivo": f.relative_to(PROJ).as_posix(), "linea": i,
                                 "patron": m.group(0), "texto": line.strip()[:140]})
    return pd.DataFrame(rows)


ART = "main_FINAL_v15.tex (ES)"
CURATED = [
    # ---- dependen del ajuste estandar y el articulo los usa
    ("notebooks/_src/nb02_f_maps.py L.249 -> outputs/posterior_maps.npz",
     "Hg_mean, Hg_median, Hg_sd_f, Hg_sd_y, Hg_p_agri, Hg_p_resid (mapas de Hg)",
     f"SI: Fig. mapaHg = FIG17_Hg (L.553); Fig9 SERRA (Fig. 8 del PDF). {ART}",
     "SI: idata_main['Hg'] = outputs/idata/main_Hg.nc (R-hat 1.0104, ESS bulk 331)",
     "RECALCULADO -> outputs/serra_r2/posterior_maps_Hg_refuerzo.npz"),
    ("notebooks/_src/nb02_f_maps.py L.257 -> figuras/TABLE20_exceedance_summary.csv",
     "% del dominio con P(Hg>6.6)>0.5 = 99.7697", f"SI: tab:exc fila Hg col. P>0.5 = 99.8 y texto L.570. {ART}",
     "SI (sale de Hg_p_agri de posterior_maps.npz)", "RECALCULADO -> hg_antes_despues.csv"),
    ("src/t2_1_area_excedida_posterior.py -> figuras/TABLE_area_excedida_posterior.csv + outputs/area_excedida_posterior_arrays.npz",
     "% del dominio con P>0.8 (96.69) y P>0.95 (76.11); area excedida: mediana, IC95 %",
     f"SI: tab:exc fila Hg cols. P>0.8 = 96.7 y P>0.95 = 76.1. {ART}",
     "SI: lee outputs/idata/main_{metal}.nc, incluido main_Hg.nc", "RECALCULADO -> hg_antes_despues.csv"),
    ("src/t1_1_prueba_varianza.py -> figuras/TABLE_T1_1_varianza_marginal.csv",
     "P(log tau < 0) de Hg = 0.2785", f"SI: L.239 'el mercurio es la excepcion (0.28)'. {ART}",
     "SI: lee main_{As,Ba,Hg,Pb,Cd}.nc", "RECALCULADO -> hg_hiperparametros_antes_despues.csv"),
    ("notebooks/_src/nb02.py L.972-985 -> figuras/TABLE11_posterior_parameters.csv",
     "IC95 % de las 4 covariables de la media de Hg (columnas beta_*_sig)",
     f"SI: L.504 'Ninguna de las cuatro covariables ... sus intervalos creibles contienen el cero en los cinco metales'. {ART}",
     "SI: idata_main['Hg'] = main_Hg.nc",
     "RECALCULADO. HALLAZGO: la frase es FALSA para Hg con AMBOS ajustes (log_dist_dump excluye el cero)"),
    ("notebooks/_src/nb02.py L.949 -> figuras/TABLE10_convergence.csv",
     "R-hat 1.0104 / ESS bulk 331 / ESS tail 460 de Hg (passes = False)",
     f"NO usa esta fila: tab:conv (L.338) trae 1.0059 / 467 / 693 = el ajuste reforzado. {ART}",
     "SI el CSV (fuente rancia para Hg); el articulo no", "NO SE RECALCULA; TRAZABILIDAD apunta al ajuste reforzado"),
    ("notebooks/_src/nb02_f_maps.py L.294 (ell_typ) + src/mapas_f3.py L.225 (ell_km=1.03 fijo)",
     "zona de extrapolacion (celdas a > 1.5 ell_typ del dato mas cercano): 0.772 % = '0.8 %'",
     "SI: pie de la Fig. 8 SERRA ('0.8 % del dominio'); leyenda 'extrapolation (x %)' de todos los mapas",
     "SI, NO ES PURA GEOMETRIA: ell_typ promedia el ell de los cinco idata_main, incluido main_Hg.nc",
     "RECALCULADO -> hg_extrapolacion.csv (0.772 % -> 0.691 % si ell_typ se recalcula)"),
    # ---- dependen del ajuste estandar; el articulo en espanol no los usa
    ("notebooks/_src/nb02.py L.1016 -> figuras/TABLE12_stratum_support_aliasing.csv + FIG7",
     "b_background, log tau, correlacion posterior de Hg", f"NO (v15 no cita TABLE12). {ART}",
     "SI: idata_main['Hg']", "RECALCULADO -> hg_hiperparametros_antes_despues.csv"),
    ("notebooks/_src/nb02.py L.1049 -> figuras/TABLE13_prior_posterior.csv + FIG8",
     "razon ancho posterior/prior de ell, sigma_tot, rho, b_background, log tau (Hg)", f"NO. {ART}",
     "SI: idata_main['Hg']", "RECALCULADO -> hg_hiperparametros_antes_despues.csv"),
    ("notebooks/_src/nb02.py L.1150-1200 -> figuras/TABLE14_posterior_predictive.csv",
     "chequeos predictivos posteriores de Hg", f"NO. {ART}", "SI: idata_main['Hg']",
     "RECALCULADO -> hg_hiperparametros_antes_despues.csv"),
    ("notebooks/_src/nb02.py L.1965 -> figuras/TABLE19_background_vs_mining.csv",
     "factor minero exp(-b_background) de Hg = 1.53 [0.74, 3.13]", f"NO (v15 declara el contraste no identificable). {ART}",
     "SI: idata_main['Hg']", "RECALCULADO -> hg_hiperparametros_antes_despues.csv"),
    ("notebooks/_src/nb02.py L.1408 -> figuras/FIG9_Hg_crps (anotacion de ell y rho)",
     "ell y rho de Hg impresos en la figura del enmascaramiento", f"NO esta entre las figuras de v15. {ART}",
     "SI (solo la anotacion de texto; las curvas son del experimento de enmascaramiento, ajustes propios)",
     "VALORES recalculados (ell, rho); figura no regenerada"),
    ("notebooks/_src/nb02.py L.2278 (DESIGN_METAL='Hg') -> outputs/monitoring_design.npz, figuras/TABLE21, TABLE22, FIG14",
     "diseno de monitoreo: reduccion de sd con 5/10/20 puntos nuevos y puntos elegidos", f"NO (v15 no incluye FIG14 ni TABLE21/22). {ART}",
     "SI: ell, eta, sigma_n de main_Hg.nc y la mascara de extrapolacion", "RECALCULADO -> hg_diseno_monitoreo_antes_despues.csv"),
    ("src/h85a_perdidas.py -> figuras/TABLE29_decision_thresholds.csv + FIG20",
     "umbrales de decision bayesiana a partir de p_agri de Hg", f"NO. {ART}",
     "INDIRECTO: lee Hg_p_agri de posterior_maps.npz", "NO RECALCULADO (no citado); se regenera si el articulo en ingles lo usa"),
    ("src/h86_loo.py -> figuras/TABLE_S5_loo.csv, TABLE_S10_pareto_k.csv",
     "elpd-LOO de Hg (reconstruido desde main_Hg.nc)", f"NO (solo el Cd se cita, tab:lmc). {ART}",
     "SI: lee main_{a}.nc", "NO RECALCULADO (suplementario, no citado)"),
    ("src/t2_3_diagnostico_campo_latente.py -> figuras/TABLE_T2_3_diagnostico_campo_latente.csv",
     "R-hat/ESS del campo latente de Hg desde main_Hg.nc", f"NO se imprime (v15 L.314 diagnostica solo hiperparametros). {ART}",
     "SI: lee main_{metal}.nc", "YA CUBIERTO por SERRA-R1 T2 (campo_latente_diag.csv usa el reforzado para Hg)"),
    ("src/h85ce_advi_jerarquico.py", "ADVI vs NUTS: referencia NUTS leida de main_{metal}.nc", f"NO. {ART}",
     "SI: main_Hg.nc como referencia NUTS", "NO RECALCULADO (suplementario, no citado)"),
    ("src/extraccion_v3_luis.py", "ESS tail de Hg leido de main_Hg.nc (cruce con TABLE10)",
     f"NO: tab:conv ESS tail de Hg = 693 viene del reforzado. {ART}", "SI esa lectura", "NO RECALCULADO"),
    ("src/mapas_f3.py / src/figserra.py::fig9 / src/v7rev_fig_axes.py", "figuras FIG17_Hg y Fig9 SERRA (leen posterior_maps.npz)",
     f"SI (son las figuras del mapa de Hg). {ART}", "SI (via posterior_maps.npz)", "REGENERADAS (T3)"),
    # ---- confirmados que no dependen
    ("src/serra_r1_campo_latente.py (SERRA-R1)", "convergencia del campo latente de Hg y area excedida de Hg",
     "SI (outputs/serra_r1)", "NO: ya usa probe_Hg_strict_2026-10-26_ta0999.nc", "SIN CAMBIO (ya correcto)"),
    ("CV espacial (TABLE17), enmascaramiento (h6_deep, TABLE_H6_MAIN), sensibilidad a priors (TABLE31), LGCP (TABLE24), mezcla, LMC, SBC y recuperacion",
     "ajustes propios de cada experimento", "SI (varias tablas)",
     "NO: sin referencias a main_Hg.nc, idata_main ni posterior_maps.npz (barrido: inventario_T1_scan.csv)", "SIN CAMBIO"),
]


def main():
    s = scan()
    s.to_csv(OUT_DIR / "inventario_T1_scan.csv", index=False, encoding="utf-8")
    inv = pd.DataFrame(CURATED, columns=["archivo_o_script -> salida", "cantidad", "la usa el articulo?",
                                         "depende de main_Hg.nc?", "accion en R2"])
    inv.to_csv(OUT_DIR / "inventario_T1.csv", index=False, encoding="utf-8")
    print(f"barrido automatico: {len(s)} referencias en {s['archivo'].nunique()} archivos")
    print(s.groupby("archivo").size().sort_values(ascending=False).to_string())
    print(f"\ninventario curado: {len(inv)} filas -> outputs/serra_r2/inventario_T1.csv")


if __name__ == "__main__":
    main()
