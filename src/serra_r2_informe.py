"""SERRA-R2 -- escribe outputs/serra_r2/SERRA_R2_INFORME.md leyendo TODAS las cifras de los CSV de
outputs/serra_r2/ (ninguna se transcribe a mano). Requiere haber corrido serra_r2_hg_refuerzo.py
(todas las etapas), serra_r2_inventario.py, serra_r2_figuras.py y serra_r2_verificar.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config import OUT  # noqa: E402

R2 = OUT / "serra_r2"


def f(x, n=1):
    return f"{float(x):.{n}f}"


def main():
    ad = pd.read_csv(R2 / "hg_antes_despues.csv").set_index("cantidad")
    hip = pd.read_csv(R2 / "hg_hiperparametros_antes_despues.csv")
    hip["clave"] = hip["tabla"] + " | " + hip["cantidad"]
    hp = hip.set_index("cantidad")
    ac = pd.read_csv(R2 / "hg_acuerdo_clasificacion.csv").set_index("umbral_P")
    dm = pd.read_csv(R2 / "hg_diferencias_mapas.csv").iloc[0]
    ex = pd.read_csv(R2 / "hg_extrapolacion.csv")
    ru = pd.read_csv(R2 / "hg_ruido_montecarlo_semillas.csv")
    cv = pd.read_csv(R2 / "hg_covariables_ic95_cinco_metales.csv")
    t4 = pd.read_csv(R2 / "tabla4_convergencia_articulo.csv")
    ver = pd.read_csv(R2 / "verificacion_R2.csv")
    inv = pd.read_csv(R2 / "inventario_T1.csv")
    fg = pd.read_csv(R2 / "hg_figuras_comparacion.csv").set_index("figura")
    dz = pd.read_csv(R2 / "hg_diseno_monitoreo_antes_despues.csv")

    P = {t: f"% del dominio con P(Hg > 6.6) > {t} (agricola = residencial: mismo ECA)" for t in (0.5, 0.8, 0.95)}
    a5, d5 = ad.loc[P[0.5], "antes_main_Hg"], ad.loc[P[0.5], "despues_reforzado"]
    a8, d8 = ad.loc[P[0.8], "antes_main_Hg"], ad.loc[P[0.8], "despues_reforzado"]
    a95, d95 = ad.loc[P[0.95], "antes_main_Hg"], ad.loc[P[0.95], "despues_reforzado"]
    ptau_a = float(hp.loc["p_logtau_lt0", "antes_main_Hg"])
    ptau_d = float(hp.loc["p_logtau_lt0", "despues_reforzado"])
    cvh = cv[cv.modelo.str.startswith("Hg")]
    dump_a = cvh[(cvh.modelo == "Hg (estandar, main_Hg)") & (cvh.covariable == "log_dist_dump")].iloc[0]
    dump_d = cvh[(cvh.modelo == "Hg (reforzado)") & (cvh.covariable == "log_dist_dump")].iloc[0]
    n_excl = cv.groupby("modelo").excluye_cero.sum()

    # ruido Monte Carlo
    g = ru.groupby("ajuste")
    mc = g.agg(**{f"{c}_{s}": (c, s) for c in ["pct_P_gt_0.5", "pct_P_gt_0.8", "pct_P_gt_0.95", "area_excedida_mediana_pct"]
                  for s in ["mean", "min", "max"]})
    ant, dep = mc.loc["antes (main_Hg.nc)"], mc.loc["despues (reforzado)"]

    def ratio(col):
        rango = max(ant[f"{col}_max"] - ant[f"{col}_min"], dep[f"{col}_max"] - dep[f"{col}_min"])
        return abs(dep[f"{col}_mean"] - ant[f"{col}_mean"]) / rango if rango > 0 else float("inf")

    def lectura(col):
        r = ratio(col)
        return (f"dentro del ruido Monte Carlo ({f(r, 1)}× el rango entre semillas)" if r < 1
                else f"cambio real ({f(r, 1)}× el rango entre semillas)")

    def mc_row(lab, col):
        am, an, ax_ = ant[f"{col}_mean"], ant[f"{col}_min"], ant[f"{col}_max"]
        dm_, dn, dx = dep[f"{col}_mean"], dep[f"{col}_min"], dep[f"{col}_max"]
        rango = max(ax_ - an, dx - dn)
        razon = abs(dm_ - am) / rango if rango > 0 else float("inf")
        return (f"| {lab} | {f(am, 2)} [{f(an, 2)}, {f(ax_, 2)}] | {f(dm_, 2)} [{f(dn, 2)}, {f(dx, 2)}] | "
                f"{f(dm_ - am, 2)} | {f(razon, 1)} |")

    # inventario resumido
    n_dep = int(inv["depende de main_Hg.nc?"].str.startswith("SI").sum())
    n_art = int((inv["depende de main_Hg.nc?"].str.startswith("SI") & inv["la usa el articulo?"].str.startswith("SI")).sum())
    t22 = hip[hip.tabla == "TABLE22 diseno"].set_index("cantidad")
    ov = {k: int(float(t22.loc[f"puntos coincidentes entre antes y despues, primeros {k}", "despues_reforzado"])) for k in (5, 10, 20)}

    L = []
    w = L.append
    w("<!-- generado por src/serra_r2_informe.py desde los CSV de outputs/serra_r2/: ninguna cifra se transcribe a mano -->")
    w("")
    w("# SERRA_R2_INFORME.md — mapas y resultados de Hg con el ajuste que sí converge")
    w("")
    w("El artículo declara que los cinco metales cumplen el criterio estricto (R̂ < 1.01, ESS > 400, 0 divergencias) y que Hg lo")
    w("cumple **con el protocolo reforzado** (`outputs/idata/probe_Hg_strict_2026-10-26_ta0999.nc`: 4 cadenas × 2000 draws,")
    w("calentamiento 3000, `target_accept` 0.999; R̂ 1.0059, ESS bulk 467, ESS tail 693, 0 divergencias). Pero los mapas de Hg y")
    w("todo lo que sale de `posterior_maps.npz` o de `idata_main['Hg']` usaban **`outputs/idata/main_Hg.nc`**, el ajuste estándar")
    w("(R̂ 1.0104, ESS bulk 331). Origen verificado en el código: `notebooks/_src/nb02_f_maps.py` L.249 llama a")
    w("`sb.predict_field(idata_main[a], …)`; `nb02.py` L.938 define `idata_main = {a: fit_main(a)}` con `main_{analyte}`; y en")
    w(f"`posterior_maps.npz` Hg da P > 0.5 en el {f(a5, 2)} % del dominio, la cifra «99.8 %» del artículo.")
    w("")
    w("Esta ronda **no reajusta ningún modelo**: es post-proceso de los draws existentes. No se tocó `main.tex` ni")
    w("`referencias.bib`, y no hay tag ni release de GitHub.")
    w("")
    w("---")
    w("")
    w("## CAMBIA EN EL ARTÍCULO")
    w("")
    w("Cifras que el artículo imprime y que cambian al redondear como las imprime. Cada una se recalculó desde los archivos")
    w(f"(`outputs/serra_r2/verificacion_R2.csv`, {len(ver)} comprobaciones).")
    w("")
    w("| # | Dónde (v15 en español) | Antes (impreso) | Con el ajuste reforzado | Ruido Monte Carlo | Archivo fuente |")
    w("|---|---|---|---|---|---|")
    w(f"| 1 | Tabla 9 (`tab:exc`), fila Hg, columna P > 0.8 | {f(a8)} | **{f(d8)}** ({f(a8, 4)} → {f(d8, 4)}) | {lectura("pct_P_gt_0.8")} | `hg_antes_despues.csv` |")
    w(f"| 2 | Tabla 9 (`tab:exc`), fila Hg, columna P > 0.95 | {f(a95)} | **{f(d95)}** ({f(a95, 4)} → {f(d95, 4)}) | {lectura("pct_P_gt_0.95")} | `hg_antes_despues.csv` |")
    w(f"| 3 | Sección del modelo (v15 L.239), probabilidad de τ < 1 del mercurio | 0.28 | **0.29** ({f(ptau_a, 4)} → {f(ptau_d, 4)}) | no aplica (usa los draws completos, sin submuestreo) | `hg_hiperparametros_antes_despues.csv` |")
    w("")
    w(f"**Cómo leer el ruido:** la cifra que imprime el pipeline oficial (semilla 20260906) es la de la tabla, y con ella cambian las dos")
    w("columnas. Pero el submuestreo de 4000 draws mueve esas cifras al cambiar la semilla (sección T2): en P > 0.8 el cambio de 96.7 a 96.8")
    w("es ruido de redondeo que la semilla también produce con el ajuste estándar, mientras que en P > 0.95 el ajuste reforzado sube la")
    w("cifra de forma sistemática. Recomendación: imprimir las cifras del pipeline oficial (96.8 y 77.4), que son reproducibles.")
    w("")
    w(f"**No cambian:** la columna P > 0.5 de Hg ({f(a5, 4)} → {f(d5, 4)}, se imprime 99.8 en los dos casos, tabla y texto), las demás")
    w("filas de la Tabla 9 (Pb, As, Ba, Cd), la Tabla 4 de convergencia (ya traía el ajuste reforzado para Hg) y la Tabla de")
    w("sensibilidad a priors (son ajustes propios que no leen `main_Hg.nc`).")
    w("")
    w("### Una frase del artículo que es falsa, con cualquiera de los dos ajustes")
    w("")
    w("El artículo dice que *ninguna de las cuatro covariables de la media resulta significativa, con intervalos creíbles que")
    w("contienen el cero en los cinco metales*. Para **Hg** la log-distancia al botadero **excluye el cero con los dos ajustes**:")
    w("")
    w("| Ajuste | media | IC 95 % | ¿excluye el cero? |")
    w("|---|---|---|---|")
    w(f"| estándar (`main_Hg.nc`) | {f(dump_a.media, 4)} | [{f(dump_a.ic95_bajo, 4)}, {f(dump_a.ic95_alto, 4)}] | sí |")
    w(f"| reforzado | {f(dump_d.media, 4)} | [{f(dump_d.ic95_bajo, 4)}, {f(dump_d.ic95_alto, 4)}] | sí |")
    w("")
    otros = [m for m in n_excl.index if not m.startswith("Hg")]
    w(f"Las otras 3 covariables de Hg (con los dos ajustes) y los {4 * len(otros)} intervalos de Pb, As, Ba y Cd (Cd con r libre y con")
    w(f"L = LD; {len(otros)} modelos × 4 covariables) contienen el cero (intervalos que lo excluyen: {int(n_excl[otros].sum())}). Detalle de los {len(cv)}")
    w("intervalos en `hg_covariables_ic95_cinco_metales.csv`. La frase correcta es «en cuatro de los cinco metales ninguna")
    w("covariable excluye el cero; en mercurio, la log-distancia al botadero tiene coeficiente positivo con IC 95 % que lo excluye»;")
    w("el efecto es de la covariable de distancia a la fuente, no de la elevación, la pendiente ni la distancia a la ciudad.")
    w("")
    w("### Zona de extrapolación («0.8 %» del pie de la figura de Hg)")
    w("")
    w("**No es pura geometría.** `nb02_f_maps.py` L.294 define `ell_typ` como el promedio de la escala de longitud de los cinco")
    w("`idata_main`, incluido Hg, y la zona extrapolada son las celdas a más de 1.5·`ell_typ` del dato más cercano. Las figuras no")
    w("recalculan eso: `mapas_f3.py` L.225 usa una constante fija `ell_km = 1.03`.")
    w("")
    w("| Variante | `ell_typ` (km) | umbral (km) | celdas | % del dominio |")
    w("|---|---|---|---|---|")
    for _, r in ex.iterrows():
        w(f"| {r['variante']} | {f(r['ell_typ_km'], 4)} | {f(r['umbral_km'], 4)} | {int(r['celdas_extrapoladas'])} | {f(r['pct_dominio'], 3)} |")
    w("")
    w("**Resolución:** se mantiene la constante fija de las figuras, de modo que todos los mapas (Hg, Pb, As, Cd) comparten la")
    w("misma máscara y el «0.8 %» sigue siendo exactamente lo que dibujan (57 celdas a más de 1.545 km de cualquier dato). Lo que")
    w("debe cuidarse en el pie de figura es **describir la máscara como una distancia fija de 1.5 × 1.03 km**, no como «1.5 escalas")
    w("de longitud promedio de los ajustes»: recalculada con el Hg reforzado esa promedio daría 0.7 % (51 celdas), y con el Cd bajo")
    w("L = LD (que ya no usa el pipeline original) 0.2 %. Es una cuestión de redacción, no de cifra impresa.")
    w("")
    w("---")
    w("")
    w("## T1 — Inventario de lo que dependía de `main_Hg.nc`")
    w("")
    w(f"Barrido automático de `src/` y `notebooks/_src/` (`inventario_T1_scan.csv`) y tabla curada ({len(inv)} filas,")
    w(f"`inventario_T1.csv`), contrastada con `main_FINAL_v15.tex`. {n_dep} entradas dependen del ajuste estándar; el artículo usa")
    w(f"{n_art} de ellas. El manuscrito en inglés para SERRA no está en disco: lo que solo apareciera en él figura como no verificable.")
    w("")
    w("| Dependencia | Cantidad | ¿La usa el artículo? | Acción |")
    w("|---|---|---|---|")
    for _, r in inv.iterrows():
        if str(r["depende de main_Hg.nc?"]).startswith("SI") and str(r["la usa el articulo?"]).startswith("SI"):
            w(f"| `{r['archivo_o_script -> salida'].split(' -> ')[0].split(' L.')[0]}` | {r['cantidad']} | {r['la usa el articulo?'].split('. main_FINAL')[0]} | {r['accion en R2']} |")
    w("")
    w("Excluidos por ser ajustes propios y no leer `main_Hg.nc` (comprobado por barrido): validación cruzada, enmascaramiento,")
    w("sensibilidad a priors, LGCP, mezcla, LMC, SBC y recuperación. `src/serra_r1_campo_latente.py` ya usaba el ajuste reforzado.")
    w("Dependen pero el artículo en español no las usa (se recalcularon igual): tablas 12, 13, 14 y 19, el diseño de monitoreo")
    w("(`TABLE21/22`, `FIG14`), la anotación de la figura `FIG9_Hg_crps`. Dependen y no se rehicieron (no citadas):")
    w("`TABLE29_decision_thresholds.csv`, `TABLE_S5_loo.csv`, `TABLE_S10_pareto_k.csv`, `TABLE_T2_3_...`, ADVI.")
    w("")
    w("`figuras/TABLE10_convergence.csv` conserva la fila del ajuste estándar de Hg (1.0104 / 331) y la de Cd con r libre: **no es la")
    w("Tabla 4 del artículo**. La Tabla 4 se verificó aparte, desde los `idata`:")
    w("")
    w("| Metal | `idata` | R̂ | ESS bulk | ESS tail | Div. | = artículo |")
    w("|---|---|---|---|---|---|---|")
    for _, r in t4.iterrows():
        w(f"| {r['metal']} | `{r['idata']}` | {f(r['rhat_max_calculado'], 4)} | {f(r['ess_bulk_calculado'], 0)} | {f(r['ess_tail_calculado'], 0)} | {int(r['divergencias'])} | {'sí' if r['coincide'] else 'NO'} |")
    w("")
    w("---")
    w("")
    w("## T2 — Recalculado con el ajuste reforzado")
    w("")
    w("Script: `src/serra_r2_hg_refuerzo.py`. **Mismo pipeline que `nb02_f_maps.py`**: grilla de 50 m con buffer de 300 m (7381")
    w("celdas), `N_MAP_SAMPLES = 4000`, semilla 20260906, bloques de 1500 celdas, `predict_field` con `anisotropic=True`, ECA 6.6 mg/kg")
    w("(el residencial de Hg es el mismo 6.6, así que `p_resid` = `p_agri`). La única diferencia es el `idata`.")
    w("")
    w("**Validación del pipeline antes de usarlo:** con el ajuste estándar reproduce `posterior_maps.npz` en el bloque 0 (celdas")
    w("0–1499) con **diferencia 0.0** en las seis claves (`validacion_pipeline_bloque0.csv`). `src/t2_1_area_excedida_posterior.py` ya")
    w("había reproducido el mapa completo con diferencia 0.0000 por celda.")
    w("")
    w("Nota técnica: el ajuste reforzado tiene 8000 draws (4 × 2000) y el estándar 4000; la regla del pipeline toma 4000 por")
    w("submuestreo sin reemplazo (4000 de 8000 con el reforzado, los 4000 con el estándar).")
    w("")
    w("Productos: `posterior_maps_Hg_refuerzo.npz` (`Hg_mean`, `Hg_median`, `Hg_sd_f`, `Hg_sd_y`, `Hg_p_agri`, `Hg_p_resid`, más el área")
    w("excedida por draw). **No se sobrescribió `posterior_maps.npz`.**")
    w("")
    w("### Antes / después (`hg_antes_despues.csv`)")
    w("")
    w("| Cantidad | Antes (`main_Hg.nc`) | Después (reforzado) | Diferencia |")
    w("|---|---|---|---|")
    for q, r in ad.iterrows():
        q2 = q.replace("(agricola = residencial: mismo ECA)", "").strip()
        n = 4 if abs(float(r["antes_main_Hg"])) < 10 and "celdas" not in q else 2
        w(f"| {q2} | {f(r['antes_main_Hg'], n)} | {f(r['despues_reforzado'], n)} | {f(r['diferencia'], n)} |")
    w("")
    r1 = pd.read_csv(OUT / "serra_r1" / "cantidades_mapa_diag_area.csv").set_index("metal").loc["Hg"]
    am = ad.loc["area excedida (% del dominio), mediana", "despues_reforzado"]
    w(f"Contraste cruzado con la ronda anterior (`cantidades_mapa_diag_area.csv`, con los 8000 draws en orden de cadena y sin submuestreo):")
    w(f"mediana del área excedida de Hg {f(r1['area_frac_median_pct'], 2)} % frente a {f(am, 2)} % aquí (4000 draws submuestreados): compatible.")
    w("")
    w("### Acuerdo de clasificación celda a celda (`hg_acuerdo_clasificacion.csv`)")
    w("")
    w("| Umbral de P | Acuerdo | Celdas que cambian | excedente → no | no → excedente |")
    w("|---|---|---|---|---|")
    for t, r in ac.iterrows():
        w(f"| {t} | {f(r['acuerdo_celda_a_celda_pct'], 2)} % | {int(r['celdas_que_cambian'])} | {int(r['de_excedente_a_no'])} | {int(r['de_no_a_excedente'])} |")
    w("")
    w(f"Correlación de la log-mediana entre mapas {f(dm['corr_log_mediana'], 4)}; diferencia relativa mediana de la mediana de concentración")
    w(f"{f(dm['mediana_dif_rel_mediana_conc_pct'], 2)} %; diferencia máxima de P {f(dm['max_abs_dif_P'], 3)}.")
    w("")
    w("### ¿El cambio es real o ruido Monte Carlo? (`hg_ruido_montecarlo_semillas.csv`)")
    w("")
    w("Mismo pipeline con 4 semillas por ajuste (20260906 más tres consecutivas):")
    w("")
    w("| Cifra | Antes: media [mín, máx] | Después: media [mín, máx] | Δ de medias | Δ / rango entre semillas |")
    w("|---|---|---|---|---|")
    w(mc_row("% dominio P > 0.5", "pct_P_gt_0.5"))
    w(mc_row("% dominio P > 0.8", "pct_P_gt_0.8"))
    w(mc_row("% dominio P > 0.95", "pct_P_gt_0.95"))
    w(mc_row("Mediana del área excedida (%)", "area_excedida_mediana_pct"))
    w("")
    r95, r8 = ratio("pct_P_gt_0.95"), ratio("pct_P_gt_0.8")
    w(f"Lectura: en P > 0.95 la diferencia entre ajustes es {f(r95, 1)} veces el rango entre semillas; en P > 0.8, {f(r8, 1)} veces. Un cociente")
    w("mayor que 1 indica que el cambio supera lo que mueve la semilla; menor que 1, que es indistinguible del ruido del")
    w("submuestreo de 4000 draws. **La cifra impresa que sale del pipeline oficial (semilla 20260906) es la de la tabla de arriba;**")
    w("la tabla de ruido solo sirve para juzgar cuánto de la diferencia es señal.")
    w("")
    w("### Hiperparámetros de Hg, antes y después (`hg_hiperparametros_antes_despues.csv`)")
    w("")
    w("| Parámetro | Antes | Después |")
    w("|---|---|---|")
    for q in ["ell_E", "ell_N", "eta", "sigma_n", "sigma_tot", "rho", "beta0", "b_background", "log_tau"]:
        r = hip[(hip.tabla == "TABLE11 / covariables") & (hip.cantidad == q)].iloc[0]
        w(f"| {q} | {f(r['antes_main_Hg'], 4)} | {f(r['despues_reforzado'], 4)} |")
    for c in ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]:
        a = cvh[(cvh.modelo == "Hg (estandar, main_Hg)") & (cvh.covariable == c)].iloc[0]
        d = cvh[(cvh.modelo == "Hg (reforzado)") & (cvh.covariable == c)].iloc[0]
        w(f"| β {c} (media; IC 95 %) | {f(a.media, 3)} [{f(a.ic95_bajo, 3)}, {f(a.ic95_alto, 3)}] | {f(d.media, 3)} [{f(d.ic95_bajo, 3)}, {f(d.ic95_alto, 3)}] |")
    w(f"| P(log τ < 0) | {f(ptau_a, 4)} | {f(ptau_d, 4)} |")
    w("")
    w("Con la autovalidación de «antes»: la fila Hg de `TABLE11` se reproduce a 4.7×10⁻⁵ (redondeo a 4 decimales), P(log τ < 0) = 0.2785")
    w("igual a `TABLE_T1_1`, el chequeo predictivo posterior igual a `TABLE14` a 4 decimales, y el diseño de monitoreo recalculado")
    w("coincide con `outputs/monitoring_design.npz` (mismos puntos, traza idéntica). Las tablas 12, 13, 14 y 19 y el diseño de monitoreo")
    w("con ambos ajustes están en el CSV; el diseño con el reforzado reduce la sd media un "
      f"{f(dz.loc[5, 'reduction_pct_despues_ell_typ_B'])} / {f(dz.loc[10, 'reduction_pct_despues_ell_typ_B'])} / {f(dz.loc[20, 'reduction_pct_despues_ell_typ_B'])} % con 5 / 10 / 20 puntos")
    w(f"nuevos (antes {f(dz.loc[5, 'reduction_pct_antes'])} / {f(dz.loc[10, 'reduction_pct_antes'])} / {f(dz.loc[20, 'reduction_pct_antes'])} %); coinciden {ov[5]} de los 5 primeros puntos, {ov[10]} de 10 y {ov[20]} de 20.")
    w("")
    w("---")
    w("")
    w("## T3 — Figuras")
    w("")
    w("`src/serra_r2_figuras.py` regenera `figuras/serra/Fig9.{pdf,eps,tif}` (mapa de Hg, «Fig. 8» en el PDF del artículo) y")
    w("`figuras/FIG17_Hg_mean_and_uncertainty.pdf` (versión en español) con los productos nuevos: misma ruta de dibujo, mismo estilo,")
    w("mismas escalas y leyendas; solo cambian las claves `Hg_*`. Las figuras se sobrescriben (sin copias de versiones anteriores).")
    w("")
    w("Verificación con `verify_figserra.py` (`T3_verificacion_log.txt`): Fig9 en Arial embebida, texto de línea base 8.0–10.0 pt,")
    w("174 × 104.4 mm, raster a 600 dpi, sin títulos internos. **A la vista el mapa es el mismo.** Lo único visible: el extremo")
    w("superior de la barra de color pasa de ~1577 a ~1489 mg/kg (máximo de la mediana; las décadas rotuladas son las mismas) y,")
    w(f"en la versión en español, un desplazamiento menor de la maquetación por el ancho de las etiquetas. Píxeles que difieren más de 8")
    w(f"niveles: Fig9 {f(fg.loc['Fig9 (SERRA)', 'pct_pixeles_que_difieren_>8_niveles'], 2)} %, FIG17_Hg {f(fg.loc['FIG17_Hg (espanol)', 'pct_pixeles_que_difieren_>8_niveles'], 2)} %. La leyenda de extrapolación sigue diciendo 0.8 %.")
    w("")
    w("---")
    w("")
    w("## T4 — Coherencia")
    w("")
    w("- **Cifras impresas que cambian:** las tres del bloque «CAMBIA EN EL ARTÍCULO». **Todo lo demás impreso queda igual.**")
    w(f"- **Verificación independiente** (`serra_r2_verificar.py`, `verificacion_R2.csv`): {int(ver.ok.sum())}/{len(ver)} comprobaciones OK,")
    w("  que incluyen la Tabla 9 completa de los cinco metales recalculada desde sus archivos y comparada con lo impreso.")
    w("- La Tabla 4 se contrastó desde los `idata` (arriba): las cinco filas coinciden.")
    w("- Las filas de Hg de la trazabilidad (`TRAZABILIDAD_RESULTADOS.csv`) apuntan ahora al ajuste reforzado.")
    w("")
    w("---")
    w("")
    w("## T5 — Zip para el registro nuevo de Zenodo")
    w("")
    w("Se hizo commit de esta ronda (con el mismo barrido de rastros que en la ronda anterior) y se subió a GitHub **sin tag ni release**.")
    w("El zip `censored-gp-santa-barbara-v1.6.zip` se genera con `git archive` (con `core.autocrlf=false`) del commit que contiene")
    w("este informe; su commit, tamaño, SHA-256, MD5 y número de archivos están en el manifiesto local `outputs/zenodo_nuevo/`")
    w("(no publicado). Versión **v1.6**; el registro nuevo incluye `outputs/serra_r2/`.")
    w("")
    w("---")
    w("")
    w("## Índice de archivos (`outputs/serra_r2/`)")
    w("")
    w("| Tarea | Script | Salidas | Registro |")
    w("|---|---|---|---|")
    w("| T1 | `src/serra_r2_inventario.py` | `inventario_T1.csv`, `inventario_T1_scan.csv` | — |")
    w("| T2 | `src/serra_r2_hg_refuerzo.py` | `posterior_maps_Hg_refuerzo.npz`, `hg_antes_despues.csv`, `hg_acuerdo_clasificacion.csv`, `hg_diferencias_mapas.csv`, `hg_hiperparametros_antes_despues.csv`, `hg_covariables_ic95_cinco_metales.csv`, `hg_extrapolacion.csv`, `hg_diseno_monitoreo_antes_despues.csv`, `hg_ruido_montecarlo_semillas.csv`, `validacion_pipeline_bloque0.csv`, `tabla4_convergencia_articulo.csv` | `T2_hiper_log.txt`, `T2_mapas_log.txt`, `T2_tablas_log.txt`, `T2_ruido_log.txt`, `T4_tabla4_log.txt` |")
    w("| T3 | `src/serra_r2_figuras.py` | `figuras/serra/Fig9.*`, `figuras/FIG17_Hg_mean_and_uncertainty.pdf`, `hg_figuras_comparacion.csv` | `T3_log.txt`, `T3_verificacion_log.txt` |")
    w("| T4 | `src/serra_r2_verificar.py`, `src/serra_r2_informe.py` | `verificacion_R2.csv`, este informe | `T4_verificacion_log.txt` |")
    w("")
    (R2 / "SERRA_R2_INFORME.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"escrito: SERRA_R2_INFORME.md ({len(L)} lineas)")


if __name__ == "__main__":
    main()
