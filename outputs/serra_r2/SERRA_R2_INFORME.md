<!-- generado por src/serra_r2_informe.py desde los CSV de outputs/serra_r2/: ninguna cifra se transcribe a mano -->

# SERRA_R2_INFORME.md — mapas y resultados de Hg con el ajuste que sí converge

El artículo declara que los cinco metales cumplen el criterio estricto (R̂ < 1.01, ESS > 400, 0 divergencias) y que Hg lo
cumple **con el protocolo reforzado** (`outputs/idata/probe_Hg_strict_2026-10-26_ta0999.nc`: 4 cadenas × 2000 draws,
calentamiento 3000, `target_accept` 0.999; R̂ 1.0059, ESS bulk 467, ESS tail 693, 0 divergencias). Pero los mapas de Hg y
todo lo que sale de `posterior_maps.npz` o de `idata_main['Hg']` usaban **`outputs/idata/main_Hg.nc`**, el ajuste estándar
(R̂ 1.0104, ESS bulk 331). Origen verificado en el código: `notebooks/_src/nb02_f_maps.py` L.249 llama a
`sb.predict_field(idata_main[a], …)`; `nb02.py` L.938 define `idata_main = {a: fit_main(a)}` con `main_{analyte}`; y en
`posterior_maps.npz` Hg da P > 0.5 en el 99.77 % del dominio, la cifra «99.8 %» del artículo.

Esta ronda **no reajusta ningún modelo**: es post-proceso de los draws existentes. No se tocó `main.tex` ni
`referencias.bib`, y no hay tag ni release de GitHub.

---

## CAMBIA EN EL ARTÍCULO

Cifras que el artículo imprime y que cambian al redondear como las imprime. Cada una se recalculó desde los archivos
(`outputs/serra_r2/verificacion_R2.csv`, 42 comprobaciones).

| # | Dónde (v15 en español) | Antes (impreso) | Con el ajuste reforzado | Ruido Monte Carlo | Archivo fuente |
|---|---|---|---|---|---|
| 1 | Tabla 9 (`tab:exc`), fila Hg, columna P > 0.8 | 96.7 | **96.8** (96.6942 → 96.8432) | dentro del ruido Monte Carlo (0.7× el rango entre semillas) | `hg_antes_despues.csv` |
| 2 | Tabla 9 (`tab:exc`), fila Hg, columna P > 0.95 | 76.1 | **77.4** (76.1143 → 77.4150) | cambio real (2.4× el rango entre semillas) | `hg_antes_despues.csv` |
| 3 | Sección del modelo (v15 L.239), probabilidad de τ < 1 del mercurio | 0.28 | **0.29** (0.2785 → 0.2854) | no aplica (usa los draws completos, sin submuestreo) | `hg_hiperparametros_antes_despues.csv` |

**Cómo leer el ruido:** la cifra que imprime el pipeline oficial (semilla 20260906) es la de la tabla, y con ella cambian las dos
columnas. Pero el submuestreo de 4000 draws mueve esas cifras al cambiar la semilla (sección T2): en P > 0.8 el cambio de 96.7 a 96.8
es ruido de redondeo que la semilla también produce con el ajuste estándar, mientras que en P > 0.95 el ajuste reforzado sube la
cifra de forma sistemática. Recomendación: imprimir las cifras del pipeline oficial (96.8 y 77.4), que son reproducibles.

**No cambian:** la columna P > 0.5 de Hg (99.7697 → 99.8103, se imprime 99.8 en los dos casos, tabla y texto), las demás
filas de la Tabla 9 (Pb, As, Ba, Cd), la Tabla 4 de convergencia (ya traía el ajuste reforzado para Hg) y la Tabla de
sensibilidad a priors (son ajustes propios que no leen `main_Hg.nc`).

### Una frase del artículo que es falsa, con cualquiera de los dos ajustes

El artículo dice que *ninguna de las cuatro covariables de la media resulta significativa, con intervalos creíbles que
contienen el cero en los cinco metales*. Para **Hg** la log-distancia al botadero **excluye el cero con los dos ajustes**:

| Ajuste | media | IC 95 % | ¿excluye el cero? |
|---|---|---|---|
| estándar (`main_Hg.nc`) | 0.4541 | [0.0727, 0.8344] | sí |
| reforzado | 0.4616 | [0.0723, 0.8439] | sí |

Las otras 3 covariables de Hg (con los dos ajustes) y los 20 intervalos de Pb, As, Ba y Cd (Cd con r libre y con
L = LD; 5 modelos × 4 covariables) contienen el cero (intervalos que lo excluyen: 0). Detalle de los 28
intervalos en `hg_covariables_ic95_cinco_metales.csv`. La frase correcta es «en cuatro de los cinco metales ninguna
covariable excluye el cero; en mercurio, la log-distancia al botadero tiene coeficiente positivo con IC 95 % que lo excluye»;
el efecto es de la covariable de distancia a la fuente, no de la elevación, la pendiente ni la distancia a la ciudad.

### Zona de extrapolación («0.8 %» del pie de la figura de Hg)

**No es pura geometría.** `nb02_f_maps.py` L.294 define `ell_typ` como el promedio de la escala de longitud de los cinco
`idata_main`, incluido Hg, y la zona extrapolada son las celdas a más de 1.5·`ell_typ` del dato más cercano. Las figuras no
recalculan eso: `mapas_f3.py` L.225 usa una constante fija `ell_km = 1.03`.

| Variante | `ell_typ` (km) | umbral (km) | celdas | % del dominio |
|---|---|---|---|---|
| A. original de nb02 (Hg estandar, Cd r libre) | 1.0294 | 1.5441 | 57 | 0.772 |
| B. Hg reforzado, resto igual (la unica diferencia de R2) | 1.0371 | 1.5556 | 51 | 0.691 |
| C. Hg reforzado y Cd L=LD (no es parte de R2) | 1.0910 | 1.6365 | 13 | 0.176 |
| D. constante fija de mapas_f3.py (la que dibujan las figuras) | 1.0300 | 1.5450 | 57 | 0.772 |

**Resolución:** se mantiene la constante fija de las figuras, de modo que todos los mapas (Hg, Pb, As, Cd) comparten la
misma máscara y el «0.8 %» sigue siendo exactamente lo que dibujan (57 celdas a más de 1.545 km de cualquier dato). Lo que
debe cuidarse en el pie de figura es **describir la máscara como una distancia fija de 1.5 × 1.03 km**, no como «1.5 escalas
de longitud promedio de los ajustes»: recalculada con el Hg reforzado esa promedio daría 0.7 % (51 celdas), y con el Cd bajo
L = LD (que ya no usa el pipeline original) 0.2 %. Es una cuestión de redacción, no de cifra impresa.

---

## T1 — Inventario de lo que dependía de `main_Hg.nc`

Barrido automático de `src/` y `notebooks/_src/` (`inventario_T1_scan.csv`) y tabla curada (21 filas,
`inventario_T1.csv`), contrastada con `main_FINAL_v15.tex`. 18 entradas dependen del ajuste estándar; el artículo usa
7 de ellas. El manuscrito en inglés para SERRA no se revisó en esa ronda: lo que solo apareciera en él figura como no verificado.

| Dependencia | Cantidad | ¿La usa el artículo? | Acción |
|---|---|---|---|
| `notebooks/_src/nb02_f_maps.py` | Hg_mean, Hg_median, Hg_sd_f, Hg_sd_y, Hg_p_agri, Hg_p_resid (mapas de Hg) | SI: Fig. mapaHg = FIG17_Hg (L.553); Fig9 SERRA (Fig. 8 del PDF) | RECALCULADO -> outputs/serra_r2/posterior_maps_Hg_refuerzo.npz |
| `notebooks/_src/nb02_f_maps.py` | % del dominio con P(Hg>6.6)>0.5 = 99.7697 | SI: tab:exc fila Hg col. P>0.5 = 99.8 y texto L.570 | RECALCULADO -> hg_antes_despues.csv |
| `src/t2_1_area_excedida_posterior.py` | % del dominio con P>0.8 (96.69) y P>0.95 (76.11); area excedida: mediana, IC95 % | SI: tab:exc fila Hg cols. P>0.8 = 96.7 y P>0.95 = 76.1 | RECALCULADO -> hg_antes_despues.csv |
| `src/t1_1_prueba_varianza.py` | P(log tau < 0) de Hg = 0.2785 | SI: L.239 'el mercurio es la excepcion (0.28)' | RECALCULADO -> hg_hiperparametros_antes_despues.csv |
| `notebooks/_src/nb02.py` | IC95 % de las 4 covariables de la media de Hg (columnas beta_*_sig) | SI: L.504 'Ninguna de las cuatro covariables ... sus intervalos creibles contienen el cero en los cinco metales' | RECALCULADO. HALLAZGO: la frase es FALSA para Hg con AMBOS ajustes (log_dist_dump excluye el cero) |
| `notebooks/_src/nb02_f_maps.py` | zona de extrapolacion (celdas a > 1.5 ell_typ del dato mas cercano): 0.772 % = '0.8 %' | SI: pie de la Fig. 8 SERRA ('0.8 % del dominio'); leyenda 'extrapolation (x %)' de todos los mapas | RECALCULADO -> hg_extrapolacion.csv (0.772 % -> 0.691 % si ell_typ se recalcula) |
| `src/mapas_f3.py / src/figserra.py::fig9 / src/v7rev_fig_axes.py` | figuras FIG17_Hg y Fig9 SERRA (leen posterior_maps.npz) | SI (son las figuras del mapa de Hg) | REGENERADAS (T3) |

Excluidos por ser ajustes propios y no leer `main_Hg.nc` (comprobado por barrido): validación cruzada, enmascaramiento,
sensibilidad a priors, LGCP, mezcla, LMC, SBC y recuperación. `src/serra_r1_campo_latente.py` ya usaba el ajuste reforzado.
Dependen pero el artículo en español no las usa (se recalcularon igual): tablas 12, 13, 14 y 19, el diseño de monitoreo
(`TABLE21/22`, `FIG14`), la anotación de la figura `FIG9_Hg_crps`. Dependen y no se rehicieron (no citadas):
`TABLE29_decision_thresholds.csv`, `TABLE_S5_loo.csv`, `TABLE_S10_pareto_k.csv`, `TABLE_T2_3_...`, ADVI.

`figuras/TABLE10_convergence.csv` conserva la fila del ajuste estándar de Hg (1.0104 / 331) y la de Cd con r libre: **no es la
Tabla 4 del artículo**. La Tabla 4 se verificó aparte, desde los `idata`:

| Metal | `idata` | R̂ | ESS bulk | ESS tail | Div. | = artículo |
|---|---|---|---|---|---|---|
| Hg | `probe_Hg_strict_2026-10-26_ta0999.nc` | 1.0059 | 467 | 693 | 0 | sí |
| Pb | `main_Pb.nc` | 1.0068 | 752 | 1272 | 0 | sí |
| As | `main_As.nc` | 1.0064 | 913 | 1205 | 0 | sí |
| Ba | `main_Ba.nc` | 1.0035 | 1026 | 1323 | 0 | sí |
| Cd | `flip_LD_Cd.nc` | 1.0051 | 767 | 845 | 0 | sí |

---

## T2 — Recalculado con el ajuste reforzado

Script: `src/serra_r2_hg_refuerzo.py`. **Mismo pipeline que `nb02_f_maps.py`**: grilla de 50 m con buffer de 300 m (7381
celdas), `N_MAP_SAMPLES = 4000`, semilla 20260906, bloques de 1500 celdas, `predict_field` con `anisotropic=True`, ECA 6.6 mg/kg
(el residencial de Hg es el mismo 6.6, así que `p_resid` = `p_agri`). La única diferencia es el `idata`.

**Validación del pipeline antes de usarlo:** con el ajuste estándar reproduce `posterior_maps.npz` en el bloque 0 (celdas
0–1499) con **diferencia 0.0** en las seis claves (`validacion_pipeline_bloque0.csv`). `src/t2_1_area_excedida_posterior.py` ya
había reproducido el mapa completo con diferencia 0.0000 por celda.

Nota técnica: el ajuste reforzado tiene 8000 draws (4 × 2000) y el estándar 4000; la regla del pipeline toma 4000 por
submuestreo sin reemplazo (4000 de 8000 con el reforzado, los 4000 con el estándar).

Productos: `posterior_maps_Hg_refuerzo.npz` (`Hg_mean`, `Hg_median`, `Hg_sd_f`, `Hg_sd_y`, `Hg_p_agri`, `Hg_p_resid`, más el área
excedida por draw). **No se sobrescribió `posterior_maps.npz`.**

### Antes / después (`hg_antes_despues.csv`)

| Cantidad | Antes (`main_Hg.nc`) | Después (reforzado) | Diferencia |
|---|---|---|---|
| % del dominio con P(Hg > 6.6) > 0.5 | 99.77 | 99.81 | 0.04 |
| % del dominio con P(Hg > 6.6) > 0.8 | 96.69 | 96.84 | 0.15 |
| % del dominio con P(Hg > 6.6) > 0.95 | 76.11 | 77.41 | 1.30 |
| area excedida (% del dominio), mediana | 96.40 | 96.55 | 0.14 |
| area excedida (% del dominio), cuantil 2.5 % | 88.85 | 89.45 | 0.60 |
| area excedida (% del dominio), cuantil 97.5 % | 98.89 | 98.93 | 0.04 |
| area excedida (% del dominio), media | 95.79 | 95.93 | 0.14 |
| mediana posterior de la concentracion (mg/kg): minimo en la grilla | 2.6402 | 2.6756 | 0.0354 |
| mediana posterior de la concentracion (mg/kg): maximo en la grilla | 1576.78 | 1489.01 | -87.77 |
| media posterior de la concentracion (mg/kg): minimo en la grilla | 5.3778 | 5.1106 | -0.2672 |
| media posterior de la concentracion (mg/kg): maximo en la grilla | 7999.10 | 7114.99 | -884.10 |
| sd predictivo (log): media sobre la grilla | 1.4219 | 1.4220 | 0.0000 |
| sd predictivo (log): maximo | 2.1104 | 2.1214 | 0.0109 |
| P(exceder) maxima en la grilla | 1.0000 | 1.0000 | 0.0000 |
| celdas con P > 0.5 | 7364.00 | 7367.00 | 3.00 |
| celdas con P > 0.9 | 6634.00 | 6662.00 | 28.00 |

Contraste cruzado con la ronda anterior (`cantidades_mapa_diag_area.csv`, con los 8000 draws en orden de cadena y sin submuestreo):
mediana del área excedida de Hg 96.53 % frente a 96.55 % aquí (4000 draws submuestreados): compatible.

### Acuerdo de clasificación celda a celda (`hg_acuerdo_clasificacion.csv`)

| Umbral de P | Acuerdo | Celdas que cambian | excedente → no | no → excedente |
|---|---|---|---|---|
| 0.5 | 99.96 % | 3 | 0 | 3 |
| 0.8 | 99.82 % | 13 | 1 | 12 |
| 0.95 | 97.48 % | 186 | 45 | 141 |

Correlación de la log-mediana entre mapas 0.9986; diferencia relativa mediana de la mediana de concentración
3.58 %; diferencia máxima de P 0.040.

### ¿El cambio es real o ruido Monte Carlo? (`hg_ruido_montecarlo_semillas.csv`)

Mismo pipeline con 4 semillas por ajuste (20260906 más tres consecutivas):

| Cifra | Antes: media [mín, máx] | Después: media [mín, máx] | Δ de medias | Δ / rango entre semillas |
|---|---|---|---|---|
| % dominio P > 0.5 | 99.77 [99.77, 99.77] | 99.79 [99.78, 99.81] | 0.02 | 0.7 |
| % dominio P > 0.8 | 96.75 [96.69, 96.80] | 96.82 [96.76, 96.86] | 0.07 | 0.7 |
| % dominio P > 0.95 | 76.25 [76.11, 76.39] | 77.63 [77.41, 77.98] | 1.38 | 2.4 |
| Mediana del área excedida (%) | 96.43 [96.40, 96.45] | 96.55 [96.50, 96.57] | 0.12 | 1.7 |

Lectura: en P > 0.95 la diferencia entre ajustes es 2.4 veces el rango entre semillas; en P > 0.8, 0.7 veces. Un cociente
mayor que 1 indica que el cambio supera lo que mueve la semilla; menor que 1, que es indistinguible del ruido del
submuestreo de 4000 draws. **La cifra impresa que sale del pipeline oficial (semilla 20260906) es la de la tabla de arriba;**
la tabla de ruido solo sirve para juzgar cuánto de la diferencia es señal.

### Hiperparámetros de Hg, antes y después (`hg_hiperparametros_antes_despues.csv`)

| Parámetro | Antes | Después |
|---|---|---|
| ell_E | 0.5322 | 0.5558 |
| ell_N | 1.7405 | 1.7935 |
| eta | 1.0461 | 1.0470 |
| sigma_n | 1.0245 | 1.0286 |
| sigma_tot | 1.4748 | 1.4787 |
| rho | 0.4995 | 0.4976 |
| beta0 | 4.0786 | 4.1018 |
| b_background | -0.4265 | -0.4356 |
| log_tau | 0.0935 | 0.0922 |
| β elev_oefa (media; IC 95 %) | -0.153 [-0.874, 0.520] | -0.142 [-0.856, 0.573] |
| β slope_deg (media; IC 95 %) | 0.117 [-0.158, 0.395] | 0.115 [-0.158, 0.378] |
| β log_dist_dump (media; IC 95 %) | 0.454 [0.073, 0.834] | 0.462 [0.072, 0.844] |
| β log_dist_city (media; IC 95 %) | -0.325 [-1.034, 0.432] | -0.315 [-1.026, 0.426] |
| P(log τ < 0) | 0.2785 | 0.2854 |

Con la autovalidación de «antes»: la fila Hg de `TABLE11` se reproduce a 4.7×10⁻⁵ (redondeo a 4 decimales), P(log τ < 0) = 0.2785
igual a `TABLE_T1_1`, el chequeo predictivo posterior igual a `TABLE14` a 4 decimales, y el diseño de monitoreo recalculado
coincide con `outputs/monitoring_design.npz` (mismos puntos, traza idéntica). Las tablas 12, 13, 14 y 19 y el diseño de monitoreo
con ambos ajustes están en el CSV; el diseño con el reforzado reduce la sd media un 10.2 / 15.0 / 21.5 % con 5 / 10 / 20 puntos
nuevos (antes 10.1 / 14.9 / 21.4 %); coinciden 5 de los 5 primeros puntos, 9 de 10 y 15 de 20.

---

## T3 — Figuras

`src/serra_r2_figuras.py` regenera `figuras/serra/Fig9.{pdf,eps,tif}` (mapa de Hg, «Fig. 8» en el PDF del artículo) y
`figuras/FIG17_Hg_mean_and_uncertainty.pdf` (versión en español) con los productos nuevos: misma ruta de dibujo, mismo estilo,
mismas escalas y leyendas; solo cambian las claves `Hg_*`. Las figuras se sobrescriben (sin copias de versiones anteriores).

Verificación con `verify_figserra.py` (`T3_verificacion_log.txt`): Fig9 en Arial embebida, texto de línea base 8.0–10.0 pt,
174 × 104.4 mm, raster a 600 dpi, sin títulos internos. **A la vista el mapa es el mismo.** Lo único visible: el extremo
superior de la barra de color pasa de ~1577 a ~1489 mg/kg (máximo de la mediana; las décadas rotuladas son las mismas) y,
en la versión en español, un desplazamiento menor de la maquetación por el ancho de las etiquetas. Píxeles que difieren más de 8
niveles: Fig9 0.44 %, FIG17_Hg 3.59 %. La leyenda de extrapolación sigue diciendo 0.8 %.

---

## T4 — Coherencia

- **Cifras impresas que cambian:** las tres del bloque «CAMBIA EN EL ARTÍCULO». **Todo lo demás impreso queda igual.**
- **Verificación independiente** (`serra_r2_verificar.py`, `verificacion_R2.csv`): 42/42 comprobaciones OK,
  que incluyen la Tabla 9 completa de los cinco metales recalculada desde sus archivos y comparada con lo impreso.
- La Tabla 4 se contrastó desde los `idata` (arriba): las cinco filas coinciden.
- Las filas de Hg de la trazabilidad (`TRAZABILIDAD_RESULTADOS.csv`) apuntan ahora al ajuste reforzado.

---

## T5 — Preparación del archivo de Zenodo

Se hizo commit de esta ronda (con el mismo barrido de rastros que en la ronda anterior) y se subió a GitHub **sin tag ni release**;
con eso el repositorio quedó listo para archivar. El registro definitivo de Zenodo (versión **v2.0.0**: el repositorio y el informe
OEFA escaneado, depósito manual, sin integración automática) se publicó después, con el DOI de versión 10.5281/zenodo.24347680 y el DOI de
concepto 10.5281/zenodo.24347679; su verificación por la API está en el informe de la ronda R1.

---

## Índice de archivos (`outputs/serra_r2/`)

| Tarea | Script | Salidas | Registro |
|---|---|---|---|
| T1 | `src/serra_r2_inventario.py` | `inventario_T1.csv`, `inventario_T1_scan.csv` | — |
| T2 | `src/serra_r2_hg_refuerzo.py` | `posterior_maps_Hg_refuerzo.npz`, `hg_antes_despues.csv`, `hg_acuerdo_clasificacion.csv`, `hg_diferencias_mapas.csv`, `hg_hiperparametros_antes_despues.csv`, `hg_covariables_ic95_cinco_metales.csv`, `hg_extrapolacion.csv`, `hg_diseno_monitoreo_antes_despues.csv`, `hg_ruido_montecarlo_semillas.csv`, `validacion_pipeline_bloque0.csv`, `tabla4_convergencia_articulo.csv` | `T2_hiper_log.txt`, `T2_mapas_log.txt`, `T2_tablas_log.txt`, `T2_ruido_log.txt`, `T4_tabla4_log.txt` |
| T3 | `src/serra_r2_figuras.py` | `figuras/serra/Fig9.*`, `figuras/FIG17_Hg_mean_and_uncertainty.pdf`, `hg_figuras_comparacion.csv` | `T3_log.txt`, `T3_verificacion_log.txt` |
| T4 | `src/serra_r2_verificar.py`, `src/serra_r2_informe.py` | `verificacion_R2.csv`, este informe | `T4_verificacion_log.txt` |

