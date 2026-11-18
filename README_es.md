# Geoestadística bayesiana con datos censurados para el mapeo de metales pesados en suelo — Santa Bárbara, Huancavelica (Perú)

**English documentation: [README.md](README.md)**

## Acerca de este trabajo

Código y datos analíticos del artículo **«Bayesian geostatistics with censored data for mapping heavy-metal soil contamination around the Santa Bárbara mine (Huancavelica, Peru)»**, enviado a *Stochastic Environmental Research and Risk Assessment* (SERRA). Autor: Diego Alonso Araujo Villegas, Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Perú — diego.araujo1@unmsm.edu.pe.

El artículo ajusta un proceso gaussiano espacial con verosimilitud censurada por intervalo (Tobit) a 114 ubicaciones de suelo en el entorno de la mina de mercurio histórica de Santa Bárbara, donde la proporción de mediciones por debajo del límite de detección $L$ va de 0 a 93 % según el metal. Lo compara con la práctica habitual de sustituir los valores censurados por $L/2$ (en un proceso gaussiano y en kriging ordinario) mediante un experimento de enmascaramiento sobre tres metales sin censura real y mediante validación cruzada espacial por bloques. La sustitución conserva la precisión puntual pero degrada la calibración de la incertidumbre (con 80 % de censura, la cobertura de los intervalos al 95 % del arsénico cae a 0.68 y a 0.53, frente a 0.94 del modelo censurado). Con el modelo calibrado se producen los mapas de concentración, de incertidumbre predictiva y de probabilidad de excedencia de los estándares de calidad ambiental (ECA) peruanos, con el cadmio, censurado en 79.8 %, como caso de aplicación.

El trabajo nació como proyecto del curso *Análisis Bayesiano* (Escuela Profesional de Estadística, Facultad de Ciencias Matemáticas, UNMSM, 2026-II).

## Estructura del repositorio

```
README.md, README_es.md     este archivo (español) y su equivalente en inglés
CITATION.cff, .zenodo.json  metadatos de cita y de archivo
LICENSE                     MIT (código); ver «Licencia» para los datos
requirements.txt            dependencias con versiones

data/final/                 dataset analítico y su diccionario
    santa_barbara_soil.csv  114 ubicaciones x 10 analitos, formato largo (también en .parquet)
    DATA_DICTIONARY.md      descripción de cada columna
    mine_workings.csv       21 labores mineras (bocaminas, desmonteras), para las covariables de distancia
notebooks/                  01_dataset_construction.ipynb, 02_bayesian_model.ipynb
notebooks/_src/             el código de ambos notebooks (sbmodel.py = el modelo) y los workers
                            del experimento de enmascaramiento
src/                        pipeline, experimentos, scripts de figuras y tablas; config.py guarda las semillas
figuras/                    figuras y tablas (CSV/TEX) del pipeline
figuras/serra/              las figuras del artículo en PDF/EPS/TIFF (Fig1-Fig11 y FigS1) y las
                            comprobaciones de fuentes y rásteres del estándar gráfico
outputs/serra_r1/           evidencia del Online Resource 1 (CSV, logs, informe)
outputs/serra_r2/           resultados de mercurio recalculados con el ajuste reforzado (CSV, logs, informe)
```

Las demás carpetas (checkpoints de MCMC, cachés, salidas intermedias) no se versionan a propósito: ver *Reproducibilidad*.

## Reproducibilidad

### Entorno

```bash
python -m venv .venv && source .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

El código se desarrolló y se corrió con Python 3.14 en Windows, sin compilador de C: PyMC 6 corre sobre **nutpie con el backend numba** (NUTS), ArviZ hace el diagnóstico (R-hat, ESS de cuerpo y de cola, divergencias, LOO), PyKrige da el baseline de kriging y geopandas/rasterio/pyproj manejan lo espacial. Las versiones exactas están en `requirements.txt`.

### Semillas y checkpoints

- Construcción del dataset: semilla 20260906 (notebook 1). Corrida de modelos: `SEED = 20260923`, `RUN_ID = F4-20260923` (`src/config.py`). Mapas posteriores: 4000 muestras, semilla 20260906.
- Los checkpoints de MCMC (`.nc`, `.npz`, `.json`) son voluminosos y **no se versionan**. Los scripts los escriben bajo `outputs/` y son reanudables: una unidad que ya tiene su checkpoint se omite.
- Rutas absolutas: `PROJ` está escrito a mano al inicio de `notebooks/_src/nb01_dataset.py`, `nb02.py` y `nb02_*.py` (y en los notebooks): edítalo a la ubicación de tu clon. Los scripts de `src/` encuentran la raíz del proyecto desde su propia ubicación.
- Entradas que no se versionan: las tablas Excel del OEFA (`data/fase1/oefa/suelo__*.xlsx`) y el modelo de elevación SRTM de 30 m (`data/fase1/covariables/santa_barbara_dem_srtm30.tif`), que necesitan el notebook 1, los mapas sobre la grilla de predicción y los scripts de Cd/Hg que reconstruyen las covariables de la grilla; los límites administrativos de los mapas (INEI) y los mosaicos del mapa base (OpenStreetMap vía `contextily`, requiere red).

### Orden de ejecución

No hay un único script maestro; las etapas siguientes se corren en este orden y cada una lee los checkpoints de las anteriores.

| # | Etapa | Cómo | Costo (medido en una estación de 20 núcleos) |
|---|---|---|---|
| 1 | Dataset | `notebooks/01_dataset_construction.ipynb` | minutos; necesita las entradas no versionadas de arriba (el CSV resultante ya está versionado) |
| 2 | Análisis exploratorio, chequeos de priors, ajustes principales, validación cruzada, mapas, coregionalización | `notebooks/02_bayesian_model.ipynb` | unos 10 min por ajuste principal; 13 min el notebook completo con todo en caché |
| 3 | Experimento de enmascaramiento: 240 ajustes MCMC (GP censurado y GP con $L/2$) más las 120 celdas de kriging | `python src/run_h31_h45.py`, luego `python src/a1_kriging_deep.py` | **caro**: unas 12 h con 10 procesos; el kriging es instantáneo |
| 4 | Calibración y recuperación: 200 réplicas de SBC, 50 de recuperación | `python src/c1_sbc.py`, `python src/c2_recovery_reps.py` | **caro**: unos 74 s por réplica, unas 5 h en total |
| 5 | Ajustes y chequeos del caso de aplicación: cadmio con $L$ = LOD, validación cruzada espacial, sensibilidad al prior, mezcla, ajuste reforzado de mercurio | `src/flip_ld_cd_f2_f4.py`, `src/cv_cd_LD.py`, `src/prior_sens_cd_LD.py`, `src/b3_mezcla_censurada.py`, `src/h_hg_strict_probe.py` | ajustes individuales del orden de 10 min o más (el de mercurio reforzado es el más lento); los mapas de los cinco metales tardan unos 35 min |
| 6 | Evidencia de reproducibilidad del Online Resource 1 y recálculo de mercurio | `src/serra_r1_*.py`, `src/serra_r2_*.py` | **rápido**: posprocesamiento de las posteriores guardadas, sin MCMC nuevo |
| 7 | Figuras del artículo y sus comprobaciones | `python src/figserra.py`, `python src/verify_figserra.py`, `python src/serra_r2_verificar.py` | **rápido**: redibujo desde arrays ya existentes, unos minutos |

Las figuras en español del pipeline en `figuras/` (`FIG*.pdf`) salen de los mismos resultados: `python src/mapas_f3.py` dibuja los mapas de media, incertidumbre y excedencia, y `python src/fig9_crps.py` las curvas de CRPS del experimento de enmascaramiento (la cabecera de cada script de `src/` indica la figura que escribe). Cada salida lleva su `RUN_ID` en los metadatos.

Solo las etapas 3 y 4 necesitan horas; todo lo que viene después de tener las posteriores es rápido. Para comprobar las cifras sin reajustar nada, el punto de entrada más barato son las etapas 6/7 sobre los resultados guardados: `python src/serra_r1_tabla6.py` recalcula la Tabla 6 y los tests de Wilcoxon desde `figuras/TABLE_H6_MAIN_con_kriging.csv`, y `python src/serra_r2_verificar.py` corre las 42 comprobaciones del recálculo de mercurio (`outputs/serra_r2/verificacion_R2.csv`).

## Mapa de resultados

Los nombres de archivo de las figuras del artículo siguen `src/figserra.py` (se dibujan once figuras): las Figs. 7–10 del manuscrito son los archivos `Fig8`–`Fig11`, y el archivo `Fig7` (gradiente de censura real frente a impuesto) se dibuja pero no se usa en el manuscrito.

### Tablas del artículo

| Elemento del artículo | Script(s) | Salida |
|---|---|---|
| Tabla 1, dataset por analito | `notebooks/_src/nb01_dataset.py` (notebook 1) | `figuras/TABLE1_dataset_summary.csv` |
| Tabla 2, distancia a las labores de ubicaciones censuradas vs detectadas | `src/a0_eda_consolidado.py` | `figuras/TABLE_A0_censura_espacial.csv` |
| Tabla 3, indicador ECA/LOD | `notebooks/_src/nb01_dataset.py` | `figuras/TABLE3_eca_lod_criterion.csv` |
| Tabla 4, diagnósticos de convergencia | `notebooks/_src/nb02_c_main.py`; fila de mercurio con el ajuste reforzado: `src/h_hg_strict_probe.py`, `src/serra_r2_hg_refuerzo.py` | `figuras/TABLE10_convergence.csv`, `outputs/serra_r2/tabla4_convergencia_articulo.csv` |
| Tabla 5, recuperación de parámetros (50 réplicas) | `src/c2_recovery_reps.py` | `figuras/TABLE9b_recovery_reps.csv` |
| Tabla 6, cobertura por metal y nivel de censura (360 filas) | `src/run_h31_h45.py` con `notebooks/_src/worker_v4.py`; `src/a1_kriging_deep.py`; recálculo independiente y tests de Wilcoxon: `src/serra_r1_tabla6.py` | `figuras/TABLE_H6_MAIN_con_kriging.csv`; `outputs/serra_r1/T6_recalculo.csv`, `T6_diff.csv`, `T6_wilcoxon.csv` |
| Tabla 7, validación cruzada espacial por bloques | `notebooks/_src/nb02_e_cv.py`; cadmio con $L$ = LOD: `src/cv_cd_LD.py` | `figuras/TABLE17_cross_validation.csv` |
| Tabla 8, sensibilidad al prior | `src/prior_sens_cd_LD.py` | `figuras/TABLE_prior_sens_Cd_LD.csv` |
| Tabla 9, excedencia por umbral y uso de suelo | `src/t2_1_area_excedida_posterior.py`, `notebooks/_src/nb02_f_maps.py`; cadmio: `src/flip_ld_cd_f2_f4.py`; mercurio: `src/serra_r2_hg_refuerzo.py` | `figuras/TABLE_area_excedida_posterior.csv`, `figuras/TABLE20_exceedance_summary.csv`, `outputs/serra_r2/hg_antes_despues.csv`; comprobaciones en `outputs/serra_r2/verificacion_R2.csv` |
| Tabla 10, coregionalización | `notebooks/_src/nb02_g_design_lmc.py` | `figuras/TABLE23_lmc_cross_correlation.csv` |
| Tabla 11, componentes de la mezcla | `src/b3_mezcla_censurada.py` | `figuras/TABLE_B3_mixture_components.csv` |

### Figuras del artículo

Todas las dibuja `src/figserra.py` en `figuras/serra/` (PDF, EPS y TIFF); `src/verify_figserra.py` comprueba fuentes, tamaños y rásteres (`figuras/serra/_verify_fonts.csv`, `_verify_rasters.csv`).

| Elemento del artículo | Archivo | Datos de entrada |
|---|---|---|
| Fig. 1, sitio de estudio | `figuras/serra/Fig1.pdf` | `data/final/` |
| Fig. 2, patrón espacial de la censura | `figuras/serra/Fig2.pdf` | `data/final/` |
| Fig. 3, diagrama de placas | `figuras/serra/Fig3.pdf` | nodos y aristas definidos en `src/c2_plate_diagram.py` |
| Fig. 4, chequeo predictivo a priori | `figuras/serra/Fig4.pdf` | muestras predictivas a priori simuladas dentro de `src/figserra.py` con los priors de `notebooks/_src/sbmodel.py` y `data/final/` |
| Fig. 5, trazas de MCMC | `figuras/serra/Fig5.pdf` | los dos ajustes de `src/a2_baseline_no_convergencia.py` (checkpoints en `outputs/a2_stress_test/`, no versionados) |
| Fig. 6, cobertura frente al nivel de censura | `figuras/serra/Fig6.pdf` | `figuras/TABLE_H6_MAIN_con_kriging.csv` |
| Fig. 7, histogramas de rangos de SBC | `figuras/serra/Fig8.pdf` | `figuras/TABLE_C1_sbc_ranks.csv` (`src/c1_sbc.py`) |
| Fig. 8, mapas de mercurio | `figuras/serra/Fig9.pdf` | `outputs/serra_r2/posterior_maps_Hg_refuerzo.npz`; la dibuja `src/serra_r2_figuras.py` |
| Fig. 9, mapas de cadmio | `figuras/serra/Fig10.pdf` | mapas de cadmio de `src/flip_ld_cd_f2_f4.py` (checkpoint, no versionado) |
| Fig. 10, excedencia de cadmio por uso de suelo | `figuras/serra/Fig11.pdf` | los mismos mapas de cadmio |

### Online Resource 1 y los informes

| Elemento | Script | Salida |
|---|---|---|
| Sec. 1, Tabla S1: umbral de enmascaramiento con y sin el bloque de prueba | `src/serra_r1_umbral.py` | `outputs/serra_r1/umbral_fuga.csv` |
| Sec. 2, Tablas S2–S3 y Fig. S1: convergencia del campo latente y de las cantidades de los mapas | `src/serra_r1_campo_latente.py` | `outputs/serra_r1/campo_latente_diag.csv`, `campo_latente_por_dimension.csv`, `cantidades_mapa_diag*.csv`; `figuras/serra/FigS1_latent_diagnostics.pdf` |
| Sec. 3, Tabla S4: estabilidad del baseline $L/2$ | `src/serra_r1_baseline.py` | `outputs/serra_r1/baseline_diag.csv`, `baseline_cobertura_restringida.csv`, `baseline_mecanismo_ties.csv`, `baseline_diff_implementacion.md` |
| Sec. 4: SBC de cantidades derivadas (no disponible, con su motivo) | `src/serra_r1_sbc.py` | `outputs/serra_r1/sbc_derivadas.csv`, `sbc_derivadas_nota.md` |
| Mercurio con el ajuste reforzado: resultados, análisis de ruido, figuras, 42 comprobaciones | `src/serra_r2_inventario.py`, `serra_r2_hg_refuerzo.py`, `serra_r2_figuras.py`, `serra_r2_verificar.py` | `outputs/serra_r2/` |
| Informes de evidencia (originales en español e inglés) | `src/serra_r2_informe.py` (R2 en español) | `outputs/serra_r1/SERRA_R1_INFORME.md` (`SERRA_R1_REPORT.md`), `outputs/serra_r2/SERRA_R2_INFORME.md` (`SERRA_R2_REPORT.md`) |

## Procedencia de los datos

- **Fuente.** Informe OEFA *N.º 00341-2018-OEFA/DEAM-STEC* (evaluación ambiental de la unidad minera Santa Bárbara, Huancavelica, 2018). Se obtuvo por solicitud de acceso a la información pública al amparo de la Ley de Transparencia y Acceso a la Información Pública (Ley N.º 27806), expediente **2026-E01-114527**, y lo entregó la Dirección de Evaluación Ambiental (DEAM) del OEFA en <https://drive.google.com/drive/folders/1Vq_lQpRip0Q1i-UiC1JqNYwl4iyh9V6Z>.
- **Portales.** Datos abiertos del OEFA: <https://datosabiertos.oefa.gob.pe>; portal nacional, conjunto «OEFA – Evaluación ambiental»: <https://www.datosabiertos.gob.pe/dataset/oefa-evaluacion-ambiental>.
- **Lo que se redistribuye aquí.** El dataset analítico `data/final/santa_barbara_soil.csv` (114 ubicaciones, 10 analitos; también en `.parquet`) con su diccionario de variables `data/final/DATA_DICTIONARY.md` (cada columna, con unidades y fuente), y `data/final/mine_workings.csv`. Los datos crudos completos del OEFA no se versionan aquí (ver los portales).
- **Lo que no va en GitHub.** El informe escaneado (unos 666 MB) es demasiado grande para el repositorio. Se archiva en Zenodo junto con este código (ver *Cita*).
- **Coordenadas.** UTM zona 18S (EPSG:32718) y WGS84 (EPSG:4326).

## Cita

Si usas el código o los datos analíticos, cita la versión archivada de este repositorio y el informe fuente.

> Araujo Villegas, D. A. (2026). *censored-gp-santa-barbara: code and analytical data for "Bayesian geostatistics with censored data for mapping heavy-metal soil contamination around the Santa Bárbara mine (Huancavelica, Peru)"* (v2.0.0) [Software]. Zenodo.
>
> **DOI: to be added on publication** (se añadirá aquí cuando el registro esté publicado).

Los metadatos para gestores de citas están en [`CITATION.cff`](CITATION.cff). **Cita de los datos:** Organismo de Evaluación y Fiscalización Ambiental (OEFA), Informe N.º 00341-2018-OEFA/DEAM-STEC, evaluación ambiental de la unidad minera Santa Bárbara, Huancavelica, 2018.

## Licencia

Código bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los datos ambientales provienen del OEFA (organismo del Estado peruano) y se redistribuyen citando la fuente; el informe escaneado del OEFA archivado en Zenodo es un documento del OEFA y no está cubierto por la licencia MIT del código.
