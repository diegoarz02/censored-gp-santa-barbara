<!-- generado: 2026-11-16, CERRADO -->

# SERRA_R1_INFORME.md — evidencia reproducible antes del envío a SERRA

Evidencia reproducible de cinco puntos (Tabla 6 y Wilcoxon, campo latente y cantidades de los
mapas, baseline GP-L/2, umbral del enmascaramiento, SBC y cantidades derivadas), preparada antes
del envío a SERRA. **No se reajustó ningún modelo principal, ningún
número del artículo cambió.** Todo lo que sigue es post-proceso o recalculo puro sobre
checkpoints/idata ya existentes, salvo donde se declara explícitamente lo contrario. No se tocó
`main.tex` ni `referencias.bib`.

---

## T1 — Tabla 6, Fig. 6 y Wilcoxon

Script: `src/serra_r1_tabla6.py`. Fuente: `figuras/TABLE_H6_MAIN_con_kriging.csv`.

1. **360 filas confirmadas** (3 modelos × 3 metales × 4 niveles × 10 réplicas), las 360 con
   `deep=True`.
2. **Las 36 medias de `picp95` recalculadas coinciden con el manuscrito**: diferencia máxima
   absoluta = **0.000497** (dentro de la tolerancia 0.0005; 0 de 36 filas fuera de tolerancia).
   `outputs/serra_r1/T6_recalculo.csv`, `T6_diff.csv`.
3. **Rangos al 80 % del censurado**: Hg [0.5714, 1.0000] (manuscrito 0.57–1.00 ✓), Pb [0.3636,
   0.9615] (manuscrito 0.36–0.96 ✓).
4. **`src/figserra.py::fig6()` lee exactamente `TABLE_H6_MAIN_con_kriging.csv`** y usa
   `groupby("level")["picp95"].mean()` — confirmado leyendo el código fuente de la figura, no
   asumido.
5. **Wilcoxon de rangos con signo** (prueba exacta, enumeración completa de 2ⁿ, sin scipy; 15
   unidades metal×bloque, promediando las 2 semillas de cada bloque — `outputs/serra_r1/T6_wilcoxon.csv`):

   | nivel | baseline | censurado gana | p bilateral |
   |---|---|---|---|
   | 80 % | GP-L/2 | 15/15 | **6.10×10⁻⁵** (manuscrito: 15/15, 6.1×10⁻⁵ ✓) |
   | 80 % | kriging | 15/15 | **6.10×10⁻⁵** (✓) |
   | 60 % | GP-L/2 | 7/9 | **0.0195** (✓) |
   | 60 % | kriging | 8/13 | **0.0325** (✓) |
   | 40 % | GP-L/2 / kriging | 4/7 | 0.203 (no significativo, ✓) |
   | 20 % | GP-L/2 / kriging | 1/4, 4/7 | 1.00 / 0.578 (no significativo, ✓) |

   **Los 8 p-valores reproducen exacto los citados en el manuscrito.**

   Cita del código que confirma que las réplicas r y r+5 comparten máscara y bloque, y difieren
   solo en la semilla (`notebooks/_src/worker_v4.py`):
   - L.102: `"block": r % n_blocks` → r y r+5 caen en el mismo bloque (`r%5 == (r+5)%5`).
   - L.128: `seed = SEED0 + 1000*rep + 97*int(level*100) + 13*(hash(metal)%1000)` → depende de
     `rep`, no de `block`: r y r+5 difieren **solo** en la semilla MCMC.
   - L.135–136: `limit = np.quantile(value, level); censored = value < limit` → depende solo de
     (metal, level), igual para las 10 réplicas — comparten máscara por construcción.

**Veredicto T1: sin diferencias.**

---

## T2 — Diagnóstico del campo latente y de las cantidades de los mapas

Script: `src/serra_r1_campo_latente.py`. **Sin MCMC nuevo** — post-proceso determinístico de los
draws ya existentes en los 5 `InferenceData` principales (As, Ba, Pb: `main_*.nc`; Hg: protocolo
reforzado `probe_Hg_strict_2026-10-26_ta0999.nc`; Cd: L=LOD, `flip_LD_Cd.nc`).

### T2.1 — Dimensiones del campo latente (`f`) y de η(s) (`mu`), 114 ubicaciones, por metal

| metal | variable | R̂ máx | % R̂≥1.01 | ESS bulk mín | % ESS bulk<400 | ESS tail mín | % ESS tail<400 |
|---|---|---|---|---|---|---|---|
| As | f | 1.0094 | 0 % | 1012 | 0 % | 1553 | 0 % |
| As | mu | 1.0018 | 0 % | 2778 | 0 % | 2832 | 0 % |
| Ba | f | 1.0035 | 0 % | 1255 | 0 % | 1652 | 0 % |
| Ba | mu | 1.0021 | 0 % | 2955 | 0 % | 3041 | 0 % |
| Pb | f | 1.0037 | 0 % | 1456 | 0 % | 1785 | 0 % |
| Pb | mu | 1.0049 | 0 % | 1572 | 0 % | 1912 | 0 % |
| Hg | f | 1.0044 | 0 % | 1555 | 0 % | 1892 | 0 % |
| Hg | mu | 1.0016 | 0 % | 1316 | 0 % | 3074 | 0 % |
| Cd | f | 1.0026 | 0 % | 1884 | 0 % | 1864 | 0 % |
| Cd | mu | 1.0026 | 0 % | 2005 | 0 % | 1989 | 0 % |

**Las 114×2×5=1140 dimensiones del campo/η(s): 0 % con R̂≥1.01, 0 % con ESS<400, en los 5
metales.** Detalle completo (mín/q05/q50/q95/máx): `outputs/serra_r1/campo_latente_diag.csv`.

### T2.2.a — Área excedida por draw, conservando cadena (Cd, Hg, Pb, As — Ba excluido por diseño de la tarea)

Reconstrucción del área agrícola excedida por draw, **conservando la estructura (chain, draw)**
(no la mezcla aleatoria que usa el pipeline de mapas para los resúmenes publicados — necesaria
aquí para que R̂ tenga sentido). Grilla oficial, 50 m, 7381 celdas.

| metal | R̂ | ESS bulk | ESS tail | mediana % | IC95% | MCSE mediana (pp) | MCSE cuantil 2.5 % (pp) | MCSE cuantil 97.5 % (pp) |
|---|---|---|---|---|---|---|---|---|
| Cd | 1.0000 | 4291 | 2782 | 40.11 | [21.93, 56.50] | 0.183 | 0.386 | 0.352 |
| Hg | 0.9999 | 3372 | 4418 | 96.53 | [89.20, 98.92] | 0.047 | 0.217 | 0.020 |
| Pb | 1.0046 | 1953 | 2434 | 84.62 | [69.20, 94.54] | 0.163 | 0.461 | 0.176 |
| As | 1.0039 | 1361 | 2095 | 71.89 | [52.70, 88.46] | 0.291 | 0.488 | 0.312 |

**Las 4: R̂<1.01, ESS bulk/tail>1300, MCSE de la mediana <0.3 puntos porcentuales (de los
cuantiles 2.5/97.5, <0.5 pp).** La mediana de
Cd (40.11 %) coincide con la ya reportada en `outputs/flip_ld_cd/maps_50m.npz` (40.09 %) — la
pequeña diferencia (0.02 pp) es la esperada entre usar TODOS los draws en orden natural (aquí,
necesario para preservar cadena) contra el submuestreo aleatorio sin reemplazo que usa el
pipeline de mapas (ambos leen la misma predicción subyacente). Detalle:
`outputs/serra_r1/cantidades_mapa_diag_area.csv`.

### T2.2.b — MCSE de P(c>ECA) por celda, Cd

**3.75 % de las 7381 celdas** tienen clasificación potencialmente inestable
(|P(exc)−0.5| < 2·MCSE), es decir, podrían cruzar el umbral de decisión 0.5 por ruido de Monte
Carlo más que por señal real (277 celdas). **Distribución del MCSE por celda** (en probabilidad,
no en porcentaje): mín 0.0017; cuantiles 5/25/50/75/95 % = 0.0035 / 0.0060 / 0.0073 / 0.0078 /
0.0089; máx 0.0117. Para dimensionar: 3451 celdas tienen P(exc) entre 0.3 y 0.7, y 277 de ellas
(≈8 %) son las inestables. Detalle completo (las 7381 celdas, con `grid_e`/`grid_n`/
`p_exceedance`/`mcse`/`unstable_0.5`): `outputs/serra_r1/cantidades_mapa_diag_cd_cellwise.csv`.

### T2.2.c — 6 ubicaciones representativas (2 fondo, 2 cerca de labores, 2 en borde de extrapolación), los 5 metales

Cantidad diagnosticada: la media posterior de log c (η(s), `mu`) por draw, en celdas de la
grilla oficial de 50 m. Reglas de selección fijadas de antemano, no ajustadas a ningún resultado;
"con soporte" = a ≤ 100 m del dato más cercano:
- **cerca de labores**: las 2 celdas con soporte más cercanas a una labor minera real
  (`data/final/mine_workings.csv`, 21 bocaminas y desmontes): a 5.1 m y 7.8 m de una labor;
- **fondo**: las 2 celdas con soporte más lejanas de cualquier labor (1190 m y 1187 m);
- **borde de extrapolación**: las 2 celdas más lejanas de cualquier dato (1712 m y 1706 m).

Para predecir solo estas 6 celdas (en vez de las 7381, que tomó horas en Hg) se verificó que la
media condicional de una celda no depende de qué otras celdas se predicen a la vez: Cd, 6 celdas
vs 6 + 100 al azar, mismo seed, **máx |ΔMU| = 1.4×10⁻⁹**
(`outputs/serra_r1/T2_reanudacion_log.txt`).

**Las 30 combinaciones (6 ubicaciones × 5 metales): R̂ máximo 1.0035 (Pb, borde de
extrapolación), ESS bulk mínimo 1484 (As, borde de extrapolación 1), ESS tail mínimo 1583 (As,
borde de extrapolación 2).** Ninguna por debajo de los umbrales habituales (R̂<1.01, ESS>400) —
incluida la zona de extrapolación, que es la más expuesta a baja información. Detalle:
`outputs/serra_r1/cantidades_mapa_diag_ubicaciones.csv`.

*Corrección respecto a la primera versión de esta sección (commit `41b285d`)*: allí "cerca de
labores" se había definido con un proxy (celdas próximas a la mediana de la distancia al dato
más cercano), que no es distancia a labores, y "fondo" eran las celdas más cercanas a un dato. Se
rehízo con las reglas de arriba. Valores anteriores → nuevos: R̂ máx 1.0035 → 1.0035 (la misma
celda de borde), ESS bulk mín 1316 → 1484, ESS tail mín 1582 → 1583. La conclusión no cambia.
Ningún número del artículo cita esta tabla.

### Figura suplementaria

`figuras/serra/FigS1_latent_diagnostics.pdf` (+ .eps/.tif, 174 × 80 mm, Arial 8–10 pt,
embebida, sin título interno): **histogramas** de R̂ (panel a) y de ESS bulk (panel b) de las 114
dimensiones del campo latente `f`, un histograma por metal (cada metal con su propio color y
estilo de línea), con la línea de referencia en R̂ = 1.01 y en ESS = 400. Los valores crudos por
dimensión (5 metales × 114 × `f`/`mu`) están en `outputs/serra_r1/campo_latente_por_dimension.csv`;
su resumen coincide con `campo_latente_diag.csv` a 10⁻¹³. *(La primera versión mostraba barras
de R̂ máx y ESS mín por metal, no histogramas como pedía el encargo; reemplazada. Respaldo local en
`outputs/serra_r1/_prev/`, no publicado.)*

### Archivos de T2 y registro de ejecución

- `campo_latente_diag.csv` (resumen por metal/variable) y `campo_latente_por_dimension.csv` (crudo).
- `cantidades_mapa_diag.csv`: **un solo archivo en formato largo** (`quantity, metal,
  location_label, statistic, value`) con el área excedida, el resumen del MCSE de Cd y las
  ubicaciones. Es la concatenación, sin recálculo, de `cantidades_mapa_diag_area.csv`,
  `_cd_summary.csv` y `_ubicaciones.csv`; el detalle por celda queda en `_cd_cellwise.csv`.
- **Logs.** `T2_log.txt` es la salida de la corrida original y **está entrelazada**: por un
  error mío se lanzaron dos procesos del mismo script a la vez y sus `stdout` se mezclaron (de ahí
  las líneas cortadas). Los CSV no se corrompieron — se verificó el número de filas de cada uno
  (10, 4, 7381, 1 y 30 en la primera versión) y la corrida se hizo reanudable por archivo.
  `T2_reanudacion_log.txt` es la salida limpia de la reanudación final (T2.2c rehecho,
  coherencia por-dimensión, invarianza) y `T2_figS1_log.txt` la de FigS1.

**Veredicto T2: convergencia limpia en las 5 ajustes, en el campo latente, en η(s), y en las
cantidades derivadas que alimentan los mapas (área excedida, probabilidad por celda, ubicaciones
representativas). El único punto a declarar honestamente es el 3.75 % de celdas de Cd con
clasificación inestable por MCSE — no es un fallo de convergencia, es la incertidumbre
Monte Carlo real de una probabilidad cerca de 0.5 con ~4000 draws, y debería mencionarse como
matiz de la figura de excedencia, no escondido.**

---

## T3 — Estabilidad del baseline GP con L/2

Script: `src/serra_r1_baseline.py`.

**a. Diagnósticos por modelo × nivel** (`outputs/serra_r1/baseline_diag.csv`): confirmado exacto
— GP-L/2 divergencias totales = **52** (esperado 52), R̂ máx = **1.192** (esperado 1.19);
censurado R̂ máx = **1.024** (esperado 1.024). Por nivel (30 celdas metal×réplica por modelo y
nivel):

| nivel | GP-L/2: divergencias (suma / máx por celda) | GP-L/2: R̂ máx (mediana / máx) | GP-L/2: ESS bulk mín (mediana / mín) | censurado: divergencias | censurado: R̂ máx |
|---|---|---|---|---|---|
| 20 % | 0 / 0 | 1.007 / 1.021 | 645 / 346 | 0 | 1.017 |
| 40 % | 8 / 4 | 1.007 / 1.031 | 613 / 171 | 0 | 1.024 |
| 60 % | 18 / 6 | 1.008 / 1.016 | 531 / 242 | 0 | 1.018 |
| 80 % | 26 / 5 | 1.052 / 1.192 | 86 / 15 | 0 | 1.016 |

(0+8+18+26 = **52**.) El kriging no usa MCMC: sin R̂/ESS, 0 divergencias por construcción.

**b. Cobertura restringida a ajustes sanos** (`outputs/serra_r1/baseline_cobertura_restringida.csv`,
todas las combinaciones metal × nivel × modelo × criterio): se conservan solo las réplicas en que
**ambos** modelos GP pasan el criterio; el kriging se evalúa sobre ese mismo conjunto de réplicas.
`n` = réplicas que sobreviven (de 10). PICP95 medio:

| criterio | metal | nivel | n | censurado | GP-L/2 | kriging |
|---|---|---|---|---|---|---|
| R̂<1.01, 0 div. | As | 60 % | 3 | 1.000 | 1.000 | 1.000 |
| R̂<1.01, 0 div. | Hg | 60 % | 5 | 0.971 | 0.914 | 0.842 |
| R̂<1.01, 0 div. | Pb | 60 % | 3 | 0.848 | 0.742 | 0.722 |
| R̂<1.01, 0 div. | As | 80 % | **1** | 0.848 | 0.515 | 0.212 |
| R̂<1.01, 0 div. | Hg | 80 % | **1** | 0.606 | 0.182 | 0.212 |
| R̂<1.01, 0 div. | Pb | 80 % | **1** | 0.364 | 0.273 | 0.061 |
| R̂<1.05, 0 div. | As | 60 % | 8 | 0.981 | 0.950 | 0.912 |
| R̂<1.05, 0 div. | Hg | 60 % | 8 | 0.964 | 0.874 | 0.811 |
| R̂<1.05, 0 div. | Pb | 60 % | 8 | 0.917 | 0.795 | 0.715 |
| R̂<1.05, 0 div. | As | 80 % | 3 | **0.949** | **0.681** | 0.330 |
| R̂<1.05, 0 div. | Hg | 80 % | 2 | 0.589 | 0.234 | 0.249 |
| R̂<1.05, 0 div. | Pb | 80 % | **1** | 0.364 | 0.273 | 0.061 |

(Los niveles 20 % y 40 % están en el CSV.) **As al 80 % con R̂<1.05 (n = 3): 0.949 frente a
0.681, coincide exacto con lo ya visto desde el CSV.** En las 12 filas el censurado queda por
encima de GP-L/2 salvo As 60 % con R̂<1.01 (empate en 1.000). Con R̂<1.01 el 80 % queda con **un
solo** ajuste por metal: la dirección se mantiene pero esa fila **no sustenta una estimación**,
solo muestra que la brecha no la crean los ajustes mal convergidos. El kriging, sin MCMC, colapsa
igual al 80 % (0.330 / 0.249 / 0.061 aquí; 0.530 / 0.497 / 0.357 con las 10 réplicas), así que la
pérdida de cobertura no depende del muestreador.

**c. Comparabilidad de implementación**: `build_censored_gp()` (`sbmodel.py`) es la **misma
función** para los dos modelos. Única diferencia (L.122–127):
```python
if substitute_half_lod:
    y_value = np.where(censored, lod / 2.0, y_value)
    censored = np.zeros(n, bool)
```
Todo lo demás es código compartido: mismos priors (`beta0~N(ȳ,sd_beta0)`, `beta~N(0,sd_beta)`,
`b_bg~N(0,sd_beta)`, `ell~InvGamma`), misma parametrización de varianza particionada
(`sigma_tot~HalfNormal`, `rho~Beta(2,2)`, `eta=sigma_tot·√rho`, `sigma_n=sigma_tot·√(1-rho)`),
mismo núcleo Matérn 5/2, mismo `gp.Latent`, mismo `fit_model()` (draws/tune/target_accept/chains/
semilla idénticos, nutpie+numba). **No hay una construcción ni una llamada de ajuste separada
para el baseline — son idénticos salvo la verosimilitud**, confirmado leyendo el código, no
asumido. Detalle completo (código fuente íntegro citado): `outputs/serra_r1/baseline_diff_implementacion.md`.

**d. Mecanismo (investigación, sin reajustar)**: de las 120 celdas de GP-L/2, **34 (28 %) tienen
R̂>1.05 o divergencias**. En esas 34 celdas problemáticas, el porcentaje del entrenamiento
empatado exactamente en L/2 va de **29.6 % a 93.2 %** (media 71.3 %) —
`outputs/serra_r1/baseline_mecanismo_ties.csv`. Esto es **consistente con** la hipótesis planteada
(empates masivos en L/2 empujan σₙ→0 → geometría difícil), sin probarla causalmente (no se
inspeccionó directamente σₙ/ρ/ℓ por draw en esas celdas porque **ninguno de los 360 JSON de
`outputs/h6_deep/` guarda el idata completo por celda**: solo trae escalares — `rhat_max`,
`ess_bulk_min`, `ess_tail_min`, `divergences` — sin información por parámetro; verificar la
hipótesis a nivel de parámetro individual requeriría reajustar, declarado así, no simulado).

**El refit opcional y acotado (máximo 6 réplicas) NO se ejecutó, y la razón es importante**: el
punto c de arriba ya estableció que GP-L/2 y el censurado usan la **misma parametrización** —
la condición planteada para el refit ("si la parametrización del baseline difiere del
censurado") **no se cumple**. Reajustar GP-L/2 "con la misma parametrización que el censurado"
no tiene una acción correspondiente: ya la tiene. La única forma de "arreglarlo" sería dejar de
sustituir por L/2, lo que simplemente lo convertiría en el modelo censurado y anularía el
propósito del baseline. Se declara así en vez de forzar un reajuste sin sentido.

---

## T4 — Umbral del enmascaramiento (posible fuga)

Script: `src/serra_r1_umbral.py`.

1. **Confirmado en el código** (`worker_v4.py::run_deep()`, L.135–138): `limit =
   np.quantile(value, level)` se calcula sobre las 114 concentraciones completas, **antes** del
   split train/test (`labels == block` se aplica después) — se aplica igual a ambos por
   construcción, no hay dos cálculos distintos.
2. **60 combinaciones metal×nivel×bloque** (3×4×5, confirmado). Umbral calculado solo con
   entrenamiento vs. el global:
   - diferencia relativa de umbral: media 10.1 % (|media| 29.2 %, rango −75 % a +244 % — los
     extremos son celdas con pocos puntos de entrenamiento en niveles bajos, donde el cuantil es
     ruidoso);
   - **puntos de test que cambiarían de estado censurado/detectado: 86 de 1368 (6.3 % del
     total), media 4.6 % por celda, máximo 34.6 % en una celda** (detalle por nivel en
     `outputs/serra_r1/umbral_fuga.csv`: el efecto es más chico a mayor censura, 5.8 % de media al
     80 % frente a 3.0 % al 20 %).
3. Salida: `outputs/serra_r1/umbral_fuga.csv`.

**Veredicto T4**: hay una fuga de información real (el umbral ve el test), pero de magnitud
**modesta**: en promedio 1 de cada 22 puntos de test cambiaría de estado bajo un umbral
"honesto" (solo entrenamiento). No se reajustó ningún modelo para corregirlo — dimensionarlo era
el alcance de esta tarea.

---

## T5 — SBC y cantidades derivadas

Script: `src/serra_r1_sbc.py`.

**NO DISPONIBLE.** Verificado leyendo las claves reales de los checkpoints (no asumido por el
nombre del script):
- `outputs/c1_sbc/rep_*.json` (200 réplicas): guarda `ranks` como **rangos escalares** por
  parámetro (`eta, ell_0, ell_1, sigma_n, beta0`), no draws del campo ni de η(s) por ubicación.
- `outputs/c2_recovery/rep_*.json` (50 réplicas): solo metadatos (censura, LOD, tiempos) — ningún
  array de draws.
- **0 archivos `.nc`** en ninguna de las dos carpetas.

**Costo estimado de una corrida que sí los guarde (no ejecutada)**: habría que modificar
`run_replica()`/`simulate_one()` (`c1_sbc.py`) y la función equivalente de
`c2_recovery_reps.py` para persistir `f`/`mu` por réplica, y **repetir las 250 réplicas**
(el campo verdadero y el posterior se generan juntos, no se puede añadir el guardado sin
reajustar). Medido en este proyecto: ~74 s/réplica → 250×74s ≈ **5.1 horas**. Espacio: cada
idata completo de este modelo pesa 10–24 MB en `outputs/idata/`; guardar 250 (aunque sea solo
f/mu) costaría varios GB. Salida: `outputs/serra_r1/sbc_derivadas.csv`,
`sbc_derivadas_nota.md`.

---

## T6 — Fig. 2 (corrección gráfica)

`figuras/serra/Fig2.{pdf,eps,tif}` regenerada con dos cambios:
1. Eje x del panel (b): "distance to mining feature (m)" → **"distance to nearest mine working
   (m)"**.
2. Las etiquetas "detected"/"censored" **colisionaban** ("detecteddcensored") incluso tras el
   primer intento de más espacio + salto de línea — 6 cajas con etiqueta de 2 líneas no caben
   legibles a 174 mm. **Arreglo real**: el estado (detected/censored) ya está codificado por
   color desde el panel (a) — se quitó el texto repetido por caja, el eje x queda con **una
   etiqueta por metal** (Cd/Sb/Ag, centrada entre su par de cajas), y una leyenda única abajo
   (azul=Detected, gris=Censored) debajo del panel, igual al patrón del panel (a). Verificado
   visualmente tras el primer intento fallido — no se dio por bueno sin mirar el resultado.

**Verificación** (`src/verify_figserra.py`, que ahora incluye FigS1; salida íntegra en
`outputs/serra_r1/T6_verificacion_log.txt`, regeneración en `T6_log.txt`): fuentes Arial
embebidas (bytes reales extraídos del PDF), texto de línea base 8.0–10.0 pt (dentro de 8–12),
0 títulos internos, 174 × 90.5 mm, raster a 600 dpi. **Ningún número cambió** (los datos del
boxplot son los mismos; solo cambió la disposición visual y la etiqueta del eje). Versión
anterior conservada en `figuras/serra/_prev/Fig2_v1.{pdf,eps,tif}` (no publicada, solo respaldo
local). Los tamaños bajo 8 pt que reporta `_verify_fonts.csv` en la columna `fs_min_all` en
otras figuras son los sub/superíndices de mathtext (≈70 % del tamaño del símbolo por diseño), no
texto de cuerpo — ver la función `font_report()` de `src/verify_figserra.py`, que separa ambos.

---

## T7 — Repositorio y release v1.5

1. **Contenido publicado.** La lista blanca del `.gitignore` publica solo `src/`, `figuras/`,
   `notebooks/`, `data/final/` y los `.csv`/`.md`/`.txt` de `outputs/serra_r1/`; ningún
   checkpoint (`.nc`, `.npz`) ni el resto de `outputs/`. Entran: los scripts y salidas de T1–T6;
   el pipeline FIGSERRA (`src/estilo_serra.py`, `src/figserra.py`, `src/verify_figserra.py`);
   `src/v14_wilcoxon_bloques.py`; `src/v7rev_fig_axes.py` junto con la corrección de
   `utm_km_ticks()` (`src/estilo_figuras.py`, `src/mapas_f3.py`, `src/a0_eda_consolidado.py`);
   las figuras SERRA (`figuras/serra/`, Fig1–Fig11 y FigS1 en PDF/EPS/TIFF) y las cinco figuras
   regeneradas (FIG17 Cd/Hg, FIG18 Cd agrícola/residencial, FIG27). Los checkpoints de entrada
   que leen los scripts (`outputs/h6_deep/`, `outputs/idata/`, `outputs/c1_sbc/`,
   `outputs/c2_recovery/`, `outputs/flip_ld_cd/`) **no se publican** por tamaño.
2. **`.zenodo.json`** en la raíz: creador Araujo Villegas, Diego Alonso (ORCID
   0000-0002-6987-3948, Universidad Nacional Mayor de San Marcos), `upload_type` `software`,
   licencia MIT, versión `v1.5`.
3. **README**: sección en inglés (*About / How to reproduce / Data provenance / Citation*) añadida
   sin borrar la española. Procedencia: Informe N.° 00341-2018-OEFA/DEAM-STEC obtenido por solicitud
   de acceso a la información pública (Ley N.° 27806, expediente 2026-E01-114527); se redistribuye
   el CSV analítico `santa_barbara_soil.csv`, no el PDF escaneado del informe (por tamaño). Se
   mantiene la insignia del DOI de concepto (10.5281/zenodo.22968165).
4. **Revisión previa a publicar**: se barrió el árbol completo (texto con `git grep`; los PDF/TIFF
   extrayendo su texto y metadatos con PyMuPDF/PIL, porque la búsqueda binaria marca como
   coincidencias secuencias de bytes comprimidos que no son texto) y los mensajes de commit.
5. **Release.** Tag y release de GitHub `v1.5`. El DOI de versión, la versión, el creador y la
   lista de archivos que devuelva la API de Zenodo se registran tal cual en la sección
   *Verificación post-release* al final de este informe, que se añade una vez archivado (por eso
   el archivo del release no puede contenerla).

---

## Índice de evidencia (tarea → script → salidas → registro)

| tarea | script | salidas en `outputs/serra_r1/` | registro de ejecución |
|---|---|---|---|
| T1 | `src/serra_r1_tabla6.py` | `T6_recalculo.csv`, `T6_diff.csv`, `T6_wilcoxon.csv` | `T1_log.txt` |
| T2 | `src/serra_r1_campo_latente.py` | `campo_latente_diag.csv`, `campo_latente_por_dimension.csv`, `cantidades_mapa_diag.csv` (+ `_area`, `_cd_cellwise`, `_cd_summary`, `_ubicaciones`); `figuras/serra/FigS1_latent_diagnostics.*` | `T2_log.txt` (entrelazado, ver T2), `T2_reanudacion_log.txt`, `T2_figS1_log.txt` |
| T3 | `src/serra_r1_baseline.py` | `baseline_diag.csv`, `baseline_cobertura_restringida.csv`, `baseline_diff_implementacion.md`, `baseline_mecanismo_ties.csv` | `T3_log.txt` |
| T4 | `src/serra_r1_umbral.py` | `umbral_fuga.csv` | `T4_log.txt` |
| T5 | `src/serra_r1_sbc.py` | `sbc_derivadas.csv`, `sbc_derivadas_nota.md` | `T5_log.txt` |
| T6 | `src/figserra.py::fig2()`, `src/verify_figserra.py` | `figuras/serra/Fig2.*` (+ `_prev/` local) | `T6_log.txt`, `T6_verificacion_log.txt` |
| T7 | `.zenodo.json`, `README.md` | — | sección *Verificación post-release* |

Los scripts son deterministas sobre los checkpoints de entrada: T4 y T5 se volvieron a ejecutar
para generar su registro y reprodujeron CSV idénticos (`git` no detectó ningún cambio).

---

## Resumen de discrepancias con el artículo

**Ninguna.** Las 36 medias de la Tabla 6, los 8 p-valores del Wilcoxon, los 3 totales de
divergencias/R̂ del baseline, y el número de celdas del enmascaramiento (60, 360) coinciden
exactos con lo ya citado o con lo ya señalado desde el CSV. Los hallazgos nuevos de
esta ronda (3.75 % de celdas de Cd con clasificación inestable por MCSE; 6.3 % de fuga del umbral
del enmascaramiento; GP-L/2 sin diferencia de parametrización respecto al censurado) son
**matices que no estaban cuantificados antes**, no contradicciones de lo publicado.
