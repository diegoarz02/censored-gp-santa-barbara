# Geoestadística bayesiana con datos censurados — Santa Bárbara, Huancavelica

Mapeo probabilístico de metales pesados en suelo bajo censura por límite de detección, mediante procesos gaussianos espaciales bayesianos, en el entorno de la mina de mercurio histórica de Santa Bárbara (Huancavelica, Perú). Proyecto del curso Análisis Bayesiano, Escuela Profesional de Estadística, Facultad de Ciencias Matemáticas, UNMSM (2026-II).

**Autor:** Diego Alonso Araujo Villegas — diego.araujo1@unmsm.edu.pe

## Qué hace este trabajo

Ajusta un proceso gaussiano espacial con verosimilitud censurada (Tobit) por intervalo y lo compara, mediante validación cruzada espacial por bloques y un experimento de enmascaramiento sobre campos de verdad conocida, contra la sustitución clásica por la mitad del límite de detección ($L/2$) y el kriging ordinario. El resultado central es que, bajo censura fuerte, la sustitución conserva la precisión pero destruye la calibración de la incertidumbre; el modelo censurado la preserva. Con el modelo calibrado se producen los mapas de exposición, de incertidumbre y de probabilidad de excedencia de los estándares de calidad ambiental.

## Datos

- **Fuente:** OEFA — Datos Abiertos, componente Suelo. Informe N.° 00341-2018-OEFA/DEAM-STEC.
  Portal: https://datosabiertos.gob.pe/dataset/oefa-evaluacion-ambiental
- **Dataset analítico:** `data/final/santa_barbara_soil.csv` — 114 ubicaciones, 10 analitos, formato largo. Diccionario de variables en `data/final/DATA_DICTIONARY.md`.
- **CRS:** UTM zona 18S (EPSG:32718) y WGS84 (EPSG:4326).
- Los datos crudos completos del OEFA no se versionan aquí (ver portal); el dataset limpio y su diccionario sí.

## Estructura del repositorio

```
notebooks/            01_dataset_construction.ipynb, 02_bayesian_model.ipynb
notebooks/_src/       sbmodel.py (modelo), y utilidades compartidas
src/                  scripts del pipeline, análisis y figuras
data/final/           dataset analítico + diccionario de variables
figuras/              figuras del artículo (PDF) y tablas (CSV/TEX)
prompts/              prompts usados para generar y auditar el análisis
requirements.txt      dependencias con versiones
README.md
```

## Reproducibilidad

- **Semillas fijadas:** construcción del dataset, semilla 20260906 (notebook 1); corrida de modelos, `SEED = 20260923` (`src/config.py`, `RUN_ID = F4-20260923`). Cada salida lleva su `RUN_ID` en los metadatos.
- **Entorno:**
  ```bash
  python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Reproducir las figuras principales:**
  ```bash
  python src/mapas_f3.py        # mapas de media, incertidumbre y excedencia
  python src/fig9_crps.py       # curvas de CRPS del experimento de enmascaramiento
  # (ver la cabecera de cada script en src/ para la figura que genera)
  ```
- **Objetos InferenceData (ArviZ):** los checkpoints de MCMC (`.nc`, `.npz`, `.json`) son voluminosos y se archivan en Zenodo con DOI (ver más abajo), no en este repositorio; los scripts los regeneran desde cero con las semillas fijadas.

## Software principal

PyMC + nutpie (NUTS, backend numba) para la inferencia; ArviZ para diagnóstico (R-hat, ESS, divergencias, LOO); PyKrige para el baseline de kriging; geopandas/pyproj/rasterio para el manejo espacial; matplotlib para las figuras. Versiones exactas en `requirements.txt`.

## Cita de los datos

Organismo de Evaluación y Fiscalización Ambiental (OEFA), Informe N.° 00341-2018-OEFA/DEAM-STEC, evaluación ambiental de la unidad minera Santa Bárbara, Huancavelica, 2018.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos ambientales provienen del OEFA (datos abiertos del Estado peruano) y se redistribuyen citando la fuente.
