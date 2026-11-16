# Geoestadística bayesiana con datos censurados — Santa Bárbara, Huancavelica

Mapeo probabilístico de metales pesados en suelo bajo censura por límite de detección, mediante procesos gaussianos espaciales bayesianos, en el entorno de la mina de mercurio histórica de Santa Bárbara (Huancavelica, Perú). Proyecto del curso Análisis Bayesiano, Escuela Profesional de Estadística, Facultad de Ciencias Matemáticas, UNMSM (2026-II).

**Autor:** Diego Alonso Araujo Villegas — diego.araujo1@unmsm.edu.pe

## Qué hace este trabajo

Ajusta un proceso gaussiano espacial con verosimilitud censurada (Tobit) por intervalo y lo compara, mediante validación cruzada espacial por bloques y un experimento de enmascaramiento sobre campos de verdad conocida, contra la sustitución clásica por la mitad del límite de detección ($L/2$) y el kriging ordinario. El resultado central es que, bajo censura fuerte, la sustitución conserva la precisión pero destruye la calibración de la incertidumbre; el modelo censurado la preserva. Con el modelo calibrado se producen los mapas de exposición, de incertidumbre y de probabilidad de excedencia de los estándares de calidad ambiental.

## Datos

- **Fuente:** OEFA — Datos Abiertos, componente Suelo. Informe N.° 00341-2018-OEFA/DEAM-STEC.
  Portal: https://datosabiertos.gob.pe/dataset/oefa-evaluacion-ambiental
- **Procedencia del informe:** el Informe N.° 00341-2018-OEFA/DEAM-STEC se obtuvo por solicitud de acceso a la información pública (Ley N.° 27806, expediente 2026-E01-114527); ver *Data provenance* más abajo.
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
- **Objetos InferenceData (ArviZ):** los checkpoints de MCMC (`.nc`, `.npz`, `.json`) son voluminosos y no se versionan en este repositorio; los scripts los regeneran desde cero con las semillas fijadas.

## Software principal

PyMC + nutpie (NUTS, backend numba) para la inferencia; ArviZ para diagnóstico (R-hat, ESS, divergencias, LOO); PyKrige para el baseline de kriging; geopandas/pyproj/rasterio para el manejo espacial; matplotlib para las figuras. Versiones exactas en `requirements.txt`.

## Cita

Snapshot de este repositorio archivado en Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22968165.svg)](https://doi.org/10.5281/zenodo.22968165)
— el DOI de concepto resuelve siempre a la versión más reciente; cada release queda además con su
propio DOI versionado.

## Cita de los datos

Organismo de Evaluación y Fiscalización Ambiental (OEFA), Informe N.° 00341-2018-OEFA/DEAM-STEC, evaluación ambiental de la unidad minera Santa Bárbara, Huancavelica, 2018.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos ambientales provienen del OEFA (datos abiertos del Estado peruano) y se redistribuyen citando la fuente.

---

## About

Bayesian spatial geostatistics with censored data for mapping heavy-metal soil contamination
around the Santa Bárbara historical mercury mine (Huancavelica, Peru). Fits a spatial Gaussian
process with an interval-censored (Tobit) likelihood and compares it, via spatial block
cross-validation and a masking experiment on fields of known truth, against the classical
substitution by half the detection limit ($L/2$) and ordinary kriging. Under strong censoring,
substitution preserves point accuracy but destroys uncertainty calibration; the censored model
preserves it. The calibrated model produces the exposure, uncertainty and exceedance-probability
maps against environmental quality standards.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/mapas_f3.py        # mean, uncertainty and exceedance maps
python src/fig9_crps.py       # CRPS curves from the masking experiment
# see each script's header in src/ for the figure it produces
```

MCMC checkpoints (`.nc`, `.npz`, `.json`) are not versioned here (too large); every script
regenerates them from scratch with the fixed seeds listed above. `outputs/serra_r1/` holds the
CSV evidence and the execution logs behind the reproducibility checks prepared ahead of
submission (Tables/Figures recomputed independently from the same source files cited in the
paper; see `outputs/serra_r1/SERRA_R1_INFORME.md`).

## Data provenance

The source report, OEFA Informe N.° 00341-2018-OEFA/DEAM-STEC, was obtained through a public
records request under Peru's Transparency and Access to Public Information Law (Ley N.° 27806),
case file 2026-E01-114527. The analytical dataset `data/final/santa_barbara_soil.csv` is derived
from that report.

**Redistributed here:** the cleaned analytical CSV (`data/final/santa_barbara_soil.csv`) and its
data dictionary — what the paper's statistical analysis is actually built on.
**Not redistributed:** the scanned OEFA PDF report itself, because of its size; it can be
requested from OEFA by citing the case file above, or obtained through Peru's open-data portal
for the underlying dataset (see *Datos* above).

## Citation

If you use this code or the analytical dataset, please cite the Zenodo archive (badge above) and
the source report (OEFA, cited above under *Cita de los datos*).

- **This release (v1.5):** https://doi.org/10.5281/zenodo.24291974 — the version that matches the
  analyses and figures of the paper; cite it when a specific version is needed.
- **All versions (concept DOI, always resolves to the latest):** https://doi.org/10.5281/zenodo.22968165
