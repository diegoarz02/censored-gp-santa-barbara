# Censored Bayesian geostatistics for heavy-metal soil mapping — Santa Bárbara, Huancavelica (Peru)

**Spanish documentation: [README_es.md](README_es.md)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24347679.svg)](https://doi.org/10.5281/zenodo.24347679)

## About

Code and analytical data for the article **"Bayesian geostatistics with censored data for mapping heavy-metal soil contamination around the Santa Bárbara mine (Huancavelica, Peru)"**, submitted to *Stochastic Environmental Research and Risk Assessment* (SERRA). Author: Diego Alonso Araujo Villegas, Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Peru — diego.araujo1@unmsm.edu.pe.

The article fits a spatial Gaussian process with an interval-censored (Tobit) likelihood to 114 soil locations around the historical Santa Bárbara mercury mine, where the share of measurements below the limit of detection $L$ ranges from 0 to 93 % depending on the metal. It compares the model against the usual practice of substituting censored values by $L/2$ (in a Gaussian process and in ordinary kriging) through a masking experiment on three metals without real censoring and through spatial block cross-validation. Substitution preserves point accuracy but degrades the calibration of the uncertainty (at 80 % censoring the coverage of the 95 % intervals for arsenic falls to 0.68 and 0.53, against 0.94 for the censored model). The calibrated model is then used to map concentration, predictive uncertainty and the probability of exceeding the Peruvian environmental quality standards (ECA), with cadmium, 79.8 % censored, as the application case.

The work started as a project of the course *Análisis Bayesiano* (Escuela Profesional de Estadística, Facultad de Ciencias Matemáticas, UNMSM, 2026-II).

## Repository structure

```
README.md, README_es.md     this file (English) and its Spanish equivalent
CITATION.cff, .zenodo.json  citation and archive metadata
LICENSE                     MIT (code); see "License" for the data
requirements.txt            pinned dependencies

data/final/                 analytical data set and its dictionary
    santa_barbara_soil.csv  114 locations x 10 analytes, long format (also as .parquet)
    DATA_DICTIONARY.md      description of every column
    mine_workings.csv       21 mine workings (adits, waste dumps), used for distance covariates
notebooks/                  01_dataset_construction.ipynb, 02_bayesian_model.ipynb
notebooks/_src/             the code behind both notebooks (sbmodel.py = the model) and the
                            workers of the masking experiment
src/                        pipeline, experiments, figure and table scripts; config.py holds the seeds
figuras/                    figures and tables (CSV/TEX) of the pipeline
figuras/serra/              the article figures in PDF/EPS/TIFF (Fig1-Fig11, FigS1) and the
                            font/raster checks of the figure standard
outputs/serra_r1/           evidence behind Online Resource 1 (CSV, logs, report)
outputs/serra_r2/           mercury results recomputed with the strengthened fit (CSV, logs, report)
```

Every other folder (MCMC checkpoints, caches, intermediate outputs) is deliberately not versioned: see *How to reproduce*.

## How to reproduce

### Environment

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code was developed and run with Python 3.14 on Windows without a C compiler: PyMC 6 runs on **nutpie with the numba backend** (NUTS), ArviZ does the diagnostics (R-hat, bulk and tail ESS, divergences, LOO), PyKrige gives the kriging baseline and geopandas/rasterio/pyproj handle the spatial data. Exact versions are in `requirements.txt`.

### Seeds and checkpoints

- Dataset construction: seed 20260906 (notebook 1). Model runs: `SEED = 20260923`, `RUN_ID = F4-20260923` (`src/config.py`). Posterior maps: 4000 draws, seed 20260906.
- The MCMC checkpoints (`.nc`, `.npz`, `.json`) are large and are **not versioned**. The scripts write them under `outputs/` and are resumable: a unit that already has its checkpoint is skipped.
- Absolute paths: `PROJ` is hard-coded at the top of `notebooks/_src/nb01_dataset.py`, `nb02.py` and `nb02_*.py` (and in the notebooks): edit it to the location of your clone. The scripts in `src/` find the project root from their own location.
- Inputs that are not versioned: the OEFA Excel tables (`data/fase1/oefa/suelo__*.xlsx`) and the SRTM 30 m elevation model (`data/fase1/covariables/santa_barbara_dem_srtm30.tif`), needed by notebook 1, by the prediction-grid maps and by the Cd/Hg scripts that rebuild the grid covariates; the administrative boundaries of the maps (INEI) and the basemap tiles (OpenStreetMap through `contextily`, needs network).

### Order of execution

There is no single driver script; the stages below are run in this order, and each stage reads the checkpoints of the previous ones.

| # | Stage | How | Cost (measured on a 20-core workstation) |
|---|---|---|---|
| 1 | Data set | `notebooks/01_dataset_construction.ipynb` | minutes; needs the unversioned inputs above (the resulting CSV is already versioned) |
| 2 | Exploratory analysis, prior checks, the main model fits, cross-validation, maps, coregionalisation | `notebooks/02_bayesian_model.ipynb` | about 10 min per main fit; 13 min for the whole notebook with everything cached |
| 3 | Masking experiment: 240 MCMC fits (censored GP and GP with $L/2$) plus the 120 kriging cells | `python src/run_h31_h45.py`, then `python src/a1_kriging_deep.py` | **expensive**: about 12 h with 10 processes; kriging is instantaneous |
| 4 | Calibration and recovery: 200 SBC replicates, 50 recovery replicates | `python src/c1_sbc.py`, `python src/c2_recovery_reps.py` | **expensive**: about 74 s per replicate, about 5 h in total |
| 5 | Application fits and checks: cadmium with $L$ = LOD, spatial cross-validation, prior sensitivity, mixture, strengthened mercury fit | `src/flip_ld_cd_f2_f4.py`, `src/cv_cd_LD.py`, `src/prior_sens_cd_LD.py`, `src/b3_mezcla_censurada.py`, `src/h_hg_strict_probe.py` | single fits of the order of 10 min or more (the strengthened mercury fit is the slowest); the maps of the five metals take about 35 min |
| 6 | Reproducibility evidence of Online Resource 1 and mercury recomputation | `src/serra_r1_*.py`, `src/serra_r2_*.py` | **fast**: post-processing of the stored posteriors, no new MCMC |
| 7 | Article figures and their checks | `python src/figserra.py`, `python src/verify_figserra.py`, `python src/serra_r2_verificar.py` | **fast** once the checkpoints of the earlier stages exist: redrawing from arrays that already exist, a few minutes |

The Spanish-language figures of the pipeline in `figuras/` (`FIG*.pdf`) come from the same results: `python src/mapas_f3.py` draws the maps of mean, uncertainty and exceedance, and `python src/fig9_crps.py` the CRPS curves of the masking experiment (the header of each script in `src/` names the figure it writes). Every output carries its `RUN_ID` in the metadata.

Stages 3 and 4 are the only ones that need hours; everything after the posteriors exist is quick. To check the numbers without refitting anything, two scripts work straight from a fresh clone: `python src/serra_r1_tabla6.py` recomputes Table 6 and the Wilcoxon tests from the versioned `figuras/TABLE_H6_MAIN_con_kriging.csv`, and `python src/verify_figserra.py` checks fonts and sizes of the figures in `figuras/serra/`. The 42 checks of the mercury recomputation (`src/serra_r2_verificar.py`) read the posterior checkpoints, which are not versioned; their recorded result is `outputs/serra_r2/verificacion_R2.csv`.

## Results map

File names of the article figures follow `src/figserra.py` (eleven figures are drawn): manuscript Figs. 7–10 are the files `Fig8`–`Fig11`, and the file `Fig7` (real versus imposed censoring gradient) is drawn but not used in the manuscript.

### Tables of the article

| Article item | Script(s) | Output |
|---|---|---|
| Table 1, data set by analyte | `notebooks/_src/nb01_dataset.py` (notebook 1) | `figuras/TABLE1_dataset_summary.csv` |
| Table 2, distance of censored vs detected locations to the workings | `src/a0_eda_consolidado.py` | `figuras/TABLE_A0_censura_espacial.csv` |
| Table 3, ECA/LOD indicator | `notebooks/_src/nb01_dataset.py` | `figuras/TABLE3_eca_lod_criterion.csv` |
| Table 4, convergence diagnostics | `notebooks/_src/nb02_c_main.py`; mercury row with the strengthened fit: `src/h_hg_strict_probe.py`, `src/serra_r2_hg_refuerzo.py` | `figuras/TABLE10_convergence.csv`, `outputs/serra_r2/tabla4_convergencia_articulo.csv` |
| Table 5, parameter recovery (50 replicates) | `src/c2_recovery_reps.py` | `figuras/TABLE9b_recovery_reps.csv` |
| Table 6, coverage by metal and censoring level (360 rows) | `src/run_h31_h45.py` with `notebooks/_src/worker_v4.py`; `src/a1_kriging_deep.py`; independent recomputation and Wilcoxon tests: `src/serra_r1_tabla6.py` | `figuras/TABLE_H6_MAIN_con_kriging.csv`; `outputs/serra_r1/T6_recalculo.csv`, `T6_diff.csv`, `T6_wilcoxon.csv` |
| Table 7, spatial block cross-validation | `notebooks/_src/nb02_e_cv.py`; cadmium with $L$ = LOD: `src/cv_cd_LD.py` | `figuras/TABLE17_cross_validation.csv` |
| Table 8, prior sensitivity | `src/prior_sens_cd_LD.py` | `figuras/TABLE_prior_sens_Cd_LD.csv` |
| Table 9, exceedance by threshold and land use | `src/t2_1_area_excedida_posterior.py`, `notebooks/_src/nb02_f_maps.py`; cadmium: `src/flip_ld_cd_f2_f4.py`; mercury: `src/serra_r2_hg_refuerzo.py` | `figuras/TABLE_area_excedida_posterior.csv`, `figuras/TABLE20_exceedance_summary.csv`, `outputs/serra_r2/hg_antes_despues.csv`; checks in `outputs/serra_r2/verificacion_R2.csv` |
| Table 10, coregionalisation | `notebooks/_src/nb02_g_design_lmc.py` | `figuras/TABLE23_lmc_cross_correlation.csv` |
| Table 11, mixture components | `src/b3_mezcla_censurada.py` | `figuras/TABLE_B3_mixture_components.csv` |

### Figures of the article

All are drawn by `src/figserra.py` into `figuras/serra/` (PDF, EPS and TIFF); `src/verify_figserra.py` checks fonts, sizes and rasters (`figuras/serra/_verify_fonts.csv`, `_verify_rasters.csv`).

| Article item | File | Input data |
|---|---|---|
| Fig. 1, study site | `figuras/serra/Fig1.pdf` | `data/final/` |
| Fig. 2, spatial pattern of censoring | `figuras/serra/Fig2.pdf` | `data/final/` |
| Fig. 3, plate diagram | `figuras/serra/Fig3.pdf` | nodes and edges defined in `src/c2_plate_diagram.py` |
| Fig. 4, prior predictive check | `figuras/serra/Fig4.pdf` | prior predictive draws simulated inside `src/figserra.py` from the priors of `notebooks/_src/sbmodel.py` and `data/final/` |
| Fig. 5, MCMC traces | `figuras/serra/Fig5.pdf` | the two fits of `src/a2_baseline_no_convergencia.py` (checkpoints in `outputs/a2_stress_test/`, not versioned) |
| Fig. 6, coverage against censoring level | `figuras/serra/Fig6.pdf` | `figuras/TABLE_H6_MAIN_con_kriging.csv` |
| Fig. 7, SBC rank histograms | `figuras/serra/Fig8.pdf` | `figuras/TABLE_C1_sbc_ranks.csv` (`src/c1_sbc.py`) |
| Fig. 8, mercury maps | `figuras/serra/Fig9.pdf` | `outputs/serra_r2/posterior_maps_Hg_refuerzo.npz`; drawn by `src/serra_r2_figuras.py` |
| Fig. 9, cadmium maps | `figuras/serra/Fig10.pdf` | cadmium maps of `src/flip_ld_cd_f2_f4.py` (checkpoint, not versioned) |
| Fig. 10, cadmium exceedance by land use | `figuras/serra/Fig11.pdf` | same cadmium maps |

### Online Resource 1 and the reports

| Item | Script | Output |
|---|---|---|
| Sect. 1, Table S1: masking threshold with and without the test block | `src/serra_r1_umbral.py` | `outputs/serra_r1/umbral_fuga.csv` |
| Sect. 2, Tables S2–S3 and Fig. S1: convergence of the latent field and of the map quantities | `src/serra_r1_campo_latente.py` | `outputs/serra_r1/campo_latente_diag.csv`, `campo_latente_por_dimension.csv`, `cantidades_mapa_diag*.csv`; `figuras/serra/FigS1_latent_diagnostics.pdf` |
| Sect. 3, Table S4: stability of the $L/2$ baseline | `src/serra_r1_baseline.py` | `outputs/serra_r1/baseline_diag.csv`, `baseline_cobertura_restringida.csv`, `baseline_mecanismo_ties.csv`, `baseline_diff_implementacion.md` |
| Sect. 4: SBC of derived quantities (not available, with the reason) | `src/serra_r1_sbc.py` | `outputs/serra_r1/sbc_derivadas.csv`, `sbc_derivadas_nota.md` |
| Mercury with the strengthened fit: results, noise analysis, figures, 42 checks | `src/serra_r2_inventario.py`, `serra_r2_hg_refuerzo.py`, `serra_r2_figuras.py`, `serra_r2_verificar.py` | `outputs/serra_r2/` |
| Evidence reports (English, with the Spanish originals alongside) | `src/serra_r2_informe.py` (R2 Spanish) | `outputs/serra_r1/SERRA_R1_REPORT.md` (`SERRA_R1_INFORME.md`), `outputs/serra_r2/SERRA_R2_REPORT.md` (`SERRA_R2_INFORME.md`) |

## Data provenance

- **Source.** OEFA report *Informe N.º 00341-2018-OEFA/DEAM-STEC* (environmental assessment of the Santa Bárbara mining unit, Huancavelica, 2018). It was obtained through a public-records request under Peru's Law on Transparency and Access to Public Information (Ley N.º 27806), case file **2026-E01-114527**, and delivered by OEFA's Environmental Assessment Directorate (DEAM) at <https://drive.google.com/drive/folders/1Vq_lQpRip0Q1i-UiC1JqNYwl4iyh9V6Z>.
- **Portals.** OEFA open data: <https://datosabiertos.oefa.gob.pe>; national portal, dataset "OEFA – Evaluación ambiental": <https://www.datosabiertos.gob.pe/dataset/oefa-evaluacion-ambiental>.
- **Redistributed here.** The analytical data set `data/final/santa_barbara_soil.csv` (114 locations, 10 analytes; also as `.parquet`) with its data dictionary `data/final/DATA_DICTIONARY.md` (every column, with units and source), and `data/final/mine_workings.csv`.
- **Not in GitHub.** The scanned report itself (about 666 MB) is too large for the repository. It is archived in the Zenodo record together with this code (see *Citation*).
- **Coordinates.** UTM zone 18S (EPSG:32718) and WGS84 (EPSG:4326).

## Citation

If you use the code or the analytical data, please cite the archived version of this repository and the source report.

> Araujo Villegas, D. A. (2026). *censored-gp-santa-barbara: code and analytical data for "Bayesian geostatistics with censored data for mapping heavy-metal soil contamination around the Santa Bárbara mine (Huancavelica, Peru)"* (v2.0.0) [Software]. Zenodo. <https://doi.org/10.5281/zenodo.24347680>

- Version DOI (v2.0.0): [10.5281/zenodo.24347680](https://doi.org/10.5281/zenodo.24347680)
- Concept DOI (all versions): [10.5281/zenodo.24347679](https://doi.org/10.5281/zenodo.24347679)

The metadata for citation managers are in [`CITATION.cff`](CITATION.cff). Data source: Organismo de Evaluación y Fiscalización Ambiental (OEFA), *Informe N.º 00341-2018-OEFA/DEAM-STEC*, evaluación ambiental de la unidad minera Santa Bárbara, Huancavelica, 2018.

## License

The code is released under the MIT licence (see [`LICENSE`](LICENSE)). The environmental data come from OEFA, an agency of the Peruvian State, and are redistributed citing the source; the scanned OEFA report archived in Zenodo is an OEFA document and is not covered by the MIT licence of the code.
