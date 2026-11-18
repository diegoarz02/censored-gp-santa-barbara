<!-- English version of SERRA_R2_INFORME.md. Every number is copied unchanged; the Spanish file is kept alongside. -->

# SERRA_R2_REPORT.md — Mercury maps and results with the fit that does converge

The article states that the five metals meet the strict criterion (R̂ < 1.01, ESS > 400, 0 divergences) and that Hg meets it
**with the reinforced protocol** (`outputs/idata/probe_Hg_strict_2026-10-26_ta0999.nc`: 4 chains × 2000 draws,
warm-up 3000, `target_accept` 0.999; R̂ 1.0059, ESS bulk 467, ESS tail 693, 0 divergences). But the Hg maps and
everything that comes out of `posterior_maps.npz` or `idata_main['Hg']` used **`outputs/idata/main_Hg.nc`**, the standard fit
(R̂ 1.0104, ESS bulk 331). Origin verified in the code: `notebooks/_src/nb02_f_maps.py` L.249 calls
`sb.predict_field(idata_main[a], …)`; `nb02.py` L.938 defines `idata_main = {a: fit_main(a)}` with `main_{analyte}`; and in
`posterior_maps.npz` Hg gives P > 0.5 over 99.77 % of the domain, the "99.8 %" figure of the article.

This round **does not refit any model**: it is post-processing of the existing draws. `main.tex` and
`referencias.bib` were not touched, and there is no GitHub tag or release.

---

## CHANGES IN THE ARTICLE

Figures that the article prints and that change when rounded the way it prints them. Each one was recomputed from the files
(`outputs/serra_r2/verificacion_R2.csv`, 42 checks).

| # | Where (v15, Spanish version) | Before (printed) | With the reinforced fit | Monte Carlo noise | Source file |
|---|---|---|---|---|---|
| 1 | Table 9 (`tab:exc`), Hg row, column P > 0.8 | 96.7 | **96.8** (96.6942 → 96.8432) | within Monte Carlo noise (0.7× the range between seeds) | `hg_antes_despues.csv` |
| 2 | Table 9 (`tab:exc`), Hg row, column P > 0.95 | 76.1 | **77.4** (76.1143 → 77.4150) | real change (2.4× the range between seeds) | `hg_antes_despues.csv` |
| 3 | Model section (v15 L.239), probability of τ < 1 for mercury | 0.28 | **0.29** (0.2785 → 0.2854) | not applicable (uses the full draws, no subsampling) | `hg_hiperparametros_antes_despues.csv` |

**How to read the noise:** the figure printed by the official pipeline (seed 20260906) is the one in the table, and with it both
columns change. But the subsampling of 4000 draws moves those figures when the seed changes (section T2): for P > 0.8 the change from 96.7 to 96.8
is rounding noise that the seed also produces with the standard fit, whereas for P > 0.95 the reinforced fit raises the
figure systematically. Recommendation: print the figures of the official pipeline (96.8 and 77.4), which are reproducible.

**Do not change:** the P > 0.5 column of Hg (99.7697 → 99.8103, 99.8 is printed in both cases, table and text), the remaining
rows of Table 9 (Pb, As, Ba, Cd), the convergence Table 4 (it already carried the reinforced fit for Hg) and the prior
sensitivity table (they are separate fits that do not read `main_Hg.nc`).

### A sentence of the article that is false, with either of the two fits

The article says that *none of the four covariates of the mean is significant, with credible intervals that
contain zero in the five metals*. For **Hg** the log-distance to the waste dump **excludes zero with both fits**:

| Fit | mean | 95 % CI | excludes zero? |
|---|---|---|---|
| standard (`main_Hg.nc`) | 0.4541 | [0.0727, 0.8344] | yes |
| reinforced | 0.4616 | [0.0723, 0.8439] | yes |

The other 3 covariates of Hg (with both fits) and the 20 intervals of Pb, As, Ba and Cd (Cd with free r and with
L = LD; 5 models × 4 covariates) contain zero (intervals that exclude it: 0). Detail of the 28
intervals in `hg_covariables_ic95_cinco_metales.csv`. The correct sentence is "in four of the five metals no
covariate excludes zero; in mercury, the log-distance to the waste dump has a positive coefficient with a 95 % CI that excludes it";
the effect belongs to the distance-to-source covariate, not to elevation, slope or distance to the city.

### Extrapolation zone ("0.8 %" in the caption of the Hg figure)

**It is not pure geometry.** `nb02_f_maps.py` L.294 defines `ell_typ` as the average of the length scale of the five
`idata_main`, Hg included, and the extrapolated zone is the cells farther than 1.5·`ell_typ` from the nearest data point. The figures do not
recompute that: `mapas_f3.py` L.225 uses a fixed constant `ell_km = 1.03`.

| Variant | `ell_typ` (km) | threshold (km) | cells | % of the domain |
|---|---|---|---|---|
| A. original of nb02 (standard Hg, free-r Cd) | 1.0294 | 1.5441 | 57 | 0.772 |
| B. reinforced Hg, everything else equal (the only difference of R2) | 1.0371 | 1.5556 | 51 | 0.691 |
| C. reinforced Hg and L=LD Cd (not part of R2) | 1.0910 | 1.6365 | 13 | 0.176 |
| D. fixed constant of mapas_f3.py (the one the figures draw) | 1.0300 | 1.5450 | 57 | 0.772 |

**Resolution:** the fixed constant of the figures is kept, so that all the maps (Hg, Pb, As, Cd) share the
same mask and the "0.8 %" remains exactly what they draw (57 cells farther than 1.545 km from any data point). What
must be taken care of in the figure caption is to **describe the mask as a fixed distance of 1.5 × 1.03 km**, not as "1.5 length scales
averaged over the fits": recomputed with the reinforced Hg that average would give 0.7 % (51 cells), and with the Cd under
L = LD (which the original pipeline no longer uses) 0.2 %. It is a matter of wording, not of printed figure.

---

## T1 — Inventory of what depended on `main_Hg.nc`

Automatic scan of `src/` and `notebooks/_src/` (`inventario_T1_scan.csv`) and curated table (21 rows,
`inventario_T1.csv`), checked against `main_FINAL_v15.tex`. 18 entries depend on the standard fit; the article uses
7 of them. The English manuscript for SERRA was not reviewed in that round: whatever appeared only in it is listed as not verified.

| Dependency | Quantity | Does the article use it? | Action |
|---|---|---|---|
| `notebooks/_src/nb02_f_maps.py` | Hg_mean, Hg_median, Hg_sd_f, Hg_sd_y, Hg_p_agri, Hg_p_resid (Hg maps) | YES: Fig. mapaHg = FIG17_Hg (L.553); Fig9 SERRA (Fig. 8 of the PDF) | RECOMPUTED -> outputs/serra_r2/posterior_maps_Hg_refuerzo.npz |
| `notebooks/_src/nb02_f_maps.py` | % of the domain with P(Hg>6.6)>0.5 = 99.7697 | YES: tab:exc Hg row col. P>0.5 = 99.8 and text L.570 | RECOMPUTED -> hg_antes_despues.csv |
| `src/t2_1_area_excedida_posterior.py` | % of the domain with P>0.8 (96.69) and P>0.95 (76.11); exceeded area: median, 95 % CI | YES: tab:exc Hg row cols. P>0.8 = 96.7 and P>0.95 = 76.1 | RECOMPUTED -> hg_antes_despues.csv |
| `src/t1_1_prueba_varianza.py` | P(log tau < 0) of Hg = 0.2785 | YES: L.239 'mercury is the exception (0.28)' | RECOMPUTED -> hg_hiperparametros_antes_despues.csv |
| `notebooks/_src/nb02.py` | 95 % CI of the 4 covariates of the mean of Hg (columns beta_*_sig) | YES: L.504 'None of the four covariates ... their credible intervals contain zero in the five metals' | RECOMPUTED. FINDING: the sentence is FALSE for Hg with BOTH fits (log_dist_dump excludes zero) |
| `notebooks/_src/nb02_f_maps.py` | extrapolation zone (cells at > 1.5 ell_typ from the nearest data point): 0.772 % = '0.8 %' | YES: caption of Fig. 8 SERRA ('0.8 % of the domain'); legend 'extrapolation (x %)' of all the maps | RECOMPUTED -> hg_extrapolacion.csv (0.772 % -> 0.691 % if ell_typ is recomputed) |
| `src/mapas_f3.py / src/figserra.py::fig9 / src/v7rev_fig_axes.py` | figures FIG17_Hg and Fig9 SERRA (they read posterior_maps.npz) | YES (they are the Hg map figures) | REGENERATED (T3) |

Excluded because they are separate fits and do not read `main_Hg.nc` (checked by scan): cross-validation, masking,
prior sensitivity, LGCP, mixture, LMC, SBC and recovery. `src/serra_r1_campo_latente.py` already used the reinforced fit.
They depend on it but the Spanish article does not use them (they were recomputed anyway): tables 12, 13, 14 and 19, the monitoring design
(`TABLE21/22`, `FIG14`), the annotation of the figure `FIG9_Hg_crps`. They depend on it and were not redone (not cited):
`TABLE29_decision_thresholds.csv`, `TABLE_S5_loo.csv`, `TABLE_S10_pareto_k.csv`, `TABLE_T2_3_...`, ADVI.

`figuras/TABLE10_convergence.csv` keeps the row of the standard fit of Hg (1.0104 / 331) and that of Cd with free r: **it is not the
Table 4 of the article**. Table 4 was verified separately, from the `idata`:

| Metal | `idata` | R̂ | ESS bulk | ESS tail | Div. | = article |
|---|---|---|---|---|---|---|
| Hg | `probe_Hg_strict_2026-10-26_ta0999.nc` | 1.0059 | 467 | 693 | 0 | yes |
| Pb | `main_Pb.nc` | 1.0068 | 752 | 1272 | 0 | yes |
| As | `main_As.nc` | 1.0064 | 913 | 1205 | 0 | yes |
| Ba | `main_Ba.nc` | 1.0035 | 1026 | 1323 | 0 | yes |
| Cd | `flip_LD_Cd.nc` | 1.0051 | 767 | 845 | 0 | yes |

---

## T2 — Recomputed with the reinforced fit

Script: `src/serra_r2_hg_refuerzo.py`. **Same pipeline as `nb02_f_maps.py`**: 50 m grid with a 300 m buffer (7381
cells), `N_MAP_SAMPLES = 4000`, seed 20260906, blocks of 1500 cells, `predict_field` with `anisotropic=True`, ECA 6.6 mg/kg
(the residential standard of Hg is the same 6.6, so `p_resid` = `p_agri`). The only difference is the `idata`.

**Validation of the pipeline before using it:** with the standard fit it reproduces `posterior_maps.npz` in block 0 (cells
0–1499) with a **difference of 0.0** in the six keys (`validacion_pipeline_bloque0.csv`). `src/t2_1_area_excedida_posterior.py` had already
reproduced the whole map with a difference of 0.0000 per cell.

Technical note: the reinforced fit has 8000 draws (4 × 2000) and the standard one 4000; the rule of the pipeline takes 4000 by
subsampling without replacement (4000 out of 8000 with the reinforced fit, all 4000 with the standard one).

Products: `posterior_maps_Hg_refuerzo.npz` (`Hg_mean`, `Hg_median`, `Hg_sd_f`, `Hg_sd_y`, `Hg_p_agri`, `Hg_p_resid`, plus the exceeded
area per draw). **`posterior_maps.npz` was not overwritten.**

### Before / after (`hg_antes_despues.csv`)

| Quantity | Before (`main_Hg.nc`) | After (reinforced) | Difference |
|---|---|---|---|
| % of the domain with P(Hg > 6.6) > 0.5 | 99.77 | 99.81 | 0.04 |
| % of the domain with P(Hg > 6.6) > 0.8 | 96.69 | 96.84 | 0.15 |
| % of the domain with P(Hg > 6.6) > 0.95 | 76.11 | 77.41 | 1.30 |
| exceeded area (% of the domain), median | 96.40 | 96.55 | 0.14 |
| exceeded area (% of the domain), 2.5 % quantile | 88.85 | 89.45 | 0.60 |
| exceeded area (% of the domain), 97.5 % quantile | 98.89 | 98.93 | 0.04 |
| exceeded area (% of the domain), mean | 95.79 | 95.93 | 0.14 |
| posterior median of the concentration (mg/kg): minimum on the grid | 2.6402 | 2.6756 | 0.0354 |
| posterior median of the concentration (mg/kg): maximum on the grid | 1576.78 | 1489.01 | -87.77 |
| posterior mean of the concentration (mg/kg): minimum on the grid | 5.3778 | 5.1106 | -0.2672 |
| posterior mean of the concentration (mg/kg): maximum on the grid | 7999.10 | 7114.99 | -884.10 |
| predictive sd (log): mean over the grid | 1.4219 | 1.4220 | 0.0000 |
| predictive sd (log): maximum | 2.1104 | 2.1214 | 0.0109 |
| maximum P(exceed) on the grid | 1.0000 | 1.0000 | 0.0000 |
| cells with P > 0.5 | 7364.00 | 7367.00 | 3.00 |
| cells with P > 0.9 | 6634.00 | 6662.00 | 28.00 |

Cross-check with the previous round (`cantidades_mapa_diag_area.csv`, with the 8000 draws in chain order and no subsampling):
median of the exceeded area of Hg 96.53 % versus 96.55 % here (4000 subsampled draws): compatible.

### Cell-by-cell classification agreement (`hg_acuerdo_clasificacion.csv`)

| P threshold | Agreement | Cells that change | exceeding → not | not → exceeding |
|---|---|---|---|---|
| 0.5 | 99.96 % | 3 | 0 | 3 |
| 0.8 | 99.82 % | 13 | 1 | 12 |
| 0.95 | 97.48 % | 186 | 45 | 141 |

Correlation of the log-median between maps 0.9986; median relative difference of the median concentration
3.58 %; maximum difference of P 0.040.

### Is the change real or Monte Carlo noise? (`hg_ruido_montecarlo_semillas.csv`)

Same pipeline with 4 seeds per fit (20260906 plus three consecutive ones):

| Figure | Before: mean [min, max] | After: mean [min, max] | Δ of means | Δ / range between seeds |
|---|---|---|---|---|
| % domain P > 0.5 | 99.77 [99.77, 99.77] | 99.79 [99.78, 99.81] | 0.02 | 0.7 |
| % domain P > 0.8 | 96.75 [96.69, 96.80] | 96.82 [96.76, 96.86] | 0.07 | 0.7 |
| % domain P > 0.95 | 76.25 [76.11, 76.39] | 77.63 [77.41, 77.98] | 1.38 | 2.4 |
| Median of the exceeded area (%) | 96.43 [96.40, 96.45] | 96.55 [96.50, 96.57] | 0.12 | 1.7 |

Reading: for P > 0.95 the difference between fits is 2.4 times the range between seeds; for P > 0.8, 0.7 times. A ratio
greater than 1 indicates that the change exceeds what the seed moves; smaller than 1, that it cannot be told apart from the noise of the
4000-draw subsampling. **The printed figure that comes out of the official pipeline (seed 20260906) is the one in the table above;**
the noise table only serves to judge how much of the difference is signal.

### Hg hyperparameters, before and after (`hg_hiperparametros_antes_despues.csv`)

| Parameter | Before | After |
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
| β elev_oefa (mean; 95 % CI) | -0.153 [-0.874, 0.520] | -0.142 [-0.856, 0.573] |
| β slope_deg (mean; 95 % CI) | 0.117 [-0.158, 0.395] | 0.115 [-0.158, 0.378] |
| β log_dist_dump (mean; 95 % CI) | 0.454 [0.073, 0.834] | 0.462 [0.072, 0.844] |
| β log_dist_city (mean; 95 % CI) | -0.325 [-1.034, 0.432] | -0.315 [-1.026, 0.426] |
| P(log τ < 0) | 0.2785 | 0.2854 |

With the self-validation of "before": the Hg row of `TABLE11` is reproduced to 4.7×10⁻⁵ (rounding to 4 decimals), P(log τ < 0) = 0.2785
equal to `TABLE_T1_1`, the posterior predictive check equal to `TABLE14` to 4 decimals, and the recomputed monitoring design
coincides with `outputs/monitoring_design.npz` (same points, identical trace). Tables 12, 13, 14 and 19 and the monitoring design
with both fits are in the CSV; the design with the reinforced fit reduces the mean sd by 10.2 / 15.0 / 21.5 % with 5 / 10 / 20 new
points (before 10.1 / 14.9 / 21.4 %); 5 of the first 5 points coincide, 9 of 10 and 15 of 20.

---

## T3 — Figures

`src/serra_r2_figuras.py` regenerates `figuras/serra/Fig9.{pdf,eps,tif}` (Hg map, "Fig. 8" in the PDF of the article) and
`figuras/FIG17_Hg_mean_and_uncertainty.pdf` (Spanish version) with the new products: same drawing path, same style,
same scales and legends; only the `Hg_*` keys change. The figures are overwritten (no copies of earlier versions).

Verification with `verify_figserra.py` (`T3_verificacion_log.txt`): Fig9 in embedded Arial, baseline text 8.0–10.0 pt,
174 × 104.4 mm, raster at 600 dpi, no internal titles. **By eye the map is the same.** The only visible change: the upper end
of the colour bar goes from ~1577 to ~1489 mg/kg (maximum of the median; the labelled decades are the same) and,
in the Spanish version, a minor layout shift due to the width of the labels. Pixels that differ by more than 8
levels: Fig9 0.44 %, FIG17_Hg 3.59 %. The extrapolation legend still says 0.8 %.

---

## T4 — Consistency

- **Printed figures that change:** the three in the block "CHANGES IN THE ARTICLE". **Everything else printed stays the same.**
- **Independent verification** (`serra_r2_verificar.py`, `verificacion_R2.csv`): 42/42 checks OK,
  which include the complete Table 9 of the five metals recomputed from their files and compared with what is printed.
- Table 4 was checked against the `idata` (above): the five rows coincide.
- The Hg rows of the traceability file (`TRAZABILIDAD_RESULTADOS.csv`) now point to the reinforced fit.

---

## T5 — Preparation of the Zenodo archive

This round was committed (with the same sweep for traces as in the previous round) and pushed to GitHub **without a tag or a release**;
the repository was thereby ready to be archived. The final Zenodo record is prepared in the next round (version
**v2.0.0**: the repository and the scanned OEFA report, manual deposit, no automatic integration).

---

## Index of files (`outputs/serra_r2/`)

| Task | Script | Outputs | Log |
|---|---|---|---|
| T1 | `src/serra_r2_inventario.py` | `inventario_T1.csv`, `inventario_T1_scan.csv` | — |
| T2 | `src/serra_r2_hg_refuerzo.py` | `posterior_maps_Hg_refuerzo.npz`, `hg_antes_despues.csv`, `hg_acuerdo_clasificacion.csv`, `hg_diferencias_mapas.csv`, `hg_hiperparametros_antes_despues.csv`, `hg_covariables_ic95_cinco_metales.csv`, `hg_extrapolacion.csv`, `hg_diseno_monitoreo_antes_despues.csv`, `hg_ruido_montecarlo_semillas.csv`, `validacion_pipeline_bloque0.csv`, `tabla4_convergencia_articulo.csv` | `T2_hiper_log.txt`, `T2_mapas_log.txt`, `T2_tablas_log.txt`, `T2_ruido_log.txt`, `T4_tabla4_log.txt` |
| T3 | `src/serra_r2_figuras.py` | `figuras/serra/Fig9.*`, `figuras/FIG17_Hg_mean_and_uncertainty.pdf`, `hg_figuras_comparacion.csv` | `T3_log.txt`, `T3_verificacion_log.txt` |
| T4 | `src/serra_r2_verificar.py`, `src/serra_r2_informe.py` | `verificacion_R2.csv`, this report | `T4_verificacion_log.txt` |
