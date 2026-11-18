<!-- English version of SERRA_R1_INFORME.md. Every number is copied unchanged; the Spanish file is kept alongside. -->

# SERRA_R1_REPORT.md — reproducible evidence before the submission to SERRA

Reproducible evidence on five points (Table 6 and Wilcoxon, latent field and map quantities,
GP-L/2 baseline, masking threshold, SBC and derived quantities), prepared before the
submission to SERRA. **No main model was refitted and no
number of the article changed.** Everything that follows is post-processing or pure recomputation on
existing checkpoints/idata, except where the opposite is explicitly stated. `main.tex` and
`referencias.bib` were not touched.

---

## T1 — Table 6, Fig. 6 and Wilcoxon

Script: `src/serra_r1_tabla6.py`. Source: `figuras/TABLE_H6_MAIN_con_kriging.csv`.

1. **360 rows confirmed** (3 models × 3 metals × 4 levels × 10 replicates), all 360 with
   `deep=True`.
2. **The 36 means of `picp95` recomputed coincide with the manuscript**: maximum
   absolute difference = **0.000497** (within the tolerance 0.0005; 0 of 36 rows outside tolerance).
   `outputs/serra_r1/T6_recalculo.csv`, `T6_diff.csv`.
3. **Ranges at 80 % of the censored model**: Hg [0.5714, 1.0000] (manuscript 0.57–1.00 ✓), Pb [0.3636,
   0.9615] (manuscript 0.36–0.96 ✓).
4. **`src/figserra.py::fig6()` reads exactly `TABLE_H6_MAIN_con_kriging.csv`** and uses
   `groupby("level")["picp95"].mean()` — confirmed by reading the source code of the figure, not
   assumed.
5. **Wilcoxon signed-rank test** (exact test, full enumeration of 2ⁿ, no scipy; 15
   metal×block units, averaging the 2 seeds of each block — `outputs/serra_r1/T6_wilcoxon.csv`):

   | level | baseline | censored wins | two-sided p |
   |---|---|---|---|
   | 80 % | GP-L/2 | 15/15 | **6.10×10⁻⁵** (manuscript: 15/15, 6.1×10⁻⁵ ✓) |
   | 80 % | kriging | 15/15 | **6.10×10⁻⁵** (✓) |
   | 60 % | GP-L/2 | 7/9 | **0.0195** (✓) |
   | 60 % | kriging | 8/13 | **0.0325** (✓) |
   | 40 % | GP-L/2 / kriging | 4/7 | 0.203 (not significant, ✓) |
   | 20 % | GP-L/2 / kriging | 1/4, 4/7 | **0.375** / 0.578 (not significant, ✓) |

   **The 4 p-values printed in the manuscript (80 %: 6.1×10⁻⁵ against both; 60 %: 0.0195 and 0.0325)
   are reproduced exactly.** Those of 20 % and 40 % are not printed and are not significant.

   *Correction with respect to the first version of this table*: the p at 20 % against GP-L/2 came out as 1.00
   because the convention of `v14_wilcoxon_bloques.py` had been reused (twice the upper
   tail), which coincides with the exact two-sided value when the censored model wins but overestimates when it
   loses (here it wins 1 of 4). The exact two-sided value is **0.375**, equal to the independent recomputation.
   `T6_wilcoxon.csv` keeps both columns (`p_bilateral` and `p_v14_doble_cola_superior`); the other
   7 rows do not change.

   Quotation of the code that confirms that replicates r and r+5 share the mask and the block, and differ
   only in the seed (`notebooks/_src/worker_v4.py`):
   - L.102: `"block": r % n_blocks` → r and r+5 fall in the same block (`r%5 == (r+5)%5`).
   - L.128: `seed = SEED0 + 1000*rep + 97*int(level*100) + 13*(hash(metal)%1000)` → depends on
     `rep`, not on `block`: r and r+5 differ **only** in the MCMC seed.
   - L.135–136: `limit = np.quantile(value, level); censored = value < limit` → depends only on
     (metal, level), the same for the 10 replicates — they share the mask by construction.

**T1 verdict: no differences.**

---

## T2 — Diagnostics of the latent field and of the map quantities

Script: `src/serra_r1_campo_latente.py`. **No new MCMC** — deterministic post-processing of the
draws already existing in the 5 main `InferenceData` (As, Ba, Pb: `main_*.nc`; Hg: reinforced
protocol `probe_Hg_strict_2026-10-26_ta0999.nc`; Cd: L=LOD, `flip_LD_Cd.nc`).

### T2.1 — Dimensions of the latent field (`f`) and of η(s) (`mu`), 114 locations, by metal

| metal | variable | R̂ max | % R̂≥1.01 | ESS bulk min | % ESS bulk<400 | ESS tail min | % ESS tail<400 |
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

**The 114×2×5=1140 dimensions of the field/η(s): 0 % with R̂≥1.01, 0 % with ESS<400, in the 5
metals.** Complete detail (min/q05/q50/q95/max): `outputs/serra_r1/campo_latente_diag.csv`.

### T2.2.a — Exceeded area per draw, preserving the chain (Cd, Hg, Pb, As — Ba excluded by design of the task)

Reconstruction of the exceeded agricultural area per draw, **preserving the (chain, draw) structure**
(not the random shuffling that the map pipeline uses for the published summaries — necessary
here for R̂ to make sense). Official grid, 50 m, 7381 cells.

| metal | R̂ | ESS bulk | ESS tail | median % | 95 % CI | MCSE median (pp) | MCSE 2.5 % quantile (pp) | MCSE 97.5 % quantile (pp) |
|---|---|---|---|---|---|---|---|---|
| Cd | 1.0000 | 4291 | 2782 | 40.11 | [21.93, 56.50] | 0.183 | 0.386 | 0.352 |
| Hg | 0.9999 | 3372 | 4418 | 96.53 | [89.20, 98.92] | 0.047 | 0.217 | 0.020 |
| Pb | 1.0046 | 1953 | 2434 | 84.62 | [69.20, 94.54] | 0.163 | 0.461 | 0.176 |
| As | 1.0039 | 1361 | 2095 | 71.89 | [52.70, 88.46] | 0.291 | 0.488 | 0.312 |

**All 4: R̂<1.01, ESS bulk/tail>1300, MCSE of the median <0.3 percentage points (of the
2.5/97.5 quantiles, <0.5 pp).** The median of
Cd (40.11 %) coincides with the one already reported in `outputs/flip_ld_cd/maps_50m.npz` (40.09 %) — the
small difference (0.02 pp) is the one expected between using ALL the draws in natural order (here,
necessary to preserve the chain) and the random subsampling without replacement used by the
map pipeline (both read the same underlying prediction). Detail:
`outputs/serra_r1/cantidades_mapa_diag_area.csv`.

### T2.2.b — MCSE of P(c>ECA) per cell, Cd

**3.75 % of the 7381 cells** have a potentially unstable classification
(|P(exc)−0.5| < 2·MCSE), that is, they could cross the decision threshold 0.5 through Monte
Carlo noise rather than real signal (277 cells). **Distribution of the MCSE per cell** (in probability,
not in percentage): min 0.0017; quantiles 5/25/50/75/95 % = 0.0035 / 0.0060 / 0.0073 / 0.0078 /
0.0089; max 0.0117. To size it: 3451 cells have P(exc) between 0.3 and 0.7, and 277 of them
(≈8 %) are the unstable ones. Complete detail (the 7381 cells, with `grid_e`/`grid_n`/
`p_exceedance`/`mcse`/`unstable_0.5`): `outputs/serra_r1/cantidades_mapa_diag_cd_cellwise.csv`.

### T2.2.c — 6 representative locations (2 background, 2 near workings, 2 at the extrapolation edge), the 5 metals

Quantity diagnosed: the posterior mean of log c (η(s), `mu`) per draw, in cells of the
official 50 m grid. Selection rules fixed beforehand, not tuned to any result;
"with support" = at ≤ 100 m from the nearest data point:
- **near workings**: the 2 cells with support closest to a real mine working
  (`data/final/mine_workings.csv`, 21 adits and waste dumps): at 5.1 m and 7.8 m from a working;
- **background**: the 2 cells with support farthest from any working (1190 m and 1187 m);
- **extrapolation edge**: the 2 cells farthest from any data point (1712 m and 1706 m).

To predict only these 6 cells (instead of the 7381, which took hours for Hg) it was verified that the
conditional mean of a cell does not depend on which other cells are predicted at the same time: Cd, 6 cells
vs 6 + 100 at random, same seed, **max |ΔMU| = 1.4×10⁻⁹**
(`outputs/serra_r1/T2_reanudacion_log.txt`).

**The 30 combinations (6 locations × 5 metals): maximum R̂ 1.0035 (Pb, extrapolation
edge), minimum ESS bulk 1484 (As, extrapolation edge 1), minimum ESS tail 1583 (As,
extrapolation edge 2).** None below the usual thresholds (R̂<1.01, ESS>400) —
including the extrapolation zone, which is the most exposed to low information. Detail:
`outputs/serra_r1/cantidades_mapa_diag_ubicaciones.csv`.

*Correction with respect to the first version of this section (before its publication)*: there "near
workings" had been defined with a proxy (cells close to the median of the distance to the nearest
data point), which is not distance to workings, and "background" were the cells closest to a data point. It was
redone with the rules above. Previous values → new: R̂ max 1.0035 → 1.0035 (the same
edge cell), ESS bulk min 1316 → 1484, ESS tail min 1582 → 1583. The conclusion does not change.
No number of the article cites this table.

### Supplementary figure

`figuras/serra/FigS1_latent_diagnostics.pdf` (+ .eps/.tif, 174 × 80 mm, Arial 8–10 pt,
embedded, no internal title): **histograms** of R̂ (panel a) and of ESS bulk (panel b) of the 114
dimensions of the latent field `f`, one histogram per metal (each metal with its own colour and
line style), with the reference line at R̂ = 1.01 and at ESS = 400. The raw values per
dimension (5 metals × 114 × `f`/`mu`) are in `outputs/serra_r1/campo_latente_por_dimension.csv`;
their summary coincides with `campo_latente_diag.csv` to 10⁻¹³. *(The first version showed bars
of R̂ max and ESS min per metal, not histograms as the task required; replaced.)*

### T2 files and execution log

- `campo_latente_diag.csv` (summary by metal/variable) and `campo_latente_por_dimension.csv` (raw).
- `cantidades_mapa_diag.csv`: **a single file in long format** (`quantity, metal,
  location_label, statistic, value`) with the exceeded area, the MCSE summary of Cd and the
  locations. It is the concatenation, without recomputation, of `cantidades_mapa_diag_area.csv`,
  `_cd_summary.csv` and `_ubicaciones.csv`; the per-cell detail stays in `_cd_cellwise.csv`.
- **Logs.** `T2_log.txt` is the output of the original run and **is interleaved**: through a
  mistake of mine two processes of the same script were launched at once and their `stdout` got mixed (hence
  the cut lines). The CSV files were not corrupted — the number of rows of each one was verified
  (10, 4, 7381, 1 and 30 in the first version) and the run was made resumable per file.
  `T2_reanudacion_log.txt` is the clean output of the final resumption (T2.2c redone,
  per-dimension coherence, invariance) and `T2_figS1_log.txt` that of FigS1.

**T2 verdict: clean convergence in the 5 fits, in the latent field, in η(s), and in the
derived quantities that feed the maps (exceeded area, probability per cell, representative
locations). The only point to be declared honestly is the 3.75 % of Cd cells with
classification unstable because of the MCSE — it is not a convergence failure, it is the real Monte
Carlo uncertainty of a probability close to 0.5 with ~4000 draws, and it should be mentioned as a
nuance of the exceedance figure, not hidden.**

---

## T3 — Stability of the GP baseline with L/2

Script: `src/serra_r1_baseline.py`.

**a. Diagnostics by model × level** (`outputs/serra_r1/baseline_diag.csv`): confirmed exactly
— GP-L/2 total divergences = **52** (expected 52), R̂ max = **1.192** (expected 1.19);
censored R̂ max = **1.024** (expected 1.024). By level (30 metal×replicate cells per model and
level):

| level | GP-L/2: divergences (sum / max per cell) | GP-L/2: R̂ max (median / max) | GP-L/2: ESS bulk min (median / min) | censored: divergences | censored: R̂ max |
|---|---|---|---|---|---|
| 20 % | 0 / 0 | 1.007 / 1.021 | 645 / 346 | 0 | 1.017 |
| 40 % | 8 / 4 | 1.007 / 1.031 | 613 / 171 | 0 | 1.024 |
| 60 % | 18 / 6 | 1.008 / 1.016 | 531 / 242 | 0 | 1.018 |
| 80 % | 26 / 5 | 1.052 / 1.192 | 86 / 15 | 0 | 1.016 |

(0+8+18+26 = **52**.) Kriging does not use MCMC: no R̂/ESS, 0 divergences by construction.

**b. Coverage restricted to healthy fits** (`outputs/serra_r1/baseline_cobertura_restringida.csv`,
all the metal × level × model × criterion combinations): only the replicates in which
**both** GP models pass the criterion are kept; kriging is evaluated on that same set of replicates.
`n` = surviving replicates (out of 10). Mean PICP95:

| criterion | metal | level | n | censored | GP-L/2 | kriging |
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

(Levels 20 % and 40 % are in the CSV.) **As at 80 % with R̂<1.05 (n = 3): 0.949 against
0.681, coincides exactly with what had already been seen from the CSV.** In the 12 rows the censored model stays
above GP-L/2 except As 60 % with R̂<1.01 (tie at 1.000). With R̂<1.01 the 80 % level is left with **a
single** fit per metal: the direction is kept but that row **does not support an estimate**,
it only shows that the gap is not created by badly converged fits. Kriging, without MCMC, collapses
equally at 80 % (0.330 / 0.249 / 0.061 here; 0.530 / 0.497 / 0.357 with the 10 replicates), so the
loss of coverage does not depend on the sampler.

**c. Comparability of the implementation**: `build_censored_gp()` (`sbmodel.py`) is the **same
function** for both models. Only difference (L.122–127):
```python
if substitute_half_lod:
    y_value = np.where(censored, lod / 2.0, y_value)
    censored = np.zeros(n, bool)
```
Everything else is shared code: same priors (`beta0~N(ȳ,sd_beta0)`, `beta~N(0,sd_beta)`,
`b_bg~N(0,sd_beta)`, `ell~InvGamma`), same partitioned-variance parametrisation
(`sigma_tot~HalfNormal`, `rho~Beta(2,2)`, `eta=sigma_tot·√rho`, `sigma_n=sigma_tot·√(1-rho)`),
same Matérn 5/2 kernel, same `gp.Latent`, same `fit_model()` (draws/tune/target_accept/chains/
seed identical, nutpie+numba). **There is no separate construction or fitting call
for the baseline — they are identical except for the likelihood**, confirmed by reading the code, not
assumed. Complete detail (full source code quoted): `outputs/serra_r1/baseline_diff_implementacion.md`.

**d. Mechanism (investigation, without refitting)**: of the 120 GP-L/2 cells, **34 (28 %) have
R̂>1.05 or divergences**. In those 34 problematic cells, the percentage of the training set
tied exactly at L/2 ranges from **29.6 % to 93.2 %** (mean 71.3 %) —
`outputs/serra_r1/baseline_mecanismo_ties.csv`. This is **consistent with** the hypothesis put forward
(massive ties at L/2 push σₙ→0 → difficult geometry), without proving it causally (σₙ/ρ/ℓ per draw was not
inspected directly in those cells because **none of the 360 JSON files of
`outputs/h6_deep/` stores the complete idata per cell**: they carry only scalars — `rhat_max`,
`ess_bulk_min`, `ess_tail_min`, `divergences` — with no per-parameter information; verifying the
hypothesis at the level of the individual parameter would require refitting, declared as such, not simulated).

**The optional, bounded refit (at most 6 replicates) was NOT run, and the reason matters**: point c
above already established that GP-L/2 and the censored model use the **same parametrisation** —
the condition put forward for the refit ("if the parametrisation of the baseline differs from the
censored one") **is not met**. Refitting GP-L/2 "with the same parametrisation as the censored model"
has no corresponding action: it already has it. The only way to "fix" it would be to stop
substituting by L/2, which would simply turn it into the censored model and defeat the
purpose of the baseline. It is declared so instead of forcing a meaningless refit.

---

## T4 — Masking threshold (possible leakage)

Script: `src/serra_r1_umbral.py`.

1. **Confirmed in the code** (`worker_v4.py::run_deep()`, L.135–138): `limit =
   np.quantile(value, level)` is computed over the 114 complete concentrations, **before** the
   train/test split (`labels == block` is applied afterwards) — it is applied equally to both by
   construction, there are not two different computations.
2. **60 metal×level×block combinations** (3×4×5, confirmed). Threshold computed with the training set only
   vs. the global one:
   - relative threshold difference: mean 10.1 % (|mean| 29.2 %, range −75 % to +244 % — the
     extremes are cells with few training points at low levels, where the quantile is
     noisy);
   - **test points that would change censored/detected status: 86 of 1368 (6.3 % of the
     total), mean 4.6 % per cell, maximum 34.6 % in one cell** (detail by level in
     `outputs/serra_r1/umbral_fuga.csv`: the effect is smaller at higher censoring, 5.8 % mean at
     80 % against 3.0 % at 20 %).
3. Output: `outputs/serra_r1/umbral_fuga.csv`.

**T4 verdict**: there is a real information leak (the threshold sees the test set), but of **modest**
magnitude: on average 1 of every 22 test points would change status under an "honest" threshold
(training set only). No model was refitted to correct it — sizing it was
the scope of this task.

---

## T5 — SBC and derived quantities

Script: `src/serra_r1_sbc.py`.

**NOT AVAILABLE.** Verified by reading the real keys of the checkpoints (not assumed from the
name of the script):
- `outputs/c1_sbc/rep_*.json` (200 replicates): stores `ranks` as **scalar ranks** per
  parameter (`eta, ell_0, ell_1, sigma_n, beta0`), not draws of the field nor of η(s) per location.
- `outputs/c2_recovery/rep_*.json` (50 replicates): only metadata (censoring, LOD, times) — no
  array of draws.
- **0 `.nc` files** in either of the two folders.

**Estimated cost of a run that does store them (not executed)**: it would be necessary to modify
`run_replica()`/`simulate_one()` (`c1_sbc.py`) and the equivalent function of
`c2_recovery_reps.py` to persist `f`/`mu` per replicate, and to **repeat the 250 replicates**
(the true field and the posterior are generated together, saving cannot be added without
refitting). Measured in this project: ~74 s/replicate → 250×74s ≈ **5.1 hours**. Space: each
complete idata of this model weighs 10–24 MB in `outputs/idata/`; storing 250 (even if only
f/mu) would cost several GB. Output: `outputs/serra_r1/sbc_derivadas.csv`,
`sbc_derivadas_nota.md`.

---

## T6 — Fig. 2 (graphical correction)

`figuras/serra/Fig2.{pdf,eps,tif}` regenerated with two changes:
1. X axis of panel (b): "distance to mining feature (m)" → **"distance to nearest mine working
   (m)"**.
2. The labels "detected"/"censored" **collided** ("detecteddcensored") even after the
   first attempt with more space + line break — 6 boxes with a 2-line label do not fit
   legibly at 174 mm. **Real fix**: the status (detected/censored) is already encoded by
   colour from panel (a) — the repeated text per box was removed, the x axis keeps **one
   label per metal** (Cd/Sb/Ag, centred between its pair of boxes), and a single legend below
   (blue=Detected, grey=Censored) under the panel, equal to the pattern of panel (a). Verified
   visually after the failed first attempt — it was not accepted without looking at the result.

**Verification** (`src/verify_figserra.py`, which now includes FigS1; complete output in
`outputs/serra_r1/T6_verificacion_log.txt`, regeneration in `T6_log.txt`): Arial fonts
embedded (real bytes extracted from the PDF), baseline text 8.0–10.0 pt (within 8–12),
0 internal titles, 174 × 90.5 mm, raster at 600 dpi. **No number changed** (the boxplot data
are the same; only the visual layout and the axis label changed). The sizes below 8 pt reported by `_verify_fonts.csv` in the column `fs_min_all` in
other figures are the mathtext sub/superscripts (≈70 % of the symbol size by design), not
body text — see the function `font_report()` of `src/verify_figserra.py`, which separates both.

---

## T7 — Repository and release v1.5

1. **Published content.** The allowlist of the `.gitignore` publishes only `src/`, `figuras/`,
   `notebooks/`, `data/final/` and the `.csv`/`.md`/`.txt` of `outputs/serra_r1/`; no
   checkpoint (`.nc`, `.npz`) nor the rest of `outputs/`. Included: the scripts and outputs of T1–T6;
   the FIGSERRA pipeline (`src/estilo_serra.py`, `src/figserra.py`, `src/verify_figserra.py`);
   `src/v14_wilcoxon_bloques.py`; `src/v7rev_fig_axes.py` together with the fix of
   `utm_km_ticks()` (`src/estilo_figuras.py`, `src/mapas_f3.py`, `src/a0_eda_consolidado.py`);
   the SERRA figures (`figuras/serra/`, Fig1–Fig11 and FigS1 in PDF/EPS/TIFF) and the five
   regenerated figures (FIG17 Cd/Hg, FIG18 Cd agricultural/residential, FIG27). The input checkpoints
   read by the scripts (`outputs/h6_deep/`, `outputs/idata/`, `outputs/c1_sbc/`,
   `outputs/c2_recovery/`, `outputs/flip_ld_cd/`) **are not published** because of their size.
2. **`.zenodo.json`** at the root: creator Araujo Villegas, Diego Alonso (ORCID
   0000-0002-6987-3948, Universidad Nacional Mayor de San Marcos), `upload_type` `software`,
   MIT licence, version `v1.5` (that of the v1.5 release; in `master` it is now `v2.0.0`).
3. **README**: English section (*About / How to reproduce / Data provenance / Citation*) added
   without deleting the Spanish one. Provenance: Report N.° 00341-2018-OEFA/DEAM-STEC obtained through a request
   for access to public information (Ley N.° 27806, file 2026-E01-114527); the analytical CSV
   `santa_barbara_soil.csv` is redistributed, not the scanned PDF of the report (because of its size). The
   `v1.5` release keeps the concept-DOI badge (10.5281/zenodo.22968165); in `master`
   the main README is now in English (the Spanish version in `README_es.md`) and the DOI citation
   of the final record (version and concept) is in the last section and in the README.
4. **Review before publishing**: the complete tree was swept (text with `git grep`; PDF/TIFF
   by extracting their text and metadata with PyMuPDF/PIL, because the binary search flags as
   matches sequences of compressed bytes that are not text) and the commit messages.
5. **Release.** GitHub tag and release `v1.5`. The version DOI, the version, the creator and the
   list of files returned by the Zenodo API are recorded as returned in the section
   *Post-release verification* at the end of this report, which is added once archived (which is why
   the release archive cannot contain it).

---

## Index of evidence (task → script → outputs → log)

| task | script | outputs in `outputs/serra_r1/` | execution log |
|---|---|---|---|
| T1 | `src/serra_r1_tabla6.py` | `T6_recalculo.csv`, `T6_diff.csv`, `T6_wilcoxon.csv` | `T1_log.txt` |
| T2 | `src/serra_r1_campo_latente.py` | `campo_latente_diag.csv`, `campo_latente_por_dimension.csv`, `cantidades_mapa_diag.csv` (+ `_area`, `_cd_cellwise`, `_cd_summary`, `_ubicaciones`); `figuras/serra/FigS1_latent_diagnostics.*` | `T2_log.txt` (interleaved, see T2), `T2_reanudacion_log.txt`, `T2_figS1_log.txt` |
| T3 | `src/serra_r1_baseline.py` | `baseline_diag.csv`, `baseline_cobertura_restringida.csv`, `baseline_diff_implementacion.md`, `baseline_mecanismo_ties.csv` | `T3_log.txt` |
| T4 | `src/serra_r1_umbral.py` | `umbral_fuga.csv` | `T4_log.txt` |
| T5 | `src/serra_r1_sbc.py` | `sbc_derivadas.csv`, `sbc_derivadas_nota.md` | `T5_log.txt` |
| T6 | `src/figserra.py::fig2()`, `src/verify_figserra.py` | `figuras/serra/Fig2.*` | `T6_log.txt`, `T6_verificacion_log.txt` |
| T7 | `.zenodo.json`, `README.md` | — | section *Post-release verification* |

The scripts are deterministic on the input checkpoints: T4 and T5 were run again
to generate their log and reproduced identical CSV files (`git` detected no change).

---

## Summary of discrepancies with the article

**None.** The 36 means of Table 6, the 4 Wilcoxon p-values that the article prints, the 3 totals of
divergences/R̂ of the baseline, and the number of masking cells (60, 360) coincide
exactly with what had already been cited or with what had already been pointed out from the CSV. The new findings of
this round (3.75 % of Cd cells with classification unstable because of the MCSE; 6.3 % leakage of the masking
threshold; GP-L/2 with no parametrisation difference with respect to the censored model) are
**nuances that had not been quantified before**, not contradictions of what is published.

---

## Post-release verification (Zenodo)

Queried on 2026-11-16 in the public Zenodo API (`https://zenodo.org/api/records/24291974`),
seconds after publishing the `v1.5` release on GitHub. Values exactly as returned by the API:

| field | value returned |
|---|---|
| version DOI | **10.5281/zenodo.24291974** |
| concept DOI | 10.5281/zenodo.22968165 (the one of the README badge of the `v1.5` release) |
| version | v1.5 |
| type | software |
| licence | mit-license |
| creator | Araujo Villegas, Diego Alonso — ORCID 0000-0002-6987-3948 — Universidad Nacional Mayor de San Marcos |
| title | identical to that of `.zenodo.json` |
| status / access | published / open |
| created | 2026-11-16T20:35:09 UTC (publication date 2026-11-16) |
| relation | `isSupplementTo` → `https://github.com/diegoarz02/censored-gp-santa-barbara/tree/v1.5` |
| files (1) | `diegoarz02/censored-gp-santa-barbara-v1.5.zip`, 78 621 376 bytes, `md5:60e31abc530ad13a324abd04e1c68a3e` |

The concept now lists 6 versions (v1.0, v1.1, v1.2, v1.3, v1.4, v1.5). The version DOI
resolves (`https://doi.org/10.5281/zenodo.24291974` → zenodo.org). Version, creator, title, type and
licence coincide with `.zenodo.json`.

The tag `v1.5` points to commit `24d077b`. This section was added in a later commit (it cannot be
in the release archive, which was generated before the DOI was known): the content archived
in Zenodo is the tree of the tag `v1.5`; this report, with this section, is the one in `master`.

---

## Zenodo record for the article

The `v1.5` release was archived automatically by the GitHub–Zenodo integration within the
concept 10.5281/zenodo.22968165 (previous section), which also groups early versions of the
repository. The final record of the article is a **new manual deposit, with its own concept
DOI** (the repository plus the scanned OEFA report; versions `v2.0.0` and `v2.0.1`), without automatic integration.
Its version DOI and the concept DOI are given in the README, in `CITATION.cff` and in this report (below).

**Update (2026-11-18).** The tags and releases `v1.0` to `v1.5` were withdrawn from GitHub and versions
1.1 to 1.5 of the Zenodo concept (including record 24291974 of the previous section) were deleted; the
commit history is kept. The current version is the `v2.0.1` release (`v2.0.0` remains as the
previous version); their Zenodo records are verified below.

### Verification of the final record (Zenodo)

Queried on 2026-11-18 in the public Zenodo API. Values exactly as returned.

**Version v2.0.1 (current)** (`https://zenodo.org/api/records/24353450`):

| field | value returned |
|---|---|
| version DOI | **10.5281/zenodo.24353450** (`https://doi.org/10.5281/zenodo.24353450`) |
| concept DOI | **10.5281/zenodo.24347679** (`https://doi.org/10.5281/zenodo.24347679`) |
| version | v2.0.1 |
| type | software |
| licence | mit-license |
| creator | Araujo Villegas, Diego Alonso — ORCID 0000-0002-6987-3948 — Universidad Nacional Mayor de San Marcos |
| title | identical to that of `.zenodo.json` |
| status / access | published / open |
| created | 2026-11-18T20:51:54 UTC (publication date 2026-11-18) |
| relation | `isSupplementTo` → `https://github.com/diegoarz02/censored-gp-santa-barbara/tree/v2.0.1` |
| files (2) | `INFORME OEFA.zip`, 665 954 061 bytes, `md5:2f0537fb905fa13c9cd5f91c60a40961`; `censored-gp-santa-barbara-v2.0.1.zip`, 78 889 987 bytes, `md5:b9e82b50953721bcadf62c38d8c7573d` |

Both MD5 values match those computed before uploading the files, and the version and concept DOIs resolve
(`doi.org` redirects to zenodo.org). The concept 10.5281/zenodo.24347679 contains 2 versions: v2.0.1
(10.5281/zenodo.24353450) and v2.0.0 (10.5281/zenodo.24347680).

**Previous version v2.0.0** (`https://zenodo.org/api/records/24347680`, queried before v2.0.1 was published): version DOI
10.5281/zenodo.24347680; created 2026-11-18T18:18:28 UTC; relation `isSupplementTo` →
`https://github.com/diegoarz02/censored-gp-santa-barbara/tree/v2.0.0`; files: `censored-gp-santa-barbara-v2.0.0.zip`,
78 888 563 bytes, `md5:81d958d6417be5929dfaf555928e0f38`; `INFORME OEFA.zip`, 665 954 061 bytes,
`md5:2f0537fb905fa13c9cd5f91c60a40961`. It remains published and its DOI resolves.
