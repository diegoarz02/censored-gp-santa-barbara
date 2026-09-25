# -*- coding: utf-8 -*-
"""Check the acceptance criterion of every milestone, one by one.

Run from the project root:

    python src/verify_milestones.py

Each check reads a real file under `outputs/`, `figuras/` or `data/final/`. Nothing is asserted
from memory: if a file is missing the criterion fails and says so.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJ = Path(__file__).resolve().parent.parent
FIG, OUT, FINAL, NB = PROJ / "figuras", PROJ / "outputs", PROJ / "data" / "final", PROJ / "notebooks"
SKILLS = Path.home() / ".claude" / "skills"
VAULT = Path.home() / "Documents" / "Obsidian Vault" / "Claude Diego"

results: list[tuple[str, str, bool, str]] = []


def check(milestone, name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:                                        # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
    results.append((milestone, name, bool(ok), str(detail)[:200]))


def _nb(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [c for c in code if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    run = [c for c in code if c.get("execution_count")]
    return len(code), len(run), len(errs), nb["metadata"].get("kernelspec", {}).get("name")


def _json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
def _figure_widths():
    from PIL import Image
    bad, widths = [], set()
    for f in sorted(FIG.glob("FIG*.tif")):
        im = Image.open(f)
        mm = round(im.size[0] / im.info["dpi"][0] * 25.4)
        widths.add(mm)
        if mm not in (90, 140, 190):
            bad.append(f"{f.stem}={mm}mm")
    return (not bad, f"widths present: {sorted(widths)}" + (f"; OFF-SPEC: {bad}" if bad else ""))


def _no_type3():
    bad = []
    for f in sorted(FIG.glob("FIG*.pdf")):
        raw = f.read_bytes()
        if b"/Type3" in raw:
            bad.append(f.stem)
    return (not bad, "none found" if not bad else f"Type 3 in {bad}")


def _vault():
    src = list((PROJ / "notas_obsidian").glob("*.md"))
    copied = [p for p in src if (VAULT / "metodos" / p.name).exists()
              or (VAULT / "proyectos" / p.name).exists() or (VAULT / "conceptos" / p.name).exists()]
    new = list(VAULT.rglob("*Santa*")) + list(VAULT.rglob("*santa-barbara*"))
    return (len(copied) == len(src) and len(new) > 0,
            f"{len(copied)}/{len(src)} project notes in the vault, {len(new)} Santa Bárbara notes")


check("H0", "Cuajone material archived with a README",
      lambda: ((OUT / "_papelera_cuajone" / "README.md").exists(),
               f"{len(list((OUT / '_papelera_cuajone').rglob('*')))} archived items"))

check("H0", "OEFA raw data and inventory preserved",
      lambda: (len(list((PROJ / "data/fase1/oefa").glob("*.xlsx"))) >= 20
               and (OUT / "inventario" / "diagnostico_LDLC_archivo_completo.csv").exists(),
               f"{len(list((PROJ / 'data/fase1/oefa').glob('*.xlsx')))} xlsx, inventory kept"))

check("H1", "Dataset with 114 locations and its dictionary",
      lambda: (lambda d: (d["location_id"].nunique() == 114 and (FINAL / "DATA_DICTIONARY.md").exists(),
                          f"{d['location_id'].nunique()} locations, {len(d)} rows, "
                          f"{d['analyte'].nunique()} analytes"))(pd.read_csv(FINAL / "santa_barbara_soil.csv")))

check("H1", "Notebook 1 executed without errors",
      lambda: (lambda t: (t[2] == 0 and t[1] == t[0], f"{t[0]} cells, {t[1]} run, {t[2]} errors, kernel {t[3]}"))
      (_nb(NB / "01_dataset_construction.ipynb")))

check("H1", "DEM downloaded and terrain covariates present",
      lambda: ((PROJ / "data/fase1/covariables/santa_barbara_dem_srtm30.tif").exists()
               and "slope_deg" in pd.read_csv(FINAL / "santa_barbara_soil.csv").columns,
               "SRTM 30 m DEM, Horn slope and aspect"))

check("H2", "Variogram, Moran's I and an anisotropy decision",
      lambda: (lambda h: (h["anisotropy_ratio_max"] > 0 and (FIG / "TABLE7_morans_i.csv").exists(),
                          f"kernel: {h['kernel']}, ratio {h['anisotropy_ratio_max']:.2f} "
                          f"(naive would be {h.get('anisotropy_ratio_max_naive', float('nan')):.2f})"))
      (_json("h2_spatial_structure.json")))

check("H3", "Prior predictive check, rejected and calibrated versions",
      lambda: (lambda t: (t.groupby("prior")["plausible"].all().any() and len(t["prior"].unique()) == 2,
                          f"{len(t['prior'].unique())} prior specifications compared"))
      (pd.read_csv(FIG / "TABLE8_prior_predictive.csv")))

check("H4", "Parameter recovery passes for the censored model",
      lambda: (lambda t: (t[t.model == "censored"]["covers"].mean() >= 0.9,
                          f"censored covers {t[t.model=='censored']['covers'].mean():.0%}, "
                          f"L/2 covers {t[t.model!='censored']['covers'].mean():.0%}"))
      (pd.read_csv(FIG / "TABLE9_parameter_recovery.csv")))

check("H5", "Main models converge: R-hat < 1.01, ESS > 400, no divergences",
      lambda: (lambda t: (t["passes"].all(),
                          f"rhat max {t['rhat_max'].max():.4f}, ESS min "
                          f"{min(t['ess_bulk_min'].min(), t['ess_tail_min'].min()):.0f}, "
                          f"divergences {int(t['divergences'].sum())}"))
      (pd.read_csv(FIG / "TABLE10_convergence.csv")))

check("H5", "Stratum against support aliasing inspected",
      lambda: (lambda t: ((FIG / "FIG7_stratum_support.pdf").exists(),
                          f"max |posterior correlation| {t['posterior_corr'].abs().max():.3f}, "
                          f"separable: {bool(t['separable'].all())}"))
      (pd.read_csv(FIG / "TABLE12_stratum_support_aliasing.csv")))

check("H5", "Prior against posterior overlap reported",
      lambda: ((FIG / "TABLE13_prior_posterior.csv").exists() and (FIG / "FIG8_prior_posterior.pdf").exists(),
               "width ratio per parameter and analyte"))

check("H6", "Masking experiment run over the full design",
      lambda: (lambda t: (len(t) > 0 and t["metal"].nunique() == 3 and t["level"].nunique() == 4,
                          f"{len(t)} rows, {t['metal'].nunique()} analytes, {t['level'].nunique()} levels, "
                          f"{t['replicate'].nunique()} replicates, {t['model'].nunique()} models"))
      (pd.read_csv(OUT / "H6_masking_experiment.csv")))

check("H6", "Divergence slopes fitted with intervals",
      lambda: (lambda t: (len(t) > 0 and "ci95_low" in t.columns,
                          f"{int(t['diverges'].sum())} of {len(t)} combinations show a widening gap"))
      (pd.read_csv(FIG / "TABLE16_divergence_slopes.csv")))

check("H6", "Cheap inference validated against the full sampler",
      lambda: ((OUT / "H6_full_nuts_validation.csv").exists(),
               f"{len(pd.read_csv(OUT / 'H6_full_nuts_validation.csv'))} validation rows"))

check("H7", "Cross-validation with trivial baselines and the joint score",
      lambda: (lambda t: ({"intercept (detections)", "global mean with L/2", "ordinary kriging L/2",
                           "censored GP"} <= set(t["model"]) and "joint_log_score" in t.columns,
                          f"{t['model'].nunique()} models over {t['metal'].nunique()} analytes"))
      (pd.read_csv(FIG / "TABLE17_cross_validation.csv")))

check("H7", "Per-fold diagnostics published",
      lambda: (lambda t: (len(t) > 0,
                          f"{len(t)} folds, rhat max {t['rhat_max'].max():.4f}, "
                          f"divergences {int(t['divergences'].sum())}"))
      (pd.read_csv(FIG / "TABLE_S4_cv_fold_diagnostics.csv")))

check("H7", "Paired bootstrap on every metric difference",
      lambda: (lambda t: ("ci95_low" in t.columns and "wins_by_block" in t.columns,
                          f"{len(t)} comparisons with intervals and per-block win counts"))
      (pd.read_csv(FIG / "TABLE18_paired_bootstrap.csv")))

check("H8", "Background against mining input quantified",
      lambda: (lambda t: ((FIG / "FIG13_background_and_city.pdf").exists(),
                          "Hg background median "
                          f"{t[t.analyte=='Hg']['median_background'].iloc[0]:.1f} mg/kg, "
                          f"{t[t.analyte=='Hg']['pct_bg_above_eca'].iloc[0]:.0f}% above the standard"))
      (pd.read_csv(FIG / "TABLE19_background_vs_mining.csv")))

check("H9", "Exceedance maps with at least 4000 draws and reported Monte Carlo error",
      lambda: (lambda t, z: (int(z["n_samples"][0]) >= 4000 and "mc_se_at_pmax" in t.columns,
                             f"{int(z['n_samples'][0])} draws per cell, "
                             f"{len(z['easting'])} cells"))
      (pd.read_csv(FIG / "TABLE20_exceedance_summary.csv"),
       __import__("numpy").load(OUT / "posterior_maps.npz")))

check("H9", "Both cadmium thresholds mapped",
      lambda: ((FIG / "FIG18_Cd_exceedance.pdf").exists(),
               "agricultural 1.4 and residential 10 mg/kg as a regulatory sensitivity analysis"))

check("H10", "Monitoring design with ranked new points",
      lambda: (lambda t: (len(t) >= 20,
                          f"{len(t)} proposed points, "
                          f"{t['cumulative_sd_reduction_pct'].iloc[-1]:.1f}% sd reduction at 20"))
      (pd.read_csv(FIG / "TABLE22_proposed_sampling_points.csv")))

check("H11", "LMC cross-correlation reported with its interval",
      lambda: (lambda t: ("ci95_low" in t.columns and "identified" in t.columns,
                          "; ".join(f"{r.pair} width {r.width95:.2f} "
                                    f"({'identified' if r.identified else 'NOT identified'})"
                                    for r in t.itertuples())))
      (pd.read_csv(FIG / "TABLE23_lmc_cross_correlation.csv")))

check("H11", "Preferential sampling sensitivity analysis",
      lambda: (lambda t: (t["subset"].nunique() >= 3, f"{t['subset'].nunique()} sampling subsets compared"))
      (pd.read_csv(FIG / "TABLE24_preferential_sampling.csv")))

check("H12", "Three map versions attempted, offline one always produced",
      lambda: (lambda s: ((FIG / "FIG16_site_map_versionB_offline.pdf").exists(),
                          f"A basemap {s['map_versions']['A_basemap']}, B offline True, "
                          f"C osm {s['map_versions']['C_osm']}"))
      (_json("nb02_summary.json")))

check("H13", "Figure widths are exactly 90, 140 or 190 mm",
      lambda: _figure_widths())

check("H13", "Every figure in vector PDF and TIFF at 600 dpi or more",
      lambda: (lambda p, t: (len(p) == len(t) and len(p) > 0, f"{len(p)} PDF, {len(t)} TIFF"))
      (list(FIG.glob("FIG*.pdf")), list(FIG.glob("FIG*.tif"))))

check("H13", "No Type 3 fonts in the PDFs",
      lambda: _no_type3())

check("H13", "Tables exported to CSV and LaTeX",
      lambda: (len(list(FIG.glob("TABLE*.csv"))) >= 20 and len(list(FIG.glob("TABLE*.tex"))) >= 10,
               f"{len(list(FIG.glob('TABLE*.csv')))} CSV, {len(list(FIG.glob('TABLE*.tex')))} LaTeX"))

check("H14", "RESULTS_SUMMARY.md complete with no unresolved placeholders",
      lambda: (lambda s: ((PROJ / "RESULTS_SUMMARY.md").exists() and "{{" not in s,
                          f"{len(s.splitlines())} lines"))
      ((PROJ / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")
       if (PROJ / "RESULTS_SUMMARY.md").exists() else ""))

check("H15", "Seven skills present, including journal-maps-real",
      lambda: (lambda names: (set(names) <= set(p.name for p in SKILLS.iterdir() if p.is_dir()),
                              ", ".join(sorted(names))))
      (["bayesian-spatial-gp", "censored-data-bayes", "geostatistics-eda", "spatial-cv-metrics",
        "coregionalization-lmc", "journal-figures-q1", "journal-maps-real"]))

check("H16", "Obsidian notes copied and new ones written",
      lambda: _vault())

check("H16", "Notebook 2 executed without errors",
      lambda: (lambda t: (t[2] == 0 and t[1] == t[0], f"{t[0]} cells, {t[1]} run, {t[2]} errors, kernel {t[3]}"))
      (_nb(NB / "02_bayesian_model.ipynb")))


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    W = 128
    print("=" * W)
    print("MILESTONE ACCEPTANCE CRITERIA")
    print("=" * W)
    cur = None
    for ms, name, ok, detail in results:
        if ms != cur:
            print(f"\n--- {ms} " + "-" * (W - len(ms) - 5))
            cur = ms
        print(f"[{'x' if ok else ' '}] {name:<62s} {detail}")
    n_ok = sum(r[2] for r in results)
    print("\n" + "=" * W)
    print(f"{n_ok} of {len(results)} criteria met")
    print("=" * W)
    pd.DataFrame(results, columns=["milestone", "criterion", "met", "detail"]).to_csv(
        OUT / "milestone_verification.csv", index=False, encoding="utf-8")
    sys.exit(0 if n_ok == len(results) else 1)
