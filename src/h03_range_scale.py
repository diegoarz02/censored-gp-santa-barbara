"""H0.3 — an effective range fitted on the log scale is not an effective range in concentration.

Diggle & Ribeiro (2007), equation (3.16): for a log-Gaussian field with log-scale variance sigma^2
and log-scale correlation rho(u), the correlation of the untransformed concentrations is

    rho_T(u) = [exp{sigma^2 * rho(u)} - 1] / [exp{sigma^2} - 1]

rho_T <= rho always, and the gap widens with sigma^2. Our variograms are fitted in log space, so
"effective range 1.66 km for lead" is a statement about log concentration. On the concentration
scale — which is what a regulator reads — correlation has already decayed further at that distance.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

H2 = json.loads((OUT / "h2_spatial_structure.json").read_text(encoding="utf-8"))
VAR = pd.read_csv(FIG / "TABLE5_variogram_fits.csv")


def rho_T(rho, s2):
    """Correlation on the concentration scale from correlation on the log scale."""
    return (np.exp(s2 * rho) - 1.0) / (np.exp(s2) - 1.0)


def spherical_rho(h, a):
    """Correlation of a spherical variogram with effective range `a` (zero nugget)."""
    h = np.asarray(h, float)
    r = 1.0 - (1.5 * h / a - 0.5 * (h / a) ** 3)
    return np.where(h >= a, 0.0, r)


def range_at(target, a, s2, scale):
    """Distance at which correlation first drops to `target`, by bisection on the model."""
    lo, hi = 0.0, a
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = spherical_rho(mid, a)
        v = r if scale == "log" else rho_T(r, s2)
        if v > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    rows = []
    for a in ["Hg", "Pb", "As", "Ba"]:
        best = VAR[(VAR.analyte == a) & (VAR.model == H2["variogram_best"][a]["model"])].iloc[0]
        rng = float(best["effective_range_km"])
        s2 = float(best["sill"])                       # log-scale variance of the fitted model
        # correlation at a few distances, on both scales
        d_half = range_at(0.5, rng, s2, "log")
        d_half_T = range_at(0.5, rng, s2, "conc")
        rows.append({
            "analyte": a, "variogram": best["model"],
            "sill_log_variance": s2,
            "effective_range_km_log": rng,
            "rho_log_at_1km": float(spherical_rho(1.0, rng)),
            "rho_conc_at_1km": float(rho_T(spherical_rho(1.0, rng), s2)),
            "half_correlation_km_log": d_half,
            "half_correlation_km_conc": d_half_T,
            "shrinkage_pct": 100 * (1 - d_half_T / d_half) if d_half > 0 else np.nan,
        })
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))

    t.insert(0, "run_id", RUN_ID)
    t.round(4).to_csv(FIG / "TABLE_S7_range_scale_correction.csv", index=False, encoding="utf-8")

    pb = t[t.analyte == "Pb"].iloc[0]
    hg = t[t.analyte == "Hg"].iloc[0]
    txt = f"""{stamp_header()}

# H0.3 — the effective range is on the log scale, and it has to say so

**Diggle & Ribeiro (2007), eq. (3.16).** For a log-Gaussian field with log-scale variance
$\\sigma^2$ and log-scale correlation $\\rho(u)$, the correlation of the untransformed
concentrations is

$$\\rho_T(u) = \\frac{{\\exp\\{{\\sigma^2 \\rho(u)\\}} - 1}}{{\\exp\\{{\\sigma^2\\}} - 1}}$$

$\\rho_T \\le \\rho$ always, and the gap grows with $\\sigma^2$. Every variogram in this work is
fitted to log concentration, so an effective range quoted from it is a log-scale quantity.

## The correction, per analyte

{t.drop(columns=['run_id']).round(3).to_markdown(index=False)}

## How to say it in the manuscript

Not *"lead has an effective range of {pb['effective_range_km_log']:.2f} km"* but **"lead has an
effective range of {pb['effective_range_km_log']:.2f} km on the log scale; on the concentration
scale the correlation has already fallen to
{pb['rho_conc_at_1km']:.2f} at 1 km, against {pb['rho_log_at_1km']:.2f} in log"**.

The distance at which correlation halves shrinks by **{pb['shrinkage_pct']:.0f} %** for lead and
**{hg['shrinkage_pct']:.0f} %** for mercury when read on the scale a regulator actually uses. With
sill variances of {pb['sill_log_variance']:.1f} and {hg['sill_log_variance']:.1f} the effect is not
cosmetic.

**Why it matters beyond wording.** The monitoring design (H1.6 / H8.5d) places points using the
log-scale covariance, which is correct for the model. But the *interpretation* offered to whoever
commissions the sampling — "a new point informs its surroundings out to about this far" — is a
concentration-scale claim, and on that scale the reach is shorter. Both numbers are reported.
"""
    (OUT / "H0.3_range_scale.md").write_text(txt, encoding="utf-8")
    observe(f"H0.3: aplicada la corrección de escala de Diggle eq. (3.16). La distancia de "
            f"correlación 0.5 se encoge {pb['shrinkage_pct']:.0f} % en Pb y "
            f"{hg['shrinkage_pct']:.0f} % en Hg al pasar de escala log a concentración. No es "
            f"cosmético: las varianzas sill son {pb['sill_log_variance']:.1f} y "
            f"{hg['sill_log_variance']:.1f}. Los mapas y el diseño usan log (correcto para el "
            f"modelo); lo que cambia es cómo se interpreta el alcance ante quien encarga el "
            f"muestreo.", "Bloque 0 - trazabilidad")
    print("\nwritten: outputs/H0.3_range_scale.md, TABLE_S7")
