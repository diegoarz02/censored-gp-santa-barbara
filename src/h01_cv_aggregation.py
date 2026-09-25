"""H0.1 — settle the cadmium discrepancy by fixing one aggregation for the whole project.

The report published -1.833 / -1.371 and the external audit recomputed -1.748 / -1.569. Both are
correct arithmetic on the same file; they differ in how folds are pooled. This script shows the two,
states which is right and why, and rewrites the cross-validation table with it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from config import FIG, OUT, RUN_ID, archive_if_exists, observe, stamp_header  # noqa: E402

CV = pd.read_csv(OUT / "cv_real_data.csv")
METRICS = ["joint_log_score", "rmse_det", "mae_det", "crps_det",
           "picp90_det", "width90_det", "picp95_det", "width95_det"]


def aggregate(df, how):
    """Pool folds. `joint_log_score` is a per-point mean, so the pooled score over all held-out
    points is the n_test-weighted mean; a simple mean over folds would weight a 7-point block the
    same as a 36-point one.

    Detection-only metrics are weighted by `n_detected`, for exactly the same reason: they are means
    over the detected subset, and a fold with no detections contributes nothing rather than a NaN.
    """
    out = []
    for (metal, model), g in df.groupby(["metal", "model"], sort=False):
        row = {"metal": metal, "model": model, "n_folds": len(g),
               "n_test_total": int(g["n_test"].sum()),
               "n_detected_total": int(g["n_detected"].sum())}
        for m in METRICS:
            v = g[m].to_numpy(float)
            w = g["n_detected" if m.endswith("_det") else "n_test"].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
            if not ok.any():
                row[m] = np.nan
            elif how == "weighted":
                row[m] = float((v[ok] * w[ok]).sum() / w[ok].sum())
            else:
                row[m] = float(v[ok].mean())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    wt = aggregate(CV, "weighted")
    sm = aggregate(CV, "simple")

    cmp_ = wt[["metal", "model", "joint_log_score"]].rename(
        columns={"joint_log_score": "weighted_by_n_test"})
    cmp_["simple_mean_of_folds"] = sm["joint_log_score"].to_numpy()
    cmp_["difference"] = cmp_["weighted_by_n_test"] - cmp_["simple_mean_of_folds"]
    print("=== the two aggregations, joint log score ===")
    print(cmp_.round(4).to_string(index=False))

    cd = cmp_[cmp_.metal == "Cd"]
    print("\n=== cadmium, the discrepancy that was reported ===")
    print(cd.round(4).to_string(index=False))

    # Which fold drives the gap: block 1 has 36 of the 114 points and zero detections.
    b = CV[(CV.metal == "Cd") & (CV.model == "censored GP")][["block", "n_test", "n_detected",
                                                              "joint_log_score"]]
    print("\ncadmium folds for the censored GP (block 1 is 36 of 114 points, all censored):")
    print(b.to_string(index=False))

    wt.insert(0, "run_id", RUN_ID)
    archive_if_exists(FIG / "TABLE17_cross_validation.csv")
    wt_out = wt[["run_id", "metal", "model", "n_folds", "n_test_total", "n_detected_total"] + METRICS]
    wt_out.round(4).to_csv(FIG / "TABLE17_cross_validation.csv", index=False, encoding="utf-8")

    cmp_.insert(0, "run_id", RUN_ID)
    cmp_.round(4).to_csv(FIG / "TABLE_S6_aggregation_comparison.csv", index=False, encoding="utf-8")

    note = f"""{stamp_header()}

# H0.1 — how folds are pooled, decided once for the whole project

**Decision: weight every fold by its number of held-out points (`n_test`), and every
detection-only metric by `n_detected`.** Used everywhere from this run onward.

## Why the two numbers differed

Both were right arithmetic on `outputs/cv_real_data.csv`. The report pooled the five spatial blocks
weighted by size; the external audit took a simple mean over blocks.

| cadmium, joint log score | censored GP | global mean with L/2 |
|---|---|---|
| weighted by `n_test` (**adopted**) | {cd[cd.model == 'censored GP']['weighted_by_n_test'].iloc[0]:.4f} | {cd[cd.model == 'global mean with L/2']['weighted_by_n_test'].iloc[0]:.4f} |
| simple mean over folds | {cd[cd.model == 'censored GP']['simple_mean_of_folds'].iloc[0]:.4f} | {cd[cd.model == 'global mean with L/2']['simple_mean_of_folds'].iloc[0]:.4f} |

## Why weighted is the correct one

`joint_log_score` is stored as a **mean log density per held-out point**, not a total — a 36-point
block and a 7-point block have scores of the same magnitude. The out-of-sample joint log score of a
model over the whole survey is therefore

    (1/N) * sum over all held-out points of log p(y_i)  =  sum_k n_k * s_k / sum_k n_k

which is exactly the `n_test`-weighted mean. A simple mean over folds answers a different question —
"the average score a block gets" — and gives a 7-point block the same say as a 36-point one.

Cadmium is where the choice bites hardest, because its blocks are very uneven (7 to 36 points) and
**block 1 holds 36 of the 114 locations with zero detections**: an all-censored block, which is the
hardest case and the one the simple mean under-weights.

The conclusion does not change either way: the censored GP loses to the global mean with L/2 on
cadmium. Only the size of the gap moves, from 0.18 to 0.46 in log score.
"""
    (OUT / "H0.1_cv_aggregation.md").write_text(note, encoding="utf-8")
    observe("H0.1: la discrepancia del cadmio era agregación, no error. El informe ponderaba por "
            "n_test y la auditoría hacía media simple de pliegues. Se adopta ponderada por n_test "
            "en todo el proyecto: joint_log_score está guardado como media por punto, así que la "
            "ponderada es el score conjunto real sobre las 114 localizaciones. La conclusión (el "
            "cadmio pierde) no cambia; la brecha pasa de 0.18 a 0.46.", "Bloque 0 - trazabilidad")
    print("\nwritten: outputs/H0.1_cv_aggregation.md, TABLE17 (weighted), TABLE_S6")
