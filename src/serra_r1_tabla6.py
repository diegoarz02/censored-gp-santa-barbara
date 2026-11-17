"""SERRA-R1 T1 -- evidencia reproducible de la Tabla 6 / Fig6 / Wilcoxon, para la revision por
pares antes del envio a SERRA. Puro recalculo desde figuras/TABLE_H6_MAIN_con_kriging.csv y
outputs/h6_deep/*.json -- cero ajustes nuevos.
"""
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIG, OUT  # noqa: E402

OUT_DIR = OUT / "serra_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUSCRIPT = {
    ("As", "censored GP"): [0.959, 0.969, 0.966, 0.938],
    ("As", "GP with L/2"): [0.969, 0.970, 0.917, 0.678],
    ("As", "ordinary kriging L/2"): [0.953, 0.953, 0.839, 0.530],
    ("Hg", "censored GP"): [0.931, 0.945, 0.899, 0.832],
    ("Hg", "GP with L/2"): [0.934, 0.878, 0.805, 0.470],
    ("Hg", "ordinary kriging L/2"): [0.890, 0.821, 0.706, 0.497],
    ("Pb", "censored GP"): [0.879, 0.882, 0.884, 0.728],
    ("Pb", "GP with L/2"): [0.889, 0.837, 0.756, 0.361],
    ("Pb", "ordinary kriging L/2"): [0.903, 0.734, 0.636, 0.357],
}
LEVELS = [0.2, 0.4, 0.6, 0.8]


def signed_rank_exact(x):
    """Prueba de Wilcoxon de rangos con signo EXACTA (enumeracion completa de 2^n, sin scipy). Rangos y
    cola superior como en v14_wilcoxon_bloques.py; se anade el p bilateral exacto (ver el return)."""
    x = [v for v in x if v != 0]
    n = len(x)
    a = [abs(v) for v in x]
    o = sorted(range(n), key=lambda i: a[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[o[j + 1]] == a[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    W = sum(rk for rk, v in zip(r, x) if v > 0)
    sums = [sum(rk for rk, s in zip(r, sg) if s) for sg in itertools.product([0, 1], repeat=n)]
    p_up = sum(w >= W - 1e-9 for w in sums) / 2 ** n
    p_lo = sum(w <= W + 1e-9 for w in sums) / 2 ** n
    # p_v14: convencion de v14_wilcoxon_bloques.py (doble de la cola superior); coincide con el
    # bilateral exacto cuando el censurado gana, pero sobrestima cuando pierde. p_two: bilateral exacto.
    return n, min(1.0, 2 * p_up), min(1.0, 2 * min(p_up, p_lo))


def main():
    full = pd.read_csv(FIG / "TABLE_H6_MAIN_con_kriging.csv")

    # ------------------------------------------------------------------------------------- paso 1
    n_rows = len(full)
    n_deep = int(full["deep"].all())
    combos = full[["metal", "model", "level", "replicate"]].drop_duplicates()
    print(f"[1] filas: {n_rows} (esperado 360) | todas deep=True: {bool(n_deep)} | "
          f"combinaciones metal x modelo x nivel x replica: {len(combos)} (esperado 360)")
    assert n_rows == 360, f"filas inesperadas: {n_rows}"
    assert full["deep"].all(), "hay filas con deep=False"

    # ------------------------------------------------------------------------------------- paso 2
    recalc_rows = []
    for (metal, model), levels_vals in MANUSCRIPT.items():
        g = full[(full.metal == metal) & (full.model == model)]
        for lvl, pub in zip(LEVELS, levels_vals):
            s = g[g.level == lvl]["picp95"]
            mean_recalc = float(s.mean())
            recalc_rows.append({"metal": metal, "model": model, "level": lvl,
                               "n_replicas": len(s), "picp95_manuscrito": pub,
                               "picp95_recalculado": mean_recalc,
                               "diff": mean_recalc - pub})
    recalc = pd.DataFrame(recalc_rows)
    recalc.to_csv(OUT_DIR / "T6_recalculo.csv", index=False, encoding="utf-8")
    diffs = recalc[["metal", "model", "level", "diff"]].copy()
    diffs.to_csv(OUT_DIR / "T6_diff.csv", index=False, encoding="utf-8")
    max_abs_diff = float(recalc["diff"].abs().max())
    n_over_tol = int((recalc["diff"].abs() > 0.0005).sum())
    print(f"[2] 36 medias recalculadas. diferencia maxima absoluta = {max_abs_diff:.6f} "
          f"(tolerancia 0.0005). filas sobre tolerancia: {n_over_tol}/36")

    # ------------------------------------------------------------------------------------- paso 3
    hg_80 = full[(full.metal == "Hg") & (full.model == "censored GP") & (full.level == 0.8)]["picp95"]
    pb_80 = full[(full.metal == "Pb") & (full.model == "censored GP") & (full.level == 0.8)]["picp95"]
    print(f"[3] censurado 80%: Hg rango [{hg_80.min():.4f}, {hg_80.max():.4f}] "
         f"(manuscrito 0.57-1.00) | Pb rango [{pb_80.min():.4f}, {pb_80.max():.4f}] "
         f"(manuscrito 0.36-0.96), n={len(hg_80)}/{len(pb_80)}")

    # ------------------------------------------------------------------------------------- paso 4
    import inspect

    import figserra
    src_fig6 = inspect.getsource(figserra.fig6)
    lee_mismo_csv = "TABLE_H6_MAIN_con_kriging.csv" in src_fig6
    usa_picp95_mean = 'groupby("level")["picp95"].mean()' in src_fig6
    print(f"[4] src/figserra.py::fig6() lee TABLE_H6_MAIN_con_kriging.csv: {lee_mismo_csv} | "
         f"usa groupby(level)[picp95].mean(): {usa_picp95_mean}")
    assert lee_mismo_csv and usa_picp95_mean, "Fig6 no coincide con la fuente/calculo verificado"

    # ------------------------------------------------------------------------------------- paso 5
    D = {}
    for f in glob.glob(str(OUT / "h6_deep" / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if "error" in d:
            continue
        D[(d["metal"], d["level"], d["replicate"], d["block"])] = D.get(
            (d["metal"], d["level"], d["replicate"], d["block"]), {})
        D[(d["metal"], d["level"], d["replicate"], d["block"])][d["model"]] = d["picp95"]

    wilcox_rows = []
    for lvl in LEVELS:
        for base in ["GP with L/2", "ordinary kriging L/2"]:
            diffs_w = []
            for m in ["As", "Hg", "Pb"]:
                for b in range(5):
                    ks = [(m, lvl, r, b) for r in range(10) if (m, lvl, r, b) in D]
                    if not ks:
                        continue
                    c = sum(abs(D[k]["censored GP"] - 0.95) for k in ks) / len(ks)
                    s = sum(abs(D[k][base] - 0.95) for k in ks) / len(ks)
                    diffs_w.append(s - c)
            n, p_v14, p_two = signed_rank_exact(diffs_w)
            wins = sum(d_ > 0 for d_ in diffs_w)
            wilcox_rows.append({"level": lvl, "baseline": base, "n_pairs_total": len(diffs_w),
                               "n_pairs_no_empate": n, "censurado_gana": wins,
                               "p_bilateral": p_two, "p_v14_doble_cola_superior": p_v14})
    wilcox = pd.DataFrame(wilcox_rows)
    wilcox.to_csv(OUT_DIR / "T6_wilcoxon.csv", index=False, encoding="utf-8")
    print("\n[5] Wilcoxon de rangos con signo (|PICP95-0.95|, 15 unidades metal x bloque, "
         "2 semillas promediadas por bloque):")
    print(wilcox.to_string(index=False))
    print("\nCita del codigo (worker_v4.py L97-103, L128, L135-138): "
         "\"block\": r % n_blocks  ->  replica r y r+5 caen en el MISMO block (r%5==(r+5)%5); "
         "seed = SEED0 + 1000*rep + 97*int(level*100) + 13*(hash(metal)%1000)  ->  depende de "
         "`rep`, no de `block`, asi que r y r+5 difieren SOLO en la semilla MCMC. "
         "limit = np.quantile(value, level); censored = value < limit  ->  depende solo de "
         "(metal, level), igual para las 10 replicas -- comparten mascara.")

    return dict(max_abs_diff=max_abs_diff, n_over_tol=n_over_tol, wilcox=wilcox)


if __name__ == "__main__":
    main()
