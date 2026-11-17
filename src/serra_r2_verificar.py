"""SERRA-R2 -- verificacion independiente: recalcula desde los archivos (NPZ, CSV, idata) las cifras que
cita el informe y la Tabla 9 completa del articulo, sin leer el informe. Escribe verificacion_R2.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIG, OUT  # noqa: E402

R2 = OUT / "serra_r2"
rows = []


def check(nombre, calculado, esperado, tol=0.0, fuente=""):
    ok = (calculado == esperado) if isinstance(esperado, (str, bool)) else abs(calculado - esperado) <= tol
    rows.append({"verificacion": nombre, "calculado": calculado, "esperado": esperado, "tolerancia": tol,
                 "ok": bool(ok), "fuente": fuente})
    print(("OK    " if ok else "FALLA ") + f"{nombre}: calculado={calculado} esperado={esperado} (tol {tol})")


def main():
    new = dict(np.load(R2 / "posterior_maps_Hg_refuerzo.npz"))
    old = dict(np.load(OUT / "posterior_maps.npz"))
    ad = pd.read_csv(R2 / "hg_antes_despues.csv").set_index("cantidad")

    # 1. integridad del NPZ nuevo
    for k in ["mean", "median", "sd_f", "sd_y", "p_agri", "p_resid"]:
        check(f"NPZ nuevo: Hg_{k} tiene 7381 celdas y es finito", bool(new[f"Hg_{k}"].shape == (7381,)
              and np.isfinite(new[f"Hg_{k}"]).all()), True, fuente="posterior_maps_Hg_refuerzo.npz")
    check("NPZ nuevo: P en [0,1]", bool(((new["Hg_p_agri"] >= 0) & (new["Hg_p_agri"] <= 1)).all()), True)
    check("NPZ nuevo: P residencial = P agricola (mismo ECA 6.6)", bool(np.array_equal(new["Hg_p_agri"], new["Hg_p_resid"])), True)
    check("NPZ nuevo: 4000 draws por celda y semilla 20260906", int(new["n_samples"][0]) * 10 ** 6 + int(new["seed"][0]),
          4000 * 10 ** 6 + 20260906)
    check("NPZ nuevo: procede del ajuste reforzado", str(new["source_idata"]), "probe_Hg_strict_2026-10-26_ta0999.nc")
    check("NPZ nuevo: las coordenadas son las del mapa oficial",
          bool(np.array_equal(new["easting"], old["easting"]) and np.array_equal(new["northing"], old["northing"])), True)

    # 2. "antes" = lo ya publicado (dos fuentes independientes)
    t20 = pd.read_csv(FIG / "TABLE20_exceedance_summary.csv").set_index(["analyte", "land_use"])
    check("antes P>0.5 = TABLE20 (Hg)", float(100 * (old["Hg_p_agri"] > 0.5).mean()),
          float(t20.loc[("Hg", "agricultural"), "pct_above_half"]), 5e-5,
          "TABLE20_exceedance_summary.csv (guarda 4 decimales)")
    tae = pd.read_csv(FIG / "TABLE_area_excedida_posterior.csv").set_index("metal")
    for t, col in [(0.8, "pct_domain_P_agri_gt_0.8"), (0.95, "pct_domain_P_agri_gt_0.95")]:
        check(f"antes P>{t} = TABLE_area_excedida_posterior (Hg)", float(100 * (old["Hg_p_agri"] > t).mean()),
              float(tae.loc["Hg", col]), 1e-9, "TABLE_area_excedida_posterior.csv")

    # 3. "despues" recalculado desde el NPZ = tabla del informe
    for t in (0.5, 0.8, 0.95):
        lab = f"% del dominio con P(Hg > 6.6) > {t} (agricola = residencial: mismo ECA)"
        check(f"despues P>{t} (NPZ) = hg_antes_despues.csv", float(100 * (new["Hg_p_agri"] > t).mean()),
              float(ad.loc[lab, "despues_reforzado"]), 1e-9)

    # 4. Tabla 9 (tab:exc) completa del articulo, valores impresos (1 decimal)
    imp = {"Hg": (99.8, 96.7, 76.1), "Pb": (98.7, 68.4, 22.4), "As": (92.2, 31.0, 10.1), "Ba": (15.5, 4.5, 0.0)}
    ld = dict(np.load(OUT / "flip_ld_cd" / "maps_50m.npz"))
    cols = {"Pb": "Pb", "As": "As", "Ba": "Ba"}
    for m, (a5, a8, a95) in imp.items():
        if m == "Hg":
            continue
        for t, v in zip((0.5, 0.8, 0.95), (a5, a8, a95)):
            check(f"tab:exc {m} agricola P>{t} impreso {v}", round(float(tae.loc[m, f"pct_domain_P_agri_gt_{t}"]), 1), v,
                  0.0, "TABLE_area_excedida_posterior.csv")
    for t, v in zip((0.5, 0.8, 0.95), (38.2, 3.1, 0.0)):
        check(f"tab:exc Cd agricola P>{t} impreso {v}", round(float(100 * (ld["p_agri"] > t).mean()), 1), v, 0.0,
              "outputs/flip_ld_cd/maps_50m.npz")
    for t, v in zip((0.5, 0.8, 0.95), (6.8, 0.0, 0.0)):
        check(f"tab:exc Cd residencial P>{t} impreso {v}", round(float(100 * (ld["p_resid"] > t).mean()), 1), v, 0.0,
              "outputs/flip_ld_cd/maps_50m.npz")
    # Hg: lo impreso hoy y lo que imprimiria con el ajuste reforzado
    hg_old = [round(float(100 * (old["Hg_p_agri"] > t).mean()), 1) for t in (0.5, 0.8, 0.95)]
    hg_new = [round(float(100 * (new["Hg_p_agri"] > t).mean()), 1) for t in (0.5, 0.8, 0.95)]
    check("tab:exc Hg impreso actual (antes) = 99.8 / 96.7 / 76.1", str(hg_old), str([99.8, 96.7, 76.1]))
    check("tab:exc Hg con el ajuste reforzado (CAMBIA) = 99.8 / 96.8 / 77.4", str(hg_new), str([99.8, 96.8, 77.4]))

    # 5. P(tau<1) de Hg
    for lab, f, imp_, esp in [("antes", "main_Hg.nc", 0.28, 0.28), ("despues", "probe_Hg_strict_2026-10-26_ta0999.nc", 0.29, 0.29)]:
        t = az.from_netcdf(OUT / "idata" / f).posterior["log_tau"].values.ravel()
        check(f"P(tau<1) Hg {lab} redondeado a 2 decimales", round(float((t < 0).mean()), 2), esp, 0.0, f)

    # 6. covariables de Hg
    for lab, f in [("antes", "main_Hg.nc"), ("despues", "probe_Hg_strict_2026-10-26_ta0999.nc")]:
        b = az.from_netcdf(OUT / "idata" / f).posterior["beta"].values.reshape(-1, 4)
        lo, hi = np.quantile(b[:, 2], [0.025, 0.975])
        check(f"covariable log_dist_dump de Hg ({lab}): el IC95 % excluye el cero", bool(lo > 0 or hi < 0), True, fuente=f)

    # 7. zona de extrapolacion
    ex = pd.read_csv(R2 / "hg_extrapolacion.csv")
    check("extrapolacion con la constante fija 1.03 (la que dibujan las figuras) = 0.8 %",
          round(float(ex.iloc[3]["pct_dominio"]), 1), 0.8, 0.0, "hg_extrapolacion.csv")
    check("extrapolacion recalculada con Hg reforzado = 0.7 %", round(float(ex.iloc[1]["pct_dominio"]), 1), 0.7, 0.0, "hg_extrapolacion.csv")

    # 8. figuras: extrap en la leyenda y rutas
    for f in ["figuras/serra/Fig9.pdf", "figuras/FIG17_Hg_mean_and_uncertainty.pdf"]:
        check(f"{f} existe", bool((Path(__file__).resolve().parent.parent / f).exists()), True)

    t = pd.DataFrame(rows)
    t.to_csv(R2 / "verificacion_R2.csv", index=False, encoding="utf-8")
    print(f"\n{int(t.ok.sum())}/{len(t)} verificaciones OK")
    if not t.ok.all():
        sys.exit(1)


if __name__ == "__main__":
    main()
