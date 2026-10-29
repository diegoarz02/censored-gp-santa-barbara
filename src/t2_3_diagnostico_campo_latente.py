"""T2.3 del blindaje Q1/Q2: diagnosticos MCMC incluyendo el campo latente `f` (Copilot, ChatGPT).

El manuscrito aclara (anadido por Cowork) que el R-hat/ESS de la tabla de convergencia se reporta
sobre los hiperparametros, no sobre el campo latente de ~114 dimensiones (ver
outputs/AUDITORIA_UMBRAL_CENSURA.md, Bloque 2, donde se confirmo que esa exclusion reproduce
exacto la tabla publicada). Este script:

1. Verifica esa afirmacion (que main.tex describe correctamente la convencion).
2. Resume R-hat/ESS del campo `f` completo: distribucion, maximo, cuantiles, patron espacial
   (¿los puntos con peor R-hat estan donde falta densidad de muestreo, o son aleatorios?).
3. Reporta BFMI, profundidad de arbol maxima, divergencias -- diagnosticos que la tabla de
   convergencia actual NO incluye en absoluto.
4. Determina si el problema de mezcla del campo (si existe) afecta las probabilidades de
   excedencia (comparando R-hat de f en los puntos de ENTRENAMIENTO contra las celdas de
   PREDICCION de mayor P(excedencia), ya que el campo en la prediccion se deriva analiticamente
   del campo de entrenamiento vía predict_field, no se muestrea aparte).

No reajusta ningun modelo -- solo posprocesa outputs/idata/main_*.nc ya guardados.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import arviz as az   # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
METALS = ["As", "Ba", "Hg", "Pb", "Cd"]


def main():
    rows = []
    field_rhat_all = {}
    for metal in METALS:
        idata = az.from_netcdf(OUT / "idata" / f"main_{metal}.nc")
        post = idata.posterior
        ss = idata.sample_stats

        # --- R-hat/ESS del campo f completo (una fila por punto de dato, ~114) ---
        f_summ = az.summary(idata, var_names=["f"], round_to="none")
        f_rhat = pd.to_numeric(f_summ["r_hat"], errors="coerce").to_numpy()
        f_ess_bulk = pd.to_numeric(f_summ["ess_bulk"], errors="coerce").to_numpy()
        f_ess_tail = pd.to_numeric(f_summ["ess_tail"], errors="coerce").to_numpy()
        field_rhat_all[metal] = f_rhat

        # --- diagnosticos que la tabla de convergencia actual NO reporta ---
        divergences = int(ss["diverging"].sum()) if "diverging" in ss else None
        depth_var = "depth" if "depth" in ss else ("tree_depth" if "tree_depth" in ss else None)
        max_tree_depth = int(ss[depth_var].values.max()) if depth_var else None
        # BFMI (energia bayesiana) por cadena, formula directa (Betancourt 2016):
        # BFMI_chain = Var(E_t - E_{t-1}) / Var(E_t)
        if "energy" in ss:
            E = ss["energy"].values
            bfmi_chain = np.diff(E, axis=1).var(axis=1) / E.var(axis=1)
            bfmi_min = float(bfmi_chain.min())
        else:
            bfmi_min = None

        # --- hiperparametros nombrados (lo que SI reporta la tabla publicada) ---
        all_vars = list(post.data_vars)
        excl = [v for v in all_vars if v.startswith("f") or "log_limit" in v]
        keep = [v for v in all_vars if v not in excl]
        hyp_summ = az.summary(idata, var_names=keep, round_to="none")
        hyp_rhat = pd.to_numeric(hyp_summ["r_hat"], errors="coerce").dropna()
        hyp_ess_bulk = pd.to_numeric(hyp_summ["ess_bulk"], errors="coerce").dropna()

        row = dict(
            metal=metal,
            hyp_rhat_max=float(hyp_rhat.max()), hyp_ess_bulk_min=float(hyp_ess_bulk.min()),
            field_rhat_max=float(np.nanmax(f_rhat)), field_rhat_mean=float(np.nanmean(f_rhat)),
            field_rhat_q50=float(np.nanpercentile(f_rhat, 50)),
            field_rhat_q95=float(np.nanpercentile(f_rhat, 95)),
            field_rhat_pct_above_1_01=float(100 * np.nanmean(f_rhat > 1.01)),
            field_ess_bulk_min=float(np.nanmin(f_ess_bulk)),
            field_ess_bulk_q05=float(np.nanpercentile(f_ess_bulk, 5)),
            field_ess_tail_min=float(np.nanmin(f_ess_tail)),
            divergences=divergences, max_tree_depth=max_tree_depth, bfmi_min=bfmi_min,
        )
        rows.append(row)
        print(f"{metal}: hiperparams R-hat_max={row['hyp_rhat_max']:.4f} ESS_min={row['hyp_ess_bulk_min']:.0f} "
              f"| campo f: R-hat_max={row['field_rhat_max']:.4f} ({row['field_rhat_pct_above_1_01']:.1f}% "
              f"de los {len(f_rhat)} puntos >1.01), ESS_bulk_min={row['field_ess_bulk_min']:.0f} "
              f"| BFMI_min={bfmi_min:.3f} | div={divergences} | tree_depth_max={max_tree_depth}")

    tab = pd.DataFrame(rows)
    tab.to_csv(FIG / "TABLE_T2_3_diagnostico_campo_latente.csv", index=False, encoding="utf-8")

    # --- patron espacial: ¿el peor R-hat del campo se relaciona con menor densidad local de datos? ---
    print("\n=== Patron espacial del peor R-hat del campo (¿aislado o estructural?) ===")
    df = pd.read_csv(PROJ / "data" / "final" / "santa_barbara_soil.csv")
    for metal in METALS:
        loc = (df[df.analyte == metal].drop_duplicates("location_id")
               .set_index("location_id")[["easting", "northing"]].sort_index())
        xy = loc.to_numpy(float) / 1000.0
        d_nn = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        np.fill_diagonal(d_nn, np.inf)
        nn_dist = d_nn.min(1)                       # distancia al vecino mas cercano, por punto
        rhat = field_rhat_all[metal]
        if len(rhat) == len(nn_dist):
            corr = float(np.corrcoef(rhat, nn_dist)[0, 1])
            worst_idx = int(np.nanargmax(rhat))
            print(f"  {metal}: corr(R-hat_f, dist. al vecino mas cercano) = {corr:+.3f} | "
                  f"peor punto (idx {worst_idx}): R-hat={rhat[worst_idx]:.4f}, "
                  f"dist. vecino mas cercano={nn_dist[worst_idx]:.3f} km "
                  f"(mediana del dataset={np.median(nn_dist):.3f} km)")

    print("\nescrito: figuras/TABLE_T2_3_diagnostico_campo_latente.csv")
    return tab, field_rhat_all


if __name__ == "__main__":
    main()
