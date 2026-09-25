"""Bloque 2 driver — render every map figure under the H2 rules and audit the result.

No MCMC: `outputs/posterior_maps.npz` already holds the posterior, so this only re-draws.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import mapas_f3 as M  # noqa: E402
from config import FIG, OUT, RUN_ID, archive_if_exists, observe, stamp_header  # noqa: E402

if __name__ == "__main__":
    D = M.load()
    rows = []

    print("--- site maps ---")
    rows.append({"figure": "FIG16_site_map_versionB_offline", **M.site_map_offline(D)})
    a = M.site_map_basemap(D, provider="OpenTopoMap")
    print("version A:", "ok via " + a.get("provider", "?") if a.get("ok") else a.get("why"))
    if a.get("ok"):
        rows.append({"figure": "FIG16_site_map_versionA_basemap", **a})

    print("\n--- posterior mean and uncertainty ---")
    for an in M.MAIN:
        rows.append({"figure": f"FIG17_{an}_mean_and_uncertainty",
                     **M.mean_and_uncertainty(D, an)})

    print("\n--- exceedance ---")
    exc = []
    for an in M.MAIN:
        r = M.exceedance(D, an, "agricultural")
        exc.append({"analyte": an, **{k: r[k] for k in ("land_use", "threshold",
                                                        "pct_above_half")}})
        rows.append({"figure": f"FIG18_{an}_exceedance", **r})
    # H4.2: cadmium gets both land uses, because that contrast is the paper's thesis
    r = M.exceedance(D, "Cd", "residential")
    exc.append({"analyte": "Cd", **{k: r[k] for k in ("land_use", "threshold",
                                                      "pct_above_half")}})
    rows.append({"figure": "FIG18_Cd_exceedance_residential", **r})

    e = pd.DataFrame(exc)
    e.insert(0, "run_id", RUN_ID)
    archive_if_exists(FIG / "TABLE20_exceedance_summary.csv")
    e.round(4).to_csv(FIG / "TABLE20_exceedance_summary.csv", index=False, encoding="utf-8")
    print("\n" + e.round(3).to_string(index=False))

    print("\n--- exceedance grid (5 metals + Cd residential) ---")
    rows.append({"figure": "FIG22_exceedance_grid", **M.exceedance_grid(D)})

    ef.contact_sheet()
    aud = ef.audit_figures()
    aud.insert(0, "run_id", RUN_ID)
    aud.to_csv(FIG / "TABLE_S8_figure_audit.csv", index=False, encoding="utf-8")
    bad_w = aud[~aud.width_ok]
    print(f"\naudit: {len(aud)} figures | off-spec widths {len(bad_w)} | "
          f"Type 3 {int(aud.type3.sum())} | with F3 run_id {int(aud.run_id_in_pdf.sum())}")
    if len(bad_w):
        print(bad_w[["figure", "width_mm"]].to_string(index=False))

    cd_a = e[(e.analyte == "Cd") & (e.land_use == "agricultural")]["pct_above_half"].iloc[0]
    cd_r = e[(e.analyte == "Cd") & (e.land_use == "residential")]["pct_above_half"].iloc[0]
    idx = f"""{stamp_header()}

# Figuras de la corrida F3

## Regeneradas en F3 (reglas H2.0 a H2.6)

Todas llevan `run_id` en los metadatos del PDF, nunca impreso en la figura.

{chr(10).join(f'- `{r["figure"]}`' for r in rows)}

## Qué cambió respecto a la Fase 2

- **Recuadro de ubicación**: colocado en el cuadrante más vacío, medido por ocupación de puntos.
  Antes tapaba terreno entre Santa Bárbara y las curvas del oeste.
- **Huancavelica**: ahora aparece en todos los mapas, con estrella, rótulo y las distancias
  medidas ({D['loc'].shape[0]} puntos, de 1.42 a 6.78 km de la plaza, 48 a menos de 3 km).
- **Relieve** a opacidad 0.18–0.35 según el mapa. Antes se tragaba los puntos.
- **Basemap** sin clave: OpenTopoMap con reintento y respaldo a Esri y OSM. CartoDB pasó a exigir
  clave y estampaba «API KEY REQUIRED» sobre la versión A.
- **Extrapolación** con hachurado además de contorno, no solo transparencia: la transparencia no
  se lee como una afirmación sobre la evidencia.
- **Umbral 0.5 marcado en la barra de color** de excedencia, que es el umbral de decisión.
- **Cadmio con sus dos usos de suelo**, que es la tesis del artículo: agrícola (ECA 1.4)
  {cd_a:.1f} % del dominio con P > 0.5, residencial (ECA 10) {cd_r:.1f} %.

## Heredadas de la Fase 2, no regeneradas en este bloque

Las de EDA, priors, recuperación, enmascaramiento, cobertura, validación cruzada y diseño
(FIG1 a FIG15). Se regeneran cuando el notebook vuelva a correr con los resultados nuevos.
"""
    (FIG / "_INDICE_F3.md").write_text(idx, encoding="utf-8")
    observe(f"Bloque 2: {len(rows)} figuras de mapa regeneradas con las reglas H2.0-H2.6. "
            f"Cadmio con sus dos umbrales: agrícola {cd_a:.1f} % del dominio con P>0.5, "
            f"residencial {cd_r:.1f} %. Ese contraste es la tesis del artículo en un par de "
            f"números: mismo metal, mismo suelo, mismo LD.", "Bloque 2 - figuras")
    print(f"\nwritten: figuras/_INDICE_F3.md, TABLE20, TABLE_S8, contact sheet")
