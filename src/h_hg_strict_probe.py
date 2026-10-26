"""Sonda puntual: re-ajusta SOLO el modelo censurado de Hg con un protocolo de muestreo mas caro,
para ver si pasa el criterio estricto de convergencia (R-hat<1.01, ESS>400, 0 divergencias) que hoy
falla por poco (R-hat=1.0104, ESS bulk=331, Cuadro IV).

Pedido explicito de Diego (2026-10-26), alcance acotado: correr y reportar, nada mas. No toca
main.tex, no toca el PDF, no toca ningun otro analito ni figura. La redaccion la hace Cowork
despues de validar estos numeros.

Replica EXACTAMENTE la preparacion de datos de `notebooks/_src/nb02.py::fit_main()` (seccion 7,
"Main hierarchical censored GP models"): mismas coordenadas, mismas covariables estandarizadas,
mismo stratum en la media, mismo kernel Matern 5/2 ARD, misma censura por intervalo, mismo limite
como parametro, misma semilla (SEED=20260906, la semilla propia de nb02.py, no la de config.py).
Lo UNICO que cambia es el protocolo de NUTS: tune/draws/chains/target_accept, segun lo pedido.

Checkpoint separado de `outputs/idata/main_Hg.nc` (el oficial, F4-20260923) para no tocarlo:
`outputs/idata/probe_Hg_strict_2026-10-26.nc`.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import sbmodel as sb              # noqa: E402
from config import OUT, observe   # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DIR_FINAL = PROJ / "data" / "final"
DIR_IDATA = OUT / "idata"
DIR_IDATA.mkdir(parents=True, exist_ok=True)

PROBE_ID = "PROBE-Hg-strict-2026-10-26"
NB02_SEED = 20260906   # semilla propia de nb02.py (fit_main), NO config.SEED (20260923)


def build_hg_context():
    """Replica identica de nb02.py lineas ~183-228 (carga, covariables, metal_arrays), solo Hg."""
    df = pd.read_csv(DIR_FINAL / "santa_barbara_soil.csv")
    loc = (df.drop_duplicates("location_id")
           .set_index("location_id")[["easting", "northing", "lon", "lat", "elev_oefa", "elev_dem",
                                      "slope_deg", "aspect_deg", "stratum", "sample_type", "sector",
                                      "dist_waste_dump_m", "dist_adit_m", "dist_city_m", "coord_flag"]]
           .sort_index())

    xy = np.c_[loc["easting"].to_numpy(float), loc["northing"].to_numpy(float)] / 1000.0
    xy = xy - xy.mean(0)

    covars = ["elev_oefa", "slope_deg", "log_dist_dump", "log_dist_city"]
    loc = loc.assign(log_dist_dump=np.log(loc["dist_waste_dump_m"].clip(lower=1.0)),
                     log_dist_city=np.log(loc["dist_city_m"]))
    cov = loc[covars].to_numpy(float)
    cov = (cov - cov.mean(0)) / cov.std(0)

    is_background = (loc["stratum"] == "background").to_numpy()
    is_composite = (loc["sample_type"] == "composite").to_numpy()

    g = df[df.analyte == "Hg"].set_index("location_id").loc[loc.index]
    censored = g["censored"].to_numpy(bool)
    value = g["value"].to_numpy(float)
    rep = g["reported_limit"].iloc[0]
    lod = float(rep) if np.isfinite(rep) else np.nan
    resolution = float(g["resolution"].iloc[0])

    ell_params, _, _ = sb.ell_prior_params(xy)

    return dict(xy=xy, cov=cov, is_background=is_background, is_composite=is_composite,
               censored=censored, value=value, lod=lod, resolution=resolution,
               ell_params=ell_params)


def main():
    ctx = build_hg_context()
    print(f"Hg: n={len(ctx['value'])}, censurados reales={int(ctx['censored'].sum())}, "
          f"lod={ctx['lod']}, resolution={ctx['resolution']}")

    min_det = float(np.nanmin(ctx["value"])) if np.isfinite(ctx["value"]).any() else None

    # ANISO_MAX > 2.0 en toda la corrida oficial (Hg 1.55, Pb 2.07, As 3.54, Ba 3.55 -- rango ya
    # publicado en main.tex linea 443) -> USE_ANISOTROPIC=True de forma global, no solo para Hg.
    model = sb.build_censored_gp(
        ctx["xy"], ctx["cov"], ctx["value"], ctx["censored"], ctx["resolution"], ctx["lod"],
        ell_params=ctx["ell_params"], is_background=ctx["is_background"],
        is_composite=ctx["is_composite"], anisotropic=True, limit_as_parameter=True,
        min_detected=min_det)

    protocols = [
        dict(draws=2000, tune=3000, chains=4, target_accept=0.995),
        dict(draws=2000, tune=3000, chains=4, target_accept=0.999),
    ]

    results = []
    for kw in protocols:
        print(f"\n=== Hg, protocolo {kw} ===")
        t0 = time.time()
        idata = sb.fit_model(model, seed=NB02_SEED, **kw)
        dt = time.time() - t0
        diag = sb.diagnose(idata)
        print(f"tiempo: {dt:.0f} s | R-hat max={diag['rhat_max']:.4f} | "
              f"ESS bulk={diag['ess_bulk_min']:.0f} | ESS tail={diag['ess_tail_min']:.0f} | "
              f"divergencias={diag['divergences']} | BFMI min={diag.get('bfmi_min', float('nan')):.3f} | "
              f"pasa={diag['passes']}")

        tag = f"ta{str(kw['target_accept']).replace('.', '')}"
        ck = DIR_IDATA / f"probe_Hg_strict_2026-10-26_{tag}.nc"
        idata.to_netcdf(ck)

        results.append({"protocol": kw, "seconds": dt, "checkpoint": str(ck), **diag})

        if diag["passes"]:
            print(f"PASA el criterio estricto con target_accept={kw['target_accept']} -- "
                  "no hace falta probar el siguiente protocolo.")
            break

    txt_rows = "\n".join(
        f"| {r['protocol']['target_accept']} | {r['seconds']:.0f} s | {r['rhat_max']:.4f} | "
        f"{r['ess_bulk_min']:.0f} | {r['ess_tail_min']:.0f} | {r['divergences']} | "
        f"{r.get('bfmi_min', float('nan')):.3f} | {'si' if r['passes'] else 'no'} |"
        for r in results
    )
    report = f"""<!-- probe_id: {PROBE_ID} | generado: {pd.Timestamp.now():%Y-%m-%d %H:%M} -->

# Sonda: Hg con protocolo de muestreo reforzado

Pedido de Diego (2026-10-26): ver si el modelo censurado de Hg pasa el criterio estricto de
convergencia (R-hat<1.01, ESS bulk y tail >400, 0 divergencias) con mas presupuesto de computo.
Hoy (Cuadro IV, corrida oficial F4-20260923, protocolo draws=1000/tune=1500/chains=4/
target_accept=0.99): **R-hat=1.0104, ESS bulk=331, ESS tail=460, 0 divergencias, NO pasa** (falla
solo por R-hat y ESS bulk).

Mismos datos, covariables, priors y semilla que la corrida oficial (SEED={NB02_SEED}, la propia de
`notebooks/_src/nb02.py`, no la de `config.py`) -- verificado reproduciendo la preparacion de datos
linea por linea desde el CSV. Unico cambio: el protocolo de NUTS.

| target_accept | tiempo | R-hat max | ESS bulk | ESS tail | divergencias | BFMI min | pasa |
|---|---|---|---|---|---|---|---|
{txt_rows}

Checkpoints: {", ".join(r["checkpoint"] for r in results)}.

**No se tocó** `main.tex`, el PDF del artículo, la Tabla IV, ni ningún otro analito/figura/cifra del
proyecto -- alcance acotado al pedido explícito de Diego. Estos números los valida Cowork antes de
decidir si reemplazan a los de la corrida oficial en la redacción final.
"""
    (OUT / "H_Hg_strict_probe.md").write_text(report, encoding="utf-8")

    final = results[-1]
    observe(f"Sonda Hg con protocolo reforzado (pedido puntual de Diego, no parte de la corrida "
            f"oficial F4): probados target_accept {[r['protocol']['target_accept'] for r in results]}, "
            f"draws=2000/tune=3000/chains=4. Mejor resultado: R-hat={final['rhat_max']:.4f}, "
            f"ESS bulk={final['ess_bulk_min']:.0f}, ESS tail={final['ess_tail_min']:.0f}, "
            f"divergencias={final['divergences']}, pasa={final['passes']}. "
            f"No se tocó main.tex ni ningún otro analito. Informe en outputs/H_Hg_strict_probe.md.",
            "Sonda Hg -- pedido puntual 2026-10-26")
    print("\nescrito: outputs/H_Hg_strict_probe.md")
    for r in results:
        print(f"  target_accept={r['protocol']['target_accept']}: R-hat={r['rhat_max']:.4f}, "
              f"ESS bulk={r['ess_bulk_min']:.0f}, ESS tail={r['ess_tail_min']:.0f}, "
              f"div={r['divergences']}, pasa={r['passes']}")


if __name__ == "__main__":
    main()
