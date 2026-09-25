"""Sensibilidad al prior: la única defensa válida cuando el prior no se diluye.

**Por qué es obligatorio y no un extra.** Zhang (2004), vía Banerjee, Carlin & Gelfand (2015)
p. 148: con covarianza Matérn de suavidad ν solo el producto σ²φ^{2ν} es identificable, no los
parámetros por separado. El prior sobre varianza y rango **no se diluye ni con infinitos datos** en
un dominio acotado. Luego elegir «mejor» el prior no demuestra nada; lo que hay que demostrar es que
**la conclusión no depende de él**.

Tres especificaciones, mismo modelo, mismos datos, misma semilla:

| variante | sigma_tot | rho | fuente |
|---|---|---|---|
| `default` | HalfNormal(0.7) | Beta(2,2) | calibrado por chequeo predictivo del prior |
| `pc` | Exponencial, λ de Prob(σ>U)=0.01 con U despejado del límite físico | Beta(1,2), encoge a «sin campo» | Riebler et al. (2016) |
| `vague` | HalfNormal(2.0) | Beta(1,1) uniforme | Banerjee p. 149: «relatively vague prior for σ²» |

El prior de `ell` **no cambia** en ninguna: está calculado de la geometría del muestreo y es el que
Banerjee p. 149 recomienda mantener informativo.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402
import prior_worker as pw  # noqa: E402

METALS = ["Hg", "Pb", "As", "Cd"]
FAMILIES = ["default", "pc", "vague"]

if __name__ == "__main__":
    units = [{"metal": m, "family": f} for m in METALS for f in FAMILIES]
    done = sum(1 for u in units if pw.ckpt_path(u).exists())
    print(f"sensibilidad al prior: {len(units)} ajustes ({len(METALS)} metales x "
          f"{len(FAMILIES)} priors), {done} ya en checkpoint")
    # Deliberadamente pocos procesos: la corrida principal de H3.1+H4.5 tiene prioridad y no debe
    # quedarse sin nucleos por un analisis secundario.
    rows = Parallel(n_jobs=2, backend="loky", verbose=5)(delayed(pw.run_prior)(u) for u in units)
    df = pd.DataFrame([r for r in rows if r])
    # El worker ya estampa run_id en cada checkpoint, asi que insertarlo aqui duplicaba la columna
    # y reventaba el script *despues* de 33 horas de computo. Los checkpoints sobrevivieron, pero
    # la leccion es que la agregacion final no debe poder tirar lo que ya esta calculado.
    if "run_id" not in df.columns:
        df.insert(0, "run_id", RUN_ID)
    df.round(5).to_csv(FIG / "TABLE31_prior_sensitivity.csv", index=False, encoding="utf-8")
    print("\n" + df.drop(columns=["run_id"]).round(4).to_string(index=False))

    ok = df[df.get("error").isna()] if "error" in df else df
    txt = [stamp_header(), "", "# Sensibilidad al prior", "",
           "Zhang (2004) vía Banerjee p. 148: con Matérn el prior sobre varianza y rango **no se",
           "diluye nunca**. La defensa no es elegir mejor el prior, es mostrar que la conclusión no",
           "depende de él.", ""]
    if len(ok):
        piv = ok.pivot_table(index="metal", columns="family",
                             values=[c for c in ("sigma_tot_mean", "rho_mean", "ell0_mean")
                                     if c in ok])
        txt += ["## Posteriores bajo las tres especificaciones", "",
                piv.round(3).to_markdown(), ""]
        for col, nombre in [("sigma_tot_mean", "sigma_tot"), ("rho_mean", "rho")]:
            if col in ok:
                w = ok.pivot_table(index="metal", columns="family", values=col)
                rng = (w.max(axis=1) - w.min(axis=1)) / w.mean(axis=1)
                txt += [f"**{nombre}**: variación relativa entre priors, "
                        f"máximo {100 * rng.max():.1f} %, mediana {100 * rng.median():.1f} %.", ""]
    (OUT / "H_sensibilidad_priors.md").write_text("\n".join(txt), encoding="utf-8")
    observe(f"Sensibilidad al prior ejecutada: {len(ok)} de {len(units)} ajustes. Tres "
            f"especificaciones (actual, PC priors con U despejado del limite fisico, y vaga segun "
            f"Banerjee p.149) sobre {len(METALS)} metales.", "Bloque 4 - artículo")
    print("\nescrito: outputs/H_sensibilidad_priors.md, TABLE31")
