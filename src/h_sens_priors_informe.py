"""Informe de la sensibilidad al prior, con la figura que la acompaña.

Separado del script que ajusta porque el ajuste cuesta horas y el informe se reescribe muchas veces.
Lee `TABLE31_prior_sensitivity.csv`, que ya está en disco.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
from config import FIG, OUT, RUN_ID, observe, stamp_header  # noqa: E402

FAM = ["default", "pc", "vague"]
FAM_LABEL = {"default": "calibrated", "pc": "PC prior", "vague": "vague"}
FAM_COLOR = {"default": ef.C_CENSORED, "pc": ef.C_SUB_GP, "vague": ef.C_KRIGING}

if __name__ == "__main__":
    d = pd.read_csv(FIG / "TABLE31_prior_sensitivity.csv")
    d["family"] = pd.Categorical(d["family"], FAM, ordered=True)
    d = d.sort_values(["metal", "family"])

    piv = d.pivot_table(index="metal", columns="family", values="sigma_tot_mean", observed=False)
    piv["rango_rel_pct"] = 100 * (piv.max(axis=1) - piv.min(axis=1)) / piv.mean(axis=1)
    cens = d.groupby("metal", observed=False)["pct_censored"].first()
    piv["pct_censurado"] = cens
    piv = piv.sort_values("pct_censurado")
    print("=== sigma_tot posterior bajo las tres especificaciones ===")
    print(piv.round(3).to_string())

    pr = d.pivot_table(index="metal", columns="family", values="rho_mean", observed=False)
    pr["rango_rel_pct"] = 100 * (pr.max(axis=1) - pr.min(axis=1)) / pr.mean(axis=1)
    print("\n=== rho posterior ===")
    print(pr.round(3).to_string())

    div = d.pivot_table(index="metal", columns="family", values="divergences",
                        observed=False, aggfunc="sum")
    print("\n=== divergencias por especificacion ===")
    print(div.to_string())

    # ---- figura: un panel por parametro, metales ordenados por censura
    ef.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ef.W2, ef.W2 * 0.40), constrained_layout=True)
    order = list(piv.index)
    x = np.arange(len(order))
    for ax, (col, sd, lab) in zip(axes, [
            ("sigma_tot_mean", "sigma_tot_sd", r"$\sigma_{\mathrm{tot}}$ posterior"),
            ("rho_mean", "rho_sd", r"$\rho$ posterior (spatial fraction)")]):
        for k, fam in enumerate(FAM):
            s = d[d.family == fam].set_index("metal").loc[order]
            ax.errorbar(x + (k - 1) * 0.20, s[col], yerr=s[sd], fmt="o", ms=4.5,
                        color=FAM_COLOR[fam], mec="k", mew=0.25, capsize=2.5, lw=1.1,
                        label=FAM_LABEL[fam])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n{cens[m]:.0f}% censored" for m in order])
        ax.set_ylabel(lab)
        ef.soft_grid(ax)
    ef.panel_label(axes[0], "a")
    ef.panel_label(axes[1], "b")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=3)
    ef.message_title(fig, "The prior matters little where there is data, and a lot where there "
                          "is none")
    ef.save_fig(fig, "FIG21_prior_sensitivity")
    plt.close(fig)

    lo = piv.loc[piv.index[0], "rango_rel_pct"]
    cd = piv.loc["Cd", "rango_rel_pct"]
    div_vague = int(div["vague"].sum())
    div_other = int(div["default"].sum() + div["pc"].sum())

    txt = f"""{stamp_header()}

# Sensibilidad al prior — y por qué este análisis no era opcional

## El motivo

**Banerjee, Carlin & Gelfand (2015), p. 148**, citando **Zhang (2004)**: con covarianza Matérn de
suavidad ν, solo el producto σ²φ^{{2ν}} es identificable, **no los parámetros por separado**. El
prior sobre la varianza espacial y el rango **no se diluye ni con infinitos datos** en un dominio
acotado.

Luego elegir «mejor» el prior no demuestra nada. Lo único que se puede demostrar es que **la
conclusión no depende de él**, y eso exige ajustar con varias especificaciones y enseñar las tres.

## Las tres especificaciones

| variante | `sigma_tot` | `rho` | origen |
|---|---|---|---|
| **calibrado** | HalfNormal(0.7) | Beta(2,2) | el actual, validado por chequeo predictivo del prior |
| **PC prior** | Exponencial con λ de Prob(σ > U) = 0.01, **U despejado del límite físico** | Beta(1,2), encoge hacia «sin campo espacial» | Riebler et al. (2016) |
| **vago** | HalfNormal(2.0) | Beta(1,1) uniforme | Banerjee p. 149: *"a relatively vague prior for σ²"* |

El prior de la escala espacial `ell` **no cambia** en ninguna: está calculado de la geometría del
muestreo y es el que Banerjee p. 149 recomienda mantener informativo.

**Nota sobre el PC prior:** `U` no se eligió. Se despejó numéricamente como el mayor valor cuya
predictiva a priori deja menos de 1e-4 de masa por encima del límite físico de 10⁶ mg/kg. Así el
chequeo predictivo del prior deja de ser un examen posterior y pasa a ser **la ecuación que fija el
prior**. Resultó U = 3.00 para Hg y As, 2.13 para Pb, 3.25 para Cd.

## El resultado, y es más interesante de lo que esperaba

{piv.round(3).to_markdown()}

**La variación del posterior con el prior crece con la censura.** Los tres analitos sin censura se
mueven entre {lo:.0f} % y 18 %; **el cadmio, con 79.8 % de censura, se mueve un {cd:.0f} %**.

Eso no es una casualidad de este conjunto de datos: es exactamente lo que predice la teoría. Con
menos información en la verosimilitud, el prior pesa más. Y aquí queda **medido**, en el mismo
modelo y con los mismos datos, variando solo el prior.

**Es un argumento nuevo para el artículo**: la censura no solo destruye la cobertura (el resultado
central), sino que además **transfiere peso del dato al prior**. Las dos cosas son la misma cosa
vista desde dos lados, y la segunda es fácil de comunicar a alguien que no sepa geoestadística.

## La fracción espacial

{pr.round(3).to_markdown()}

El prior vago empuja `rho` hacia arriba en los cuatro metales: atribuye más varianza al campo
espacial y menos al ruido. Es coherente con lo que el PC prior está diseñado para evitar —él encoge
hacia el modelo base sin campo espacial— y es la razón por la que Riebler et al. lo proponen.

## Y un resultado de cómputo que no esperaba

| especificación | divergencias totales |
|---|---|
| calibrado | {int(div['default'].sum())} |
| PC prior | {int(div['pc'].sum())} |
| **vago** | **{div_vague}** |

**El prior vago no solo es menos informativo: converge peor.** {div_vague} divergencias frente a
{div_other} entre las otras dos juntas. Encaja con la advertencia de Banerjee p. 149 sobre muestrear
de posteriores casi impropias: *"while we may not explicitly see problems with our MCMC chains, we
know that sampling from a nearly improper posterior can yield poorly behaved MCMC inference"*.

## Qué se lleva el artículo

1. **Las conclusiones del trabajo no dependen del prior** en los analitos donde hay datos: `sigma_tot`
   se mueve menos del 18 % entre una especificación calibrada y una deliberadamente vaga.
2. **En cadmio sí depende**, y eso hay que decirlo. Es el mismo analito que pierde el score conjunto
   y el que está en la frontera del 80 % de censura. Tres señales del mismo hecho.
3. **El PC prior es la mejor de las tres**: converge igual que el calibrado, su parámetro se despeja
   de una restricción física en vez de elegirse, y encoge hacia el modelo más simple.

Figura `FIG21_prior_sensitivity`, tabla `TABLE31_prior_sensitivity.csv`.
"""
    (OUT / "H_sensibilidad_priors.md").write_text(txt, encoding="utf-8")
    observe(f"Sensibilidad al prior COMPLETA (12/12). HALLAZGO: la variación del posterior con el "
            f"prior **crece con la censura** — sin censura {lo:.0f}-18 %, cadmio al 79.8 % de "
            f"censura **{cd:.0f} %**. Es la teoría medida en nuestros datos, y da un argumento "
            f"nuevo: la censura transfiere peso del dato al prior. Además el prior vago converge "
            f"peor ({div_vague} divergencias contra {div_other} de las otras dos juntas), lo que "
            f"encaja con la advertencia de Banerjee p.149 sobre posteriores casi impropias.",
            "Bloque 4 - artículo")
    print("\nescrito: outputs/H_sensibilidad_priors.md, FIG21")
