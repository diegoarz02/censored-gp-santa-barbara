"""FIGSERRA -- redibuja las 11 figuras del articulo al estandar grafico de SERRA (Springer
Nature). SOLO redibujo: ningun modelo se
reajusta, ninguna prediccion nueva, todos los arrays/idata/CSV ya existen en disco. No se
sobrescribe ninguna figura de `figuras/` (la version en espanol de main_FINAL_v13.tex) -- todo
sale a `figuras/serra/`.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "_src"))

import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import estilo_figuras as ef  # noqa: E402
import estilo_serra as es  # noqa: E402
import mapas_f3 as M  # noqa: E402
import masking_worker as mw  # noqa: E402
import sbmodel as sb  # noqa: E402
from config import DATA, FIG, OUT  # noqa: E402

RESULTS = []


def record(info, width_mm):
    info["width_mm_target"] = width_mm
    RESULTS.append(info)
    return info


# =============================================================================================
# Fig1 <- FIG1_sampling_design_and_mercury (174 mm: dos paneles lado a lado necesitan el ancho
# completo para que la leyenda y la barra de color queden legibles a 8-9 pt).
# =============================================================================================
def fig1():
    from fig1_control import north_arrow, scale_bar

    ECA_HG_AGRI = 6.6
    final = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    workings = pd.read_csv(DATA / "final" / "mine_workings.csv")
    loc1 = final.drop_duplicates("location_id")
    hg = final[final.analyte == "Hg"].set_index("location_id")

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.58),
                             constrained_layout=True)

    ax = axes[0]
    for st, mk, col, lab in [("potential_interest", "o", es.C_CENSORED,
                              "Potential interest (simple)"),
                             ("background", "s", es.C_KRIGING, "Background (composite)"),
                             ("unlabelled", "^", es.C_REF, "Unlabelled")]:
        s = loc1[loc1.stratum == st]
        ax.scatter(s.easting / 1000, s.northing / 1000, marker=mk, s=11, facecolor=col,
                  edgecolor="k", linewidth=0.2, label=lab, zorder=3)
    w = workings[workings.feature_type == "waste_dump"]
    ax.scatter(w.easting / 1000, w.northing / 1000, marker="x", s=16, c="0.25", linewidth=0.8,
              label="Waste dump", zorder=4)
    ax.set_xlabel("Easting UTM 18S (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2,
             handletextpad=0.4, borderaxespad=0, fontsize=es.FS_LEGEND)
    scale_bar(ax, 1.0, pos=(0.06, 0.03))
    north_arrow(ax, pos=(0.88, 0.86), length=0.06)
    es.panel_label(ax, "a")

    ax = axes[1]
    v = hg.loc[loc1.location_id, "value"].to_numpy()
    over = v > ECA_HG_AGRI
    sc = ax.scatter(loc1.easting / 1000, loc1.northing / 1000, c=np.log10(v), cmap="viridis",
                    s=13, edgecolor="k", linewidth=0.2, zorder=3)
    ax.scatter(loc1.easting[over] / 1000, loc1.northing[over] / 1000, marker="o", s=38,
              facecolor="none", edgecolor="#D55E00", linewidth=0.7, zorder=4,
              label=f"Above ECA {ECA_HG_AGRI:g} mg kg$^{{-1}}$ (n = {int(over.sum())})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"log$_{10}$ Hg (mg kg$^{-1}$)", fontsize=es.FS_LABEL)
    cb.ax.tick_params(labelsize=es.FS_TICK)
    ax.set_xlabel("Easting UTM 18S (km)")
    ax.set_ylabel("Northing UTM 18S (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, borderaxespad=0,
             fontsize=es.FS_LEGEND)
    scale_bar(ax, 1.0, pos=(0.06, 0.03))
    north_arrow(ax, pos=(0.88, 0.86), length=0.06)
    es.panel_label(ax, "b")

    info = es.save_fig_serra(fig, "Fig1", 174)
    plt.close(fig)
    info["n_over_eca"] = int(over.sum())
    return record(info, 174)


# =============================================================================================
# Fig2 <- FIG27_censura_espacial (174 mm: panel b tiene 6 cajas, necesita ancho completo).
# =============================================================================================
def fig2():
    CENSORED_METALS = ["Cd", "Sb", "Ag"]
    df = pd.read_csv(DATA / "final" / "santa_barbara_soil.csv")
    loc = df.drop_duplicates("location_id").set_index("location_id").sort_index()
    D = M.load()
    e0, e1, n0, n1 = D["extent"]

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.52),
                             constrained_layout=True)

    ax = axes[0]
    if D["hill"] is not None:
        ax.imshow(D["hill"], cmap="gray", extent=(e0, e1, n0, n1), origin="lower",
                  alpha=0.22, zorder=1, interpolation="bilinear")
    gcd = df[df.analyte == "Cd"].set_index("location_id").loc[loc.index]
    cen_mask = gcd["censored"].to_numpy(bool)
    ax.scatter(loc.easting[~cen_mask], loc.northing[~cen_mask], s=16, facecolor=es.C_CENSORED,
              edgecolor="k", linewidth=0.3, zorder=4, label=f"Detected (n={int((~cen_mask).sum())})")
    ax.scatter(loc.easting[cen_mask], loc.northing[cen_mask], s=16, facecolor=es.C_REF,
              edgecolor="k", linewidth=0.3, zorder=3, marker="x",
              label=f"Censored (n={int(cen_mask.sum())})")
    ax.set_xlim(e0, e1); ax.set_ylim(n0, n1); ax.set_aspect("equal")
    ax.set_xlabel("Easting UTM 18S (km)"); ax.set_ylabel("Northing UTM 18S (km)")
    ef.utm_km_ticks(ax, explicit_ticks=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
             fontsize=es.FS_LEGEND)
    es.panel_label(ax, "a")

    ax = axes[1]
    # SERRA-R1 T6, segundo intento: el primero (mas espacio + salto de linea) seguia colisionando
    # ("detecteddcensored") porque las etiquetas de 2 lineas por caja no caben a 174mm con 6 cajas.
    # Fix real: el estado (detected/censored) ya esta codificado por color igual que en el panel
    # (a) -- no hace falta repetirlo como texto en cada caja. El eje x solo lleva el metal (1
    # etiqueta por par, centrada), y una leyenda unica abajo declara que azul=detected,
    # gris=censored, igual que la leyenda del panel (a).
    positions, group_centers, metal_labels, data, colors = [], [], [], [], []
    pos = 0
    for a in CENSORED_METALS:
        p0 = pos
        for estado, col in [("detected", es.C_CENSORED), ("censored", es.C_REF)]:
            vals = df[(df.analyte == a) & (df.censored == (estado == "censored"))]["dist_any_working_m"].dropna()
            positions.append(pos); data.append(vals.to_numpy()); colors.append(col)
            pos += 1
        group_centers.append((p0 + pos - 1) / 2)
        metal_labels.append(a)
        pos += 1.0
    bp = ax.boxplot(data, positions=positions, widths=0.75, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("k")
    ax.set_xticks(group_centers); ax.set_xticklabels(metal_labels, fontsize=es.FS_LABEL)
    ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    ax.set_ylabel("distance to nearest mine working (m)")
    ax.grid(True, axis="y", color="#dedbd6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    h_status = [mpatches.Patch(facecolor=es.C_CENSORED, alpha=0.55, edgecolor="k", label="Detected"),
               mpatches.Patch(facecolor=es.C_REF, alpha=0.55, edgecolor="k", label="Censored")]
    ax.legend(handles=h_status, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
             ncol=2, fontsize=es.FS_LEGEND)
    es.panel_label(ax, "b")

    info = es.save_fig_serra(fig, "Fig2", 174)
    plt.close(fig)
    return record(info, 174)


# =============================================================================================
# Fig3 <- FIG30_plate_diagram (84 mm: un solo diagrama compacto, sin leyenda ancha).
# =============================================================================================
def fig3():
    from c2_plate_diagram import (EDGES, GLOBAL, GLOBAL_LABEL, GLOBAL_OBSERVED, PLATE,
                                  PLATE_DETERMINISTIC, PLATE_LABEL, PLATE_OBSERVED, draw_edge,
                                  draw_node)

    es.apply_style_serra()
    fig, ax = plt.subplots(figsize=(es.W1_SERRA, es.W1_SERRA * 1.0), constrained_layout=True)
    pos = {**GLOBAL, **PLATE}
    for a, b in EDGES:
        draw_edge(ax, pos, a, b)
    for k, xy in GLOBAL.items():
        draw_node(ax, xy, GLOBAL_LABEL[k], observed=(k in GLOBAL_OBSERVED))
    for k, xy in PLATE.items():
        draw_node(ax, xy, PLATE_LABEL[k], observed=(k in PLATE_OBSERVED),
                 deterministic=(k in PLATE_DETERMINISTIC))

    xs = [x for x, y in PLATE.values()]; ys = [y for x, y in PLATE.values()]
    pad = 0.55
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor="none", edgecolor="#3a3835", linewidth=1.1,
                                        zorder=1))
    ax.text(x0 + 0.12, y0 + 0.12, "$s = 1 \\ldots N$\n$N = 114$", ha="left", va="bottom",
           fontsize=es.FS_TICK, color="#3a3835")

    h = [
        mlines.Line2D([], [], marker="o", ms=9, mfc="white", mec=es.C_CENSORED, mew=1.1, lw=0,
                     label="latent"),
        mlines.Line2D([], [], marker="o", ms=9, mfc="#d9d5d0", mec="#52514e", mew=0.9, lw=0,
                     label="observed"),
        mlines.Line2D([], [], marker="o", ms=10, mfc="white", mec=es.C_CENSORED, mew=1.6, lw=0,
                     label="deterministic"),
    ]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.03), frameon=False,
             ncol=1, fontsize=es.FS_LEGEND)
    ax.set_xlim(0.2, 6.8); ax.set_ylim(0.3, 6.0)
    ax.set_aspect("equal"); ax.axis("off")
    info = es.save_fig_serra(fig, "Fig3", 84)
    plt.close(fig)
    return record(info, 84)


# =============================================================================================
# Fig4 <- FIG5b_prior_predictive_concentrations (174 mm: 5 histogramas superpuestos + rug plot +
# leyenda de 5 entradas no caben legibles a 1 columna).
# No se reajusta ningun modelo: la prior predictive es simulacion directa del prior (sin MCMC,
# sin verosimilitud), con las MISMAS escalas calibradas ya publicadas (sb.SD_BETA/SIGMA/BETA0).
# El texto "Ticks below the axis are the observed values" sale de la figura (va al pie de
# figura). Escala sin cambios: log10.
# =============================================================================================
def _stats_invgamma_rvs(alpha, beta, rng):
    return beta / rng.gamma(alpha, 1.0)


def _prior_predictive_concentrations(ctx, analyte, sd_beta, sd_sigma, sd_beta0, ell_params,
                                     n_sim=400, seed=20260906):
    xy, cov, is_bg = ctx["xy"], ctx["cov"], ctx["is_bg"]
    g = ctx["df"][ctx["df"].analyte == analyte].set_index("location_id").loc[ctx["loc"].index]
    value = g["value"].to_numpy(float)
    censored = g["censored"].to_numpy(bool)
    y_bar = float(np.mean(np.log(value[~censored])))
    rng = np.random.default_rng(seed)
    out = np.empty((n_sim, len(xy)))
    for s in range(n_sim):
        beta0 = rng.normal(y_bar, sd_beta0)
        beta = rng.normal(0, sd_beta, size=cov.shape[1])
        b_bg = rng.normal(0, sd_beta)
        ell = np.array([_stats_invgamma_rvs(ell_params["alpha"], ell_params["beta"], rng),
                        _stats_invgamma_rvs(ell_params["alpha"], ell_params["beta"], rng)])
        sigma_tot = abs(rng.normal(0, sd_sigma))
        rho = rng.beta(2, 2)
        eta, sn = sigma_tot * np.sqrt(rho), sigma_tot * np.sqrt(1 - rho)
        a = xy / ell
        K = eta ** 2 * sb._matern52(((a[:, None] - a[None]) ** 2).sum(-1), ell)
        K[np.diag_indices(len(xy))] += 1e-8
        f = np.linalg.cholesky(K) @ rng.standard_normal(len(xy))
        out[s] = beta0 + cov @ beta + b_bg * is_bg.astype(float) + f + rng.standard_normal(len(xy)) * sn
    return np.exp(out)


def fig4():
    PHYSICAL_MAX = 1e6
    MAIN_METALS = ["Hg", "Pb", "As", "Ba", "Cd"]
    ctx = mw._context()
    ell_params, _, _ = sb.ell_prior_params(ctx["xy"])
    pp_draws = {a: _prior_predictive_concentrations(ctx, a, sb.SD_BETA, sb.SD_SIGMA, sb.SD_BETA0,
                                                     ell_params)
               for a in MAIN_METALS}

    es.apply_style_serra()
    fig, ax = plt.subplots(figsize=(es.W2_SERRA, es.W2_SERRA * 0.42), constrained_layout=True)
    for k, a in enumerate(MAIN_METALS):
        c = np.log10(pp_draws[a].ravel())
        ax.hist(c, bins=60, histtype="step", density=True, color=es.METAL_COLOR[a], lw=1.1,
               label=f"{a} prior")
    ax.axvline(np.log10(PHYSICAL_MAX), color="k", lw=1.0, ls="-.",
              label="Physical bound (10$^6$ mg kg$^{-1}$)")
    for k, a in enumerate(MAIN_METALS):
        o = np.log10(ctx["df"][ctx["df"].analyte == a]["value"].dropna())
        ax.plot(np.sort(o), np.full(len(o), -0.05 - 0.035 * k), "|", color=es.METAL_COLOR[a],
               ms=3, mew=0.7)
    ax.set_ylim(bottom=-0.24)
    ax.set_xlabel(r"log$_{10}$ concentration (mg kg$^{-1}$)")
    ax.set_ylabel("Prior predictive density")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=3,
             fontsize=es.FS_LEGEND)
    info = es.save_fig_serra(fig, "Fig4", 174)
    plt.close(fig)
    return record(info, 174)


# =============================================================================================
# Fig5 <- FIG29_trazas_no_convergencia (174 mm: 2 paneles lado a lado). Reutiliza los idata YA
# AJUSTADOS en outputs/a2_stress_test/ (checkpoint con run_id vigente) -- sin refit. Quita los
# titulos de panel ("GP with L/2 (baseline)", "Censored GP") -- van al pie. eta -> $\sigma_f$.
# =============================================================================================
def fig5():
    import a2_baseline_no_convergencia as a2

    idata_std, _ = a2.fit_one("GP with L/2", tune=2000, target_accept=0.99, tag="baseline_std")
    idata_cens, _ = a2.fit_one("censored GP", tune=2000, target_accept=0.99, tag="censored_std")

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.42),
                             constrained_layout=True)
    labels_moved_to_caption = []
    for ax, idata_i, label, letter in [(axes[0], idata_std, "GP with L/2 (baseline)", "a"),
                                       (axes[1], idata_cens, "Censored GP", "b")]:
        var = "eta" if "eta" in idata_i.posterior else list(idata_i.posterior.data_vars)[0]
        vals = idata_i.posterior[var].values
        for c in range(vals.shape[0]):
            ax.plot(vals[c], lw=0.5, alpha=0.8, color=es.C_CENSORED)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\sigma_f$" if var == "eta" else r"$\sigma_n$" if var == "sigma_n" else var)
        ax.grid(True, axis="y", color="#dedbd6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        es.panel_label(ax, letter)
        labels_moved_to_caption.append(f"({letter}) {label}")
    info = es.save_fig_serra(fig, "Fig5", 174)
    plt.close(fig)
    info["caption_text"] = labels_moved_to_caption
    return record(info, 174)


# =============================================================================================
# Fig6 <- FIG28_coverage_three_models (174 mm: 2 leyendas -- modelo y metal -- necesitan el ancho
# completo). Reutiliza figuras/TABLE_H6_MAIN_con_kriging.csv tal cual, sin recomputar nada.
# Accesibilidad: cada MODELO ahora tambien tiene su propio estilo de linea, no solo color.
# =============================================================================================
def fig6():
    METALS = ["As", "Hg", "Pb"]
    full = pd.read_csv(FIG / "TABLE_H6_MAIN_con_kriging.csv")

    es.apply_style_serra()
    fig, ax = plt.subplots(figsize=(es.W2_SERRA, es.W2_SERRA * 0.60), constrained_layout=True)
    for model in ["censored GP", "GP with L/2", "ordinary kriging L/2"]:
        for metal in METALS:
            s = (full[(full.model == model) & (full.metal == metal)]
                .groupby("level")["picp95"].mean().sort_index())
            if s.empty:
                continue
            ax.plot(s.index * 100, s.values, ls=es.MODEL_LINESTYLE[model], lw=1.2, alpha=0.9,
                   color=es.MODEL_COLOR[model], marker=es.METAL_MARKER[metal], ms=4.4,
                   mec="k", mew=0.3)
    h_model = [mlines.Line2D([], [], color=es.MODEL_COLOR[m], ls=es.MODEL_LINESTYLE[m], lw=1.6,
                             label=es.MODEL_SHORT[m])
              for m in ["censored GP", "GP with L/2", "ordinary kriging L/2"]]
    h_metal = [mlines.Line2D([], [], color="0.35", marker=es.METAL_MARKER[m], ms=4.6, lw=0,
                             label=m) for m in METALS]
    leg1 = ax.legend(handles=h_model, loc="upper center", bbox_to_anchor=(0.26, -0.16),
                     frameon=False, fontsize=es.FS_LEGEND, title="model",
                     title_fontsize=es.FS_LEGEND)
    ax.add_artist(leg1)
    ax.legend(handles=h_metal, loc="upper center", bbox_to_anchor=(0.76, -0.16), frameon=False,
             fontsize=es.FS_LEGEND, title="metal", title_fontsize=es.FS_LEGEND, ncol=1)
    ax.axhline(0.95, color="k", lw=1.0, ls="--")
    ax.text(81, 0.955, "nominal 0.95", fontsize=es.FS_TICK, color=es.C_REF, ha="right", va="bottom")
    ax.set_xlabel("imposed censoring (%)")
    ax.set_ylabel("95% interval coverage")
    ax.set_xticks([20, 40, 60, 80])
    ax.grid(True, axis="y", color="#dedbd6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    info = es.save_fig_serra(fig, "Fig6", 174)
    plt.close(fig)
    return record(info, 174)


# =============================================================================================
# Fig7 <- FIG26_gradiente_real_vs_impuesto (174 mm: muchas etiquetas de metal con lineas guia
# necesitan espacio). Reutiliza score_folds() (lee checkpoints de pliegues ya existentes, sin
# refit) y outputs/h6_deep/*.json (ya en disco). Accesibilidad: linea solida/discontinua por
# modelo ademas del color.
# =============================================================================================
def fig7():
    import glob
    import json

    from h41_analisis import GRADIENT, MODELS, score_folds

    real = score_folds()
    deep = pd.DataFrame([json.load(open(f, encoding="utf-8"))
                        for f in glob.glob(str(OUT / "h6_deep" / "*.json"))])
    deep = deep[deep.get("error").isna()] if "error" in deep else deep
    imp = (deep.groupby(["level", "model"])["picp95"].mean().reset_index()
          .rename(columns={"level": "frac"}))
    imp["pct_censored"] = 100 * imp["frac"]

    es.apply_style_serra()
    fig, ax = plt.subplots(figsize=(es.W2_SERRA, es.W2_SERRA * 0.56), constrained_layout=True)
    for model in MODELS:
        c = es.MODEL_COLOR[model]
        ls = es.MODEL_LINESTYLE[model]
        s = imp[imp.model == model].sort_values("pct_censored")
        ax.plot(s["pct_censored"], s["picp95"], ls=ls, lw=1.4, color=c, alpha=0.7, zorder=2)
        ax.scatter(s["pct_censored"], s["picp95"], s=26, facecolor="white", edgecolor=c,
                  linewidth=1.3, zorder=3, label=f"{es.MODEL_SHORT[model]} · imposed")
        r = real[real.model == model].sort_values("pct_censored")
        ax.scatter(r["pct_censored"], r["picp95"], s=50, marker="D", facecolor=c,
                  edgecolor="k", linewidth=0.4, zorder=4, label=f"{es.MODEL_SHORT[model]} · real")
    ax.axhline(0.95, color="k", lw=1.0, ls="--")
    ax.text(12, 0.955, "nominal 0.95", fontsize=es.FS_TICK, color=es.C_REF, ha="left", va="bottom")

    real_cens = real[real.model == "censored GP"].sort_values("pct_censored").reset_index(drop=True)
    xr = real_cens["pct_censored"].max() - real_cens["pct_censored"].min()
    yr = real_cens["picp95"].max() - real_cens["picp95"].min()
    xs = real_cens["pct_censored"].to_numpy()
    ys = real_cens["picp95"].to_numpy() * (xr / max(yr, 1e-6))
    groups, used = [], set()
    for i in range(len(real_cens)):
        if i in used:
            continue
        g = [i]
        for j in range(i + 1, len(real_cens)):
            if j in used:
                continue
            if abs(xs[j] - xs[i]) < 0.12 * xr:
                g.append(j)
        used.update(g)
        groups.append(g)
    FAN = [(0, 11), (16, 5), (-16, -18), (16, -24), (-16, 20)]
    for g in groups:
        for slot, idx in enumerate(g):
            r_ = real_cens.iloc[idx]
            x_, y_ = r_["pct_censored"], r_["picp95"]
            dx, dy = FAN[slot % len(FAN)]
            leader = len(g) > 1
            ax.annotate(r_["metal"], (x_, y_), xytext=(dx, dy), textcoords="offset points",
                       fontsize=es.FS_TICK, ha="center", va="center", color=es.C_REF,
                       arrowprops=(dict(arrowstyle="-", lw=0.5, color=es.C_REF, shrinkA=0,
                                       shrinkB=2) if leader else None))
    ax.set_xlabel("censoring (%)")
    ax.set_ylabel("95% interval coverage")
    ax.set_xlim(10, 100)
    ax.grid(True, axis="y", color="#dedbd6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2,
             fontsize=es.FS_LEGEND)
    info = es.save_fig_serra(fig, "Fig7", 174)
    plt.close(fig)
    return record(info, 174)


# =============================================================================================
# Fig8 <- FIG35_sbc_ranks (174 mm: 2 paneles). Reutiliza figuras/TABLE_C1_sbc_ranks.csv (200
# replicas ya calculadas), sin ninguna simulacion nueva. eta -> $\sigma_f$.
# =============================================================================================
def fig8():
    rank_df = pd.read_csv(FIG / "TABLE_C1_sbc_ranks.csv")
    n_total = int(rank_df["n_draws_total"].iloc[0])

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.46),
                             constrained_layout=True)
    label_map = {"eta": r"$\sigma_f$", "sigma_n": r"$\sigma_n$"}
    for ax, p, letter in zip(axes, ["eta", "sigma_n"], ["a", "b"]):
        ranks = rank_df[p].to_numpy() / n_total
        ax.hist(ranks, bins=10, range=(0, 1), color=es.C_CENSORED, edgecolor="k", linewidth=0.4,
               alpha=0.85)
        ax.axhline(len(ranks) / 10, color="k", lw=1.0, ls="--")
        ax.set_xlabel(f"normalised rank, {label_map[p]}")
        ax.set_ylabel("replicates")
        ax.grid(True, axis="y", color="#dedbd6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        es.panel_label(ax, letter)
    info = es.save_fig_serra(fig, "Fig8", 174)
    plt.close(fig)
    return record(info, 174)


# =============================================================================================
# Fig9/Fig10 <- FIG17_Hg/Cd_mean_and_uncertainty (174 mm: 2 paneles lado a lado, mapa + barra).
# Reutiliza M.base_axes (misma furniture de mapa: hillshade, desmonteras, Huancavelica, escala,
# norte, ticks en km) -- sin reajustar nada. "red line = ECA x" sale de la barra de color: la
# linea del ECA se declara en una leyenda aparte, sin nombrar el color.
# =============================================================================================
def _mean_and_uncertainty_serra(D, analyte, stem):
    mp, sh = D["mp"], D["shape"]
    e0, e1, n0, n1 = D["extent"]
    med = mp[f"{analyte}_median"].reshape(sh)
    sdy = mp[f"{analyte}_sd_y"].reshape(sh)
    eca = M.ECA_AGRI[analyte]

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.60),
                             constrained_layout=True)
    frac = None
    for ax, arr, lab, cmap, letter in [
            (axes[0], med, f"posterior median {analyte} (mg kg$^{{-1}}$)", es.CMAP_MAG, "a"),
            (axes[1], sdy, "predictive sd (log units)", "cividis", "b")]:
        M.base_axes(D, ax, hill_alpha=0.18, scale_x=0.05)
        kw = dict(extent=(e0, e1, n0, n1), origin="lower", zorder=2, interpolation="bilinear")
        if letter == "a":
            from matplotlib.colors import LogNorm
            im = ax.imshow(arr, cmap=cmap, norm=LogNorm(vmin=max(arr.min(), 1e-3), vmax=arr.max()),
                          alpha=0.85, **kw)
        else:
            im = ax.imshow(arr, cmap=cmap, alpha=0.85, **kw)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label(lab, fontsize=es.FS_LABEL)
        cb.ax.tick_params(labelsize=es.FS_TICK)
        if letter == "a":
            try:
                cb.ax.axhline(eca, color="k", lw=1.4)
            except Exception:                                       # noqa: BLE001
                pass
        frac = M.extrapolation_overlay(ax, D)
        es.panel_label(ax, letter)
        ax.tick_params(labelsize=es.FS_TICK)
        if letter == "b":
            ax.set_ylabel("")

    axes[0].scatter(D["loc"].easting, D["loc"].northing, s=4, facecolor="none",
                    edgecolor="k", linewidth=0.25, zorder=7)
    h_eca = mlines.Line2D([], [], color="k", lw=1.4, label=f"ECA {eca:g} mg kg$^{{-1}}$")
    h_extrap = mpatches.Patch(facecolor="white", edgecolor=es.C_REF, hatch="////",
                              label=f"extrapolation ({100 * frac:.1f} %)")
    axes[0].legend(handles=[h_eca], loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="#cfccc7", fontsize=es.FS_LEGEND)
    axes[1].legend(handles=[h_extrap], loc="upper right", frameon=True, framealpha=0.92,
                  edgecolor="#cfccc7", fontsize=es.FS_LEGEND)
    info = es.save_fig_serra(fig, stem, 174)
    plt.close(fig)
    info["extrap_frac"] = frac
    return info


def fig9():
    D = M.load()
    return record(_mean_and_uncertainty_serra(D, "Hg", "Fig9"), 174)


def fig10():
    r50 = dict(np.load(OUT / "flip_ld_cd" / "maps_50m.npz"))
    D = M.load()
    D_ld = dict(D)
    mp_ld = dict(D["mp"])
    mp_ld["Cd_median"] = r50["median"]
    mp_ld["Cd_sd_y"] = r50["sd_y"]
    mp_ld["Cd_p_agri"] = r50["p_agri"]
    mp_ld["Cd_p_resid"] = r50["p_resid"]
    D_ld["mp"] = mp_ld
    return record(_mean_and_uncertainty_serra(D_ld, "Cd", "Fig10"), 174)


# =============================================================================================
# Fig11 <- FIG18_Cd_exceedance + FIG18_Cd_exceedance_residential, UNIDAS en (a) agricola / (b)
# residencial de una sola figura (pedido explicito de SERRA). Mismo contorno 0.5, misma escala
# de color 0-1, mismos numeros (38.2/6.8 %) -- reutiliza outputs/flip_ld_cd/maps_50m.npz.
# =============================================================================================
def fig11():
    r50 = dict(np.load(OUT / "flip_ld_cd" / "maps_50m.npz"))
    D = M.load()
    D_ld = dict(D)
    mp_ld = dict(D["mp"])
    mp_ld["Cd_median"] = r50["median"]
    mp_ld["Cd_sd_y"] = r50["sd_y"]
    mp_ld["Cd_p_agri"] = r50["p_agri"]
    mp_ld["Cd_p_resid"] = r50["p_resid"]
    D_ld["mp"] = mp_ld
    e0, e1, n0, n1 = D_ld["extent"]
    sh = D_ld["shape"]

    es.apply_style_serra()
    fig, axes = plt.subplots(1, 2, figsize=(es.W2_SERRA, es.W2_SERRA * 0.62),
                             constrained_layout=True)
    pct_above = {}
    for ax, land_use, key, thr, letter in [
            (axes[0], "agricultural", "p_agri", M.ECA_AGRI["Cd"], "a"),
            (axes[1], "residential", "p_resid", M.ECA_RESID["Cd"], "b")]:
        p = D_ld["mp"][f"Cd_{key}"].reshape(sh)
        M.base_axes(D_ld, ax, hill_alpha=0.18)
        im = ax.imshow(p, cmap=es.CMAP_MAG, vmin=0, vmax=1, extent=(e0, e1, n0, n1),
                      origin="lower", zorder=2, alpha=0.88, interpolation="bilinear")
        ax.contour(np.linspace(e0, e1, sh[1]), np.linspace(n0, n1, sh[0]), p, levels=[0.5],
                  colors=["k"], linewidths=1.0, zorder=4)
        M.extrapolation_overlay(ax, D_ld)
        ax.scatter(D_ld["loc"].easting, D_ld["loc"].northing, s=4, facecolor="none",
                  edgecolor="k", linewidth=0.25, zorder=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label(f"P(Cd > {thr:g} mg kg$^{{-1}}$)", fontsize=es.FS_LABEL)
        cb.ax.tick_params(labelsize=es.FS_TICK)
        ef.mark_half_on_colorbar(cb)
        es.panel_label(ax, letter)
        if letter == "b":
            ax.set_ylabel("")
        pct_above[land_use] = float(100 * (p > 0.5).mean())

    info = es.save_fig_serra(fig, "Fig11", 174)
    plt.close(fig)
    info["pct_above_half"] = pct_above
    return record(info, 174)


def main():
    t0 = time.time()
    results = {}
    for name, fn in [("Fig1", fig1), ("Fig2", fig2), ("Fig3", fig3), ("Fig4", fig4),
                     ("Fig5", fig5), ("Fig6", fig6), ("Fig7", fig7), ("Fig8", fig8),
                     ("Fig9", fig9), ("Fig10", fig10), ("Fig11", fig11)]:
        t1 = time.time()
        results[name] = fn()
        print(f"  {name}: {time.time()-t1:.1f}s")
    print(f"\nFIGSERRA: 11 figuras en {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    main()
