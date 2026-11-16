"""FIGSERRA — estilo de figura para SERRA (Stochastic Environmental Research and Risk Assessment,
Springer Nature). Modulo separado de `estilo_figuras.py` a proposito: la version en espanol de las
figuras (90/140/190 mm, la que usa `main_FINAL_v13.tex`) no se toca ni se sobrescribe -- este
modulo escribe exclusivamente a `figuras/serra/`.

Reglas de SERRA aplicadas aqui:
- Ancho final exacto: 84 mm (una columna) o 174 mm (ancho completo), alto maximo 234 mm. La figura
  se crea DIRECTAMENTE a ese tamano con `figsize` en pulgadas -- nunca se reescala despues.
- Tipografia: Arial/Helvetica en todo el texto (incluido mathtext), ticks a 8 pt, etiquetas de eje
  y leyendas a 8-9 pt, etiquetas de panel (a)/(b) a 9-10 pt en negrita. Nada fuera de [8, 12] pt.
  Fuentes embebidas (`pdf.fonttype`/`ps.fonttype` = 42).
- Sin titulos dentro de la figura: ni `suptitle` ni titulos de panel descriptivos, solo (a)/(b).
- Notacion identica al texto: simbolos ($\\sigma_f$, $\\sigma_n$, etc.), no nombres de variable de
  codigo (`eta`, `sigma_n`).
- Accesibilidad: cada modelo con su propio estilo de linea ademas de su color, paleta verificable
  en escala de grises.
- Salida: PDF vectorial + EPS + TIFF 600 dpi RGB 8 bits, por figura, en `figuras/serra/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJ, RUN_ID  # noqa: E402
import estilo_figuras as ef  # noqa: E402

# --------------------------------------------------------------------------------- page geometry
MM = 1 / 25.4
SERRA_FIG = PROJ / "figuras" / "serra"
SERRA_FIG.mkdir(parents=True, exist_ok=True)

W1_SERRA = 84 * MM     # one column, 3.3071 in
W2_SERRA = 174 * MM    # full width, 6.8504 in
MAX_H_SERRA = 234 * MM  # 9.2126 in

FS_TICK = 8
FS_LABEL = 9
FS_LEGEND = 8
FS_PANEL = 10

# Same hues as estilo_figuras.py (rule 2 of H2.0: one colour per entity across the whole
# manuscript) -- SERRA only changes typography/layout/accessibility, not the colour identity a
# reader already learned from the Spanish-language figures.
C_CENSORED, C_SUB_GP, C_KRIGING, C_REF = ef.C_CENSORED, ef.C_SUB_GP, ef.C_KRIGING, ef.C_REF
MODEL_COLOR = ef.MODEL_COLOR
MODEL_SHORT = ef.MODEL_SHORT
METAL_MARKER = ef.METAL_MARKER
METAL_COLOR = ef.METAL_COLOR
CMAP_MAG = ef.CMAP_MAG

# Accessibility (SERRA rule 5): each MODEL gets its own line style and marker, not colour alone --
# verified by converting to greyscale (see `grayscale_thumbnail` below).
MODEL_LINESTYLE = {
    "censored GP": "-",
    "GP with L/2": "--",
    "ordinary kriging L/2": ":",
    "global mean with L/2": "-.",
    "intercept (detections)": (0, (1, 1)),
}
MODEL_MARKER = {
    "censored GP": "o",
    "GP with L/2": "s",
    "ordinary kriging L/2": "^",
    "global mean with L/2": "D",
    "intercept (detections)": "v",
}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": FS_LABEL, "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.2, "lines.markersize": 3.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": "#dedbd6", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "savefig.dpi": 600, "savefig.bbox": "standard", "figure.dpi": 140,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": mpl.cycler(color=[C_CENSORED, C_SUB_GP, C_KRIGING, C_REF]),
}


def apply_style_serra() -> None:
    plt.rcParams.update(RC)


apply_style_serra()


def panel_label(ax, letter: str, dx: float = -0.02, dy: float = 1.06) -> None:
    """(a), (b), ... only -- SERRA forbids a descriptive title inside the figure."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", ha="left", va="bottom")


def save_fig_serra(fig, stem: str, width_mm: float, verify: bool = True) -> dict:
    """PDF (vector, embedded fonts) + EPS + TIFF (600 dpi, RGB 8-bit), all to figuras/serra/.

    The figure must already be AT final size (`figsize` set in inches before this is called) --
    this function never rescales. `declared_mm`/`actual_mm` are both measured from what reached
    disk, the same discipline `estilo_figuras.py::save_fig` already uses for the Spanish figures.
    """
    apply_style_serra()
    pdf = SERRA_FIG / f"{stem}.pdf"
    eps = SERRA_FIG / f"{stem}.eps"
    tif = SERRA_FIG / f"{stem}.tif"
    fig.savefig(pdf, metadata={"Creator": RUN_ID, "Producer": RUN_ID})
    eps_ok = True
    try:
        fig.savefig(eps)
    except Exception as exc:                                        # noqa: BLE001
        eps_ok = False
        print(f"  EPS FALLO para {stem}: {exc!r}")
    fig.savefig(tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    # TIFF del backend Agg sale RGBA; SERRA pide RGB de 8 bits -- se convierte explicitamente.
    from PIL import Image
    im = Image.open(tif)
    if im.mode != "RGB":
        im.convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
        im = Image.open(tif)

    declared = fig.get_size_inches() * 25.4
    info = {"stem": stem, "declared_w_mm": float(declared[0]), "declared_h_mm": float(declared[1]),
            "eps_ok": eps_ok}
    if verify:
        actual_w = im.size[0] / 600 * 25.4
        actual_h = im.size[1] / 600 * 25.4
        info["actual_w_mm"] = actual_w
        info["actual_h_mm"] = actual_h
        off = abs(actual_w - width_mm) > 0.6
        info["width_ok"] = not off
        info["height_ok"] = actual_h <= 234.6
        print(f"saved: {stem} | {actual_w:.0f}x{actual_h:.0f} mm | 600 dpi | eps={'ok' if eps_ok else 'FAIL'}"
              + (f"  <-- CHECK ancho, declarado {width_mm:.0f} mm" if off else ""))
    return info


def grayscale_thumbnail(stem: str, out_png: str | None = None):
    """Convierte el TIFF a escala de grises y guarda una miniatura -- la unica forma real de
    verificar accesibilidad: no basta con razonar sobre los codigos de color."""
    from PIL import Image
    tif = SERRA_FIG / f"{stem}.tif"
    im = Image.open(tif).convert("L")
    out = Path(out_png) if out_png else SERRA_FIG / f"{stem}_gray.png"
    im.save(out)
    return out
