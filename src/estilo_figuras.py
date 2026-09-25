"""H2.0 — the single place where figure style lives. No colour is defined anywhere else.

The rules below are not taste, they are the reasons a reviewer can or cannot read the figure:

1. One figure, one message, and the message is the title — an affirmative sentence saying what to
   see. If that sentence cannot be written, the figure does not know what it is for.
2. One colour per entity across the whole manuscript, colour-blind safe in every pair. Colour
   follows the entity, never the plotting order.
3. Never two vertical axes.
4. Sequential colour for magnitude, diverging with a neutral centre for differences, never rainbow.
5. Direct labelling at the end of a curve when there are three series or fewer; the legend stays,
   because identity must not depend on colour alone.
6. The conclusion is annotated on the figure itself.
7. Recessive grid and axes; no top or right spine.
8. Every number carries its uncertainty.
9. At most two panels, `(a)` `(b)` bold upper left.
10. Exact widths 90 / 140 / 190 mm, vector PDF plus 600 dpi TIFF, `pdf.fonttype=42`,
    `bbox_inches=None` with `constrained_layout` — `bbox="tight"` silently pushed 190 mm to 193.
11. A contact sheet at real column size is the only check that catches label collisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, RUN_ID  # noqa: E402

# --------------------------------------------------------------------------------- page geometry
MM = 1 / 25.4
FS_MIN = 8          # never annotate below this: rule 10, measured in the PDF
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM        # single, 1.5 and double column, exact

# ------------------------------------------------------------------------------- entity palette
# Verified for deuteranopia, protanopia and tritanopia in every pair, which is what scatter and
# maps need — not merely "distinguishable from the others in a row".
C_CENSORED = "#2a78d6"      # censored GP, ours
C_SUB_GP = "#eb6834"        # GP with L/2
C_KRIGING = "#1baf7a"       # ordinary kriging with L/2
C_REF = "#52514e"           # reference lines, nominal levels, zero. Never a data series.

MODEL_COLOR = {
    "censored GP": C_CENSORED,
    "GP with L/2": C_SUB_GP,
    "ordinary kriging L/2": C_KRIGING,
    "global mean with L/2": "#8c6bb1",
    "intercept (detections)": "#b0aca6",
}
MODEL_SHORT = {
    "censored GP": "censored GP",
    "GP with L/2": "GP, L/2",
    "ordinary kriging L/2": "kriging, L/2",
    "global mean with L/2": "global mean, L/2",
    "intercept (detections)": "intercept",
}
METAL_MARKER = {"Hg": "o", "Pb": "s", "As": "^", "Cd": "D", "Ba": "v",
                "Co": "P", "Sb": "X", "Ag": "*", "Bi": "<", "Ni": ">"}
METAL_LS = {"Hg": "-", "Pb": "--", "As": ":", "Cd": "-.", "Ba": (0, (3, 1, 1, 1))}

# One colour per metal, fixed across the whole manuscript (rule 2), anchored on the same three hues
# as MODEL_COLOR so a reader who has seen one figure recognises the family in the next one, instead
# of the un-branded Okabe-Ito set (`OI`) the notebook figures used on their own, disconnected from
# this module. Extended to ten entries — every analyte in METAL_MARKER — and kept distinguishable
# under deuteranopia/protanopia by alternating hue and lightness rather than hue alone.
PALETTE10 = ["#2a78d6", "#eb6834", "#1baf7a", "#8c6bb1", "#d6a72a",
            "#c94c4c", "#4ba3a3", "#8c8c39", "#b0559e", "#52514e"]
METAL_COLOR = dict(zip(["Hg", "Pb", "As", "Cd", "Ba", "Co", "Sb", "Ag", "Bi", "Ni"], PALETTE10))

# Sequential for magnitude: one hue, light to dark. Diverging for differences: neutral grey centre.
CMAP_MAG = LinearSegmentedColormap.from_list("mag", ["#f2f6fb", "#9dc3e6", "#2a78d6", "#0d2f55"])
CMAP_DIFF = LinearSegmentedColormap.from_list("diff", ["#1baf7a", "#e8e6e3", "#eb6834"])

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Rule 10 is 8 pt minimum **at final print size**, and it applies to ticks and legends too,
    # not only to body text. The audit found the previous values (7 pt ticks, 6 pt annotations)
    # right across the set; at 90 mm column width that is illegible on paper.
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": "#dedbd6", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "savefig.dpi": 600, "savefig.bbox": "standard", "figure.dpi": 140,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": mpl.cycler(color=[C_CENSORED, C_SUB_GP, C_KRIGING, C_REF]),
}


def apply_style() -> None:
    """Call after every third-party import. `import preliz` silently resets savefig.bbox to
    "tight", which crops to content and turns a declared 190 mm into 193 mm on disk."""
    plt.rcParams.update(RC)


apply_style()


# ------------------------------------------------------------------------------------- helpers
def panel_label(ax, letter: str, dx: float = -0.02, dy: float = 1.06) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontweight="bold",
            ha="left", va="bottom")


def message_title(fig_or_ax, sentence: str, **kw) -> None:
    """Rule 1. The title is what to see, not what is plotted."""
    tgt = fig_or_ax
    (tgt.suptitle if hasattr(tgt, "suptitle") else tgt.set_title)(
        sentence, ha="left", x=kw.pop("x", 0.01), **{"fontweight": "bold", **kw})


def utm_km_ticks(ax, axis: str = "both") -> None:
    """UTM northing is ~8.58e6 m: matplotlib's default `ScalarFormatter` factors that out as a
    literal "1e6" text box floating over the axis instead of proper "x10^6" notation, which reads
    as a rendering bug on the page. Format the ticks as plain kilometres instead — the real value,
    not an auto-scaled one — so nothing needs a multiplier at all.
    """
    from matplotlib.ticker import FuncFormatter
    # `ax.ticklabel_format(style="plain")` only works on the default ScalarFormatter, and raises
    # once it is replaced below — the FuncFormatter already writes plain text, so there is nothing
    # left for it to turn off.
    fmt = FuncFormatter(lambda v, _pos: f"{v / 1000:.0f}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def soft_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color="#dedbd6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def label_line_end(ax, x, y, text, color, dx: float = 0.15, **kw) -> None:
    """Rule 5. Name at the end of the curve, in the series colour, so the eye does not travel."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y), color=color, fontsize=FS_MIN,
                va="center", ha="left", annotation_clip=False, **kw)


def annotate_conclusion(ax, xy, text, xytext, color=C_REF, **kw) -> None:
    """Rule 6. The number that carries the argument, written on the figure."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=FS_MIN, color=color,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=color,
                                shrinkA=0, shrinkB=2), **kw)


def save_fig(fig, stem: str, dpi_tiff: int = 600, verify: bool = True) -> dict:
    """Vector PDF plus TIFF, RUN_ID in the PDF metadata (never printed on the figure), and the
    width that actually reached disk measured rather than assumed."""
    apply_style()
    pdf = FIG / f"{stem}.pdf"
    tif = FIG / f"{stem}.tif"
    fig.savefig(pdf, metadata={"Creator": RUN_ID, "Producer": RUN_ID})
    fig.savefig(tif, dpi=dpi_tiff, pil_kwargs={"compression": "tiff_lzw"})
    declared = fig.get_size_inches()[0] * 25.4
    info = {"stem": stem, "declared_mm": declared}
    if verify:
        from PIL import Image
        actual = Image.open(tif).size[0] / dpi_tiff * 25.4
        info["actual_mm"] = actual
        off = abs(actual - declared) > 0.6
        info["ok"] = not off
        print(f"saved: {stem} | {actual:.0f} mm | {dpi_tiff} dpi"
              + (f"  <-- CHECK, declared {declared:.0f} mm" if off else ""))
    return info


# ------------------------------------------------------------------- H2.1 the locator inset
def add_locator_inset(ax, draw, *, corner: str = "auto", size: float = 0.17,
                      pad: float = 0.015, points=None):
    """Place the country locator in the emptiest quadrant instead of on top of the data.

    In the previous run the Peru inset sat in the middle of the map, covering real ground between
    Santa Bárbara and the western contours. `draw(inset_ax)` renders whatever the inset should
    contain, so this function owns placement only.

    `points` are the data coordinates in axes fraction; if omitted the quadrant occupancy is read
    from what is already drawn on `ax`.
    """
    if points is None:
        pts = []
        for coll in ax.collections:
            off = coll.get_offsets()
            if off is not None and len(off):
                pts.append(np.asarray(off))
        for ln in ax.lines:
            if len(ln.get_xdata()):
                pts.append(np.c_[ln.get_xdata(), ln.get_ydata()])
        points = np.vstack(pts) if pts else np.empty((0, 2))
    points = np.asarray(points, float)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = {"lower left": (0.0, 0.0), "lower right": (1 - size, 0.0),
               "upper left": (0.0, 1 - size), "upper right": (1 - size, 1 - size)}

    if corner == "auto" and len(points):
        fx = (points[:, 0] - x0) / (x1 - x0)
        fy = (points[:, 1] - y0) / (y1 - y0)
        counts = {}
        for name, (cx, cy) in corners.items():
            inside = ((fx >= cx - pad) & (fx <= cx + size + pad)
                      & (fy >= cy - pad) & (fy <= cy + size + pad))
            counts[name] = int(inside.sum())
        corner = min(counts, key=counts.get)
        # A filled raster — an exceedance surface, a posterior mean — covers every corner, so a
        # corner that is empty of *points* still hides the surface underneath. Only a map with a
        # genuinely blank corner keeps the inset inside; otherwise it hangs outside the axes.
        has_raster = any(im.get_array() is not None for im in ax.images)
        if counts[corner] > 0 or has_raster:
            corner = "outside"

    if corner == "outside":
        iax = ax.inset_axes([1.04, 0.02, size * 1.15, size * 1.15 * 1.55])
    else:
        cx, cy = corners.get(corner, corners["lower left"])
        iax = ax.inset_axes([cx + pad, cy + pad, size, size * 1.9])

    draw(iax)
    iax.set_xticks([])
    iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(C_REF)
    iax.patch.set_alpha(0.92)
    return iax, corner


# ------------------------------------------------------------------- H2.5 exceedance colour bar
def exceedance_norm():
    """Magnitude scale for a probability, with the decision threshold marked at 0.5."""
    return TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)


def mark_half_on_colorbar(cb) -> None:
    """A rule across the bar and a bold 0.5 tick. No floating words.

    "decision threshold" written beside the bar collided with the bar's own axis label on every
    exceedance map, and moving it only moved the collision: the space to the right of a colorbar
    belongs to its label. What the line means goes in the caption, where there is room.
    """
    cb.ax.axhline(0.5, color="k", lw=1.4)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cb.ax.get_yticklabels()[2].set_fontweight("bold")


# ---------------------------------------------------------------------- H2.11 the contact sheet
def contact_sheet(stems=None, cols: int = 4, out: str = "_CONTACT_SHEET_F3") -> Path:
    """Rule 11. Every figure at real column size on one page. No rule replaces looking at it."""
    from PIL import Image
    stems = stems or sorted(p.stem for p in FIG.glob("FIG*.tif"))
    n = len(stems)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, stem in zip(np.atleast_1d(axes).ravel(), stems):
        try:
            ax.imshow(Image.open(FIG / f"{stem}.tif"))
        except Exception:                                           # noqa: BLE001
            ax.text(0.5, 0.5, f"{stem}\n(unreadable)", ha="center", va="center", fontsize=5)
        ax.set_title(stem.replace("_", " "), fontsize=4.5, pad=1.5)
    fig.suptitle(f"Contact sheet — {n} figures — {RUN_ID}", fontsize=8, fontweight="bold")
    fig.tight_layout()
    p = FIG / f"{out}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"contact sheet: {p.name} ({n} figures)")
    return p


# ------------------------------------------------------------------------------ audit utilities
def audit_figures() -> "pd.DataFrame":                              # noqa: F821
    """H2.6 — widths, Type 3 fonts and panel counts measured from the files on disk."""
    import pandas as pd
    from PIL import Image
    rows = []
    for tif in sorted(FIG.glob("FIG*.tif")):
        pdf = tif.with_suffix(".pdf")
        im = Image.open(tif)
        mm = im.size[0] / im.info["dpi"][0] * 25.4
        raw = pdf.read_bytes() if pdf.exists() else b""
        rows.append({"figure": tif.stem, "width_mm": round(mm, 1),
                     "width_ok": int(round(mm)) in (90, 140, 190),
                     "dpi": int(im.info["dpi"][0]),
                     "has_pdf": pdf.exists(),
                     "type3": b"/Type3" in raw,
                     "run_id_in_pdf": RUN_ID.encode() in raw})
    return pd.DataFrame(rows)
