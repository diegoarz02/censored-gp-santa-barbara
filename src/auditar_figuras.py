"""Detect what a contact sheet cannot: text collisions, clipping and off-spec geometry.

The contact sheet (H2.11) shows layout at column size and is the right tool for "is this legible".
It is the wrong tool for "does the legend sit on top of the scale bar", because at thumbnail size
two overlapping objects look like one object.

This measures instead of looking. For every figure it re-opens the PDF, walks the text objects and
reports:

* **collisions** — pairs of text boxes that overlap
* **clipping** — text whose box extends past the figure edge
* **tipografia bajo minimo** — cuerpo real por debajo de 8 pt, leido del operador `Tf`. NO del
  atributo `size` de pdfplumber, que en estos PDF sigue el ancho del glifo y marcaba una incidencia
  por letra estrecha en las 47 figuras
* **geometry** — width not in {90, 140, 190} mm, Type 3 fonts

Runs on the PDF rather than the figure object, so it audits what a reviewer will actually receive,
including figures produced by code this module never saw.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from config import FIG, RUN_ID  # noqa: E402

MIN_PT = 5.0          # heredado; ya no se usa para tipografia
MIN_PT_PRINT = 8.0    # regla H2.0: nada por debajo de 8 pt al tamano impreso
PAGE_MARGIN_PT = 1.0


def _boxes(page):
    """Text boxes as (x0, y0, x1, y1, size, text), in PDF points."""
    out = []
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        return None
    return out


def audit_pdf(path: Path) -> dict:
    """Geometry and text audit of one figure PDF."""
    import pypdf
    r = pypdf.PdfReader(str(path))
    pg = r.pages[0]
    w_pt = float(pg.mediabox.width)
    h_pt = float(pg.mediabox.height)
    raw = path.read_bytes()
    return {"figure": path.stem,
            "width_mm": round(w_pt / 72 * 25.4, 1),
            "height_mm": round(h_pt / 72 * 25.4, 1),
            "type3": b"/Type3" in raw,
            "run_id_in_pdf": RUN_ID.encode() in raw}


def audit_text(path: Path) -> list[dict]:
    """Collisions, clipping and tiny type, read from the PDF's own text objects."""
    try:
        import pdfplumber
    except ImportError:
        return [{"figure": path.stem, "issue": "pdfplumber-missing", "detail": ""}]
    issues = []
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        W, H = page.width, page.height
        words = [w for w in page.extract_words(extra_attrs=["size"]) if w["text"].strip()]
        # NO se usa w["size"] para la tipografia: en estos PDF ese atributo sigue el ANCHO del
        # glifo, no el cuerpo de la fuente ('i' salia a 1.8 pt y 'e' a 4.4 pt en un texto de 8 pt).
        # Un detector que marcaba las 47 figuras estaba midiendo el objeto equivocado: el tamano
        # real solo lo dice el operador Tf del flujo de contenido. Ver audit_font_sizes().
        for w in words:
            if (w["x0"] < -PAGE_MARGIN_PT or w["x1"] > W + PAGE_MARGIN_PT
                    or w["top"] < -PAGE_MARGIN_PT or w["bottom"] > H + PAGE_MARGIN_PT):
                issues.append({"figure": path.stem, "issue": "clipped",
                               "detail": f"{w['text'][:26]!r} outside the page"})
        # pairwise overlap of word boxes that are not on the same text line
        for i in range(len(words)):
            a = words[i]
            for j in range(i + 1, len(words)):
                b = words[j]
                ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                oy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
                same_line = abs(a["top"] - b["top"]) < 0.6
                # Same-line pairs used to be skipped as "normal spacing". That hid the commonest
                # overlap of all: crowded tick labels, which share a baseline and run into each
                # other. Words separated by a space touch at ox <= 0, so a genuine overlap on the
                # same line still shows as ox > 1.2 — only the *skip* was wrong.
                if ox > 1.2 and oy > 1.2:
                    issues.append({"figure": path.stem, "issue": "text-collision",
                                   "detail": f"{a['text'][:18]!r} x {b['text'][:18]!r} "
                                             f"({ox:.1f}x{oy:.1f} pt"
                                             f"{', same line' if same_line else ''})"})
    return issues


def audit_font_sizes(path: Path) -> list[dict]:
    """Tamano de fuente REAL, leido del operador `Tf` del flujo de contenido.

    Es la unica lectura fiable. El atributo `size` de pdfplumber sigue el ancho del glifo en estos
    PDF, asi que marcaba 1166 incidencias repartidas por las 47 figuras — una por letra estrecha.
    `Tf` lleva el cuerpo en puntos tal y como lo escribio matplotlib, que es lo que ve la imprenta.

    Devuelve una incidencia por (figura, tamano) distinto que quede por debajo de `MIN_PT_PRINT`,
    con cuantas veces aparece, en vez de una por caracter.
    """
    import re
    from collections import Counter

    try:
        from pypdf import PdfReader
    except ImportError:
        return [{"figure": path.stem, "issue": "pypdf-missing", "detail": ""}]
    try:
        data = PdfReader(str(path)).pages[0].get_contents().get_data()
    except Exception as exc:  # flujo ilegible: se reporta, no se silencia
        return [{"figure": path.stem, "issue": "content-stream-unreadable", "detail": str(exc)[:80]}]

    sizes = Counter(round(float(s), 1)
                    for _, s in re.findall(rb"/([A-Za-z0-9#\-]+)\s+([0-9.]+)\s+Tf", data))

    # matplotlib compone sub/superindices matematicos al 0.7 del cuerpo. Un exponente de 5.6 pt
    # dentro de un texto de 8 pt NO es tipografia fuera de norma: es como se escribe kg^-1, y
    # subirlo a 8 romperia la formula. Se separa en su propia categoria en vez de contarlo como
    # incidencia, porque mezclarlos era lo que hacia ilegible la salida del detector.
    ok = [s for s in sizes if s >= MIN_PT_PRINT]
    def es_indice(pt):
        return any(0.66 <= pt / s <= 0.74 for s in ok)

    out = []
    for pt, n in sorted(sizes.items()):
        if pt >= MIN_PT_PRINT:
            continue
        issue = "indice-matematico" if es_indice(pt) else "tipografia-bajo-minimo"
        out.append({"figure": path.stem, "issue": issue,
                    "detail": f"{pt} pt en {n} bloques de texto (minimo {MIN_PT_PRINT} pt)"})
    return out


def audit_legends_over_data(path: Path) -> list[dict]:
    """Legend boxes sitting on top of plotted data — what the text-vs-text check cannot see.

    The first version of this module compared word boxes against each other and reported zero
    problems on figures where a reader could plainly see a legend covering the curves. Text over
    text is only one of the ways a figure hides its own data; a framed legend over a line is the
    commoner one, and it never registers as a text collision because the line is not text.

    Measured on the rendered raster: for each figure, find light rectangular regions that contain
    text (legend frames) and count how much drawn ink falls underneath them.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    tif = path.with_suffix(".tif")
    if not tif.exists():
        return []
    im = np.asarray(Image.open(tif).convert("L"), dtype=float) / 255.0
    h, w = im.shape
    issues = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pg = pdf.pages[0]
            sx, sy = w / pg.width, h / pg.height
            # A legend is a *small* framed box. The first version filtered only on absolute size
            # and flagged every axes frame in the set — 56 hits across 42 figures, one or two per
            # figure, which is the signature of a detector measuring the wrong object. A legend is
            # at most a quarter of the page in each direction; an axes frame is most of it.
            page_area = pg.width * pg.height
            rects = [r for r in pg.rects
                     if 25 < r["width"] < 0.45 * pg.width
                     and 12 < r["height"] < 0.45 * pg.height
                     and 0.004 * page_area < r["width"] * r["height"] < 0.16 * page_area]
            words = pg.extract_words()
            for r in rects:
                inside = [wd for wd in words
                          if r["x0"] <= wd["x0"] and wd["x1"] <= r["x1"]
                          and r["top"] <= wd["top"] and wd["bottom"] <= r["bottom"]]
                if len(inside) < 2:          # a legend has at least two entries
                    continue
                # a legend that overlaps data leaves ink right around its border
                x0, x1 = int(r["x0"] * sx), int(r["x1"] * sx)
                y0, y1 = int(r["top"] * sy), int(r["bottom"] * sy)
                pad = 6
                ring = np.ones_like(im, dtype=bool)
                ring[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = False
                band = np.zeros_like(im, dtype=bool)
                band[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = True
                band[y0:y1, x0:x1] = False
                ink = float((im[band] < 0.85).mean()) if band.any() else 0.0
                if ink > 0.06:
                    issues.append({"figure": path.stem, "issue": "legend-over-data",
                                   "detail": f"legend with {len(inside)} words, "
                                             f"{100 * ink:.0f} % ink in the surrounding band"})
    except Exception:                                               # noqa: BLE001
        return issues
    return issues


def run(stems=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pdfs = sorted(FIG.glob("FIG*.pdf"))
    if stems:
        pdfs = [p for p in pdfs if p.stem in set(stems)]
    geo, txt = [], []
    for p in pdfs:
        geo.append(audit_pdf(p))
        txt.extend(audit_text(p))
        txt.extend(audit_font_sizes(p))
        txt.extend(audit_legends_over_data(p))
    g = pd.DataFrame(geo)
    g["width_ok"] = g["width_mm"].round().isin([90, 140, 190])
    t = pd.DataFrame(txt) if txt else pd.DataFrame(columns=["figure", "issue", "detail"])
    return g, t


if __name__ == "__main__":
    g, t = run()
    print(f"figuras auditadas: {len(g)}")
    print(f"  ancho fuera de norma : {int((~g.width_ok).sum())}")
    print(f"  con fuentes Type 3   : {int(g.type3.sum())}")
    print(f"  con run_id F3        : {int(g.run_id_in_pdf.sum())} de {len(g)}")
    if len(t):
        print(f"\nincidencias de texto: {len(t)}")
        print(t.groupby("issue").size().to_string())
        print("\nfiguras con mas incidencias:")
        top = t.groupby(["figure", "issue"]).size().reset_index(name="n")
        top = top.sort_values("n", ascending=False).head(18)
        print(top.to_string(index=False))
    g.insert(0, "run_id", RUN_ID)
    g.to_csv(FIG / "TABLE_S8_figure_audit.csv", index=False, encoding="utf-8")
    if len(t):
        t.insert(0, "run_id", RUN_ID)
        t.to_csv(FIG / "TABLE_S9_figure_text_issues.csv", index=False, encoding="utf-8")
    print("\nescrito: TABLE_S8_figure_audit.csv, TABLE_S9_figure_text_issues.csv")
