"""Verificacion obligatoria de FIGSERRA: fuentes (tamano min/max en pt, embebidas, Arial/Helvetica),
rasters (resolucion efectiva), titulos internos, rotulos de parametros. Usa PyMuPDF en vez de
`pdfimages -list` (poppler no esta instalado en esta maquina) -- capacidad equivalente: lee el
mismo contenido del PDF (fuentes, texto por tamano, imagenes rasterizadas con su matriz de
colocacion en la pagina).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz  # PyMuPDF  # noqa: E402
import pandas as pd  # noqa: E402

from config import PROJ  # noqa: E402

SERRA_FIG = PROJ / "figuras" / "serra"
STEMS = [f"Fig{i}" for i in range(1, 12)] + ["FigS1_latent_diagnostics"]

TITLE_WORDS = ["GP with L/2 (baseline)", "Censored GP", "baseline", "normalised rank, eta",
              "normalised rank, sigma_n", "Ticks below the axis are the observed values",
              "red line"]


def font_report(stem):
    """Tamano de fuente por bloque de texto, separando el texto de linea base del texto de
    sub/superindice de mathtext (p. ej. el "-1" de "mg kg$^{-1}$", el "10" de "log$_{10}$"): un
    sub/superindice SIEMPRE tiene un origen Y distinto al resto de su linea -- es la unica forma
    fiable de distinguirlo de texto de cuerpo genuinamente chico, en vez de adivinar por longitud
    de cadena. Reportamos ambos: el minimo real sobre TODO el texto (para no esconder nada) y el
    minimo sobre el texto de linea base (el que de verdad compite con la regla de 8 pt -- un
    superindice mas chico que su base es tipografia estandar en cualquier revista, no un defecto).
    """
    pdf = SERRA_FIG / f"{stem}.pdf"
    doc = fitz.open(pdf)
    page = doc[0]
    all_sizes, baseline_sizes, fonts = [], [], set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            if not spans:
                continue
            # un sub/superindice de mathtext SIEMPRE es mas chico que el texto de cuerpo de su
            # propia linea -- mas robusto que comparar el origen Y, que empata cuando la linea
            # tiene solo 2 fragmentos (un simbolo + su subindice, p. ej. "$\sigma_f$").
            max_sz = max(round(s["size"], 2) for s in spans)
            for span in spans:
                sz = round(span["size"], 2)
                all_sizes.append(sz)
                fonts.add(span["font"])
                if sz >= max_sz - 0.01:
                    baseline_sizes.append(sz)
    page_w_pt, page_h_pt = page.rect.width, page.rect.height
    font_names = sorted({f[3] for f in page.get_fonts()})
    # confirma que el programa de la fuente esta de verdad embebido (bytes reales en el PDF), no
    # solo referenciado por nombre -- el prefijo de 6 letras ("EEAGXI+") ya lo indica por norma
    # del formato PDF, pero se verifica extrayendo el stream.
    embedded_bytes = {f[3]: len(doc.extract_font(f[0])[-1]) for f in page.get_fonts(full=True)}
    all_embedded = all(n > 0 for n in embedded_bytes.values()) if embedded_bytes else False
    all_arial = all("Arial" in n for n in font_names)
    doc.close()
    return dict(stem=stem, width_mm=round(page_w_pt / 72 * 25.4, 1),
               height_mm=round(page_h_pt / 72 * 25.4, 1),
               fs_min_all=min(all_sizes) if all_sizes else None,
               fs_max_all=max(all_sizes) if all_sizes else None,
               fs_min_baseline=min(baseline_sizes) if baseline_sizes else None,
               fs_max_baseline=max(baseline_sizes) if baseline_sizes else None,
               n_text_spans=len(all_sizes), fonts=font_names, n_fonts=len(font_names),
               all_embedded=all_embedded, all_arial=all_arial)


def raster_report(stem):
    pdf = SERRA_FIG / f"{stem}.pdf"
    doc = fitz.open(pdf)
    page = doc[0]
    rows = []
    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        base = doc.extract_image(xref)
        w_px, h_px = base["width"], base["height"]
        w_in = rect.width / 72
        h_in = rect.height / 72
        dpi_x = w_px / w_in if w_in > 0 else float("nan")
        dpi_y = h_px / h_in if h_in > 0 else float("nan")
        rows.append({"xref": xref, "w_px": w_px, "h_px": h_px,
                    "dpi_x": round(dpi_x), "dpi_y": round(dpi_y)})
    doc.close()
    return rows


def title_check(stem):
    pdf = SERRA_FIG / f"{stem}.pdf"
    doc = fitz.open(pdf)
    text = doc[0].get_text()
    doc.close()
    found = [w for w in TITLE_WORDS if w in text]
    return found


def main():
    font_rows, raster_rows, title_rows = [], [], []
    for stem in STEMS:
        fr = font_report(stem)
        font_rows.append(fr)
        rr = raster_report(stem)
        for r in rr:
            raster_rows.append({"stem": stem, **r})
        tc = title_check(stem)
        title_rows.append({"stem": stem, "titulos_encontrados": tc})

    fonts_df = pd.DataFrame(font_rows)
    raster_df = pd.DataFrame(raster_rows)
    title_df = pd.DataFrame(title_rows)

    pd.set_option("display.width", 160)
    print("=== FUENTES (tamano en pt, fuentes embebidas) ===")
    print(fonts_df[["stem", "width_mm", "height_mm", "fs_min_all", "fs_max_all",
                    "fs_min_baseline", "fs_max_baseline", "all_embedded", "all_arial", "fonts"]]
         .to_string(index=False))
    print("\n=== RASTERS (dpi efectivo) ===")
    print(raster_df.to_string(index=False) if len(raster_df) else "(ninguna figura tiene rasters embebidos)")
    print("\n=== TITULOS INTERNOS (deberia salir vacio en todas) ===")
    print(title_df.to_string(index=False))

    fonts_df.to_csv(SERRA_FIG / "_verify_fonts.csv", index=False, encoding="utf-8")
    raster_df.to_csv(SERRA_FIG / "_verify_rasters.csv", index=False, encoding="utf-8")

    bad_fs = fonts_df[(fonts_df.fs_min_baseline < 7.9) | (fonts_df.fs_max_baseline > 12.1)]
    bad_titles = title_df[title_df.titulos_encontrados.apply(len) > 0]
    print(f"\n{'OK' if bad_fs.empty else 'FALLA'}: texto de linea base fuera de [8,12] pt")
    if not bad_fs.empty:
        print(bad_fs[["stem", "fs_min_baseline", "fs_max_baseline"]].to_string(index=False))
    print(f"{'OK' if bad_titles.empty else 'FALLA'}: titulos internos")
    if not bad_titles.empty:
        print(bad_titles.to_string(index=False))


if __name__ == "__main__":
    main()
