"""Auditoria de idioma de las figuras -- corrida 3, item 3 del cierre.

Extrae TODO el texto de cada FIG*.pdf y lo clasifica en espanol/ingles/mixto por deteccion de
marcadores: tildes y enye (inequivocos de espanol), y una lista de palabras funcionales frecuentes
de cada idioma. No decide "correcto o no" -- solo mide, para que la correccion sea dirigida.
"""
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import pdfplumber  # noqa: E402

from config import FIG, RUN_ID  # noqa: E402

ES_MARKERS = {"el", "la", "los", "las", "de", "del", "en", "con", "por", "para", "que", "es",
             "un", "una", "y", "sin", "sobre", "al", "no", "si", "más", "mas", "entre", "cada",
             "dos", "tres", "censura", "censurado", "muestreo", "réplicas", "replicas",
             "cuadrícula", "cuadricula", "intervalo", "modelo", "cobertura", "metal", "nivel",
             "componente", "componentes", "posterior", "mapa", "localización", "localizacion",
             "distancia", "punto", "puntos", "profundidad", "según", "segun", "sí", "vs"}
EN_MARKERS = {"the", "of", "in", "with", "for", "and", "is", "at", "on", "from", "to", "vs",
             "coverage", "model", "level", "component", "components", "posterior", "map",
             "location", "distance", "point", "points", "depth", "censoring", "sampling",
             "replicate", "replicates", "grid", "interval"}
ACCENTED = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")


def classify(text: str) -> str:
    words = re.findall(r"[a-záéíóúñA-ZÁÉÍÓÚÑ]+", text.lower())
    es_hits = sum(1 for w in words if w in ES_MARKERS)
    en_hits = sum(1 for w in words if w in EN_MARKERS)
    has_accent = bool(ACCENTED.search(text))
    if not words:
        return "sin_texto"
    if es_hits == 0 and en_hits == 0 and not has_accent:
        return "ambiguo"          # solo numeros, simbolos matematicos, unidades
    if (es_hits > 0 or has_accent) and en_hits == 0:
        return "espanol"
    if en_hits > 0 and es_hits == 0 and not has_accent:
        return "ingles"
    return "mixto"


def main():
    rows = []
    for f in sorted(glob.glob(str(FIG / "FIG*.pdf"))):
        stem = Path(f).stem
        with pdfplumber.open(f) as pdf:
            text = " ".join(w["text"] for w in pdf.pages[0].extract_words() if w["text"].strip())
        lang = classify(text)
        rows.append({"figure": stem, "idioma_detectado": lang, "sample": text[:140]})
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", RUN_ID)
    df.to_csv(FIG / "TABLE_S11_figure_language.csv", index=False, encoding="utf-8")
    print(df["idioma_detectado"].value_counts().to_string())
    print("\n=== español o mixto (necesitan revision) ===")
    need = df[df.idioma_detectado.isin(["espanol", "mixto"])]
    for _, r in need.iterrows():
        print(f"  {r['figure']:45s} [{r['idioma_detectado']:8s}] {r['sample']}")
    print(f"\nescrito: TABLE_S11_figure_language.csv ({len(df)} figuras, {len(need)} a revisar)")


if __name__ == "__main__":
    main()
