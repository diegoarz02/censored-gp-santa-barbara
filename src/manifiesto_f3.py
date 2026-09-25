"""Censo final de la corrida F3 y comparación contra el inventario congelado de la fase 2.

Regla 7 de H0.0: al cerrar hay que poder decir **qué fichero es nuevo, cuál cambió y cuál
desapareció** respecto al estado que se congeló antes de empezar. Sin eso, «está todo» es una
afirmación sin respaldo.

Escribe `MANIFIESTO_F3.csv` (censo completo con hash) y `MANIFIESTO_F3.md` (el resumen legible).
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config import PROJ, RUN_ID, observe, stamp_header  # noqa: E402

# lo que no se censa: cachés, temporales y el propio archivo histórico
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", "tmp"}
SKIP_SUFFIX = {".pyc", ".pyo", ".tmp"}


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def census(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        st = p.stat()
        rows.append({
            "path": rel,
            "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                .astimezone().strftime("%Y-%m-%dT%H:%M:%S"),
            "sha256": sha256(p),
        })
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def main():
    root = PROJ
    now = census(root)
    now = now[~now.path.str.startswith("archivo/")].reset_index(drop=True)

    old = pd.read_csv(root / "archivo" / "INVENTARIO_FASE2.csv")
    old = old[~old.path.str.startswith("archivo/")]

    a, b = set(old.path), set(now.path)
    nuevos = sorted(b - a)
    idos = sorted(a - b)
    comunes = sorted(a & b)

    h_old = old.set_index("path")["sha256"].to_dict()
    h_new = now.set_index("path")["sha256"].to_dict()
    cambiados = [p for p in comunes if h_old.get(p) != h_new.get(p)]
    iguales = [p for p in comunes if h_old.get(p) == h_new.get(p)]

    now.insert(0, "run_id", RUN_ID)
    now["estado"] = now.path.map(
        lambda p: "nuevo" if p in set(nuevos) else
        ("modificado" if p in set(cambiados) else "sin cambios"))
    now.to_csv(root / "MANIFIESTO_F3.csv", index=False, encoding="utf-8")

    total_mb = now.bytes.sum() / 1e6

    def top(paths, n=12):
        s = now.set_index("path").loc[[p for p in paths if p in h_new], "bytes"]
        return s.sort_values(ascending=False).head(n)

    grandes = now.nlargest(10, "bytes")[["path", "bytes"]]
    grandes["MB"] = (grandes.bytes / 1e6).round(1)

    txt = f"""{stamp_header()}

# Manifiesto de cierre F3

Censo completo del proyecto al cerrar la corrida, comparado contra el inventario que se congeló
**antes** de empezar (`archivo/INVENTARIO_FASE2.csv`). La carpeta `archivo/` queda fuera del cómputo
en los dos lados, para no contar dos veces lo que se movió allí.

| | |
|---|---|
| Ficheros censados ahora | **{len(now)}** |
| Ficheros en el inventario de la fase 2 | {len(old)} |
| **Nuevos en F3** | **{len(nuevos)}** |
| **Modificados** (mismo nombre, hash distinto) | **{len(cambiados)}** |
| Sin cambios | {len(iguales)} |
| Ya no están (archivados o borrados) | {len(idos)} |
| Tamaño total | **{total_mb:.0f} MB** |

## Los diez ficheros más grandes

{grandes[['path', 'MB']].to_markdown(index=False)}

## Ficheros que estaban en la fase 2 y ya no están

{chr(10).join('* `' + p + '`' for p in idos) if idos else '_ninguno_'}

**Si alguno de estos no fue archivado a propósito, es una pérdida y hay que recuperarlo desde
`archivo/`.** La regla de la corrida era mover, nunca borrar.

## Cómo usar esto al auditar

`MANIFIESTO_F3.csv` tiene una fila por fichero con su hash y la columna `estado`. Para responder
«¿qué fichero no menciona ningún informe?», cruza la columna `path` contra el texto de los `.md` de
`outputs/`. Todo lo que aparezca en el censo y en ningún informe es un cabo suelto.
"""
    (root / "MANIFIESTO_F3.md").write_text(txt, encoding="utf-8")

    print(f"censados {len(now)} ficheros, {total_mb:.0f} MB")
    print(f"nuevos {len(nuevos)} · modificados {len(cambiados)} · "
          f"sin cambios {len(iguales)} · ausentes {len(idos)}")
    if idos:
        print("\nausentes:")
        for p in idos[:30]:
            print("  ", p)

    observe(f"Manifiesto de cierre: {len(now)} ficheros, {total_mb:.0f} MB. {len(nuevos)} nuevos, "
            f"{len(cambiados)} modificados, {len(idos)} ausentes respecto al inventario congelado de "
            f"la fase 2. Escrito MANIFIESTO_F3.csv con hash y columna de estado, para que la "
            f"auditoría pueda cruzar el censo contra los informes y encontrar cabos sueltos.",
            "Bloque 0 - trazabilidad")


if __name__ == "__main__":
    main()
