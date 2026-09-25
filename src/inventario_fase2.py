"""H0.0 — snapshot of what Phase 2 left on disk, before this run writes anything.

`archivo/INVENTARIO_FASE2.csv` is written exactly once and never overwritten: it is what lets the
final manifest say, with certainty, which files this run created, replaced or left untouched.
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARCHIVE, FIG, OUT, PROJ, RUN_ID  # noqa: E402

import pandas as pd  # noqa: E402

DEST = ARCHIVE / "INVENTARIO_FASE2.csv"


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def scan(roots=(OUT, FIG)) -> pd.DataFrame:
    rows = []
    for root in roots:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix == ".tmp":
                continue
            st = p.stat()
            rows.append({
                "path": p.relative_to(PROJ).as_posix(),
                "bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256(p),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if DEST.exists():
        d = pd.read_csv(DEST)
        print(f"[skip] {DEST.name} ya existe con {len(d)} ficheros; no se toca (regla H0.0.1)")
        sys.exit(0)
    df = scan()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST, index=False, encoding="utf-8")
    mb = df["bytes"].sum() / 1e6
    print(f"inventario escrito: {len(df)} ficheros, {mb:.0f} MB -> {DEST.relative_to(PROJ)}")
    print(df.groupby(df["path"].str.split("/").str[0]).agg(
        n=("path", "size"), MB=("bytes", lambda s: round(s.sum() / 1e6, 1))).to_string())
