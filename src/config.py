"""Single source of truth for run identity, paths and durability helpers.

Everything this run writes carries RUN_ID, so a number in the manuscript can always be traced to the
run that produced it. Phase 2 outputs are never deleted: a file about to be replaced is *moved* to
`archivo/fase2/` first, keeping its relative path.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------------- run identity
RUN_ID = "F4-20260923"
RUN_DESC = "corrida 3: cierra rubrica sobre F3 (bloques A-C de PROMPT_CORRIDA_3.md)"
SEED = 20260923
RUN_ID_PREV = "F3-20260908"      # para leer checkpoints/figuras de la corrida anterior sin invalidarlos

# ------------------------------------------------------------------------------------------ paths
PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs"
FIG = PROJ / "figuras"
DATA = PROJ / "data"
NB = PROJ / "notebooks"
ARCHIVE = PROJ / "archivo"
ARCHIVE_F2 = ARCHIVE / "fase2"
VAULT = Path.home() / "Documents" / "Obsidian Vault" / "Claude Diego"

for _d in (OUT, FIG, ARCHIVE, ARCHIVE_F2):
    _d.mkdir(parents=True, exist_ok=True)


def stamp_header(kind: str = "md") -> str:
    """First line of a generated report, so its provenance survives copy-paste."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"<!-- run_id: {RUN_ID} | generado: {now} -->"


# ------------------------------------------------------------------- archive-before-overwrite
def archive_if_exists(path: str | Path) -> Path | None:
    """Move an existing file to `archivo/fase2/<same relative path>` before it is replaced.

    Moving rather than copying: it costs no extra space and only files that are genuinely replaced
    end up archived, so `archivo/` is an exact record of what this run overwrote.
    """
    path = Path(path)
    if not path.is_absolute():                 # accept "outputs/x.csv" as well as a full path
        path = PROJ / path
    if not path.exists():
        return None
    rel = path.resolve().relative_to(PROJ)
    dest = ARCHIVE_F2 / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():                       # already archived once; keep the first copy
        return dest
    shutil.move(str(path), str(dest))
    return dest


# ------------------------------------------------------------------------------- atomic writing
def atomic_write(dest: str | Path, writer, retries: int = 3):
    """Write through a temporary file and rename, so a crash never leaves a truncated file.

    A half-written `.nc` or `.npz` *exists and looks complete*, and a resume would accept it. The
    rename is atomic on the same volume, Windows included. OneDrive can hold a lock on the target
    exactly when the rename fires, so `os.replace` is retried with a growing wait.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    writer(tmp)
    last = None
    for attempt in range(retries):
        try:
            os.replace(tmp, dest)
            return dest
        except PermissionError as exc:      # OneDrive sync holding the handle
            last = exc
            time.sleep(2 ** attempt)
    raise PermissionError(
        f"could not rename {tmp.name} after {retries} attempts (OneDrive sync?): {last}")


def clean_tmp(root: str | Path = None) -> int:
    """Remove `*.tmp` left behind by a crash mid-write. Run this at start-up, never mid-run."""
    root = Path(root) if root else PROJ
    n = 0
    for f in root.rglob("*.tmp"):
        if f.is_file():
            f.unlink()
            n += 1
    return n


# ------------------------------------------------------------------------------------- run state
RUNSTATE = OUT / "_RUNSTATE.json"


def state_read() -> dict:
    if RUNSTATE.exists():
        try:
            return json.loads(RUNSTATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"run_id": RUN_ID, "started": datetime.now().isoformat(timespec="seconds"),
            "milestones": {}}


def state_update(milestone: str, done: int = None, total: int = None, note: str = "") -> dict:
    s = state_read()
    m = s["milestones"].setdefault(milestone, {})
    if done is not None:
        m["done"] = done
    if total is not None:
        m["total"] = total
    if note:
        m["note"] = note
    m["last_progress"] = datetime.now().isoformat(timespec="seconds")
    atomic_write(RUNSTATE, lambda p: p.write_text(json.dumps(s, indent=2, ensure_ascii=False),
                                                  encoding="utf-8"))
    return s


def state_summary() -> str:
    s = state_read()
    if not s["milestones"]:
        return f"[{RUN_ID}] fresh start, no milestones recorded"
    lines = [f"[{RUN_ID}] resuming, state from {s.get('started', '?')}"]
    for k, v in sorted(s["milestones"].items()):
        d, t = v.get("done"), v.get("total")
        prog = f"{d} of {t}" if d is not None and t is not None else v.get("note", "")
        lines.append(f"  {k}: {prog}  (last {v.get('last_progress', '?')})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- observation log
OBS = OUT / "OBSERVACIONES.md"


def observe(text: str, section: str = None) -> None:
    """Append one timestamped observation. This file is part of the deliverable."""
    if not OBS.exists():
        OBS.write_text(f"{stamp_header()}\n\n# Observaciones de la corrida {RUN_ID}\n\n"
                       "Una línea por observación, con la hora. Decisiones tomadas y por qué,\n"
                       "supuestos, cifras que no cuadran, cosas que sorprendieron.\n\n",
                       encoding="utf-8")
    t = OBS.read_text(encoding="utf-8")
    entry = f"- **{datetime.now().strftime('%H:%M')}** — {text}\n"
    if section:
        head = f"\n## {section}\n\n"
        if head not in t:
            t = t.rstrip() + "\n" + head
        i = t.index(head) + len(head)
        j = t.find("\n## ", i)
        j = len(t) if j == -1 else j
        t = t[:j].rstrip() + "\n" + entry + t[j:]
    else:
        t = t.rstrip() + "\n" + entry
    OBS.write_text(t, encoding="utf-8")


def log_attempt(what: str, failed: str, changed: str) -> None:
    """Try, fail, try again: one line per attempt. Only three failed variants count as a blocker."""
    p = OUT / "bitacora_intentos.md"
    if not p.exists():
        p.write_text(f"{stamp_header()}\n\n# Bitácora de intentos — {RUN_ID}\n\n"
                     "| hora | qué se probó | qué pasó | qué se cambió |\n|---|---|---|---|\n",
                     encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"| {datetime.now().strftime('%H:%M')} | {what} | {failed} | {changed} |\n")
