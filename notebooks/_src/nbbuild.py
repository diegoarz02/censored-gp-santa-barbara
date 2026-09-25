# -*- coding: utf-8 -*-
"""Convierte un .py en formato percent (# %% / # %% [markdown]) a .ipynb."""
import sys, json, io, os

def build(src_path, out_path):
    with io.open(src_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    cells, cur, kind = [], [], None
    def flush():
        if kind is None: return
        src = "\n".join(cur).strip("\n")
        if kind == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in src.split("\n"))
        if not src.strip(): return
        if kind == "markdown":
            cells.append({"cell_type":"markdown","metadata":{},"source":src.split("\n")})
        else:
            cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.split("\n")})
    for ln in lines:
        if ln.startswith("# %%"):
            flush(); cur = []
            kind = "markdown" if "[markdown]" in ln else "code"
        else:
            cur.append(ln)
    flush()
    # normalizar source a lista con \n
    for c in cells:
        s = c["source"]
        c["source"] = [x + "\n" for x in s[:-1]] + [s[-1]]
    nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3.14 (metales bayesiano)","language":"python","name":"metales-bayes"},
          "language_info":{"name":"python","version":"3.14.5","pygments_lexer":"ipython3","file_extension":".py",
          "mimetype":"text/x-python","nbconvert_exporter":"python","codemirror_mode":{"name":"ipython","version":3}}},
          "nbformat":4,"nbformat_minor":5}
    with io.open(out_path,"w",encoding="utf-8") as f:
        json.dump(nb,f,ensure_ascii=False,indent=1)
    print("OK ->", out_path, "|", len(cells), "celdas")

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
