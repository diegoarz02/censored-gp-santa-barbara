import sys, glob, os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OEFA = os.path.join(ROOT, "data", "fase1", "oefa")
OUT  = os.path.join(ROOT, "outputs", "inventario")
os.makedirs(OUT, exist_ok=True)

def procesar(f, matriz):
    d = pd.read_excel(f)
    d.columns = [c.strip() for c in d.columns]
    need = ["Nombre de la Evaluación","Componente ambiental","Tipo de análisis",
            "Procedencia de la Muestra","Nombre del punto","Este","Norte","Fecha",
            "Valor","Parámetro","Unidad de medida"]
    for c in need:
        if c not in d.columns: d[c] = np.nan
    v = d["Valor"].astype(str).str.strip()
    d["cens"] = v.str.startswith("<")
    d["num"]  = pd.to_numeric(v.where(~d["cens"]), errors="coerce")
    d["LD"]   = pd.to_numeric(v.where(d["cens"]).str.lstrip("<"), errors="coerce")
    d["par"]  = d["Parámetro"].astype(str).str.strip()
    d["ev"]   = d["Nombre de la Evaluación"].astype(str).str.strip()
    d["tipo"] = d["Tipo de análisis"].astype(str).str.strip()
    d["proc"] = d["Procedencia de la Muestra"].astype(str).str.strip()
    d["fecha"]= pd.to_datetime(d["Fecha"], errors="coerce")
    d["matriz"] = matriz
    d["archivo"] = os.path.basename(f)
    rows = []
    for (ev, par, tipo), g in d.groupby(["ev","par","tipo"]):
        n = len(g)
        if n < 30: continue
        det = g["num"].dropna()
        xy = g.dropna(subset=["Este","Norte"]).drop_duplicates(["Este","Norte"])
        ld = g["LD"].dropna()
        rows.append(dict(
            matriz=matriz, evaluacion=ev, parametro=par, tipo_analisis=tipo,
            archivo=os.path.basename(f), n=n, pts_unicos=len(xy),
            pct_cens=round(100*g["cens"].mean(),1), n_det=len(det),
            min_det=det.min() if len(det) else np.nan,
            med_det=det.median() if len(det) else np.nan,
            max_det=det.max() if len(det) else np.nan,
            sd_log_det=round(float(np.log(det[det>0]).std(ddof=1)),3) if (det>0).sum()>2 else np.nan,
            factor_rango=round(float(det.max()/det.min()),2) if len(det)>1 and det.min()>0 else np.nan,
            LD_reportado=ld.mode().iloc[0] if len(ld) else np.nan,
            min_det_sobre_LD=round(float(det.min()/ld.mode().iloc[0]),2) if len(det) and len(ld) and ld.mode().iloc[0]>0 else np.nan,
            unidad=str(g["Unidad de medida"].iloc[0]),
            fecha_min=str(g["fecha"].min())[:10], fecha_max=str(g["fecha"].max())[:10],
            n_informes=g["Número de informe"].nunique() if "Número de informe" in g else np.nan,
        ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    matriz = sys.argv[1]
    files = sorted(glob.glob(os.path.join(OEFA, f"{matriz}__*.xlsx")))
    todo = []
    for f in files:
        try:
            r = procesar(f, matriz)
            todo.append(r); print(f"OK {os.path.basename(f)}: {len(r)} combinaciones", flush=True)
        except Exception as e:
            print(f"ERR {os.path.basename(f)}: {e}", flush=True)
    if todo:
        t = pd.concat(todo, ignore_index=True)
        t.to_csv(os.path.join(OUT, f"inventario_{matriz}.csv"), index=False, encoding="utf-8")
        print(f"\nGuardado inventario_{matriz}.csv con {len(t)} filas")
