import glob, os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OEFA=os.path.join(ROOT,"data","fase1","oefa"); OUT=os.path.join(ROOT,"outputs","inventario")
KEEP=["Nombre de la Evaluación","Componente ambiental","Tipo de análisis","Procedencia de la Muestra",
      "Nombre del punto","Este","Norte","Altitud","Zona","Fecha","Valor","Parámetro","Unidad de medida",
      "Descripción de ubicación","Número de informe"]
todo=[]
for f in sorted(glob.glob(os.path.join(OEFA,"suelo__*.xlsx"))):
    d=pd.read_excel(f); d.columns=[c.strip() for c in d.columns]
    if "Nombre de la Evaluación" not in d.columns:
        print("SKIP (sin columna evaluación):",os.path.basename(f)); continue
    for c in KEEP:
        if c not in d.columns: d[c]=np.nan
    d=d[KEEP].copy(); d["archivo"]=os.path.basename(f); todo.append(d)
    print("OK",os.path.basename(f),len(d),flush=True)
d=pd.concat(todo,ignore_index=True)
v=d["Valor"].astype(str).str.strip()
d["cens"]=v.str.startswith("<")
d["num"]=pd.to_numeric(v.where(~d["cens"]),errors="coerce")
d["LD"]=pd.to_numeric(v.where(d["cens"]).str.lstrip("<"),errors="coerce")
d["par"]=d["Parámetro"].astype(str).str.strip()
d["ev"]=d["Nombre de la Evaluación"].astype(str).str.strip()
d.to_csv(os.path.join(OUT,"suelo_consolidado.csv.gz"),index=False,compression="gzip")
print("\nguardado suelo_consolidado.csv.gz:",len(d),"filas,",d.ev.nunique(),"evaluaciones")
