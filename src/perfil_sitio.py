import sys, glob, os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OEFA=os.path.join(ROOT,"data","fase1","oefa")
ECA={"Arsénico":(50,140),"Plomo":(70,800),"Cadmio":(1.4,22),"Mercurio":(6.6,24)}
def cargar(matriz, patron):
    out=[]
    for f in sorted(glob.glob(os.path.join(OEFA,f"{matriz}__*.xlsx"))):
        d=pd.read_excel(f); d.columns=[c.strip() for c in d.columns]
        m=d["Nombre de la Evaluación"].astype(str).str.contains(patron,case=False,na=False,regex=True)
        if m.any(): out.append(d[m])
    if not out: return None
    d=pd.concat(out,ignore_index=True)
    d=d[d["Tipo de análisis"].astype(str).str.contains("total",case=False,na=False)]
    d=d[d["Procedencia de la Muestra"].astype(str).str.strip().str.lower()=="suelo"] if matriz=="suelo" else d
    v=d["Valor"].astype(str).str.strip()
    d["cens"]=v.str.startswith("<"); d["num"]=pd.to_numeric(v.where(~d["cens"]),errors="coerce")
    d["par"]=d["Parámetro"].astype(str).str.strip()
    return d
def perfil(nombre, matriz, patron):
    d=cargar(matriz,patron)
    if d is None or len(d)==0: print(f"\n### {nombre}: sin datos"); return
    xy=d.dropna(subset=["Este","Norte"]).drop_duplicates(["Este","Norte"])[["Este","Norte"]].astype(float).values
    zonas=d["Zona"].astype(str).unique() if "Zona" in d else []
    print(f"\n### {nombre}  [{matriz}]  puntos={len(xy)}  zonas UTM={list(zonas)[:3]}  fechas {str(pd.to_datetime(d['Fecha'],errors='coerce').min())[:10]}..{str(pd.to_datetime(d['Fecha'],errors='coerce').max())[:10]}")
    if len(xy)>3:
        D=np.sqrt(((xy[:,None]-xy[None])**2).sum(-1)); np.fill_diagonal(D,np.inf)
        ext=np.sqrt(((xy[:,None]-xy[None])**2).sum(-1)).max()
        print(f"    extensión={ext/1000:.2f} km | vecino más cercano: med={np.median(D.min(1)):.0f} m, max={D.min(1).max():.0f} m")
    for par,(ag,ex) in ECA.items():
        g=d[d["par"]==par]
        if len(g)==0: continue
        det=g["num"].dropna(); npts=g.drop_duplicates(["Este","Norte"]).shape[0]
        n_ag=int((det>ag).sum()); n_ex=int((det>ex).sum())
        print(f"    {par:12s} pts={npts:4d} cens={100*g['cens'].mean():5.1f}%  max={det.max() if len(det) else np.nan:>10.1f}  >ECA_agri={n_ag:4d} ({100*n_ag/max(len(g),1):4.1f}%)  >ECA_extr={n_ex:4d}")
if __name__=="__main__":
    for nombre,matriz,patron in [
        ("Cuajone","suelo","^Cuajone$"),
        ("San Juan / delta Upamayo (Cerro de Pasco)","suelo","San Juan y delta Upamayo"),
        ("ETNA Ventanilla-Mi Peru (2022)","suelo","ETNA"),
        ("Vigilancia Ventanilla-Mi Peru (2019)","suelo","zona industrial de Ventanilla"),
        ("Santa Barbara","suelo","unidad minera Santa Bárbara"),
        ("Cerro S.A.C","suelo","^Cerro S.A.C$"),
        ("Quiulacocha","suelo","^Quiulacocha$"),
        ("Antapaccay","suelo","unidad minera Antapaccay"),
        ("Toromocho","suelo","^Toromocho$"),
    ]:
        perfil(nombre,matriz,patron)
