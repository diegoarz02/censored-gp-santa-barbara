import sys,os,re,json,urllib.request
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cat=json.load(open(os.path.join(root,"data/raw/oefa/_catalogo_streams.json")))
arr=cat if isinstance(cat,list) else cat.get('results',cat)
link={x['guid']:(x.get('link') or '') for x in arr}
outdir=os.path.join(root,"data/raw/oefa")
def comp(g):
    u=g.upper()
    return 'suelo' if 'SUELO' in u else 'sedimento' if 'SEDIM' in u else 'agua' if 'AGUA' in u else 'otro'
for g in sys.argv[1:]:
    dv=link.get(g,'').replace('http://','https://')
    if not dv: print(g,"SIN link en catalogo"); continue
    try: html=urllib.request.urlopen(dv,timeout=30).read().decode('utf-8','replace')
    except Exception as e: print(g,"ERR dataview",e); continue
    m=re.search(r'/datasets/(\d+)-([^"\']+?)\.download/',html)
    if not m: print(g,"sin .download"); continue
    dl="https://datosabiertos.oefa.gob.pe"+m.group(0)
    out=os.path.join(outdir,f"{comp(g)}__{g}.xlsx")
    try:
        data=urllib.request.urlopen(dl,timeout=90).read(); open(out,'wb').write(data)
        print(f"{g}  {comp(g)}  bytes={len(data)}  xlsx={data[:2]==b'PK'}")
    except Exception as e: print(g,"ERR download",e)
