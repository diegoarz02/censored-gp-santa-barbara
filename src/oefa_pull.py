import sys, os, urllib.request
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K=""
for l in open(os.path.join(root,".env"),encoding="utf-8"):
    if l.startswith("OEFA_KEY="): K=l.strip().split("=",1)[1]
guid=sys.argv[1]; out=sys.argv[2]
base=f"https://api.datosabiertos.oefa.gob.pe/api/v2/datastreams/{guid}/data.csv/?auth_key={K}&limit=999"
off=0; header=None; rows=[]; pages=0; err=None; prev_first=None
while True:
    url=base+f"&offset={off}"; data=None
    for t in range(4):
        try: data=urllib.request.urlopen(url,timeout=25).read().decode("utf-8","replace"); break
        except Exception as e: err=str(e)
    if data is None: print("ERR",guid,"off",off,err); break
    lines=data.splitlines()
    body=lines[1:] if lines else []
    if header is None and lines: header=lines[0]
    if not body: break
    if body[0]==prev_first: break          # API ignoró offset -> evita loop
    prev_first=body[0]
    rows+=body; pages+=1; off+=len(body)
    if off>300000: print("SAFETY STOP",guid); break
open(out,"w",encoding="utf-8").write((header or "")+"\n"+"\n".join(rows)+"\n")
print(f"{guid}  total_rows={len(rows)}  pages={pages}")
