"""V14 (2026-11-13) -- Wilcoxon de rangos con signo sin pseudorreplicacion.

En h6_deep las replicas r y r+5 comparten mascara (cuantil determinista) y bloque de prueba
(block = r % 5); solo cambia la semilla MCMC. La unidad independiente es (metal, bloque).
Se promedian las dos semillas por bloque y se prueba |PICP95-0.95| baseline - censurado.
Lectura pura de outputs/h6_deep/*.json: cero ajustes nuevos. Prueba exacta (sin scipy).
"""
import json, glob, itertools, collections
D = collections.defaultdict(dict)
for f in glob.glob('outputs/h6_deep/*.json'):
    d = json.load(open(f)); D[(d['metal'], d['level'], d['replicate'], d['block'])][d['model']] = d['picp95']

def signed_rank_exact(x):
    x = [v for v in x if v != 0]; n = len(x); a = [abs(v) for v in x]
    o = sorted(range(n), key=lambda i: a[i]); r = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and a[o[j+1]] == a[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2+1
        i = j+1
    W = sum(rk for rk, v in zip(r, x) if v > 0)
    p1 = sum(sum(rk for rk, s in zip(r, sg) if s) >= W-1e-9 for sg in itertools.product([0,1], repeat=n)) / 2**n
    return n, min(1.0, 2*p1)

for lvl in [0.2, 0.4, 0.6, 0.8]:
    for base in ['GP with L/2', 'ordinary kriging L/2']:
        diffs = []
        for m in ['As', 'Hg', 'Pb']:
            for b in range(5):
                ks = [(m, lvl, r, b) for r in range(10) if (m, lvl, r, b) in D]
                c = sum(abs(D[k]['censored GP']-0.95) for k in ks)/len(ks)
                s = sum(abs(D[k][base]-0.95) for k in ks)/len(ks)
                diffs.append(s-c)
        n, p = signed_rank_exact(diffs)
        print(f"nivel {lvl:.0%} vs {base}: gana censurado {sum(d>0 for d in diffs)}/15 (pares no empatados {n}), p bilateral = {p:.2e}")
