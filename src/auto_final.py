"""Overnight automation: find CNN runs with all 5 folds, build candidate final blends (src/final_blend.py),
rank them by fake-PKS CV and write the best as submissions/cand_*.csv + logs/candidates.txt.
usage: auto_final.py [MAXCAND]"""
import sys, os, re, subprocess, itertools, time
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
PY = r'C:\Users\clark\AppData\Local\Programs\Python\Python311\python.exe'
GB = os.environ.get('GBDT', 'A_rs,A_lvl_PL,A_trk3')
tags = sorted({m.group(1) for f in os.listdir(D) for m in [re.match(r'oofdpart_(.+)_f0\.npy$', f)] if m})
complete = [t for t in tags if all(os.path.exists(os.path.join(D, f'{p}_{t}_f{k}.npy')) for p in ('oofpart', 'oofdpart', 'ptepart') for k in range(5))]
fo = folds(); y = y_train()
single = {}
for t in complete:
    o = np.zeros((len(y), 9))
    for k in range(5): o[fo == k] = np.load(os.path.join(D, f'oofdpart_{t}_f{k}.npy'))
    single[t] = mf1(y, o.argmax(1))
ranked = sorted(complete, key=lambda t: -single[t])
print('complete CNNs (fake-PKS CV):', {t: round(single[t], 4) for t in ranked}, flush=True)
cands = []
for n in range(1, min(len(ranked), 5) + 1): cands.append(ranked[:n])            # top-n by single score
if len(ranked) > 1: cands.append(ranked)                                         # everything
seen, results = set(), []
for c in cands:
    key = tuple(sorted(c))
    if key in seen: continue
    seen.add(key)
    name = 'cand_' + '_'.join(t.rstrip('g') for t in c) + '.csv'
    out = subprocess.run([PY, '-X', 'utf8', os.path.join(ROOT, 'src', 'final_blend.py'), name] + list(c), capture_output=True, text=True,
                         env={**os.environ, 'GBDT': GB}).stdout
    m = re.search(r'FINAL avg\(meta,blend\) fPKS ([0-9.]+)\s+clean ([0-9.]+)', out)
    if m: results.append((float(m.group(1)), float(m.group(2)), name, c)); print(f'{name}: fPKS {m.group(1)} clean {m.group(2)}', flush=True)
results.sort(reverse=True)
with open(os.path.join(ROOT, 'logs', 'candidates.txt'), 'w') as fh:
    fh.write(f'{time.strftime("%Y-%m-%d %H:%M")}  feature models: {GB}\n')
    for r in results: fh.write(f'fPKS {r[0]:.4f}  clean {r[1]:.4f}  submissions/{r[2]}  CNNs {r[3]}\n')
print(open(os.path.join(ROOT, 'logs', 'candidates.txt')).read())
