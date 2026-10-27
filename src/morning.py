"""Morning build: every pretrained CNN run with all 5 folds goes INTO the stacked meta (final_blend.py, SEP=1);
runs with only some folds are mixed on top with one validated weight (mix_partial.py).
usage: morning.py OUTNAME   (env GBDT = feature models to average; env EXCL = comma list of CNN tags to leave out)"""
import sys, os, re, subprocess
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
PY = [r'C:\Users\clark\AppData\Local\Programs\Python\Python311\python.exe', '-X', 'utf8']
OUT = sys.argv[1]; OLD = ['B3pl2g', 'B3wg', 'B3plwg']; EXCL = set(os.environ.get('EXCL', '').split(','))
have = lambda t, k: all(os.path.exists(os.path.join(D, f'{p}_{t}_f{k}.npy')) for p in ('oofpart', 'oofdpart', 'ptepart'))
tags = sorted({m.group(1) for f in os.listdir(D) for m in [re.match(r'oofdpart_([PE]\w*g)_f\d\.npy$', f)] if m} - EXCL)
full = [t for t in tags if all(have(t, k) for k in range(5))]; part = [t for t in tags if t not in full and any(have(t, k) for k in range(5))]
print('complete (into meta):', full, '| partial (mixed):', {t: [k for k in range(5) if have(t, k)] for t in part}, flush=True)
run = lambda args, env: subprocess.run(PY + args, capture_output=True, text=True, env={**os.environ, **env}).stdout
base = f'{OUT}_base'
o = run([os.path.join(ROOT, 'src', 'final_blend.py'), base + '.csv'] + OLD + full, {})
print('\n'.join(l for l in o.splitlines() if 'FINAL' in l or 'CNN ' in l or 'fold' in l), flush=True)
if part:
    o = run([os.path.join(ROOT, 'src', 'mix_partial.py'), base, OUT + '.csv'], {'PTAGS': ','.join(part), 'ANYFOLD': '1'}); print(o, flush=True)
else:
    os.replace(os.path.join(ROOT, 'submissions', base + '.csv'), os.path.join(ROOT, 'submissions', OUT + '.csv')); print('no partial runs: base is final')
print('->', os.path.join(ROOT, 'submissions', OUT + '.csv'))
