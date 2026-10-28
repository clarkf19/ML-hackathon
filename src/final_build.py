"""Final build: morning.py (complete CNNs inside the GSEP LightGBM meta, partial ones mixed) + a logistic-regression
stacker on the same inputs averaged in at weight LRW (validated: +0.0012 nested, every fold up) + optional bias step.
usage: final_build.py NAME   -> submissions/NAME.csv  (env as morning.py: GBDT, GSEP, EXCL, META_NEST, META_LR; LRW, BIAS=1)"""
import sys, os, re, subprocess
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
PY = [r'C:\Users\clark\AppData\Local\Programs\Python\Python311\python.exe', '-X', 'utf8']
NAME = sys.argv[1]; S = os.path.join(ROOT, 'src'); LRW = float(os.environ.get('LRW', 0.2))
run = lambda args, env={}: subprocess.run(PY + args, capture_output=True, text=True, env={**os.environ, **env}).stdout
o = run([os.path.join(S, 'morning.py'), f'{NAME}_m']); print(o, flush=True)
full = eval(re.search(r'complete \(into meta\): (\[.*?\])', o).group(1))
o = run([os.path.join(S, 'meta_lr.py'), f'{NAME}_m_base', 'B3pl2g', 'B3wg', 'B3plwg'] + full, {'SAVE': '1', 'LRC': '0.05'}); print(o, flush=True)
pm = os.path.join(D, f'pte_{NAME}_m.npy')
if not os.path.exists(pm): pm = os.path.join(D, f'pte_{NAME}_m_base.npy')   # no partial CNNs: the meta base is the blend
P = (1 - LRW) * np.load(pm) + LRW * np.load(os.path.join(D, f'pte_lr_{NAME}_m_base.npy'))
np.save(os.path.join(D, f'pte_{NAME}.npy'), P); print(f'LR stacker at {LRW}:', write_sub(P.argmax(1), f'{NAME}.csv'))
if os.environ.get('BIAS') == '1':   # bias tuned on the same mixture of OOF predictions
    F = (1 - LRW) * np.load(os.path.join(D, f'oofd_final_{NAME}_m_base.npy')) + LRW * np.load(os.path.join(D, f'oofd_lr_{NAME}_m_base.npy'))
    np.save(os.path.join(D, f'oofd_final_{NAME}.npy'), F)
    print(run([os.path.join(S, 'bias_final.py'), NAME, NAME, f'{NAME}b.csv'], {'MINGAIN': '0'}))
