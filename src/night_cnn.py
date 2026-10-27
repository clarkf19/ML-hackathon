import sys, os; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
y = y_train(); fo = folds(); S = LABELS.index('scintillating_star')
F = np.load(os.path.join(D, 'oofd_final_base8_3.npy')); G = np.load(os.path.join(D, 'oofd_A_rsswp_PL4.npy'))
TAGS = 'P1g,E1g,E2g,E3g,E4g,P3g,B3pl2g,B3wg,B3plwg'.split(',')
def f1s(yy, p): from sklearn.metrics import f1_score; return f1_score(yy == S, p == S)
for k in range(4):
    v = fo == k; row = [f'f{k} base {mf1(y[v], F[v].argmax(1)):.4f}/{f1s(y[v], F[v].argmax(1)):.3f}', f'gbdt {mf1(y[v], G[v].argmax(1)):.4f}/{f1s(y[v], G[v].argmax(1)):.3f}']
    cs = []
    for t in TAGS:
        p = os.path.join(D, f'oofdpart_{t}_f{k}.npy')
        if os.path.exists(p):
            c = np.load(p); row.append(f'{t} {mf1(y[v], c.argmax(1)):.4f}/{f1s(y[v], c.argmax(1)):.3f}')
            if t[0] in 'PE': cs.append(c)
    a = np.mean(cs, 0); row.append(f'PREavg {mf1(y[v], a.argmax(1)):.4f}/{f1s(y[v], a.argmax(1)):.3f}')
    print(' | '.join(row))
