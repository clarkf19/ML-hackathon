"""Error analysis of best GBDT (TAG) on clean and fake-PKS val + CNN fold-0 comparison. -> logs/error_analysis.txt"""
import sys; sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
from sklearn.metrics import confusion_matrix
TAG = sys.argv[1]; CNN = sys.argv[2] if len(sys.argv) > 2 else 'B'
y = y_train(); fo = folds(); tr = meta()[0]
oc = np.load(D + rf'\oofc_{TAG}.npy'); od = np.load(D + rf'\oofd_{TAG}.npy')
Fc = pd.read_csv(D + r'\fe_phys_train.csv'); Fd = pd.read_csv(D + r'\fe_phys_train_deg2.csv'); Tk = pd.read_csv(D + r'\fe_trk_train.csv')
o = open(ROOT + r'\logs\error_analysis.txt', 'w')
def P(*a): print(*a, file=o)
for nm, pr, Fx in [('clean', oc, Fc), ('fakePKS', od, Fd)]:
    p = pr.argmax(1); P(f'===== {TAG} on {nm} val: macro-F1 {mf1(y, p):.4f}')
    cm = confusion_matrix(y, p); off = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(9) for j in range(9) if i != j]
    P('top confusions (true -> pred):', sorted(off, reverse=True)[:10])
    # macro-F1 cost of each class = (1-F1)/9
    f = f1_score(y, p, average=None); P('macro-F1 lost per class:', dict(zip(LABELS, np.round((1 - f) / 9, 4))))
    err = p != y
    gl = (Fx.bb0_max > 10).values
    sl = {'glitch(bb0>10)': gl, 'no glitch': ~gl, 'missing rows>0': (Fx.n_missing_rows > 0).values, 'no missing': (Fx.n_missing_rows == 0).values}
    for g in (3, 4, 5, 6): sl[f'slew {g}'] = (tr.slew_gap_rows == g).values
    P('error rate by slice:', {k: (round(err[m].mean(), 4), int(m.sum())) for k, m in sl.items()})
    P('macro-F1 by slice:', {k: round(mf1(y[m], p[m]), 4) for k, m in sl.items()})
    conf = pr.max(1); P('accuracy by confidence decile:', np.round([1 - err[(conf >= a) & (conf < b)].mean() for a, b in zip(np.quantile(conf, np.linspace(0, .9, 10)), np.quantile(conf, np.linspace(.1, 1, 10)) + 1e-9)], 3).tolist())
# ETI: strength vs recall
e = y == 0; s = Tk.t_obj.values
bins = np.quantile(s[e], [0, .25, .5, .75, 1])
P('ETI recall by track-score quartile (clean / fPKS):', [(round(a, 1), round((oc.argmax(1)[e & (s >= a) & (s <= b)] == 0).mean(), 3), round((od.argmax(1)[e & (s >= a) & (s <= b)] == 0).mean(), 3)) for a, b in zip(bins[:-1], bins[1:])])
# CNN vs GBDT on fold 0
try:
    v = fo == 0; c = np.load(D + rf'\oofpart_{CNN}_f0.npy'); g = oc[v]; yy = y[v]
    cp, gp = c.argmax(1), g.argmax(1)
    P(f'\n===== fold0 clean: GBDT {mf1(yy, gp):.4f}  CNN {mf1(yy, cp):.4f}  blend {mf1(yy, (g + c).argmax(1)):.4f}  oracle(either right) acc {np.mean((cp == yy) | (gp == yy)):.4f} vs GBDT acc {np.mean(gp == yy):.4f}')
    P('CNN right & GBDT wrong, by true class:', dict(zip(LABELS, np.bincount(yy[(cp == yy) & (gp != yy)], minlength=9))))
    P('GBDT right & CNN wrong, by true class:', dict(zip(LABELS, np.bincount(yy[(gp == yy) & (cp != yy)], minlength=9))))
    cd = np.load(D + rf'\oofdpart_{CNN}_f0.npy'); gd = od[v]
    P(f'fold0 fakePKS: GBDT {mf1(yy, gd.argmax(1)):.4f}  CNN {mf1(yy, cd.argmax(1)):.4f}  blend {mf1(yy, (gd + cd).argmax(1)):.4f}')
except Exception as ex:
    P('cnn compare failed', ex)
o.close(); print(open(ROOT + r'\logs\error_analysis.txt').read())
