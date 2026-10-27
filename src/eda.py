import numpy as np, pandas as pd, sys
D = r'C:\Users\clark\hack4hack-r2\data\signal-hunters-round2'
Xtr = np.load(f'{D}/train_cadences.npy', mmap_mode='r')
Xte = np.load(f'{D}/test_cadences.npy', mmap_mode='r')
tr = pd.read_csv(f'{D}/train_meta.csv'); te = pd.read_csv(f'{D}/test_meta.csv')
ss = pd.read_csv(f'{D}/sample_submission.csv')
out = open(r'C:\Users\clark\hack4hack-r2\logs\eda.txt', 'w')
def P(*a): print(*a, file=out)
P('shapes', Xtr.shape, Xte.shape, tr.shape, te.shape, ss.shape, 'ss ids==test ids', (ss.id.values == te.id.values).all())
P(tr.label.value_counts(normalize=True).round(4).to_dict())
for c in ['telescope', 'channel_width_hz', 'tsamp_s', 'slew_gap_rows']:
    P(c, 'train', tr[c].value_counts().to_dict(), 'test', te[c].value_counts().to_dict())
P('center freq train', tr.center_freq_mhz.describe().round(1).to_dict())
P('center freq test', te.center_freq_mhz.describe().round(1).to_dict())
P('targets train', tr.target.nunique(), 'test', te.target.nunique(), 'overlap', len(set(tr.target) & set(te.target)))
P('id prefixes', tr.id.str[:2].unique(), te.id.str[:2].unique(), 'dup ids', tr.id.duplicated().sum())
# label vs meta
P(pd.crosstab(tr.slew_gap_rows, tr.label, normalize='index').round(3).to_string())
tr['fbin'] = pd.qcut(tr.center_freq_mhz, 5)
P(pd.crosstab(tr.fbin, tr.label, normalize='index').round(3).to_string())
tr['tc'] = tr.target.map(tr.target.value_counts())
P('target count desc', tr.tc.describe().to_dict())
# label by target repeated? purity
g = tr.groupby('target').label.agg(lambda s: s.value_counts(normalize=True).iloc[0])
P('mean target label purity', g[tr.target.value_counts() > 3].mean())
# row order/id hex leak
tr['idnum'] = tr.id.str[2:].apply(lambda h: int(h, 16))
P('label by id quintile', pd.crosstab(pd.qcut(tr.idnum, 5), tr.label, normalize='index').round(3).to_string())
P('label by row position quintile', pd.crosstab(pd.qcut(np.arange(len(tr)), 5), tr.label, normalize='index').round(3).to_string())
# pixel stats
def stats(X, n=3000):
    x = X[:n].astype(np.float32)
    z = (x == 0)
    rowdrop = (x == 0).all(axis=3).mean()
    xx = x.copy(); xx[z] = np.nan
    bp = np.nanmedian(xx, axis=(0, 1, 2))
    return dict(zero_frac=z.mean(), rowdrop=rowdrop, med=np.nanmedian(xx), std=np.nanstd(xx), p99=np.nanpercentile(xx, 99), max=np.nanmax(xx)), bp, xx
s1, bp1, x1 = stats(Xtr); s2, bp2, x2 = stats(Xte)
P('train px', {k: round(float(v), 4) for k, v in s1.items()})
P('test px', {k: round(float(v), 4) for k, v in s2.items()})
P('train bandpass', np.round(bp1, 1).tolist())
P('test bandpass', np.round(bp2, 1).tolist())
# panel gain spread
for nm, xx in [('train', x1), ('test', x2)]:
    pm = np.nanmedian(xx, axis=(2, 3))
    P(nm, 'panel median spread (std over 6 panels) mean', np.nanmean(pm.std(1)).round(3), 'cad median std', np.nanstd(pm.mean(1)).round(3))
    # partial row dropped? zeros not in full rows
    z = (xx != xx)
    P(nm, 'zeros in full rows frac', (z.all(axis=3, keepdims=True) & z).sum() / max(z.sum(), 1))
    # noise std after per-channel median removal per panel
    r = xx - np.nanmedian(xx, axis=2, keepdims=True)
    P(nm, 'robust noise std', np.nanmedian(np.abs(r)) * 1.4826)
# exact duplicates train/test
h1 = pd.util.hash_array(np.ascontiguousarray(Xtr[:]).reshape(len(Xtr), -1).view(f'V{6*16*128}').ravel()) if False else None
import hashlib
htr = [hashlib.md5(Xtr[i].tobytes()).hexdigest() for i in range(len(Xtr))]
hte = [hashlib.md5(Xte[i].tobytes()).hexdigest() for i in range(len(Xte))]
P('dup cadences in train', len(htr) - len(set(htr)), 'test', len(hte) - len(set(hte)), 'train-test', len(set(htr) & set(hte)))
# panel-level duplicates (shared OFF panels?)
ph = {}
cnt = 0
for i in range(len(Xtr)):
    for p in range(6):
        k = hashlib.md5(Xtr[i, p].tobytes()).hexdigest()
        if k in ph: cnt += 1
        ph[k] = (i, p)
P('dup panels within train', cnt)
out.close()
