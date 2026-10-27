"""Write a column subset of a stats .npz (X_clean, X_test, X_deg2, X_deg4, names) for src/bundle_eval.py.
usage: python src/subset_npz.py in.npz out.npz [name,name,...|@prefix,...] [--pairauc results/resid_T_pairauc.csv --k 40]
       [--shift results/shift_T.csv --maxshift 0.75]   (top-k = best rank of |AUC_M-.5| minus null q95 over all confused pairs)"""
import sys, argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('names', nargs='?', default='')
ap.add_argument('--pairauc'); ap.add_argument('--k', type=int, default=40); ap.add_argument('--shift'); ap.add_argument('--maxshift', type=float, default=1.0)
a = ap.parse_args()
z = np.load(a.src, allow_pickle=True); names = list(z['names'])
keep = set()
for t in [t for t in a.names.split(',') if t]: keep |= {n for n in names if n.startswith(t[1:])} if t.startswith('@') else {t}
if a.pairauc:
    P = pd.read_csv(a.pairauc); P['excess'] = P.effect - P.null_q95
    keep |= set(P.groupby('stat').excess.max().sort_values(ascending=False).head(a.k).index)
if a.shift:
    S = pd.read_csv(a.shift).set_index('stat').shift_deg; keep = {n for n in keep if S[n] <= a.maxshift}
idx = [j for j, n in enumerate(names) if n in keep]
np.savez_compressed(a.dst, names=np.array([names[j] for j in idx]), **{k: z[k][:, idx] for k in z.files if k.startswith('X_')})
print(a.dst, len(idx), 'stats:', ', '.join(names[j] for j in idx))
