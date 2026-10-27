"""Robustness of the decision-layer variants (from results/cache/):
 1. paired bootstrap (cadence resampling, 1000x) of the fakePKS-val macro-F1 difference vs the variant's own base model
    (harness baseline argmax; submission2-model argmax for 'meta' rows) -> 90% interval and P(gain > 0);
 2. fakePKS-val macro-F1 under the estimated TEST class mix (importance-weighted confusion matrix). Most layers push more
    rows to scintillation, and the test mix has about half the scint share of train, so a gain that only holds at the train
    mix is suspect. Test mix = EM (Saerens et al.) on the submission2 model's test probabilities.
writes results/robustness.json.  usage: python src/cascade/robustness.py"""
import re, json, numpy as np
from common import y, K, R, SHORT, meta_probs

z = meta_probs()
pi_tr = np.bincount(y, minlength=K) / len(y)
pi_model = z['oofd'].mean(0)  # prior the (balanced-weight) model's posteriors encode on fake-PKS rows
pi = pi_model.copy(); P = z['pte']
for _ in range(200):
    q = P * (pi / pi_model); q /= q.sum(1, keepdims=True); pi = q.mean(0)
print('train mix', dict(zip(SHORT, np.round(pi_tr, 3))))
print('test mix (EM)', dict(zip(SHORT, np.round(pi, 3))))
w_cls = pi / pi_tr; w = w_cls[y]


def wf1(lab, wt):
    cm = np.zeros((K, K)); np.add.at(cm, (y, lab), wt); tp = np.diag(cm)
    return np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-12))


def lab(name, s='d'):
    return np.load(R / 'cache' / (re.sub(r'[^A-Za-z0-9_.+-]+', '_', name) + '.npz'))[f'lab_{s}']


rows = json.load(open(R / 'variants.json'))
rng = np.random.default_rng(0)
B = [np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float) for _ in range(1000)]
# reference = the variant's own level-1 base: the stored submission2 model for 'meta ...' rows, else the harness baseline
REFS = {k: lab(k) for k in ['H argmax (harness baseline)', 'meta argmax (submission2 model)']}
ref_b = {k: np.array([wf1(l, b) for b in B]) for k, l in REFS.items()}; ref_w = {k: wf1(l, w) for k, l in REFS.items()}
out = []
for q in rows:
    rk = 'meta argmax (submission2 model)' if q['name'].startswith('meta') or '(meta' in q['name'] else 'H argmax (harness baseline)'
    l = lab(q['name']); d = np.array([wf1(l, b) for b in B]) - ref_b[rk]
    o = dict(name=q['name'], ref=rk, fpks=q['fpks'], d_fpks_vs_ref=float(wf1(l, np.ones(len(y))) - wf1(REFS[rk], np.ones(len(y)))),
             ci90=np.round(np.percentile(d, [5, 95]), 4).tolist(), p_better=float((d > 0).mean()), fpks_testmix=wf1(l, w),
             d_fpks_testmix_vs_ref=wf1(l, w) - ref_w[rk], test_scint_share=float((lab(q['name'], 't') == 3).mean()))
    out.append(o)
    print(f"{q['name']:62s} vs {rk.split()[0]:4s} dfPKS {o['d_fpks_vs_ref']:+.4f} 90%CI [{o['ci90'][0]:+.4f},{o['ci90'][1]:+.4f}] "
          f"P>0 {o['p_better']:.2f} | test-mix fPKS {o['fpks_testmix']:.4f} ({o['d_fpks_testmix_vs_ref']:+.4f}) | "
          f"test scint share {o['test_scint_share']:.3f}")
json.dump(dict(test_mix=dict(zip(SHORT, pi.tolist())), rows=out), open(R / 'robustness.json', 'w'), indent=1)
