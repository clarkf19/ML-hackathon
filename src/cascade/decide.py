"""Score every decision-layer variant with nested CV (params tuned on 4 folds, scored on the 5th; clean-val and fakePKS-val).
A variant = a level-1 score bundle + a chain of steps (bias / cascade / Platt / pairwise override / stacking).
Writes results/variants.md + results/variants.json; per-variant labels/probs are cached in results/cache/.
usage: python src/cascade/decide.py [--force] [--only substr] [--save-best NAME]"""
import sys, re, json, numpy as np
from common import y, folds, K, R, SHORT, f1m, f1_per, load_probs, meta_probs
import rules as r
from train_base import PAIRS

CACHE = R / 'cache'; CACHE.mkdir(exist_ok=True)
W = ['c', 'd', 't']  # clean-val OOF, fakePKS-val OOF, test


def models():
    out = {}
    for n in ['baseline', 'cond', 'ovr', 'pairs', 'xgb', 'cat', 'mlp']:
        try:
            z = load_probs(n); out[n] = dict(zip(W, [z['oofc'], z['oofd'], z['pte']]))
        except FileNotFoundError:
            pass
    z = meta_probs(); out['meta'] = dict(zip(W, [z['oofc'], z['oofd'], z['pte']]))
    return out


# ---- steps: fit(Ss, ys) -> params ; tr(params, S) -> S' -------------------------------------------------------------
def with_(S, **kw):
    S = dict(S); S.update(kw); return S


class Bias:
    name = 'bias'
    def fit(self, Ss, ys): return r.bias_fit(Ss, ys)
    def tr(self, b, S):
        p = r.norm(S['p'] * np.exp(b)); return with_(S, p=p, base=p.argmax(1))


class Cascade:
    def __init__(self, fallback): self.fb = fallback; self.name = f'cascade[{fallback}]'
    def fit(self, Ss, ys): return r.cascade_fit(Ss, ys, self.fb)
    def tr(self, t, S): return with_(S, base=r.cascade_apply(t, S, self.fb))


class Platt:
    name = 'platt'
    def fit(self, Ss, ys): return r.platt_fit(Ss, ys)
    def tr(self, ab, S):
        p = r.platt_probs(ab, S['s']); return with_(S, p=p, base=p.argmax(1))


class Pairs:
    def __init__(self, idx=None): self.idx = list(range(len(PAIRS))) if idx is None else idx; self.name = 'pairs'
    def _S(self, S):
        return with_(S, q=S['q'][:, self.idx], base=S['base'] if 'base' in S else S['p'].argmax(1))
    def fit(self, Ss, ys): return r.pair_fit([self._S(S) for S in Ss], ys, [PAIRS[i] for i in self.idx])
    def tr(self, th, S): return with_(S, base=r.pair_apply(th, self._S(S), [PAIRS[i] for i in self.idx]))


class Stack:
    def __init__(self, C=0.05): self.C = C; self.name = f'stackLR(C={C})'
    def fit(self, Ss, ys): return r.stack_fit(Ss, ys, self.C)
    def tr(self, p, S):
        q = r.stack_proba(p, S); return with_(S, p=q, base=q.argmax(1))


def run_chain(steps, Sc, Sd, St):
    """nested over folds; returns labels/probs for c, d (OOF) and t (params fit on all rows), per-fold params"""
    def fit(Ss, ys):
        ps = []
        for st in steps:
            p = st.fit(Ss, ys); ps.append(p); Ss = [st.tr(p, S) for S in Ss]
        return ps
    def apply_full(ps, S):
        for st, p in zip(steps, ps): S = st.tr(p, S)
        return S
    out = {w: {} for w in W}
    lab = {w: np.zeros(len(y), int) for w in 'cd'}; prob = {w: np.zeros((len(y), K)) for w in 'cd'}
    for k in range(5):
        tr, v = folds != k, folds == k
        sub = lambda S, m: {n: a[m] for n, a in S.items()}
        ps = fit([sub(Sc, tr), sub(Sd, tr)], [y[tr], y[tr]]) if steps else []
        for w, S in zip('cd', [Sc, Sd]):
            o = apply_full(ps, sub(S, v))
            lab[w][v] = o['base'] if 'base' in o else o['p'].argmax(1); prob[w][v] = o.get('p', np.zeros((v.sum(), K)))
    ps = fit([Sc, Sd], [y, y]) if steps else []
    o = apply_full(ps, St)
    out['t'] = dict(lab=o['base'] if 'base' in o else o['p'].argmax(1), p=o.get('p', np.zeros((len(next(iter(St.values()))), K))))
    for w in 'cd': out[w] = dict(lab=lab[w], p=prob[w])
    return out, ps


def cond_s(p):  # multiclass probs -> conditional "class k given nothing of higher priority" scores for the cascade
    return p / np.cumsum(p[:, ::-1], 1)[:, ::-1]


def chain(c):  # cond detector scores -> joint distribution P(k) = c_k prod_{j<k}(1-c_j)
    surv = np.cumprod(np.c_[np.ones(len(c)), 1 - c[:, :K - 1]], 1)
    return surv * np.c_[c[:, :K - 1], np.ones(len(c))]


def avg(*ps): return np.mean([r.norm(p) for p in ps], 0)
def gavg(*ps): return r.norm(np.exp(np.mean([r.lg(r.norm(p)) for p in ps], 0)))
def logit(s): return np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))


def variants(M):
    """name -> (group, builder(w) -> S, steps). Only variants whose models exist are returned."""
    V = {}
    def add(name, group, need, build, steps):
        if all(n in M for n in need): V[name] = (group, build, steps)
    H = lambda w: M['baseline'][w]
    # --- reference + (d) biases on the harness baseline
    add('H argmax (harness baseline)', 'ref', ['baseline'], lambda w: {'p': H(w)}, [])
    add('H + bias', 'd', ['baseline'], lambda w: {'p': H(w)}, [Bias()])
    # --- (a) priority cascades
    add('H cond-prob cascade (pure)', 'a', ['baseline'], lambda w: {'p': H(w), 's': cond_s(H(w))}, [Cascade('noise')])
    add('H cond-prob cascade (upward overrides of argmax)', 'a', ['baseline'],
        lambda w: {'p': H(w), 's': cond_s(H(w))}, [Cascade('argmax')])
    add('H + bias -> cond-prob upward overrides', 'a+d', ['baseline'],
        lambda w: {'p': H(w), 's': cond_s(H(w))}, [Bias(), Cascade('argmax')])
    C = lambda w: M['cond'][w]
    add('cond detectors: pure cascade', 'a', ['cond'], lambda w: {'s': C(w)}, [Cascade('noise')])
    add('cond detectors: chain-rule joint argmax', 'a', ['cond'], lambda w: {'p': chain(C(w))}, [])
    add('cond detectors: chain-rule joint + bias', 'a+d', ['cond'], lambda w: {'p': chain(C(w))}, [Bias()])
    add('H argmax + cond-detector upward overrides', 'a', ['baseline', 'cond'],
        lambda w: {'p': H(w), 's': C(w)}, [Cascade('argmax')])
    add('H + bias + cond-detector upward overrides', 'a+d', ['baseline', 'cond'],
        lambda w: {'p': H(w), 's': C(w)}, [Bias(), Cascade('argmax')])
    add('avg(H, cond joint) + bias', 'a+d', ['baseline', 'cond'], lambda w: {'p': avg(H(w), chain(C(w)))}, [Bias()])
    # --- (b) one-vs-rest
    O = lambda w: M['ovr'][w]
    add('OvR raw argmax', 'b', ['ovr'], lambda w: {'p': O(w)}, [])
    add('OvR normalised + bias', 'b+d', ['ovr'], lambda w: {'p': r.norm(O(w))}, [Bias()])
    add('OvR Platt-calibrated argmax', 'b', ['ovr'], lambda w: {'s': O(w)}, [Platt()])
    add('OvR Platt + bias', 'b+d', ['ovr'], lambda w: {'s': O(w)}, [Platt(), Bias()])
    add('OvR Platt + priority rule (upward overrides)', 'b', ['ovr'], lambda w: {'s': O(w)}, [Platt(), Cascade('argmax')])
    add('OvR pure cascade (raw scores)', 'b', ['ovr'], lambda w: {'s': O(w)}, [Cascade('noise')])
    add('H argmax + OvR upward overrides', 'b', ['baseline', 'ovr'], lambda w: {'p': H(w), 's': O(w)}, [Cascade('argmax')])
    add('avg(H, OvR norm) + bias', 'b+d', ['baseline', 'ovr'], lambda w: {'p': avg(H(w), O(w))}, [Bias()])
    # --- (c) pairwise specialists
    Q = lambda w: M['pairs'][w]
    add('H argmax + pair overrides', 'c', ['baseline', 'pairs'], lambda w: {'p': H(w), 'q': Q(w)}, [Pairs()])
    add('H + bias + pair overrides', 'c+d', ['baseline', 'pairs'], lambda w: {'p': H(w), 'q': Q(w)}, [Bias(), Pairs()])
    add('H + bias + cond-prob overrides + pair overrides', 'a+c+d', ['baseline', 'pairs'],
        lambda w: {'p': H(w), 's': cond_s(H(w)), 'q': Q(w)}, [Bias(), Cascade('argmax'), Pairs()])
    # --- (e) other learners and averages
    for n in ['xgb', 'cat', 'mlp']:
        add(f'{n} argmax', 'e', [n], lambda w, n=n: {'p': M[n][w]}, [])
        add(f'avg(H, {n})', 'e', ['baseline', n], lambda w, n=n: {'p': avg(H(w), M[n][w])}, [])
        add(f'avg(H, {n}) + bias', 'e+d', ['baseline', n], lambda w, n=n: {'p': avg(H(w), M[n][w])}, [Bias()])
    add('avg(H, xgb, cat)', 'e', ['baseline', 'xgb', 'cat'], lambda w: {'p': avg(H(w), M['xgb'][w], M['cat'][w])}, [])
    add('avg(H, xgb, cat) + bias', 'e+d', ['baseline', 'xgb', 'cat'],
        lambda w: {'p': avg(H(w), M['xgb'][w], M['cat'][w])}, [Bias()])
    GB = ['baseline', 'xgb', 'cat', 'mlp']
    add('avg(H, xgb, cat, mlp)', 'e', GB, lambda w: {'p': avg(*[M[n][w] for n in GB])}, [])
    add('avg(H, xgb, cat, mlp) + bias', 'e+d', GB, lambda w: {'p': avg(*[M[n][w] for n in GB])}, [Bias()])
    add('geo-avg(H, xgb, cat, mlp) + bias', 'e+d', GB, lambda w: {'p': gavg(*[M[n][w] for n in GB])}, [Bias()])
    add('avg(H, xgb, cat, mlp) + bias + cond-prob overrides + pairs', 'all', GB + ['pairs'],
        lambda w: {'p': avg(*[M[n][w] for n in GB]), 's': cond_s(avg(*[M[n][w] for n in GB])), 'q': Q(w)},
        [Bias(), Cascade('argmax'), Pairs()])
    # --- stacking all level-1 scores (multinomial LR, nested)
    ST = ['baseline', 'xgb', 'cat', 'mlp', 'ovr', 'cond', 'pairs']
    zf = lambda w: np.hstack([r.lg(r.norm(M[n][w])) for n in ['baseline', 'xgb', 'cat', 'mlp']]
                             + [logit(M['ovr'][w]), logit(M['cond'][w][:, :K - 1]), logit(M['pairs'][w])])
    add('stack LR (all level-1 scores)', 'stack', ST, lambda w: {'z': zf(w)}, [Stack(0.05)])
    add('stack LR (all level-1 scores) + bias', 'stack+d', ST, lambda w: {'z': zf(w)}, [Stack(0.05), Bias()])
    # --- the same decision layers on the stored submission2 model (meta: 4 fake-PKS copies, not the harness)
    Mt = lambda w: M['meta'][w]
    add('meta argmax (submission2 model)', 'meta', ['meta'], lambda w: {'p': Mt(w)}, [])
    add('meta + bias', 'meta+d', ['meta'], lambda w: {'p': Mt(w)}, [Bias()])
    add('meta cond-prob cascade (pure)', 'meta+a', ['meta'], lambda w: {'p': Mt(w), 's': cond_s(Mt(w))}, [Cascade('noise')])
    add('meta + bias -> cond-prob upward overrides', 'meta+a+d', ['meta'],
        lambda w: {'p': Mt(w), 's': cond_s(Mt(w))}, [Bias(), Cascade('argmax')])
    add('meta + bias + pair overrides', 'meta+c+d', ['meta', 'pairs'], lambda w: {'p': Mt(w), 'q': Q(w)}, [Bias(), Pairs()])
    MB = ['meta', 'xgb', 'cat', 'mlp']
    add('avg(meta, xgb, cat, mlp) + bias', 'meta+e+d', MB, lambda w: {'p': avg(*[M[n][w] for n in MB])}, [Bias()])
    add('avg(meta, H, xgb, cat, mlp) + bias', 'meta+e+d', MB + ['baseline'],
        lambda w: {'p': avg(*[M[n][w] for n in MB + ['baseline']])}, [Bias()])
    return V


def score(out):
    fc, fd = f1m(y, out['c']['lab']), f1m(y, out['d']['lab'])
    per_fold = np.array([f1m(y[folds == k], out['d']['lab'][folds == k]) for k in range(5)])
    return fc, fd, per_fold


if __name__ == '__main__':
    a = sys.argv[1:]
    force, only = '--force' in a, a[a.index('--only') + 1] if '--only' in a else None
    M = models(); V = variants(M); rows = []
    hb = None
    for name, (group, build, steps) in V.items():
        if only and only not in name and not name.startswith('H argmax (harness'): continue
        f = CACHE / (re.sub(r'[^A-Za-z0-9_.+-]+', '_', name) + '.npz')
        if f.exists() and not force:
            z = np.load(f, allow_pickle=True); out = {w: dict(lab=z[f'lab_{w}'], p=z[f'p_{w}']) for w in W}
        else:
            Sc, Sd, St = build('c'), build('d'), build('t')
            out, ps = run_chain(steps, Sc, Sd, St)
            np.savez_compressed(f, **{f'lab_{w}': out[w]['lab'] for w in W}, **{f'p_{w}': out[w]['p'] for w in W})
        fc, fd, pf = score(out)
        if hb is None: hb = (fc, fd, pf)
        pc = f1_per(y, out['d']['lab'])
        rows.append(dict(name=name, group=group, clean=fc, fpks=fd, d_clean=fc - hb[0], d_fpks=fd - hb[1],
                         folds_up=int((pf > hb[2] + 1e-12).sum()), per_class_fpks=dict(zip(SHORT, np.round(pc, 3).tolist())),
                         steps=' -> '.join(s.name for s in steps) or 'argmax'))
        print(f'{name:62s} clean {fc:.4f} ({fc - hb[0]:+.4f})  fPKS {fd:.4f} ({fd - hb[1]:+.4f})  folds+ {rows[-1]["folds_up"]}/5',
              flush=True)
    if not only:
        json.dump(rows, open(R / 'variants.json', 'w'), indent=1)
        with open(R / 'variants.md', 'w') as fh:
            fh.write('| Variant | idea | clean-val | Δ | fakePKS-val | Δ | fPKS folds better |\n|---|---|---|---|---|---|---|\n')
            for q in rows:
                fh.write(f"| {q['name']} | {q['group']} | {q['clean']:.4f} | {q['d_clean']:+.4f} | {q['fpks']:.4f} | "
                         f"{q['d_fpks']:+.4f} | {q['folds_up']}/5 |\n")
