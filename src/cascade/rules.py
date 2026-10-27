"""Decision rules on top of level-1 scores, each with a tuner. All tuning is nested: nested() tunes on the OOF rows of 4 folds
(clean-val and fakePKS-val rows together, objective = mean of the two macro-F1s) and applies to the 5th fold.
A rule is a pair fit(Ss, ys) -> params / apply(params, S) -> labels, where S is a dict of per-row score arrays."""
import numpy as np
from common import K, f1m, folds

EPS = 1e-9


def lg(P):
    return np.log(np.clip(P, EPS, None))


def norm(P):
    return P / P.sum(1, keepdims=True)


def objective(preds, ys):
    return np.mean([f1m(yy, p) for p, yy in zip(preds, ys)])


def nested(fit, apply, Sc, Sd, y, full=None):
    """Sc / Sd: dict name -> (N, ...) arrays for the clean-val / fakePKS-val rows. Returns predc, predd (each row predicted by
    params tuned on the other 4 folds), per-fold params, and params tuned on all rows (for test) if full is True."""
    pc, pd = np.zeros(len(y), int), np.zeros(len(y), int); ps = []
    for k in range(5):
        tr, v = folds != k, folds == k
        sub = lambda S, m: {n: a[m] for n, a in S.items()}
        p = fit([sub(Sc, tr), sub(Sd, tr)], [y[tr], y[tr]]); ps.append(p)
        pc[v], pd[v] = apply(p, sub(Sc, v)), apply(p, sub(Sd, v))
    pf = fit([Sc, Sd], [y, y]) if full else None
    return pc, pd, ps, pf


def coord_ascent(obj, x0, grids, passes=3):
    x = np.array(x0, float); best = obj(x)
    for _ in range(passes):
        moved = False
        for i, g in enumerate(grids):
            for val in g:
                xx = x.copy(); xx[i] = val; s = obj(xx)
                if s > best + 1e-9: best, x, moved = s, xx, True
        if not moved: break
    return x


# ---- (d) per-class additive log-probability biases ----------------------------------------------------------------------
def bias_apply(b, S):
    return (lg(S['p']) + b).argmax(1)


def bias_fit(Ss, ys, grid=np.round(np.arange(-2, 2.01, 0.1), 2)):
    obj = lambda b: objective([bias_apply(b, S) for S in Ss], ys)
    return coord_ascent(obj, np.zeros(K), [grid] * K)


def argmax_fit(Ss, ys):
    return None


def argmax_apply(_, S):
    return S['p'].argmax(1)


# ---- (a)/(b) priority cascade: first class k (top-down) with score s_k > t_k, else fallback -------------------------------
TGRID = np.r_[np.round(np.arange(0.05, 0.96, 0.05), 2), 0.98, 0.995, 1.01]  # 1.01 = stage switched off


def cascade_apply(t, S, fallback='noise'):
    """S['s']: (N, 9) detector scores (column 8 unused). fallback 'noise' = pure cascade; 'argmax' = only override the
    argmax of S['p'] upward (a higher-priority class j beats the argmax class if s_j > t_j)."""
    s = S['s'][:, :K - 1]
    hit = s > t[:K - 1]
    if fallback == 'argmax':
        base = S['p'].argmax(1)
        hit &= np.arange(K - 1)[None, :] < base[:, None]
    else:
        base = np.full(len(s), K - 1)
    first = hit.argmax(1)
    return np.where(hit.any(1), first, base)


def cascade_fit(Ss, ys, fallback='noise', t0=0.5):
    obj = lambda t: objective([cascade_apply(t, S, fallback) for S in Ss], ys)
    return coord_ascent(obj, np.full(K, t0 if fallback == 'noise' else 1.01), [TGRID] * (K - 1) + [[1.01]])


# ---- (b) per-class Platt calibration of one-vs-rest scores, then normalise --------------------------------------------
def platt_fit(Ss, ys):
    from sklearn.linear_model import LogisticRegression
    s = np.vstack([S['s'] for S in Ss]); yy = np.concatenate(ys); ab = []
    for k in range(K):
        z = np.log(np.clip(s[:, k], EPS, 1 - EPS) / np.clip(1 - s[:, k], EPS, 1))[:, None]
        m = LogisticRegression(C=1e3).fit(z, yy == k); ab.append((m.coef_[0, 0], m.intercept_[0]))
    return np.array(ab)


def platt_probs(ab, s):
    z = np.log(np.clip(s, EPS, 1 - EPS) / np.clip(1 - s, EPS, 1))
    return norm(1 / (1 + np.exp(-(z * ab[:, 0] + ab[:, 1]))))


# ---- (c) pairwise specialists: where the current label is a or b, switch to a if q > hi, to b if q < lo -----------------
HI = [1.01, 0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
LO = [-0.01, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def pair_apply(th, S, pairs):
    pred = S['base'].copy()
    for j, (a, b) in enumerate(pairs):
        q = S['q'][:, j]; m = (pred == a) | (pred == b)
        pred = np.where(m & (q > th[2 * j]), a, np.where(m & (q < th[2 * j + 1]), b, pred))
    return pred


def pair_fit(Ss, ys, pairs):
    obj = lambda th: objective([pair_apply(th, S, pairs) for S in Ss], ys)
    return coord_ascent(obj, np.r_[[1.01, -0.01] * len(pairs)], [HI, LO] * len(pairs))


# ---- stacking: multinomial logistic regression on log-scores of several level-1 models --------------------------------
def stack_fit(Ss, ys, C=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Z = np.vstack([S['z'] for S in Ss]); yy = np.concatenate(ys)
    sc = StandardScaler().fit(Z)
    m = LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(sc.transform(Z), yy)
    return sc, m


def stack_proba(p, S):
    sc, m = p
    return m.predict_proba(sc.transform(S['z']))
