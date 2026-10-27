"""Reverse-engineered generator of labelled SETI cadences (Hack4Hack R2 'Signal Hunters') for GBT or PKS conditions.

Cadence = 6 panels ON/OFF/ON/OFF/ON/OFF x 16 rows (time) x 128 channels, uint8 log power:
    byte = round(96 + 24*log2(P)), clipped to 1..255; dropped rows = 0.
Global time of row r in panel p: t = p*(16+gap) + r, gap = slew gap rows in {3,4,5,6}.
Power model (linear units of the per-channel noise mean):
    P = bandpass(c) * gain(p) * (chi2_k/k + S(p,t,c)) * glitch(p,t,c)
S = sum of the rendered sources. Each cadence gets a primary source (its label) plus lower-priority secondary sources;
label = highest-priority source present (LABELS order). Instrument effects (noise dof, bandpass, ch-64 spike, panel gain
jitter, glitch rows, dropped rows) depend on the telescope setting.

All numbers in TEL / SRC are estimates: noise/gain/glitch/drop from the logged pixel statistics of the real data and the
bundle features (see results/REPORT.md); signal parameters calibrated by matching per-class feature quantiles of
generated vs real train (src/gen_calib.py).
usage: python src/generator.py N TEL SEED OUTPREFIX   -> OUTPREFIX_cadences.npy, _gaps.npy, _y.npy, _info.csv"""
import sys, numpy as np, pandas as pd

LABELS = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
          'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']   # priority order
L2I = {l: i for i, l in enumerate(LABELS)}
TRAIN_MIX = np.array([0.0266, 0.0579, 0.0905, 0.0795, 0.1003, 0.1027, 0.1479, 0.1607, 0.2339])
CH = np.arange(128, dtype=np.float32)
ON, OFF = (0, 2, 4), (1, 3, 5)

# ---------------- instrument settings ----------------
TEL = {
    'GBT': dict(dof=18.0,                       # ln-power noise std 0.343 -> log-chi2 with 18 dof
                gain_sd=1.45,                   # panel gain jitter, bytes (lvl_off_spread median 2.06: 1.25->1.46, 1.8->2.71)
                drop_rate=0.147,                # dropped rows / cadence, Poisson (0.147 measured)
                glitch_panel=0.10,              # P(panel has a glitch row): #glitched panels ~ Binomial(6, 0.10)
                glitch_amp=(18.0, 46.0),        # bytes added to the glitch row
                glitch_partial=0.3,             # fraction of glitch rows covering only part of the band
                spike=12.0,                     # centre spike at ch 64 (bytes)
                level=100.0, bp_edge=10.0, bp_ripple=2.0, bp_tilt=4.0),
    'PKS': dict(dof=11.0,                       # ln-power noise std 0.45 -> ~11 dof (heavier negative tail)
                gain_sd=2.3,                    # test lvl_off_spread median 3.55
                drop_rate=0.355,                # test 0.355 dropped rows / cadence
                glitch_panel=0.18,              # test: #glitched panels ~ Binomial(6, 0.18) (fits 3037/4000/2193/640/121/9)
                glitch_amp=(18.0, 48.0),
                glitch_partial=0.3,
                spike=0.0,
                drift_scale=0.88,               # strong ON-track drift q75/q90 test 0.219/0.387 vs fake-PKS 0.262/0.45
                level=100.0, bp_edge=14.0, bp_ripple=3.0, bp_tilt=6.0),
}

# ---------------- source parameters (calibrated, see gen_calib.py) ----------------
SRC = dict(
    eti=dict(amp=(0.25, 2.5), drift=(0.04, 0.8), width=(0.5, 1.4)),
    line=dict(amp=(0.6, 5.0), width=(0.5, 1.6)),
    pers=dict(amp=(1.0, 8.0), width=(1.0, 2.8), p_zero_drift=0.65, drift=(0.01, 0.35), p_fade=0.30),
    inter=dict(mix=[0.20, 0.13, 0.10, 0.30, 0.27], drift_p_zero=0.4, drift=(0.02, 0.6)),
    sat=dict(amp=(0.8, 6.0), width=(0.6, 1.8), p_one=0.8, drift=(0.1, 8.0), curv=(0.0, 0.5)),
    radar=dict(amp=(0.3, 2.0), width=(1.5, 4.0), period=(4.0, 16.0), span=(10.0, 90.0), p_panelrel=0.5),   # slope 0.6-20 ch/row
    pulsar=dict(amp=(0.15, 1.2), period=(2.5, 10.0), width=(0.3, 1.0), sweep=(0.0, 9.0), lvl=(0.0, 0.0)),
    frb=dict(amp=(0.4, 3.0), width=(0.5, 1.5), sweep=(2.0, 14.0), tail=(0.0, 3.0)),
    scint=dict(flux=(0.03, 0.3), depth=(0.15, 0.5), tscale=(1.0, 3.0), fscale=(2.0, 10.0)),   # tex v2x4..v16x64 -> scintle area ~100 px
)
# probability that a cadence of class k also carries each lower-priority secondary source
# (real panel patterns: 'xxxxxx' = persistent line in 49% of ETI, 9-14% of frb/pulsar/scint/sat/radar, 23% of rfi_int;
#  single-panel lines ~2% per panel in frb/pulsar/scint/radar)
SECONDARY = {
    'eti_candidate': dict(rfi_persistent=0.52, rfi_intermittent=0.08),
    'frb': dict(rfi_persistent=0.13, rfi_intermittent=0.08, satellite_pass=0.04),
    'pulsar': dict(rfi_persistent=0.10, rfi_intermittent=0.08, satellite_pass=0.04),
    'scintillating_star': dict(rfi_persistent=0.14, rfi_intermittent=0.08, satellite_pass=0.04),
    'satellite_pass': dict(rfi_persistent=0.15, rfi_intermittent=0.05),
    'radar_chirp': dict(rfi_persistent=0.13, rfi_intermittent=0.08),
    'rfi_intermittent': dict(rfi_persistent=0.25),
    'rfi_persistent': dict(),
    'noise': dict(),
}


def logu(rng, lo, hi): return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def tgrid(gap):
    """global time (6,16) of every row"""
    return (np.arange(6)[:, None] * (16 + gap) + np.arange(16)[None, :]).astype(np.float32)


def add_track(S, rng, gap, panels, c_ref, drift, amp, width, offs=None, curv=0.0, t_ref=None, pamp=None):
    """Straight (or curved) narrowband track in global time through `panels`. amp in noise-mean units."""
    T = tgrid(gap); t_ref = 2 * (16 + gap) + 7.5 if t_ref is None else t_ref
    for p in panels:
        t = T[p] - t_ref
        c = c_ref + drift * t + curv * t ** 2 + (0.0 if offs is None else offs.get(p, 0.0))
        a = amp * (1.0 if pamp is None else pamp.get(p, 1.0))
        S[p] += a * np.exp(-0.5 * ((CH[None, :] - c[:, None]) / width) ** 2)


# ---------------- sources ----------------
def src_eti(S, rng, gap, P, primary=True):
    q = P['eti']; d = logu(rng, *q['drift']) * rng.choice([-1, 1]); span = abs(d) * (2 * (16 + gap) + 8)
    c0 = rng.uniform(4 + span, 123 - span) if span < 55 else rng.uniform(4, 123)
    add_track(S, rng, gap, ON, c0, d, logu(rng, *q['amp']), rng.uniform(*q['width']))
    return dict(drift=d, c0=c0)


def src_rfi_persistent(S, rng, gap, P, primary=True):
    q = P['pers']; ql = P['line']
    d = 0.0 if rng.random() < q['p_zero_drift'] else logu(rng, *q['drift']) * rng.choice([-1, 1])
    c0 = rng.uniform(3, 124); A = logu(rng, *q['amp'])
    pamp = {p: rng.uniform(0.6, 1.0) for p in range(6)}
    if rng.random() < q['p_fade']: pamp[int(rng.integers(6))] = rng.uniform(0.0, 0.4)   # one faded panel
    add_track(S, rng, gap, range(6), c0, d, A, rng.uniform(*q['width']), pamp=pamp)
    return dict(drift=d, c0=c0)


def src_rfi_intermittent(S, rng, gap, P, primary=True):
    """Line present in a subset of panels. As the primary label the real panel patterns are strongly ON-biased
    (x.x.x. 21%, ON subsets ~15%, OFF-including rare) -> mixture of the stated ETI traps and subsets:
    zero: ON-only, zero drift | disc: ON-only, same drift, panels offset | onoff: ON track + one OFF panel |
    onsub: 1-2 ON panels | rand: any 1-4 panels. As a secondary source only 'rand'."""
    q = P['inter']; ql = P['line']
    kind = rng.choice(['zero', 'disc', 'onoff', 'onsub', 'rand'], p=q['mix']) if primary else 'rand'
    d = 0.0 if (kind == 'zero' or rng.random() < q['drift_p_zero']) else logu(rng, *q['drift']) * rng.choice([-1, 1])
    offs = None
    if kind in ('zero', 'disc'): panels = [0, 2, 4]
    elif kind == 'onoff': panels = [0, 2, 4, int(rng.choice(OFF))]
    elif kind == 'onsub': panels = sorted(rng.choice(ON, int(rng.integers(1, 3)), replace=False).tolist())
    else: panels = sorted(rng.choice(6, int(rng.integers(1, 5)), replace=False).tolist())
    if kind == 'disc':
        offs = {p: rng.choice([-1, 1]) * rng.uniform(4, 30) for p in panels}; offs[int(rng.choice(panels))] = 0.0
    span = abs(d) * (2 * (16 + gap) + 8); c0 = rng.uniform(4 + span, 123 - span) if span < 55 else rng.uniform(4, 123)
    add_track(S, rng, gap, panels, c0, d, logu(rng, *ql['amp']), rng.uniform(*ql['width']), offs=offs)
    return dict(drift=d, c0=c0, kind=kind, panels=''.join(map(str, panels)))


def src_satellite(S, rng, gap, P, primary=True):
    """Fast-drifting, slightly curved Doppler track c(t) = c0 + d*t + k*d*t*|t| (t from the pass centre) visible in one
    panel (real patterns: single panels ~8% each, ON and OFF alike) or 2-3 consecutive panels. Real detected drifts:
    median 0.38 ch/row, 38% |d|>1, steep (|d|>3) line score p90 11 -> |d| log-uniform 0.1-8 ch/row.
    (A tanh S-curve with steep centre was tried: mostly undetectable by the line searches, 48% of cadences blank vs 17%.)"""
    q = P['sat']; n = 1 if rng.random() < q['p_one'] else int(rng.integers(2, 4)); p0 = int(rng.integers(0, 7 - n))
    T = tgrid(gap); A = logu(rng, *q['amp']); w = rng.uniform(*q['width'])
    d = logu(rng, *q['drift']) * rng.choice([-1, 1]); k = rng.uniform(*q['curv']) * rng.choice([-1, 1])
    tc = T[p0, 0] + rng.uniform(0, 16 * n + gap * (n - 1)); c0 = rng.uniform(10, 118)
    for p in range(p0, p0 + n):
        t = T[p] - tc; c = c0 + d * t + k * d * t * np.abs(t) / 16.0
        S[p] += A * rng.uniform(0.7, 1.0) * np.exp(-0.5 * ((CH[None, :] - c[:, None]) / w) ** 2)
    return dict(drift=d, npanel=n)


def src_radar(S, rng, gap, P, primary=True):
    """sawtooth chirp: frequency ramps over `span` channels in `per` rows, then wraps, through all panels; half of them
    in panel-relative time (same streaks at the same place in every panel: real rep_k3/k6 of radar are raised, but
    all-panel-relative overshoots rep_k3 2.85 vs 2.06), half continuous in global time."""
    q = P['radar']; per = rng.uniform(*q['period']); span = rng.uniform(*q['span']); slope = rng.choice([-1, 1])
    A = logu(rng, *q['amp']); w = rng.uniform(*q['width']); ph = rng.uniform(0, per)
    lo = rng.uniform(0, 128 - span); hi = lo + span
    T = np.tile(np.arange(16, dtype=np.float32), (6, 1)) if rng.random() < q['p_panelrel'] else tgrid(gap)
    for p in range(6):
        frac = ((T[p] + ph) % per) / per                           # (16,)
        c = (lo + frac * span) if slope > 0 else (hi - frac * span)
        S[p] += A * np.exp(-0.5 * ((CH[None, :] - c[:, None]) / w) ** 2)
    return dict(period=per)


def dispersed_pulse(t_arr, t0, sweep, width, tail, shape=0.0):
    """(16,128) broadband pulse arriving at t0 at the top channel; delay(c) = sweep*shp(u), u = 1 - c/127 (ch 0 = lowest
    freq, arrives last); Gaussian width (rows) + one-sided exponential scattering tail (stronger at low freq)."""
    u = 1 - CH / 127.0; shp = (u + shape * u ** 2) / (1 + shape)
    ta = t0 + sweep * shp                                           # (128,)
    dt = t_arr[:, None] - ta[None, :]
    g = np.exp(-0.5 * (dt / width) ** 2)
    if tail > 0:
        tau = np.maximum(tail * (0.3 + u) ** 2, 1e-3)              # longer tail at low frequency
        e = np.exp(-np.clip(dt, 0, None) / tau[None, :]) * (dt > 0)
        g = np.maximum(g, e)
    return g


def src_pulsar(S, rng, gap, P, primary=True):
    q = P['pulsar']; per = rng.uniform(*q['period']); A = logu(rng, *q['amp']); w = rng.uniform(*q['width'])
    sw = rng.uniform(*q['sweep']); shape = rng.choice([0.0, 1.5]); T = tgrid(gap); ph = rng.uniform(0, per)
    base = rng.uniform(*q['lvl'])
    r = np.arange(16, dtype=np.float32)
    for p in ON:
        t0s = np.arange(T[p, 0] - sw - per - (T[p, 0] - ph) % per, T[p, -1] + 1, per)
        for t0 in t0s:
            S[p] += A * rng.uniform(0.6, 1.4) * dispersed_pulse(T[p], t0, sw, w, 0.0, shape)
        S[p] += base
    return dict(period=per, sweep=sw)


def src_frb(S, rng, gap, P, primary=True):
    q = P['frb']; p = int(rng.choice(ON)); A = logu(rng, *q['amp']); sw = rng.uniform(*q['sweep'])
    t0 = rng.uniform(-sw * 0.5, 12.0); T = np.arange(16, dtype=np.float32)
    S[p] += A * dispersed_pulse(T, t0, sw, rng.uniform(*q['width']), rng.uniform(*q['tail']), rng.choice([0.0, 1.5]))
    return dict(panel=p, sweep=sw)


def smooth_field(rng, tsc, fsc, shape=(16, 128)):
    """zero-mean unit-variance smooth Gaussian random field (scintles of tsc rows x fsc channels)."""
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(rng.normal(size=shape), (tsc, fsc), mode='wrap')
    return (g - g.mean()) / (g.std() + 1e-9)


def src_scint(S, rng, gap, P, primary=True):
    """ON-panel brightening F (mean) modulated by scintles: F*(1 + depth*g), g smooth random field, independent per ON panel."""
    q = P['scint']; F = logu(rng, *q['flux']); tsc = rng.uniform(*q['tscale']); fsc = logu(rng, *q['fscale'])
    dep = rng.uniform(*q['depth'])
    for p in ON:
        S[p] += F * np.clip(1 + dep * smooth_field(rng, tsc, fsc), 0, None)
    return dict(flux=F, tscale=tsc, fscale=fsc, depth=dep)


RENDER = {'eti_candidate': src_eti, 'frb': src_frb, 'pulsar': src_pulsar, 'scintillating_star': src_scint,
          'satellite_pass': src_satellite, 'radar_chirp': src_radar, 'rfi_intermittent': src_rfi_intermittent,
          'rfi_persistent': src_rfi_persistent, 'noise': lambda S, rng, gap, P, primary=True: {}}


# ---------------- instrument ----------------
def bandpass(rng, t):
    u = CH / 127.0
    edge = -t['bp_edge'] * (np.exp(-u / 0.05) + np.exp(-(1 - u) / 0.05))
    rip = t['bp_ripple'] * np.sin(2 * np.pi * (u * rng.uniform(2, 6) + rng.uniform(0, 1)))
    tilt = t['bp_tilt'] * rng.uniform(-1, 1) * (u - 0.5)
    fine = rng.normal(0, t.get('bp_fine', 0.0), 128)               # static channel-scale ripple (survives prep's 9-ch median)
    bp = t['level'] + rng.normal(0, 4) + edge + rip + tilt + fine
    if t['spike']: bp[64] += t['spike']
    return bp.astype(np.float32)                                    # bytes


def glitch_rows(rng, t):
    """(6,16,128) bytes added by glitch rows: each panel independently glitched w.p. glitch_panel (binomial count of
    glitched panels matches real train/test exactly), 1 row (2 rows w.p. 0.07)."""
    G = np.zeros((6, 16, 128), np.float32); n = 0
    for p in np.where(rng.random(6) < t['glitch_panel'])[0]:
        for r in rng.choice(16, 2 if rng.random() < 0.07 else 1, replace=False):
            a = rng.uniform(*t['glitch_amp'])
            if rng.random() < t['glitch_partial']:
                lo = int(rng.integers(0, 64)); hi = min(128, lo + int(rng.integers(40, 64)))
            else: lo, hi = 0, 128
            G[p, r, lo:hi] += a; n += 1
    return G, n


def render_one(rng, label, tel='GBT', gap=None, P=SRC, secondary=SECONDARY):
    t = TEL[tel]; gap = int(rng.integers(3, 7)) if gap is None else gap
    S = np.zeros((6, 16, 128), np.float32)
    info = dict(label=label, gap=gap); info.update({f'p_{k}': v for k, v in RENDER[label](S, rng, gap, P).items()})
    sec = []
    for s, pr in secondary.get(label, {}).items():
        if L2I[s] > L2I[label] and rng.random() < pr:
            RENDER[s](S, rng, gap, P, primary=False); sec.append(s)
    info['secondary'] = '+'.join(sec)
    k = t['dof']; noise = rng.chisquare(k, (6, 16, 128)).astype(np.float32) / k
    # radiometer model: signal and noise fluctuate together, P = (1+S)*chi2_k/k (an additive S compresses the
    # log-variance of brightened pixels: scint ON panels then show LESS small-scale texture than OFF, opposite to real)
    v = 96.0 + 24.0 * np.log2((1.0 + S) * noise)
    v += bandpass(rng, t)[None, None, :] - 96.0
    v += rng.normal(0, t['gain_sd'], (6, 1, 1)).astype(np.float32)
    if t.get('gain_slope', 0):                                       # slow gain drift within a panel (bytes over 16 rows)
        v += (rng.normal(0, t['gain_slope'], (6, 1, 1)) * (np.arange(16) - 7.5)[None, :, None] / 16).astype(np.float32)
    G, ng = glitch_rows(rng, t); v += G; info['n_glitch'] = ng
    v = np.clip(np.round(v), 1, 255).astype(np.uint8)
    nd = rng.poisson(t['drop_rate'])
    for _ in range(nd): v[int(rng.integers(6)), int(rng.integers(16))] = 0
    info['n_drop'] = nd
    return v, gap, info


def scaled_params(P, ds):
    """pixel-drift scale of the telescope (PKS channel width / tsamp differ): scale every drift / sweep-slope range."""
    import copy
    Q = copy.deepcopy(P)
    if ds != 1.0:
        for k, f in [('eti', 'drift'), ('pers', 'drift'), ('inter', 'drift'), ('radar', 'span'), ('sat', 'drift')]:
            Q[k][f] = tuple(v * ds for v in Q[k][f])
    return Q


def generate(n, tel='GBT', seed=0, mix=TRAIN_MIX, P=SRC, secondary=SECONDARY, labels=None):
    rng = np.random.default_rng(seed); P = scaled_params(P, TEL[tel].get('drift_scale', 1.0))
    y = rng.choice(9, n, p=np.asarray(mix) / np.sum(mix)) if labels is None else np.asarray(labels)
    X = np.zeros((n, 6, 16, 128), np.uint8); gaps = np.zeros(n, int); infos = []
    for i in range(n):
        X[i], gaps[i], inf = render_one(rng, LABELS[y[i]], tel, P=P, secondary=secondary); infos.append(inf)
    return X, gaps, y, pd.DataFrame(infos)


if __name__ == '__main__':
    n, tel, seed, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    X, g, y, info = generate(n, tel, seed)
    np.save(out + '_cadences.npy', X); np.save(out + '_gaps.npy', g); np.save(out + '_y.npy', y); info.to_csv(out + '_info.csv', index=False)
    print('generated', X.shape, tel, dict(zip(LABELS, np.bincount(y, minlength=9))))
