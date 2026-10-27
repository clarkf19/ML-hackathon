"""B: CNN on the global-time canvas. Panels are placed at their real global time
(p*(16+gap)+r) in a 128x128 image, so an ETI track is ONE straight line; channels = [z, ON mask, valid mask].
Augment: freq flip, freq shift, SNR reduction (PKS is noisier), freq zoom-out (PKS drift in pixels is smaller), row drop.
Usage: cnn.py FOLD [EPOCHS] [TAG]   (FOLD=-1 -> train on all, predict test)"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, sys, time, os
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
torch.set_num_threads(int(os.environ.get('NT', 8))); torch.manual_seed(0); np.random.seed(0)
FOLD = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 24; TAG = sys.argv[3] if len(sys.argv) > 3 else 'B'
AUG = os.environ.get('AUG', 'full')

def canvas(z, gaps):
    b = len(z); C = torch.zeros(b, 3, 128, 128)
    for i in range(b):
        g = int(gaps[i])
        for p in range(6):
            t0 = p * (16 + g)
            C[i, 0, t0:t0 + 16] = z[i, p]
            C[i, 1, t0:t0 + 16] = 1.0 if p % 2 == 0 else -1.0
            C[i, 2, t0:t0 + 16] = (z[i, p] != 0).any(-1, keepdim=True).float()
    return C

def augment(z):  # z: (b,6,16,128) float tensor
    b = len(z)
    flip = torch.rand(b) < 0.5; z[flip] = z[flip].flip(-1)
    if AUG == 'none': return z
    a = torch.empty(b, 1, 1, 1).uniform_(0.65, 1.0)                     # SNR reduction
    z = z * a + torch.randn_like(z) * torch.sqrt(1 - a ** 2)
    zoom = torch.rand(b) < 0.5                                          # freq zoom-out by 0.6-1.0
    if zoom.any():
        zz = z[zoom]; n = len(zz); s = torch.empty(n).uniform_(0.6, 1.0)
        grid_x = torch.linspace(-1, 1, 128)[None, :] / s[:, None]          # sample outside -> noise fill
        gx = grid_x[:, None, :].expand(n, 96, 128); gy = torch.linspace(-1, 1, 96)[None, :, None].expand(n, 96, 128)
        src = zz.reshape(n, 1, 96, 128)
        out = F.grid_sample(src, torch.stack([gx, gy], -1), align_corners=True, padding_mode='zeros').reshape(n, 6, 16, 128)
        outside = (grid_x.abs() > 1)[:, None, None, :].expand_as(out)
        out[outside] = torch.randn(int(outside.sum()))
        z[zoom] = out
    sh = torch.randint(-20, 21, (b,))
    for i in range(b):
        if sh[i] != 0: z[i] = torch.roll(z[i], int(sh[i]), -1)
    return z

class Net(nn.Module):
    def __init__(s, w=32, nf=0):
        super().__init__(); s.nf = nf
        def blk(i, o, st): return nn.Sequential(nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                               nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        s.f = nn.Sequential(blk(3, w, 1), blk(w, w, 2), blk(w, 2 * w, 2), blk(2 * w, 4 * w, 2), blk(4 * w, 4 * w, 2))
        if nf: s.m = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        s.h = nn.Linear(8 * w + (64 if nf else 0), 9)
    def forward(s, x, f=None):
        h = s.f(x); h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        if s.nf: h = torch.cat([h, s.m(f)], 1)
        return s.h(h)

SETS = ['phys', 'tex', 'trk', 'rep', 'rep2']
def featmat(which):
    fn = {'clean': lambda q: f'fe_{q}_train.csv', 'test': lambda q: f'fe_{q}_test.csv', 'inj1': lambda q: f'fe_{q}_inj1.csv',
          'deg2': lambda q: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv'}.get(q, f'fe_{q}_train_deg2.csv')}[which]
    return pd.concat([pd.read_csv(os.path.join(D, fn(q))) for q in SETS], axis=1).values.astype(np.float32)

if __name__ == '__main__':
    tr, te = meta(); y = y_train(); fo = folds()
    Z = np.load(D + r'\z_train.npy', mmap_mode='r'); gtr = tr.slew_gap_rows.values
    if FOLD >= 0: tri, vai = np.where(fo != FOLD)[0], np.where(fo == FOLD)[0]
    else: tri, vai = np.arange(len(y)), None
    Zall = torch.from_numpy(np.asarray(Z, dtype=np.float32))
    EXTRA = os.environ.get('EXTRA', ''); DEGMIX = float(os.environ.get('DEGMIX', 0))
    Zd = np.load(D + r'\z_train_deg2.npy', mmap_mode='r') if (DEGMIX > 0 or EXTRA) else None
    Zdall = torch.from_numpy(np.asarray(Zd, dtype=np.float32)) if DEGMIX > 0 else None
    ylab = y.copy()
    if EXTRA:
        info = pd.read_csv(D + rf'\{EXTRA}_info.csv'); Zi = torch.from_numpy(np.load(D + rf'\z_{EXTRA}.npy').astype(np.float32))
        keep = np.where(fo[info.base.values] != FOLD)[0] if FOLD >= 0 else np.arange(len(info))
        Zall = torch.cat([Zall, Zi]); gtr = np.r_[gtr, info.slew_gap_rows.values]; ylab = np.r_[y, info.y.values]
        tri = np.r_[tri, len(y) + keep]
        if Zdall is not None: Zdall = torch.cat([Zdall, Zi])
    USEF = bool(os.environ.get('FEATS'))
    Fall = Fdall = Fv = Fdv = Ft = None
    if USEF:
        Fc = featmat('clean'); med = np.median(Fc, 0); iqr = np.subtract(*np.percentile(Fc, [75, 25], 0)) + 1e-3
        nrm = lambda a: torch.from_numpy(np.clip(np.nan_to_num((a - med) / iqr), -10, 10).astype(np.float32))
        Fall = nrm(Fc); Fdall = nrm(featmat('deg2')); Ft = nrm(featmat('test')); Fv, Fdv = Fall, Fdall
        if EXTRA: Fi = nrm(featmat(EXTRA)); Fall = torch.cat([Fall, Fi]); Fdall = torch.cat([Fdall, Fi])
    net = Net(nf=Fall.shape[1] if USEF else 0); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    BS = 64; steps = EP * (len(tri) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    cw = torch.tensor(len(y) / (9 * np.bincount(y[fo != FOLD] if FOLD >= 0 else y, minlength=9)), dtype=torch.float32) ** 0.5
    ck = D + rf'\ck_{TAG}_f{FOLD}.pt'; ep0 = 0
    if os.path.exists(ck):
        st = torch.load(ck); net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sch']); ep0 = st['ep'] + 1
    def predict(idx, Zs, gaps, Fs=None):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                b = idx[s:s + 256]; zb = torch.from_numpy(np.asarray(Zs[b], dtype=np.float32)); fb = Fs[b] if Fs is not None else None
                p = F.softmax(net(canvas(zb, gaps[b]), fb), 1) + F.softmax(net(canvas(zb.flip(-1), gaps[b]), fb), 1)
                out.append(p.numpy() / 2)
        return np.concatenate(out)
    log = open(ROOT + rf'\logs\cnn_{TAG}_f{FOLD}.log', 'a')
    for ep in range(ep0, EP):
        net.train(); t = time.time(); perm = np.random.permutation(tri); tl = 0
        for s in range(0, len(perm) - BS + 1, BS):
            b = perm[s:s + BS]; zb = Zall[b].clone(); fb = Fall[b].clone() if USEF else None
            if Zdall is not None:
                md_ = torch.rand(len(b)) < DEGMIX; zb[md_] = Zdall[b][md_]
                if USEF: fb[md_] = Fdall[b][md_]
            if USEF: fb = fb * (torch.rand_like(fb) > 0.15)
            xb = canvas(augment(zb), gtr[b])
            loss = F.cross_entropy(net(xb, fb), torch.from_numpy(ylab[b]), weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        msg = f'ep {ep} loss {tl / (len(perm) // BS):.4f} {time.time() - t:.0f}s'
        if vai is not None and (ep % 4 == 3 or ep == EP - 1):
            pv = predict(vai, Z, gtr, Fv); msg += f' val mF1 {mf1(y[vai], pv.argmax(1)):.4f}'
            np.save(D + rf'\oofpart_{TAG}_f{FOLD}.npy', pv)
            if Zd is not None or os.environ.get('EVALDEG'):
                Zd_ = Zd if Zd is not None else np.load(D + r'\z_train_deg2.npy', mmap_mode='r')
                pvd = predict(vai, Zd_, gtr, Fdv); msg += f' fPKS {mf1(y[vai], pvd.argmax(1)):.4f}'; np.save(D + rf'\oofdpart_{TAG}_f{FOLD}.npy', pvd)
            if ep == EP - 1: msg += ' ' + str(per_class(y[vai], pv.argmax(1)))
        print(msg, file=log, flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sched.state_dict(), ep=ep), ck)
    Zt = np.load(D + r'\z_test.npy', mmap_mode='r')
    np.save(D + rf'\ptepart_{TAG}_f{FOLD}.npy', predict(np.arange(len(te)), Zt, te.slew_gap_rows.values, Ft))
    print('done', file=log, flush=True)
