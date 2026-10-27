"""B: CNN on the global-time canvas. Panels are placed at their real global time
(p*(16+gap)+r) in a 128x128 image, so an ETI track is ONE straight line; channels = [z, ON mask, valid mask].
Augment: freq flip, freq shift, SNR reduction (PKS is noisier), freq zoom-out (PKS drift in pixels is smaller), row drop.
Usage: cnn.py FOLD [EPOCHS] [TAG]   (FOLD=-1 -> train on all, predict test)"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, sys, time, os
sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')
from common import *
torch.set_num_threads(int(os.environ.get('NT', 8))); SEED = int(os.environ.get('SEED', 0)); torch.manual_seed(SEED); np.random.seed(SEED)
FOLD = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 24; TAG = sys.argv[3] if len(sys.argv) > 3 else 'B'
AUG = os.environ.get('AUG', 'full'); ZP = os.environ.get('ZPFX', 'z')

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

def online_pks(z, sd):
    """Fresh fake-PKS rendering in the normalised (zr) domain. z: (b,6,16,128) clean, sd: (b,) GBT noise scale in bytes."""
    b = len(z); miss = (z == 0).all(-1, keepdim=True)
    x = z * sd[:, None, None, None]                                                    # bytes relative to bandpass/level
    K = torch.empty(b).uniform_(10.0, 15.5)[:, None, None, None]
    x = x + 24 * torch.log2(torch.distributions.Gamma(K.expand_as(x), K.expand_as(x)).sample())
    x = x + torch.randn(b, 6, 1, 1) * 1.9                                              # panel gain jitter (test-matched)
    gl = torch.rand(b, 6, 16, 1) < torch.empty(b, 1, 1, 1).uniform_(0.0057, 0.009)    # glitch rows
    amp = torch.empty(b, 6, 16, 1).uniform_(18, 60)
    lo = torch.randint(0, 64, (b, 6, 16, 1)); part = torch.rand(b, 6, 16, 1) < 0.3
    ch = torch.arange(128)[None, None, None, :]
    band = torch.where(part, (ch >= lo) & (ch < lo + torch.randint(40, 64, (b, 6, 16, 1))), torch.ones_like(ch, dtype=torch.bool))
    x = x + (gl & band).float() * amp
    drop = miss | (torch.rand(b, 6, 16, 1) < 0.0023)
    v = x.masked_fill(drop, float('nan')).reshape(b, -1)
    med = v.nanmedian(1).values; sd2 = 1.4826 * (v - med[:, None]).abs().nanmedian(1).values
    zz = (x - med[:, None, None, None]) / sd2[:, None, None, None]
    zz = interp(zz, QA, QB).clamp(-20, 40)
    return zz.masked_fill(drop.expand_as(zz), 0.0)

def interp(x, xp, fp):
    i = torch.bucketize(x.contiguous(), xp).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    out = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    return torch.where(x > xp[-1], fp[-1] + (x - xp[-1]), torch.where(x < xp[0], fp[0] + (x - xp[0]), out))

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

class BasicBlock(nn.Module):          # parameter names identical to torchvision's -> ImageNet weights load directly
    def __init__(s, i, o, st):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.bn1 = nn.BatchNorm2d(o); s.relu = nn.ReLU(inplace=True)
        s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.bn2 = nn.BatchNorm2d(o)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if (st != 1 or i != o) else None
    def forward(s, x):
        idt = x if s.downsample is None else s.downsample(x)
        return s.relu(s.bn2(s.conv2(s.relu(s.bn1(s.conv1(x))))) + idt)

class ResNetTV(nn.Module):
    """torchvision-compatible ResNet18/34 on the canvas, upsampled 2x; optional ImageNet init (PRETRAIN=1, needs torchvision)."""
    def __init__(s, arch='rn34'):
        super().__init__(); s.nf = 0
        layers = {'rn18': (2, 2, 2, 2), 'rn34': (3, 4, 6, 3)}[arch]
        s.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); s.bn1 = nn.BatchNorm2d(64); s.relu = nn.ReLU(inplace=True); s.maxpool = nn.MaxPool2d(3, 2, 1)
        inp = 64
        for li, (n, c) in enumerate(zip(layers, [64, 128, 256, 512])):
            setattr(s, f'layer{li + 1}', nn.Sequential(*([BasicBlock(inp, c, 1 if li == 0 else 2)] + [BasicBlock(c, c, 1) for _ in range(n - 1)]))); inp = c
        s.fc = nn.Linear(512, 9)
        if os.environ.get('PRETRAIN'):
            import torchvision
            sd = getattr(torchvision.models, {'rn18': 'resnet18', 'rn34': 'resnet34'}[arch])(weights='IMAGENET1K_V1').state_dict()
            miss, unexp = s.load_state_dict({k: v for k, v in sd.items() if not k.startswith('fc.')}, strict=False)
            print('ImageNet weights loaded; missing', miss, 'unexpected', unexp, flush=True)
    def forward(s, x, f=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = s.layer4(s.layer3(s.layer2(s.layer1(s.maxpool(s.relu(s.bn1(s.conv1(x))))))))
        return s.fc(x.mean((2, 3)))

def make_net(USEF, Fall):
    a = os.environ.get('ARCH', '')
    if a: return ResNetTV(a)
    return Net(w=int(os.environ.get('WIDTH', 32)), nf=Fall.shape[1] if USEF else 0)

SETS = ['phys', 'tex', 'trk', 'rep', 'rep2']
def featmat(which):
    fn = {'clean': lambda q: f'fe_{q}_train.csv', 'test': lambda q: f'fe_{q}_test.csv', 'inj1': lambda q: f'fe_{q}_inj1.csv',
          'deg2': lambda q: {'tex': 'fe_texd_deg2.csv', 'trk': 'fe_trkd_deg2.csv'}.get(q, f'fe_{q}_train_deg2.csv')}[which]
    return pd.concat([pd.read_csv(os.path.join(D, fn(q))) for q in SETS], axis=1).values.astype(np.float32)

if __name__ == '__main__':
    tr, te = meta(); y = y_train(); fo = folds()
    Z = np.load(D + rf'\{ZP}_train.npy', mmap_mode='r'); gtr = tr.slew_gap_rows.values
    if FOLD >= 0: tri, vai = np.where(fo != FOLD)[0], np.where(fo == FOLD)[0]
    else: tri, vai = np.arange(len(y)), None
    Zall = torch.from_numpy(np.asarray(Z, dtype=np.float32))
    ONLINE = bool(os.environ.get('ONLINE'))
    if ONLINE:
        qz = np.load(D + r'\qmap_zr.npz'); global QA, QB
        QA, QB = torch.tensor(qz['q_deg'], dtype=torch.float32), torch.tensor(qz['q_test'], dtype=torch.float32)
        SD = torch.from_numpy(np.load(D + r'\sig_train.npy').astype(np.float32))
    EXTRA = os.environ.get('EXTRA', ''); DEGMIX = float(os.environ.get('DEGMIX', 0))
    Zd = np.load(D + rf'\{ZP}_train_deg2.npy', mmap_mode='r') if (DEGMIX > 0 or EXTRA) else None
    Zdall = torch.from_numpy(np.asarray(Zd, dtype=np.float32)) if DEGMIX > 0 else None
    ylab = y.copy()
    if EXTRA:
        info = pd.read_csv(D + rf'\{EXTRA}_info.csv'); Zi = torch.from_numpy(np.load(D + rf'\{ZP}_{EXTRA}.npy').astype(np.float32))
        keep = np.where(fo[info.base.values] != FOLD)[0] if FOLD >= 0 else np.arange(len(info))
        Zall = torch.cat([Zall, Zi]); gtr = np.r_[gtr, info.slew_gap_rows.values]; ylab = np.r_[y, info.y.values]
        tri = np.r_[tri, len(y) + keep]
        if Zdall is not None: Zdall = torch.cat([Zdall, Zi])
        if ONLINE: SD = torch.cat([SD, torch.full((len(Zi),), float(SD.median()))])
    if os.environ.get('PL'):   # confident pseudo-labelled REAL test cadences (labels -1 = unused); never in any val fold
        pl = np.load(D + rf"\{os.environ['PL']}"); ip = np.where(pl >= 0)[0]
        Zp = torch.from_numpy(np.asarray(np.load(D + rf'\{ZP}_test.npy', mmap_mode='r')[ip], dtype=np.float32))
        n0 = len(Zall); Zall = torch.cat([Zall, Zp]); gtr = np.r_[gtr, te.slew_gap_rows.values[ip]]; ylab = np.r_[ylab, pl[ip]]
        tri = np.r_[tri, n0 + np.arange(len(ip))]
        if Zdall is not None: Zdall = torch.cat([Zdall, Zp])
        if ONLINE: SD = torch.cat([SD, torch.zeros(len(Zp))])   # sd 0 marks real PKS rows: never re-degraded
    USEF = bool(os.environ.get('FEATS'))
    Fall = Fdall = Fv = Fdv = Ft = None
    if USEF:
        Fc = featmat('clean'); med = np.median(Fc, 0); iqr = np.subtract(*np.percentile(Fc, [75, 25], 0)) + 1e-3
        nrm = lambda a: torch.from_numpy(np.clip(np.nan_to_num((a - med) / iqr), -10, 10).astype(np.float32))
        Fall = nrm(Fc); Fdall = nrm(featmat('deg2')); Ft = nrm(featmat('test')); Fv, Fdv = Fall, Fdall
        if EXTRA: Fi = nrm(featmat(EXTRA)); Fall = torch.cat([Fall, Fi]); Fdall = torch.cat([Fdall, Fi])
    net = make_net(USEF, Fall); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    BS = 64; steps = EP * (len(tri) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, float(os.environ.get('LR', 3e-3)), total_steps=steps)
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
            if ONLINE:
                md_ = (torch.rand(len(b)) < DEGMIX) & (SD[b] > 0)
                if md_.any(): zb[md_] = online_pks(zb[md_], SD[b][md_])
            elif Zdall is not None:
                md_ = torch.rand(len(b)) < DEGMIX; zb[md_] = Zdall[b][md_]
                if USEF: fb[md_] = Fdall[b][md_]
            if USEF: fb = fb * (torch.rand_like(fb) > 0.15)
            xb = canvas(augment(zb), gtr[b])
            MIX = float(os.environ.get('MIXUP', 0))
            if MIX > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(MIX, MIX)); pm = torch.randperm(len(b)); xb = lam * xb + (1 - lam) * xb[pm]
            else:
                lam, pm = 1.0, torch.arange(len(b))
            out = net(xb, fb); yt = torch.from_numpy(ylab[b])
            loss = lam * F.cross_entropy(out, yt, weight=cw, label_smoothing=0.05) + (1 - lam) * F.cross_entropy(out, yt[pm], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        msg = f'ep {ep} loss {tl / (len(perm) // BS):.4f} {time.time() - t:.0f}s'
        if vai is not None and (ep % 4 == 3 or ep == EP - 1):
            pv = predict(vai, Z, gtr, Fv); msg += f' val mF1 {mf1(y[vai], pv.argmax(1)):.4f}'
            np.save(D + rf'\oofpart_{TAG}_f{FOLD}.npy', pv)
            if Zd is not None or os.environ.get('EVALDEG'):
                Zd_ = Zd if Zd is not None else np.load(D + rf'\{ZP}_train_deg2.npy', mmap_mode='r')
                pvd = predict(vai, Zd_, gtr, Fdv); msg += f' fPKS {mf1(y[vai], pvd.argmax(1)):.4f}'; np.save(D + rf'\oofdpart_{TAG}_f{FOLD}.npy', pvd)
            if ep == EP - 1: msg += ' ' + str(per_class(y[vai], pv.argmax(1)))
        print(msg, file=log, flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sched.state_dict(), ep=ep), ck)
    Zt = np.load(D + rf'\{ZP}_test.npy', mmap_mode='r')
    np.save(D + rf'\ptepart_{TAG}_f{FOLD}.npy', predict(np.arange(len(te)), Zt, te.slew_gap_rows.values, Ft))
    print('done', file=log, flush=True)
