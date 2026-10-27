"""Predict-only pass over a trained CNN fold checkpoint: AdaBN (re-estimate BatchNorm statistics on the target images,
no labels) + richer test-time augmentation (frequency flip x frequency shifts). Device-agnostic (CPU or GPU).
Writes oofpart/oofdpart/ptepart with tag suffix 'a' (e.g. P1g -> P1ga) so the original predictions are untouched.
usage: same env as training (ZPFX, ARCH, WIDTH, ...) + CK=<checkpoint path>; python cnn_predict.py FOLD TAG
env ADABN=1 (default) / 0, SHIFTS='0,-8,8'"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], sys.argv[1], '1', sys.argv[2]]
import cnn as C                                        # defines canvas, make_net, D, meta, folds ... (main is guarded)
FOLD, TAG = int(sys.argv[1]), sys.argv[3]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ZP = os.environ.get('ZPFX', 'z'); ADA = os.environ.get('ADABN', '1') == '1'
SHIFTS = [int(s) for s in os.environ.get('SHIFTS', '0,-8,8').split(',')]
tr, te = C.meta(); y = C.y_train(); fo = C.folds()
net = C.make_net(False, None).to(dev)
st = torch.load(os.environ['CK'], map_location=dev); net.load_state_dict(st['net']); net.eval()
base = {k: v.clone() for k, v in net.state_dict().items()}

def adabn(Zs, gaps, idx):
    net.load_state_dict(base)
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.reset_running_stats(); m.momentum = None
    net.train()
    with torch.no_grad():
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]; zb = torch.from_numpy(np.asarray(Zs[b], dtype=np.float32))
            net(C.canvas(zb, gaps[b]).to(dev))
    net.eval()

def predict(Zs, gaps, idx):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]; zb = torch.from_numpy(np.asarray(Zs[b], dtype=np.float32)); p = 0
            for flip in (False, True):
                zf = zb.flip(-1) if flip else zb
                for sh in SHIFTS:
                    zs = torch.roll(zf, sh, -1) if sh else zf
                    p = p + F.softmax(net(C.canvas(zs, gaps[b]).to(dev)), 1).float().cpu()
            out.append((p / (2 * len(SHIFTS))).numpy())
    return np.concatenate(out)

vai = np.where(fo == FOLD)[0]; g = tr.slew_gap_rows.values; gt = te.slew_gap_rows.values
Zc = np.load(os.path.join(C.D, f'{ZP}_train.npy'), mmap_mode='r'); Zd = np.load(os.path.join(C.D, f'{ZP}_train_deg2.npy'), mmap_mode='r')
Zt = np.load(os.path.join(C.D, f'{ZP}_test.npy'), mmap_mode='r'); allt = np.arange(len(te))
res = {}
for nm, Zs, gg, idx in [('oofpart', Zc, g, vai), ('oofdpart', Zd, g, vai), ('ptepart', Zt, gt, allt)]:
    if ADA: adabn(Zs, gg, idx)                     # adapt BN to THIS image set (clean val / fake-PKS val / real test)
    else: net.load_state_dict(base)
    res[nm] = predict(Zs, gg, idx); np.save(os.path.join(C.D, f'{nm}_{TAG}a_f{FOLD}.npy'), res[nm])
print(f'fold {FOLD} {TAG}a: clean {C.mf1(y[vai], res["oofpart"].argmax(1)):.4f}  fPKS {C.mf1(y[vai], res["oofdpart"].argmax(1)):.4f}  (ADABN={ADA}, shifts={SHIFTS})', flush=True)
