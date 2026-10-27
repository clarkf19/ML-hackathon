"""Builds colab/signal_hunters_gpu.ipynb embedding the local src files (common, cnn).
Colab flow: mount Drive -> GPU check -> unzip h4h_bundle.zip (made by colab/make_bundle.py) -> train CNN folds with
per-epoch checkpoints in Drive (resume after disconnect) -> oof/test probabilities saved as .npy in Drive."""
import json, os
ROOT = r'C:\Users\clark\hack4hack-r2'
def src(n): return open(os.path.join(ROOT, 'src', n), encoding='utf-8').read()
def colabify(s):  # local Windows paths -> Colab paths
    return (s.replace(r"r'C:\Users\clark\hack4hack-r2'", "'/content/work'")
             .replace(r"ROOT + r'\data'", "'/content/work/data'")
             .replace(r"D + r'\signal-hunters-round2'", "'/content/work/data/signal-hunters-round2'")
             .replace(r"sys.path.insert(0, r'C:\Users\clark\hack4hack-r2\src')", "sys.path.insert(0, '/content/work/src')")
             .replace("\\", "/"))
cells = []
def md(t): cells.append(dict(cell_type='markdown', metadata={}, source=t))
def code(t): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=t))
md("""# Signal Hunters - GPU CNN (Hack4Hack R2)
1. Upload `colab/h4h_bundle.zip` (made locally by `colab/make_bundle.py`) to `MyDrive/h4h_r2/`.
2. Runtime -> Change runtime type -> T4 GPU. Run all cells.
3. Checkpoints and outputs are written to `MyDrive/h4h_r2/out/`; after a disconnect just re-run all cells, training resumes from the last finished epoch.
4. Copy `out/oofpart_*`, `out/oofdpart_*`, `out/ptepart_*` back into the local `data/` folder for blending.

CFG: **`B3` (use this)** = B2 on level-preserving input (panel brightness kept); `B2` = canvas CNN + ETI/trap injection + 50% fake-PKS mixing.""")
code("from google.colab import drive; drive.mount('/content/drive')\n!nvidia-smi -L\nimport torch; print('cuda', torch.cuda.is_available())")
code("import os\nDRV = '/content/drive/MyDrive/h4h_r2'; os.makedirs(DRV + '/out', exist_ok=True)\n"
     "for d in ['src', 'data', 'logs']: os.makedirs('/content/work/' + d, exist_ok=True)\n"
     "if not os.path.exists('/content/work/data/z_train.npy'):\n"
     "    !cp $DRV/h4h_bundle.zip /content/ && cd /content/work && unzip -q -o /content/h4h_bundle.zip\n"
     "!ls /content/work/data | head -60")
for n in ['common.py', 'cnn.py']:
    code(f"%%writefile /content/work/src/{n}\n" + colabify(src(n)))
code("# patch cnn.py for GPU + checkpoints in Drive\n"
     "s = open('/content/work/src/cnn.py').read()\n"
     "s = s.replace('    net = make_net(USEF, Fall); opt', '    dev = \"cuda\"; net = make_net(USEF, Fall).to(dev); opt')\n"
     "assert 'dev = \"cuda\"; net = make_net(USEF, Fall).to(dev); opt' in s, 'GPU patch failed: dev not defined'\n"
     "s = s.replace('net(xb, fb)', 'net(xb.to(dev), fb.to(dev) if fb is not None else None)').replace('torch.from_numpy(ylab[b])', 'torch.from_numpy(ylab[b]).to(dev)')\n"
     "s = s.replace('weight=cw,', 'weight=cw.to(dev),')\n"
     "s = s.replace('F.softmax(net(canvas(zb, gaps[b]), fb), 1)', 'F.softmax(net(canvas(zb, gaps[b]).to(dev), fb.to(dev) if fb is not None else None), 1).cpu()')\n"
     "s = s.replace('F.softmax(net(canvas(zb.flip(-1), gaps[b]), fb), 1)', 'F.softmax(net(canvas(zb.flip(-1), gaps[b]).to(dev), fb.to(dev) if fb is not None else None), 1).cpu()')\n"
     "s = s.replace('torch.load(ck)', 'torch.load(ck, map_location=dev)')\n"
     "s = s.replace(\"ck = D + rf'/ck_{TAG}_f{FOLD}.pt'\", \"ck = '/content/drive/MyDrive/h4h_r2/out/' + f'ck_{TAG}_f{FOLD}.pt'\")\n"
     "assert s.count('.to(dev)') >= 6, 'GPU patch failed'\n"
     "open('/content/work/src/cnn.py', 'w').write(s); print('patched')")
code("# copy the pseudo-label file if it is in Drive (needed only for CFG='B3pl')\nimport os, shutil\n"
     "for f in ['pl_test.npy', 'pl_test2.npy', 'pl_test3.npy', 'pl_test4.npy', 'qmap_zr.npz', 'sig_train.npy']:\n"
     "    if os.path.exists(DRV + '/' + f): shutil.copy(DRV + '/' + f, '/content/work/data/' + f); print(f, 'copied')")
code("CFGS = ['P4']   # one or more configs, run one after another (each over all FOLDS), e.g. ['P4', 'E2'] for the night\n"
     "EPOCHS = 30\n"
     "FOLDS = [0, 1, 2, 3, 4]\n"
     "ENVS = {'B2': 'EXTRA=inj1 DEGMIX=0.5', 'B3': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5', 'B3w': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=1 WIDTH=48',\n"
     "        'B3pl': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=2 PL=pl_test.npy', 'B4': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 ONLINE=1 SEED=3',\n"
     "        'B3plw': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=4 WIDTH=48 PL=pl_test.npy', 'B3pl2': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=5 PL=pl_test2.npy',\n"
     "        'B3pl3w': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=8 WIDTH=48 PL=pl_test3.npy',\n"
     "        'P1': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=6 PL=pl_test2.npy ARCH=rn34 PRETRAIN=1 LR=5e-4',\n"
     "        'P2': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=7 PL=pl_test2.npy ARCH=rn34 PRETRAIN=1 LR=5e-4 MIXUP=0.4',\n"
     "        'P3': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=9 PL=pl_test3.npy ARCH=rn34 PRETRAIN=1 LR=5e-4',\n"
     "        'E1': 'ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=10 PL=pl_test3.npy ARCH=tv_efficientnet_b0 PRETRAIN=1 LR=1e-3',\n"
     "        'P4': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=11 PL=pl_test4.npy ARCH=rn34 PRETRAIN=1 LR=5e-4',\n"
     "        'E2': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=12 PL=pl_test4.npy ARCH=tv_efficientnet_b2 PRETRAIN=1 LR=1e-3',\n"
     "        'C1': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=13 PL=pl_test4.npy ARCH=tv_convnext_tiny PRETRAIN=1 LR=3e-4',\n"
     "        'P5': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=14 PL=pl_test4.npy ARCH=tv_resnet34 PRETRAIN=1 LR=5e-4 UPS=3',\n"
     "        'P6': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=15 PL=pl_test4.npy ARCH=rn34 PRETRAIN=1 LR=5e-4 MIXUP=0.4',\n"
     "        'E3': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=16 PL=pl_test4.npy ARCH=tv_efficientnet_b0 PRETRAIN=1 LR=1e-3',\n"
     "        'R50': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=17 PL=pl_test4.npy ARCH=tv_resnet50 PRETRAIN=1 LR=5e-4',\n"
     "        'E4': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=18 PL=pl_test4.npy ARCH=tv_efficientnet_b1 PRETRAIN=1 LR=1e-3',\n"
     "        'E5': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=19 PL=pl_test4.npy ARCH=tv_efficientnet_b0 PRETRAIN=1 LR=1e-3 UPS=3',\n"
     "        'E6': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=20 PL=pl_test4.npy ARCH=tv_efficientnet_b0 PRETRAIN=1 LR=1e-3',\n"
     "        'E7': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=21 PL=pl_test4.npy ARCH=tv_efficientnet_b3 PRETRAIN=1 LR=1e-3',\n"
     "        'E8': 'AMP=1 ZPFX=zr EXTRA=inj1 DEGMIX=0.5 SEED=22 PL=pl_test4.npy ARCH=tv_efficientnet_b2 PRETRAIN=1 LR=1e-3 UPS=3'}\n"
     "EPS = {'P1': 20, 'P2': 24, 'P3': 20, 'E1': 20, 'P4': 20, 'E2': 20, 'C1': 20, 'P5': 20, 'P6': 24, 'E3': 20, 'R50': 20, 'E4': 20, 'E5': 20, 'E6': 30, 'E7': 20, 'E8': 20}\n"
     "import os\n"
     "for CFG in CFGS:\n"
     "    ENV = ENVS[CFG]; TAG = CFG + 'g'; EP = EPS.get(CFG, EPOCHS)\n"
     "    if 'pl_test4.npy' in ENV and not os.path.exists('/content/work/data/pl_test4.npy'):\n"
     "        ENV = ENV.replace('pl_test4', 'pl_test3'); print('pl_test4.npy not found -> using pl_test3.npy')\n"
     "    print('=== running', CFG, ENV, flush=True)\n"
     "    for fold in FOLDS:\n"
     "        !cd /content/work && {ENV} NT=4 python src/cnn.py {fold} {EP} {TAG}\n"
     "        !cp /content/work/data/*part_{TAG}_f{fold}.npy $DRV/out/ ; tail -2 /content/work/logs/cnn_{TAG}_f{fold}.log")
nb = dict(cells=cells, metadata=dict(accelerator='GPU', kernelspec=dict(name='python3', display_name='Python 3')), nbformat=4, nbformat_minor=5)
json.dump(nb, open(os.path.join(ROOT, 'colab', 'signal_hunters_gpu.ipynb'), 'w'), indent=1)
print('written', len(cells), 'cells')
