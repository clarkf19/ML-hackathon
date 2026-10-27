"""Builds colab/signal_hunters_kaggle.ipynb: the Colab training code for a Kaggle 'Save & Run All' background commit
(no browser, no Drive). Inputs: a private Kaggle Dataset with h4h_bundle.zip and pl_test*.npy.
Work dir /tmp/work (not saved); results copied to /kaggle/working/out (Output tab). Kaggle runs use tag suffix 'k'
and a different seed (+1000) than the Colab runs, so both can be ensembled."""
import json, os
ROOT = r'C:\Users\clark\hack4hack-r2'
exec(open(os.path.join(ROOT, 'colab', 'build_colab.py'), encoding='utf-8').read().split('cells = []')[0])   # src(), colabify()
nb_colab = json.load(open(os.path.join(ROOT, 'colab', 'signal_hunters_gpu.ipynb')))
C = [''.join(c['source']) for c in nb_colab['cells']]
patch = [c for c in C if 'patch cnn.py' in c][0]
last = C[-1]
W = '/tmp/work'
cells = []
def md(t): cells.append(dict(cell_type='markdown', metadata={}, source=t))
def code(t): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=t))
md("""# Signal Hunters - Kaggle background run
Right panel: **Accelerator = GPU T4 x2 (or P100)**, **Internet = On**, **Add Input -> your private dataset**
(with `h4h_bundle.zip` + `pl_test4.npy`). Set `CFGS` in the last cell, then **Save Version -> Save & Run All (Commit)**.
Results: notebook page -> Output -> `out/` (files `oofpart_*k_f*.npy`, `oofdpart_*`, `ptepart_*`).""")
code("import torch, glob, os, shutil, subprocess\n"
     "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
     "assert torch.cuda.is_available(), 'No GPU: set Accelerator to GPU in the right panel'\n"
     f"W = '{W}'\n"
     "for d in ['src', 'data', 'logs']: os.makedirs(W + '/' + d, exist_ok=True)\n"
     "os.makedirs('/kaggle/working/out', exist_ok=True)\n"
     "z = glob.glob('/kaggle/input/**/h4h_bundle.zip', recursive=True)\n"
     "zr = glob.glob('/kaggle/input/**/zr_train.npy', recursive=True)   # Kaggle may auto-extract zips\n"
     "print('bundle zip', z, '| extracted', zr)\n"
     "if not os.path.exists(W + '/data/zr_train.npy'):\n"
     "    if z: subprocess.run(['unzip', '-q', '-o', z[0], '-d', W], check=True)\n"
     "    else: src_dir = os.path.dirname(zr[0]); shutil.copytree(os.path.dirname(src_dir), W, dirs_exist_ok=True)\n"
     "for pat in ['pl_test*.npy', 'qmap_zr.npz', 'sig_train.npy']:\n"
     "    for f in glob.glob('/kaggle/input/**/' + pat, recursive=True):\n"
     "        if '/data/' not in f or not os.path.exists(W + '/data/' + os.path.basename(f)): shutil.copy(f, W + '/data/'); print('copied', os.path.basename(f))\n"
     "print(sorted(os.listdir(W + '/data')))")
for n in ['common.py', 'cnn.py']:
    code(f"%%writefile {W}/src/{n}\n" + colabify(src(n)).replace('/content/work', W))
kpatch = patch.replace('/content/work', W)
drop = "s = s.replace(\"ck = D + rf'/ck_{TAG}_f{FOLD}.pt'\", \"ck = '/content/drive/MyDrive/h4h_r2/out/' + f'ck_{TAG}_f{FOLD}.pt'\")\n"
assert drop in kpatch, 'ck line'
code(kpatch.replace(drop, ''))
klast = last.replace("CFGS = ['P4']", "CFGS = ['E3']")
klast = klast.replace("ENV = ENVS[CFG]; TAG = CFG + 'g'; EP = EPS.get(CFG, EPOCHS)",
                      "ENV = ENVS[CFG].replace('SEED=', 'SEED=10'); TAG = CFG + 'k'; EP = EPS.get(CFG, EPOCHS)   # Kaggle: other seed, tag suffix k")
klast = klast.replace('/content/work/data/pl_test4.npy', W + '/data/pl_test4.npy').replace('!cd /content/work &&', f'!cd {W} &&')
klast = klast.replace('!cp /content/work/data/*part_{TAG}_f{fold}.npy $DRV/out/ ; tail -2 /content/work/logs/cnn_{TAG}_f{fold}.log',
                      f'!cp {W}/data/*part_{{TAG}}_f{{fold}}.npy /kaggle/working/out/ ; tail -2 {W}/logs/cnn_{{TAG}}_f{{fold}}.log')
assert "TAG = CFG + 'k'" in klast and '/kaggle/working/out/' in klast and '/content' not in klast and '$DRV' not in klast
code(klast)
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), nbformat=4, nbformat_minor=5)
json.dump(nb, open(os.path.join(ROOT, 'colab', 'signal_hunters_kaggle.ipynb'), 'w'), indent=1)
print('written', len(cells), 'cells')
