"""Gallery of generated cadences (2 per class, normalised like prep.py) -> results/gen_gallery_{TEL}.png
usage: python src/gen_gallery.py [TEL] [SEED]"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import generate, LABELS
from prep import normalise

tel = sys.argv[1] if len(sys.argv) > 1 else 'GBT'; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
lab = np.repeat(np.arange(9), 2)
X, g, y, info = generate(len(lab), tel, seed, labels=lab)
z, _ = normalise(X, spike=(tel == 'GBT'))
fig, ax = plt.subplots(len(lab), 6, figsize=(12, 0.95 * len(lab)))
for r in range(len(lab)):
    for p in range(6):
        ax[r, p].imshow(z[r, p].astype(np.float32), aspect='auto', cmap='magma', vmin=-2, vmax=5)
        ax[r, p].set_xticks([]); ax[r, p].set_yticks([])
    sec = info.secondary[r]
    ax[r, 0].set_ylabel(LABELS[y[r]][:12] + ('\n+' + sec[:14] if sec else '') + f'\ngap {g[r]}', fontsize=5, rotation=0, ha='right')
plt.subplots_adjust(wspace=0.03, hspace=0.08, left=0.12, right=0.99, top=0.99, bottom=0.01)
Path('results').mkdir(exist_ok=True); plt.savefig(f'results/gen_gallery_{tel}.png', dpi=80); print('saved')
