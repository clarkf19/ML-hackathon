"""Pixel-level smoke test of generator.py: byte noise std (robust), ln-power std, dropped/glitch rows, label mix.
usage: python src/gen_smoke.py [N]"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import generate, LABELS

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for tel in ('GBT', 'PKS'):
    X, g, y, info = generate(n, tel, 0, labels=np.full(n, 8))
    x = X.astype(np.float32); x[x == 0] = np.nan
    r = x - np.nanmedian(x, axis=2, keepdims=True)
    rob = 1.4826 * np.nanmedian(np.abs(r))
    print(f'{tel}: noise-class robust byte std {rob:.2f} (target GBT 11.9 / PKS 15.6), ln std {np.nanstd(r) / 24 * np.log(2):.3f} '
          f'(0.343/0.45), dropped/cad {info.n_drop.mean():.3f}, glitch rows/cad {info.n_glitch.mean():.3f}, '
          f'cad with glitch {(info.n_glitch > 0).mean():.3f}, median byte {np.nanmedian(x):.1f}, '
          f'skew {np.nanmean(((r - np.nanmean(r)) / np.nanstd(r)) ** 3):.3f}')
X, g, y, info = generate(900, 'GBT', 1)
print('label mix', dict(zip(LABELS, np.bincount(y, minlength=9))), 'secondary', info.secondary.value_counts().head(6).to_dict())
