"""Check that the fast featurize path reproduces the original phys / track / trk2 code (max abs difference per column).
usage: python src/gen_check_port.py [N]"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import generate
from featurize import check_equal

n = int(sys.argv[1]) if len(sys.argv) > 1 else 90
X, g, y, _ = generate(n, 'GBT', 7, labels=np.arange(n) % 9)
d = check_equal(X, g)
print('max |fast - original| over', len(d), 'columns:', float(d.max()), ' worst:', d.head(5).round(6).to_dict())
