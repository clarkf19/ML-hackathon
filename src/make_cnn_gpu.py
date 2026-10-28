"""Writes src/cnn_gpu.py: src/cnn.py with the same CUDA patch the Colab notebook applies, for the local RTX GPU.
Run cnn_gpu.py with a CUDA-enabled Python (the drowsiness-project venv); pandas/lightgbm come from the main Python311
site-packages (same Python 3.11 / numpy version), appended AFTER the venv's own packages so the CUDA torch wins."""
import os
SRC = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(SRC, 'cnn.py'), encoding='utf-8').read()
head = "import sys; import torch, torchvision; sys.path.insert(0, r'C:\\Users\\clark\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages')   # CUDA torch from the venv, sklearn/pandas from Python311 (venv sklearn DLLs are blocked)\n"
s = s.replace('    net = make_net(USEF, Fall); opt', '    dev = "cuda"; net = make_net(USEF, Fall).to(dev); opt')
assert 'dev = "cuda"; net = make_net(USEF, Fall).to(dev); opt' in s, 'GPU patch failed: dev not defined'
s = s.replace('net(xb, fb)', 'net(xb.to(dev), fb.to(dev) if fb is not None else None)').replace('torch.from_numpy(ylab[b])', 'torch.from_numpy(ylab[b]).to(dev)')
s = s.replace('weight=cw,', 'weight=cw.to(dev),')
s = s.replace('F.softmax(net(canvas(zb, gaps[b]), fb), 1)', 'F.softmax(net(canvas(zb, gaps[b]).to(dev), fb.to(dev) if fb is not None else None), 1).cpu()')
s = s.replace('F.softmax(net(canvas(zb.flip(-1), gaps[b]), fb), 1)', 'F.softmax(net(canvas(zb.flip(-1), gaps[b]).to(dev), fb.to(dev) if fb is not None else None), 1).cpu()')
s = s.replace('torch.load(ck)', 'torch.load(ck, map_location=dev)')
s = s.replace('BS = 64;', "BS = int(os.environ.get('BS', 64));")   # 8 GB laptop GPU: BS=32 avoids spilling into shared memory
assert s.count('.to(dev)') >= 6, 'GPU patch failed'
open(os.path.join(SRC, 'cnn_gpu.py'), 'w', encoding='utf-8').write(head + s); print('written src/cnn_gpu.py')
