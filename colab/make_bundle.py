"""Zip what the Colab notebook needs for CFG='B3' -> colab/h4h_bundle.zip (data/ paths preserved).
B3 needs only the level-preserving arrays (zr_*), injection info, folds and metadata."""
import zipfile, os
R = r'C:\Users\clark\hack4hack-r2'
files = ['zr_train.npy', 'zr_test.npy', 'zr_train_deg2.npy', 'zr_inj1.npy', 'inj1_info.csv', 'folds.npy',
         'signal-hunters-round2/train_meta.csv', 'signal-hunters-round2/test_meta.csv', 'signal-hunters-round2/sample_submission.csv']
with zipfile.ZipFile(os.path.join(R, 'colab', 'h4h_bundle.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in files:
        z.write(os.path.join(R, 'data', f), 'data/' + f)
print('bundle', round(os.path.getsize(os.path.join(R, 'colab', 'h4h_bundle.zip')) / 1e6), 'MB', len(files), 'files')
