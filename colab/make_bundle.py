"""Zip the files the Colab notebook needs -> colab/h4h_bundle.zip (data/ paths preserved)."""
import zipfile, os, glob
R = r'C:\Users\clark\hack4hack-r2'
files = ['z_train.npy', 'z_test.npy', 'z_train_deg2.npy', 'z_inj1.npy', 'zr_train.npy', 'zr_test.npy', 'zr_train_deg2.npy', 'zr_inj1.npy', 'inj1_info.csv', 'folds.npy',
         'signal-hunters-round2/train_meta.csv', 'signal-hunters-round2/test_meta.csv', 'signal-hunters-round2/sample_submission.csv']
for q in ['phys', 'tex', 'trk', 'rep', 'rep2']:
    files += [f'fe_{q}_train.csv', f'fe_{q}_test.csv', f'fe_{q}_inj1.csv']
files += ['fe_phys_train_deg2.csv', 'fe_texd_deg2.csv', 'fe_trkd_deg2.csv', 'fe_rep_train_deg2.csv', 'fe_rep2_train_deg2.csv']
with zipfile.ZipFile(os.path.join(R, 'colab', 'h4h_bundle.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in files:
        z.write(os.path.join(R, 'data', f), 'data/' + f)
print('bundle', round(os.path.getsize(os.path.join(R, 'colab', 'h4h_bundle.zip')) / 1e6), 'MB', len(files), 'files')
