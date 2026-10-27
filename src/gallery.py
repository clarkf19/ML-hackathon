import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
D=r'C:\Users\clark\hack4hack-r2\data'
z=np.load(D+r'\z_train.npy',mmap_mode='r'); tr=pd.read_csv(D+r'\signal-hunters-round2\train_meta.csv')
labs=['eti_candidate','rfi_intermittent','rfi_persistent','satellite_pass','pulsar','frb','radar_chirp','scintillating_star','noise']
k=int(sys.argv[1]) if len(sys.argv)>1 else 2
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 0)
rows=[(l,i) for l in labs for i in rng.choice(np.where(tr.label==l)[0],k,replace=False)]
fig,ax=plt.subplots(len(rows),6,figsize=(12,0.95*len(rows)))
for r,(l,i) in enumerate(rows):
    for p in range(6):
        ax[r,p].imshow(z[i,p].astype(np.float32),aspect='auto',cmap='magma',vmin=-2,vmax=5); ax[r,p].set_xticks([]); ax[r,p].set_yticks([])
    ax[r,0].set_ylabel(l[:10]+f'\n{i}',fontsize=6,rotation=0,ha='right')
plt.subplots_adjust(wspace=0.03,hspace=0.08,left=0.1,right=0.99,top=0.99,bottom=0.01)
plt.savefig(sys.argv[3] if len(sys.argv)>3 else r'C:\Users\clark\hack4hack-r2\logs\gallery.png',dpi=70)
