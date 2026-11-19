import numpy as np
from glob import glob
import os

for fpath in sorted(glob('output/checkpoints/*.npz')):
    with np.load(fpath, allow_pickle=True) as d:
        feats = d['feats_cams']           # object array (list of 4 arrays)
        feats = np.stack(feats)           # → proper (4, N, 2048) float32
    out = fpath.replace('.npz', '.npy')
    np.save(out, feats)
    print(f'  {os.path.basename(out)}  {feats.shape}')
