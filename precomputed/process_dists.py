import os
import os.path as osp
import numpy as np


N_SONGS = 4166
DISTS_DIR = 'precomputed/dists'
RES_DIR = 'precomputed/results'


dists = np.zeros((N_SONGS, N_SONGS))

dist_files = sorted(os.listdir(DISTS_DIR))
for dist_file in dist_files:
    dist = open(osp.join(DISTS_DIR, dist_file))
    for line in dist:
        i, j, dij = line.strip().split('\t')
        dists[int(i), int(j)] = dij
    dist.close()

dists += dists.T
np.savetxt(osp.join(RES_DIR, 'dists.txt'), dists, delimiter='\t')