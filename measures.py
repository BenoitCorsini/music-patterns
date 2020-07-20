import numpy as np
import os
import os.path as osp
from time import time

from pattern_matrix import process

class DistanceMatrix(object):

    def __init__(self, cmd):
        self.mat_dir = cmd['mat_dir']
        self.res_dir = cmd['res_dir']
        if not osp.exists(self.res_dir):
            os.makedirs(self.res_dir)

        self.normalized_size = cmd['normalized_size']
        self.batch_size = cmd['batch_size']
        self.n_iterations = cmd['n_iterations']
        self.initialize_distances = cmd['initialize_distances']

        self.__initialize__()

    def __initialize__(self):
        if self.initialize_distances or (not osp.exists(osp.join(self.res_dir, 'song_list.txt'))):
            self.mat_list_indexed = [(i,m) for (i,m) in enumerate(os.listdir(self.mat_dir))]
            self.n_songs = len(self.mat_list_indexed)
            self.dists = np.zeros((self.n_songs, self.n_songs))

            with open(osp.join(self.res_dir, 'song_list.txt'), 'w') as songs:
                for (index_mat, mat) in self.mat_list_indexed:
                    songs.write(str(index_mat) + '\t' + mat + '\n')

        else:
            self.mat_list_indexed = []
            for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
                index_mat, mat = line.split('\n')[0].split('\t')
                index_mat = int(index_mat)
                self.mat_list_indexed.append((index_mat, mat))
            self.n_songs = len(self.mat_list_indexed)

            if osp.exists(osp.join(self.res_dir, 'dists.txt')):
                self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
            else:
                self.dists = np.zeros((self.n_songs, self.n_songs))

    def distance(self, pat_mat1, pat_mat2):
        return 100*np.mean(np.abs(pat_mat1 - pat_mat2))

    def compute_batch(self):
        uncomputed_columns = np.all(self.dists == 0, axis=0)
        if np.any(uncomputed_columns):
            start_index = np.where(uncomputed_columns)[0][0]
            indices = np.arange(0, self.normalized_size, dtype=int)
            indices = np.reshape(indices, (1,self.normalized_size))
            indices = np.repeat(indices, self.normalized_size, axis=0)

            dists_to_compute = self.mat_list_indexed[start_index:start_index+self.batch_size]
            batch_pat_mat = []
            for (index_mat,mat) in dists_to_compute:
                pat_mat = process(np.loadtxt(osp.join(self.mat_dir, mat), delimiter='\t'))
                n_measures = np.size(pat_mat, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat = pat_mat[norm_indices,norm_indices.T]
                batch_pat_mat.append((index_mat, pat_mat))

            for (index_mat1,mat1) in self.mat_list_indexed[:start_index]:
                pat_mat1 = process(np.loadtxt(osp.join(self.mat_dir, mat1), delimiter='\t'))
                n_measures = np.size(pat_mat1, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat1 = pat_mat1[norm_indices,norm_indices.T]
                for (index_mat2,pat_mat2) in batch_pat_mat:
                    self.dists[index_mat1,index_mat2] = self.distance(pat_mat1, pat_mat2)

            for (index_mat1,pat_mat1) in batch_pat_mat:
                for (index_mat2,pat_mat2) in batch_pat_mat:
                    self.dists[index_mat1,index_mat2] = self.distance(pat_mat1, pat_mat2)

            for (index_mat1,mat1) in self.mat_list_indexed[start_index+self.batch_size:]:
                pat_mat1 = process(np.loadtxt(osp.join(self.mat_dir, mat1), delimiter='\t'))
                n_measures = np.size(pat_mat1, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat1 = pat_mat1[norm_indices,norm_indices.T]
                for (index_mat2,pat_mat2) in batch_pat_mat:
                    self.dists[index_mat1,index_mat2] = self.distance(pat_mat1, pat_mat2)

        np.savetxt(osp.join(self.res_dir, 'dists.txt'), self.dists, delimiter='\t')

    def compute(self):
        for i in range(self.n_iterations):
            self.compute_batch()


if __name__ == '__main__':
    cmd = {}
    cmd['mat_dir'] = 'data/matrices'
    cmd['res_dir'] = 'data/results'
    cmd['normalized_size'] = 500
    cmd['batch_size'] = 200
    cmd['n_iterations'] = 5
    cmd['initialize_distances'] = True
    dm = DistanceMatrix(cmd)
    dm.compute()
