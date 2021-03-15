import argparse
import os
import os.path as osp
import sys
from time import time
import numpy as np

from utils import process, time_to_string


class DistanceMatrix(object):

    def __init__(self, cmd):
        self.mat_dir = cmd['mat_dir']
        self.res_dir = cmd['res_dir']
        if not osp.exists(self.res_dir):
            os.makedirs(self.res_dir)

        self.initialize_distances = cmd['initialize_distances']
        self.normalized_size = cmd['normalized_size'] # The normalized matrix size used to compare pattern matrices
        self.batch_size = cmd['batch_size'] # The number of distances to be computed in a batch
        self.n_batch = cmd['n_batch'] # The number of batch to be computed
        self.p_norm = cmd['p_norm']

        self.__initialize__()

    def __initialize__(self):
        '''
        This function initializes the distance matrix and uploads some of the relevant information.
        It adds 4 new parameters to the class:
        - self.output: a text to print when running the algorithm.
        - self.song_list_indexed: a list of pairs '(index_song, song)' where 'song' is the name of the file and 'index_song' is its index.
        - self.n_songs: the number of songs.
        - self.dists: an array of size (n_songs x n_songs) corresponding to the distance between two songs. 
        '''
        if self.initialize_distances or (not osp.exists(osp.join(self.res_dir, 'song_list.txt'))):
            self.output = 'Song list and distance matrix initialized'
            self.song_list_indexed = [(i,s[:-4]) for (i,s) in enumerate(os.listdir(self.mat_dir))]
            self.n_songs = len(self.song_list_indexed)
            self.dists = np.zeros((self.n_songs, self.n_songs))

            with open(osp.join(self.res_dir, 'song_list.txt'), 'w') as song_list:
                for (index_song, song) in self.song_list_indexed:
                    song_list.write(str(index_song) + '\t' + song + '\n')
                song_list.close()

        else:
            self.song_list_indexed = []
            with open(osp.join(self.res_dir, 'song_list.txt'), 'r') as song_list:
                for line in song_list:
                    index_song, song = line.split('\n')[0].split('\t')
                    index_song = int(index_song)
                    self.song_list_indexed.append((index_song, song))
            self.n_songs = len(self.song_list_indexed)

            if osp.exists(osp.join(self.res_dir, 'dists.txt')):
                self.output = 'Song list and distance matrix uploaded'
                self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
            else:
                self.output = 'Song list uploaded and distance matrix created'
                self.dists = np.zeros((self.n_songs, self.n_songs))

    def distance(self, pat_mat1, pat_mat2):
        '''
        This function defines the distance we use between two pattern matrices.
        '''
        return np.mean(np.abs(pat_mat1 - pat_mat2)**self.p_norm)**(1/self.p_norm)

    def compute_batch(self):
        '''
        This function computes the distances of a single batch.
        It starts by normalizing all the matrices of the batch in the list 'batch_pat_mat'.
        Then it computes the distance between the songs of the batch and all the other songs.
        This process computes 'dists' by columns.
        '''
        start_time = time()
        uncomputed_columns = np.all(self.dists == 0, axis=0)

        if np.any(uncomputed_columns):
            start_index = np.where(uncomputed_columns)[0][0]

            # 'indices' is used to normalize the pattern matrices
            indices = np.arange(0, self.normalized_size, dtype=int)
            indices = np.reshape(indices, (1,self.normalized_size))
            indices = np.repeat(indices, self.normalized_size, axis=0)

            dists_to_compute = self.song_list_indexed[start_index:start_index+self.batch_size]
            batch_pat_mat = []
            for index_song, song in dists_to_compute:
                pat_mat = process(np.loadtxt(osp.join(self.mat_dir, song + '.txt'), delimiter='\t'))
                n_measures = np.size(pat_mat, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat = pat_mat[norm_indices,norm_indices.T]
                # After this last step, pat_mat is now a matrix of size (normalized_size x normalized_size)
                batch_pat_mat.append((index_song, pat_mat))

            time_spent = time_to_string(time() - start_time)
            print('Batch ready, start computing the distance ({})'.format(time_spent))
            start_time = time()
            index_song = 1

            # The rest of the algorithm fills the columns [start_index, start_index+batch_size-1] of 'dists'
            # It does so by computing one row at a time

            # Computing the rows from '0' to 'start_index - 1'
            for index_song1, song1 in self.song_list_indexed[:start_index]:
                pat_mat1 = process(np.loadtxt(osp.join(self.mat_dir, song1 + '.txt'), delimiter='\t'))
                n_measures = np.size(pat_mat1, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat1 = pat_mat1[norm_indices,norm_indices.T]

                for index_song2, pat_mat2 in batch_pat_mat:
                    self.dists[index_song1,index_song2] = self.distance(pat_mat1, pat_mat2)

                time_spent = time_to_string(time() - start_time)
                perc = int((100.*index_song)/self.n_songs)
                sys.stdout.write('\033[F\033[K')
                print('{}% of the distance computed ({})'.format(perc,time_spent))
                index_song += 1

            # Computing the rows from 'start_index' to 'start_index + batch_size - 1'
            for index_song1, pat_mat1 in batch_pat_mat:
                for index_song2, pat_mat2 in batch_pat_mat:
                    self.dists[index_song1,index_song2] = self.distance(pat_mat1, pat_mat2)

                time_spent = time_to_string(time() - start_time)
                perc = int((100.*index_song)/self.n_songs)
                sys.stdout.write('\033[F\033[K')
                print('{}% of the distance computed ({})'.format(perc,time_spent))
                index_song += 1

            # Computing the rows from 'start_index + batch_size' to the end
            for index_song1, song1 in self.song_list_indexed[start_index+self.batch_size:]:
                pat_mat1 = process(np.loadtxt(osp.join(self.mat_dir, song1 + '.txt'), delimiter='\t'))
                n_measures = np.size(pat_mat1, axis=0)
                norm_indices = np.floor(indices*n_measures/self.normalized_size).astype(int)
                pat_mat1 = pat_mat1[norm_indices,norm_indices.T]

                for index_song2, pat_mat2 in batch_pat_mat:
                    self.dists[index_song1,index_song2] = self.distance(pat_mat1, pat_mat2)

                time_spent = time_to_string(time() - start_time)
                perc = int((100.*index_song)/self.n_songs)
                sys.stdout.write('\033[F\033[K')
                print('{}% of the distance computed ({})'.format(perc,time_spent))
                index_song += 1

            sys.stdout.write('\033[F\033[K')

        np.savetxt(osp.join(self.res_dir, 'dists.txt'), self.dists, delimiter='\t')

        return time_to_string(time() - start_time)

    def compute(self):
        '''
        This function computes the distance matrix one batch a time.
        Note that if 'batch_size' x 'n_batch' < 'n_songs' then the distance matrix will not be filled.
        In this situation, the algorithm can be started from the previous stopping time by setting 'intialize_distances' to False
        '''
        start_time = time()
        print('Distance Matrix starting...')
        print(self.output)
        for i in range(self.n_batch):
            time_spent = time_to_string(time() - start_time)
            print('Batch {} of {} starting... ({})'.format(i+1,self.n_batch,time_spent))
            batch_time = self.compute_batch()
            sys.stdout.write('\033[F\033[K')
            print('Batch {} of {} done ({})'.format(i+1,self.n_batch,batch_time))
        time_algorithm = time_to_string(time() - start_time)
        print('Distance Matrix executed in {}'.format(time_algorithm))
        print('Matrix available as \'{}\''.format(osp.join(self.res_dir, 'dists.txt')))

        self.check()

    def check_completion(self):
        '''
        This function checks if all the columns of 'dists' are computed.
        '''
        uncomputed_columns = np.all(self.dists == 0, axis=0)
        if np.any(uncomputed_columns):
            print('\033[1;37;46m!!!\033[0;38;40m The matrix is not fully computed \033[1;37;46m!!!\033[0;38;40m')
            return False

        else:
            return True

    def check_values(self):
        '''
        This function checks if all the entries of 'dists' are between 0 and 1.
        '''
        check_values = (self.dists >= 0) & (self.dists < 1)
        if np.all(check_values):
            print('\033[1;37;42mCheck\033[0;38;40m no values outside of the range')
        else:
            for index_song1, song1 in self.song_list_indexed:
                for index_song2, song2 in self.song_list_indexed:
                    if index_song1 <= index_song2:
                        if not check_values[index_song1, index_song2]:
                            print(
                                '\033[1;31;43mERROR!\033[0;38;40m The distance between \'{}\' and \'{}\' is {}'.format(
                                    song1, song2, self.dists[index_song1, index_song2]
                                )
                            )

    def check_symmetry(self):
        '''
        This function checks if the matrix 'dists' is symmetric.
        '''
        check_symmetry = self.dists == self.dists.T
        if np.all(check_symmetry):
            print('\033[1;37;42mCheck\033[0;38;40m the matrix is symmetric')
        else:
            for index_song1, song1 in self.song_list_indexed:
                for index_song2, song2 in self.song_list_indexed:
                    if index_song1 <= index_song2:
                        if not check_symmetry[index_song1, index_song2]:
                            print(
                                '\033[1;31;43mERROR!\033[0;38;40m There is an asymetry between \'{}\' and \'{}\' : {} and {}'.format(
                                    song1, song2, self.dists[index_song1, index_song2], self.dists[index_song2, index_song1]
                                )
                            )

    def check(self):
        '''
        This function runs the different checks.
        '''
        if self.check_completion():
            self.check_values()
            self.check_symmetry()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, default='data/matrices')
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--initialize_distances', type=int, default=1)
    parser.add_argument('--normalized_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--p_norm', type=int, default=2)
    cmd = vars(parser.parse_args())
    dm = DistanceMatrix(cmd)
    dm.compute()