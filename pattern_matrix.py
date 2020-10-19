import argparse
import os
import os.path as osp
from time import time
import random   

import numpy as np
import matplotlib.pyplot as plt
import guitarpro

from utils import process, time_to_string
from song import Song


COLOUR_DICT = {
    'white': np.array([1, 1, 1]),
    'black': np.array([1, 1, 1]),
    'gray': np.array([1, 1, 1]),
    'grey': np.array([1, 1, 1]),
    'blue': np.array([0, 1, 2]),
    'red': np.array([1, 0, 0]),
    'yellow': np.array([1, 1, 0]),
    'green': np.array([0, 1, 0]),
    'orange': np.array([2, 1, 0]),
    'purple': np.array([1, 0, 2]),
    'cyan': np.array([0, 1, 1]),
    'salmon': np.array([2, 1, 1]),
    'pink': np.array([1, 0, 1])
}


class PatternMatrix(object):

    def __init__(self, cmd):
        self.tab_dir = cmd['tab_dir']
        self.tab_list = os.listdir(self.tab_dir)

        self.mat_dir = cmd['mat_dir']
        if not osp.exists(self.mat_dir):
            os.makedirs(self.mat_dir)

        self.im_dir = cmd['im_dir']
        self.save_im = cmd['save_im']
        if self.save_im & (not osp.exists(self.im_dir)):
            os.makedirs(self.im_dir)

        self.overwrite_mat = cmd['overwrite_mat']
        self.overwrite_im = cmd['overwrite_im']
        self.colour = cmd['colour']

    def get_indices_from_colour(self):
        '''
        This function returns indices according to the value of the parameter 'colour'.
        '''
        colour_list = list(COLOUR_DICT.keys())
        if self.colour in colour_list:
            return COLOUR_DICT[self.colour]
        elif self.colour == 'random':
            random_colour = random.choice(colour_list)
            return COLOUR_DICT[random_colour]
        else:
            return COLOUR_DICT['blue']

    def save_pattern_matrix(self, pat_mat, song_name):
        '''
        This function takes a pattern matrix and saves the results.
        The matrix will be saved in any case but the image can be chosen to be saved or not.
        '''
        if self.overwrite_mat or (not osp.exists(osp.join(self.mat_dir, song_name + '.txt'))):
            np.savetxt(osp.join(self.mat_dir, song_name + '.txt'), pat_mat, delimiter='\t')

        im_to_be_saved = self.overwrite_im or (not osp.exists(osp.join(self.im_dir, song_name + '.png')))
        if self.save_im & im_to_be_saved:
            (n1,n2) = np.shape(pat_mat)
            color_mat = np.zeros((n1,n2,3))

            processed = process(pat_mat)**.5 # The sqrt here is used to improve the clarity of the figures.

            color_mat[:,:,0] = 2*(0.5 - processed)*(processed < 0.5)
            color_mat[:,:,1] = 1 - processed
            color_mat[:,:,2] = 1 - 2*(processed - 0.5)*(processed > 0.5)
            color_mat = color_mat[:,:,self.get_indices_from_colour()]

            plt.figure(figsize=(5,5))
            plt.imshow(color_mat, interpolation='nearest')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.savefig(osp.join(self.im_dir, song_name + '.png'), dpi=2*n1)
            plt.close()

    def run(self):
        '''
        This function computes the pattern matrices of the songs in 'tab_dir' and saves the results.
        '''
        start_time = time()
        print('Pattern Matrices starting')
        for tab in self.tab_list:
            if tab[-3:-1] != 'gp':
                print('\033[1;31;43mERROR!\033[0;38;40m There is a non guitarpro file: {}'.format(tab))
            else:
                time_spent = time_to_string(time() - start_time)
                print('Processing {}... ({})'.format(tab, time_spent))

                # This part deletes the id of the song if downloaded through 'TabScroller'.
                if ' (id=' in tab:
                    song_name = tab.split(' (id=')[0]
                else:
                    song_name = tab[:-4]

                try:
                    song_gp = guitarpro.parse(osp.join(self.tab_dir, tab))
                    song = Song(song_gp)
                    self.save_pattern_matrix(song.pattern_matrix(), song_name)
                except:
                    print('\033[1;31;43mERROR!\033[0;38;40m There is an issue with file {}'.format(tab))

        time_algorithm = time_to_string(time() - start_time)
        print('Pattern Matrices executed in {}'.format(time_algorithm))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tab_dir', type=str, default='data/tablatures')
    parser.add_argument('--mat_dir', type=str, default='data/matrices')
    parser.add_argument('--im_dir', type=str, default='data/images')
    parser.add_argument('--save_im', type=int, default=1)
    parser.add_argument('--overwrite_mat', type=int, default=1)
    parser.add_argument('--overwrite_im', type=int, default=1)
    parser.add_argument('--colour', type=str, default='random')
    cmd = vars(parser.parse_args())
    pm = PatternMatrix(cmd)
    pm.run()  
