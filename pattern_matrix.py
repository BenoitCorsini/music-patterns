import os
import os.path as osp
from time import time
import random   

import numpy as np
import matplotlib.pyplot as plt
import guitarpro

from song import Song


def process(pat_mat):
    processed = 0*pat_mat.copy()

    for p in range(1,101):
        processed += pat_mat >= np.percentile(pat_mat, p)

    processed = processed - np.min(processed)
    processed = processed / np.max(processed)

    return processed


class PatternMatrix(object):

    def __init__(self, cmd):
        self.tab_dir = cmd['tab_dir']
        self.tabs_list = os.listdir(self.tab_dir)

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
        colour_dict = {
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
        colour_list = ['white', 'blue', 'red', 'yellow', 'green', 'orange', 'purple', 'cyan', 'salmon', 'pink', 'black', 'grey', 'gray']
        if self.colour in colour_list:
            return colour_dict[colour_list]
        elif self.colour == 'random':
            random_colour = random.choice(colour_list[:-3])
            return colour_dict[random_colour]
        else:
            return colour_dict['blue']


    def save_pattern_matrix(self, pat_mat, song_name):
        im_to_be_saved = self.overwrite_im or (not osp.exists(osp.join(self.im_dir, song_name + '.png')))
        if self.save_im & im_to_be_saved:
            (n1,n2) = np.shape(pat_mat)
            color_mat = np.zeros((n1,n2,3))

            processed = process(pat_mat)**.5

            color_mat[:,:,0] = 2*(0.5 - processed)*(processed < 0.5)
            color_mat[:,:,1] = 1 - processed
            color_mat[:,:,2] = 1 - 2*(processed - 0.5)*(processed > 0.5)
            colors_mat = color_mat[:,:,self.get_indices_from_colour()]

            plt.figure(figsize=(5,5))
            plt.imshow(colors_mat, interpolation='nearest')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.savefig(osp.join(self.im_dir, song_name + '.png'), dpi=2*n1)
            plt.close()

        if self.overwrite_mat or (not osp.exists(osp.join(self.mat_dir, song_name + '.txt'))):
            np.savetxt(osp.join(self.mat_dir, song_name + '.txt'), pat_mat, delimiter='\t')

    def run(self):
        for tab in self.tabs_list:
            if tab[-3:-1] != 'gp':
                print('\033[1;31;43mERROR!\033[0;38;40m There is a non guitarpro file: {}'.format(tab))
            else:
                song_name = tab[:-4]
                song_gp = guitarpro.parse(osp.join(self.tab_dir, tab))
                song = Song(song_gp)
                self.save_pattern_matrix(song.pattern_matrix(), song_name)

if __name__ == '__main__':
    cmd = {}
    cmd['tab_dir'] = 'data/tablatures'
    cmd['mat_dir'] = 'data/matrices'
    cmd['im_dir'] = 'data/images'
    cmd['save_im'] = True
    cmd['overwrite_mat'] = True
    cmd['overwrite_im'] = True
    cmd['colour'] = 'random'
    pm = PatternMatrix(cmd)
    pm.run()  
