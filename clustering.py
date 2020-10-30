import os
import os.path as osp
import numpy as np
from time import time
from shutil import copyfile, rmtree
from random import sample
from sklearn.cluster import AgglomerativeClustering as AC
import argparse
import json


class SongClustering(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']
        self.clusters_dir = cmd['clusters_dir']

        self.AC_division = cmd['AC_division']
        self.clusters_threshold = cmd['clusters_threshold']
        self.clusters_ordering = cmd['clusters_ordering']
        self.write_clusters_info = cmd['write_clusters_info']

        self.clusters = {}
        self.clusters_info = {}

        self.__get_measures__()

    def __get_measures__(self):
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[index_song] = song.split('.txt')[0]
        self.n_songs = len(self.song_dict)

    def get_clusters_recursive(self, labels, cluster_id):
        cluster_dists = self.dists[labels,:][:,labels]
        cluster_size = np.sum(labels)
        if cluster_size <= self.clusters_threshold:
            cluster_dists = np.mean(cluster_dists)
            self.clusters_info[cluster_id] = {
                  'dist' : cluster_dists,
                  'size' : int(cluster_size),
                  'depth' : len(cluster_id)
            }
            for index_song in np.where(labels)[0]:
                self.clusters[self.song_dict[index_song]] = cluster_id
        else:
            cluster_labels = AC(n_clusters=self.AC_division, affinity='precomputed', linkage='complete').fit_predict(cluster_dists)
            for l in range(self.AC_division):
                new_labels = labels.copy()
                new_labels[labels] = cluster_labels == l
                self.get_clusters_recursive(new_labels, cluster_id + str(l))

    def get_clusters(self):
        self.get_clusters_recursive(np.ones(self.n_songs) == 1, '')

        for (index_cluster, (cluster_id, cluster_info)) in enumerate(sorted(self.clusters_info.items(), key=lambda x:x[1].get(self.clusters_ordering, x[0]))):
            if cluster_info['size'] == 1:
                self.clusters_info[cluster_id]['name'] = 'Singleton Cluster'
            else:
                self.clusters_info[cluster_id]['name'] = 'Cluster {}'.format(index_cluster + 1)
                if self.write_clusters_info:
                    self.clusters_info[cluster_id]['name'] += ' (dist={:.2f},size={},depth={})'.format(100*cluster_info['dist'],cluster_info['size'],cluster_info['depth'])

    def move_images(self):
        if osp.exists(self.clusters_dir):
            rmtree(self.clusters_dir)
        os.makedirs(self.clusters_dir)
        for (song, cluster_id) in self.clusters.items():
            cluster_path = osp.join(self.clusters_dir, self.clusters_info[cluster_id]['name'])
            if not osp.exists(cluster_path):
                os.makedirs(cluster_path)
            im = song + '.png'
            copyfile(osp.join(self.im_dir, im), osp.join(cluster_path, im))

    def run(self):
        self.get_clusters()
        self.move_images()

        with open(osp.join(self.res_dir, 'clusters.json'), 'w') as clusters:
            json.dump(self.clusters, clusters, indent=2)
            clusters.close()
        with open(osp.join(self.res_dir, 'clusters_info.json'), 'w') as clusters_info:
            json.dump(self.clusters_info, clusters_info, indent=2)
            clusters_info.close()



class SongNeighbouring(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']
        self.neighbours_dir = cmd['neighbours_dir']

        self.n_neighbours = cmd['n_neighbours']
        self.write_neighbours_info = cmd['write_neighbours_info']
        self.sub_sample = cmd['sub_sample']
        self.random_sub_sample = cmd['random_sub_sample']

        self.neighbours = {}

        self.__get_measures__()

    def __get_measures__(self):
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[index_song] = song.split('.txt')[0]
        self.n_songs = len(self.song_dict)
        if self.sub_sample > self.n_songs:
            self.sub_sample = None

    def get_neighbours(self):
        neighbours = {}
        for index_song, song in self.song_dict.items():
            self.neighbours[song] = {}
            neighbours = {}
            neighbours_dists = self.dists[index_song, :]
            neighs = neighbours_dists <= np.percentile(neighbours_dists, (100.*(self.n_neighbours + 1))/self.n_songs)
            neighbours_dists = np.mean(neighbours_dists[neighs])
            for index_n in np.where(neighs)[0]:
                if index_n != index_song:
                    neighbours[self.song_dict[index_n]] = self.dists[index_song, index_n]
            self.neighbours[song]['dist'] = neighbours_dists
            self.neighbours[song]['neighbours'] = neighbours

    def move_images(self):
        if osp.exists(self.neighbours_dir):
            rmtree(self.neighbours_dir)
        os.makedirs(self.neighbours_dir)
        if self.sub_sample is None:
            list_neighbours = list(self.neighbours.items())
        elif self.random_sub_sample:
            list_neighbours = sample(list(self.neighbours.items()), self.sub_sample)
        else:
            list_neighbours = sorted(self.neighbours.items(), key=lambda x:x[1].get('dist', 1))[:self.sub_sample]
        
        for (song, neigh_infos) in list_neighbours:
            neighbours_dists = neigh_infos['dist']
            neighbours = neigh_infos['neighbours']
            dir_name = song
            if self.write_neighbours_info:
                dir_name += ' (dist={:.2f})'.format(100*neighbours_dists)
            if not osp.exists(osp.join(self.neighbours_dir, dir_name)):
                os.makedirs(osp.join(self.neighbours_dir, dir_name))

            im = song + '.png'
            copyfile(osp.join(self.im_dir, im), osp.join(self.neighbours_dir, dir_name, 'CENTER: ' + im))
            for (index_neighbour, (neighbour,_)) in enumerate(sorted(neighbours.items(), key=lambda x:x[1])):
                im = neighbour + '.png'
                copyfile(osp.join(self.im_dir, im), osp.join(self.neighbours_dir, dir_name, 'N{}: '.format(index_neighbour + 1) + im))
        

    def run(self):
        self.get_neighbours()
        self.move_images()

        with open(osp.join(self.res_dir, 'neighbours.json'), 'w') as neighbours:
            json.dump(self.neighbours, neighbours, indent=2)
            neighbours.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', type=str, default='images')
    parser.add_argument('--res_dir', type=str, default='results')

    parser.add_argument('--clusters_dir', type=str, default='clusters')
    parser.add_argument('--AC_division', type=int, default=2)
    parser.add_argument('--clusters_threshold', type=int, default=10)
    parser.add_argument('--clusters_ordering', type=str, default='size')
    parser.add_argument('--write_clusters_info', type=int, default=1)

    parser.add_argument('--neighbours_dir', type=str, default='neighbours')
    parser.add_argument('--n_neighbours', type=int, default=10)
    parser.add_argument('--write_neighbours_info', type=int, default=1)
    parser.add_argument('--sub_sample', type=int, default=20)
    parser.add_argument('--random_sub_sample', type=int, default=1)
    cmd = vars(parser.parse_args())
    sc = SongClustering(cmd)
    sc.run()
    #sn = SongNeighbouring(cmd)
    #sn.run()
