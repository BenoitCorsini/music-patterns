import os
import os.path as osp
import numpy as np
from time import time
from shutil import copyfile, rmtree
from random import sample, random
from sklearn.cluster import AgglomerativeClustering as AC
import argparse
import json
import matplotlib.pyplot as plt


class ClusteringParams(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']

        self.order_by = cmd['order_by']
        self.n_folders = cmd['n_folders']
        self.write_info = cmd['write_info']

        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[index_song] = song

        self.n_songs = len(self.song_dict)
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')

        if self.n_folders is None:
            self.n_folders = self.n_songs

    def save_results(self, grouping_dict, name, title, xlabel, plot_size):
        with open(osp.join(self.res_dir, name + '.json'), 'w') as grouping:
            json.dump(grouping_dict, grouping, indent=2)
            grouping.close()

        y = []
        std = []
        size = []
        for group_info in sorted(grouping_dict.values(), key=lambda x:x['number']):
            if group_info.get('size', 2) > 1:
                y.append(group_info['dist'])
                std.append(group_info['std_dist'])
                size.append(group_info.get('size', 1))

        if y:
            x = 1 + np.arange(len(y))
            y = np.array(y)
            std = np.array(std)
            size = np.array(size)
            size = size*np.max(std)/np.max(size)

            xmin = 0
            xmax = np.size(x) + 1
            ymin = np.min(y) - .55*np.max(std)
            ymax = np.max(y) + .55*np.max(std)

            plt.figure(figsize=(16,9))
            plt.axis(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
            plt.plot(x, y, color='navy', marker='.', markersize=20, linewidth=0, label='mean')
            plt.errorbar(x, y, .5*std, color='royalblue', label='std', linewidth=5, ls='', alpha=.5)
            if plot_size:
                plt.bar(x, ymin+size, color='red', alpha=.2, width=.7, label='size')
            else:
                plt.bar(x, ymin+std, color='red', alpha=.2, width=.7, label='std')
            plt.plot(x, y, color='navy', marker='.', markersize=20, linewidth=0)
            plt.legend()
            plt.xlabel(xlabel)
            plt.title(title)
            plt.savefig(osp.join(self.res_dir, name + '.png'), dpi=250)
            plt.close()


class SongClustering(ClusteringParams):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.clusters_dir = cmd['clusters_dir']
        self.AC_division = cmd['AC_division']
        self.cluster_size_threshold = cmd['cluster_size_threshold']
        self.clusters = {}

    def get_clusters_recursive(self, labels, cluster_id):
        cluster_dists = self.dists[labels,:][:,labels]
        cluster_size = np.sum(labels)
        if cluster_size <= self.cluster_size_threshold:
            cluster_songs = []
            for index_song in np.where(labels)[0]:
                cluster_songs.append(self.song_dict[index_song])

            self.clusters[cluster_id] = {
                  'dist' : np.mean(cluster_dists),
                  'std_dist' : np.std(cluster_dists),
                  'songs' : cluster_songs,
                  'size' : int(cluster_size),
                  'depth' : len(cluster_id)
            }
        else:
            cluster_labels = AC(n_clusters=self.AC_division, affinity='precomputed', linkage='complete').fit_predict(cluster_dists)
            for l in range(self.AC_division):
                new_labels = labels.copy()
                new_labels[labels] = cluster_labels == l
                self.get_clusters_recursive(new_labels, cluster_id + str(l))

    def get_clusters(self):
        self.get_clusters_recursive(np.ones(self.n_songs) == 1, '')

        index_cluster = 1
        for cluster_info in sorted(self.clusters.values(), key=lambda x:x.get(self.order_by, random())):
            if cluster_info['size'] == 1:   
                cluster_info['name'] = 'Singleton Cluster'
                cluster_info['info'] = ''
                cluster_info['number'] = -1
            else:
                cluster_info['name'] = 'Cluster {}'.format(index_cluster)
                cluster_info['info'] = ' (dist={:.2f},size={},depth={})'.format(100*cluster_info['dist'],cluster_info['size'],cluster_info['depth'])
                cluster_info['number'] = index_cluster
                index_cluster += 1

    def move_images(self):
        if osp.exists(self.clusters_dir):
            rmtree(self.clusters_dir)
        os.makedirs(self.clusters_dir)
        for cluster_info in self.clusters.values():
            if cluster_info['number'] < self.n_folders:
                if self.write_info:
                    cluster_path = osp.join(self.clusters_dir, cluster_info['name'] + cluster_info['info'])
                else:
                    cluster_path = osp.join(self.clusters_dir, cluster_info['name'])
                if not osp.exists(cluster_path):
                    os.makedirs(cluster_path)
                for song in cluster_info['songs']:
                    im = song + '.png'
                    copyfile(osp.join(self.im_dir, im), osp.join(cluster_path, im))

    def run(self):
        self.get_clusters()
        self.move_images()
        self.save_results(self.clusters, 'clusters', 'Properties of the ordered clusters', 'Clusters', True)



class SongNeighbouring(ClusteringParams):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.neighbours_dir = cmd['neighbours_dir']
        self.n_neighbours = min(cmd['n_neighbours'], self.n_songs - 1)
        self.neighbours = {}

    def get_neighbours(self):
        neighbours = {}
        for index_song, song in self.song_dict.items():
            self.neighbours[song] = {}
            neighbours = {}
            neighbours_dists = self.dists[index_song, :]
            neighs = neighbours_dists <= np.percentile(neighbours_dists, (100.*(self.n_neighbours + 1))/self.n_songs)
            neighbour_songs = {}
            for index_n in np.where(neighs)[0]:
                if index_n != index_song:
                    neighbour_songs[self.song_dict[index_n]] = {
                          'dist' : self.dists[index_song, index_n],
                    }
            for index_song, song_info in enumerate(sorted(neighbour_songs.values(), key=lambda x:x['dist'])):
                song_info['number'] = index_song + 1

            neighbours_dists = neighbours_dists[neighs]
            self.neighbours[song] = {
                  'dist' : np.mean(neighbours_dists),
                  'std_dist' : np.std(neighbours_dists),
                  'center' : song,
                  'neighbours' : neighbour_songs
            }

        index_song = 1
        for song_info in sorted(self.neighbours.values(), key=lambda x:x.get(self.order_by, random())):
            song_info['name'] = 'Song {}: {}'.format(index_song, song_info['center'])
            song_info['info'] = ' (dist={:.2f})'.format(100*song_info['dist'])
            song_info['number'] = index_song
            index_song += 1

    def move_images(self):
        if osp.exists(self.neighbours_dir):
            rmtree(self.neighbours_dir)
        os.makedirs(self.neighbours_dir)
        for song_info in self.neighbours.values():
            if song_info['number'] <= self.n_folders:
                if self.write_info:
                    song_path = osp.join(self.neighbours_dir, song_info['name'] + song_info['info'])
                else:
                    song_path = osp.join(self.neighbours_dir, song_info['name'])
                if not osp.exists(song_path):
                    os.makedirs(song_path)
                im = song_info['center'] + '.png'
                im_copy = 'Center: ' + im
                copyfile(osp.join(self.im_dir, im), osp.join(song_path, im_copy))
                for neigh, neigh_info in song_info['neighbours'].items():
                    im = neigh + '.png'
                    im_copy = 'N{}: '.format(neigh_info['number']) + im
                    copyfile(osp.join(self.im_dir, im), osp.join(song_path, im_copy))

    def run(self):
        self.get_neighbours()
        self.move_images()
        self.save_results(self.neighbours, 'neighbours', 'Properties of the songs ordered according to their neighbourhood', 'Songs', False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', type=str, default='data/images')
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--order_by', type=str, default='dist')
    parser.add_argument('--n_folders', type=int, default=None)
    parser.add_argument('--write_info', type=int, default=1)

    parser.add_argument('--clusters_dir', type=str, default='results/clusters')
    parser.add_argument('--AC_division', type=int, default=2)
    parser.add_argument('--cluster_size_threshold', type=int, default=3)

    parser.add_argument('--neighbours_dir', type=str, default='results/neighbours')
    parser.add_argument('--n_neighbours', type=int, default=3)
    cmd = vars(parser.parse_args())
    sc = SongClustering(cmd)
    sc.run()
    sn = SongNeighbouring(cmd)
    sn.run()
