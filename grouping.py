import argparse
import os
import os.path as osp
import json
import shutil
import sys
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids

from utils import time_to_string


MEAN_COLOUR = 'navy'
STD_COLOUR = 'royalblue'
SIZE_COLOUR = 'red'


class Grouping(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']

        self.order_by = cmd['order_by']
        self.n_folders = cmd['n_folders'] # The number of groups saved
        self.write_group_info = cmd['write_group_info']

        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[index_song] = song

        self.n_songs = len(self.song_dict)
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')

        if self.n_folders is None:
            self.n_folders = self.n_songs
        self.n_folders = min(self.n_folders, self.n_songs)

    def save_results(self, grouping_dict, name, title, xlabel, plot_size):
        '''
        This function saves the result of a grouping technique.
        It does so by first saving the grouping dictionary as a json file and then plots a summary of the results in a figure.
        Its inputs are:
        - grouping_dict: a dictionary of dictionaries, where each subdictionary contains the information of a single group.
        - name: one of 'clusters' or 'neighbours', to indentify the dictionary.
        - title: the title of the figure to be plotted.
        - xlabel: one of 'cluster' or 'song', to be used as label for the horizontal axis.
        - plot_size: boolean. If True, the size of the groups are represented in the figure.

        '''
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
            plt.plot(x, y, color=MEAN_COLOUR, marker='.', markersize=20, linewidth=0, label='mean')
            plt.errorbar(x, y, .5*std, color=STD_COLOUR, label='std', linewidth=5, ls='', alpha=.5)
            if plot_size:
                plt.bar(x, ymin+size, color=SIZE_COLOUR, alpha=.2, width=.7, label='size')
            else:
                plt.bar(x, ymin+std, color=SIZE_COLOUR, alpha=.2, width=.7, label='std')
            plt.plot(x, y, color=MEAN_COLOUR, marker='.', markersize=20, linewidth=0)
            plt.legend()
            plt.xlabel(xlabel)
            plt.title(title)
            plt.savefig(osp.join(self.res_dir, name + '.png'), dpi=250)
            plt.close()


class SongClustering(Grouping):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.clusters_dir = cmd['clusters_dir']
        self.n_clusters = cmd['n_clusters']
        self.cluster_size_threshold = cmd['cluster_size_threshold']
        self.max_iter = cmd['max_iter']
        if self.max_iter is None:
            self.max_iter = self.n_songs
        self.max_iter = max(self.max_iter, 1)
        self.clusters = {}

        if cmd['clustering_method'].lower() in ['sc', 'spectral clustering', 'spectral', 'spectralclustering']:
            self.method = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed')
        elif cmd['clustering_method'].lower() in ['km', 'k medoids', 'medoids', 'kmedoids', 'k-medoids']:
            self.method = KMedoids(n_clusters=self.n_clusters, metric='precomputed')
        else:
            self.method = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed', linkage='complete')

        self.total_clusters = 0

    def get_clusters_recursive(self, labels, cluster_id):
        '''
        This function applies a new iteration of clustering to the cluster defined by 'labels'.
        If the size of this cluster is below the size threshold, the algorithm stops.
        If the number of iteration of the clustering algorithm is higher than 'max_iter', the algorithm stops.
        When the algorithm stops, it adds the information of the cluster to its dictionary.
        Otherwise, it applies the clustering method another time.
        '''
        cluster_dists = self.dists[labels,:][:,labels]
        cluster_size = np.sum(labels)
        cluster_iter = len(cluster_id.split('.')) - 2
        if (cluster_size <= self.cluster_size_threshold) or (cluster_iter >= self.max_iter):
            cluster_songs = []
            for index_song in np.where(labels)[0]:
                cluster_songs.append(self.song_dict[index_song])

            self.clusters[cluster_id[1:-1]] = {
                'dist' : np.mean(cluster_dists),
                'std_dist' : np.std(cluster_dists),
                'songs' : cluster_songs,
                'size' : int(cluster_size),
                'iter' : cluster_iter
            }

            self.total_clusters += 1
            sys.stdout.write('\033[F\033[K')
            print('{} clusters done'.format(self.total_clusters))

        else:
            cluster_labels = self.method.fit_predict(cluster_dists)
            for l in range(self.n_clusters):
                new_labels = labels.copy()
                new_labels[labels] = cluster_labels == l
                self.get_clusters_recursive(new_labels, cluster_id + str(l) + '.')

    def get_clusters(self):
        '''
        This function uses 'get_clusters_recursive' to cluster the songs.
        Once all the clusters are obtained, it sorts them according to 'order_by' (or randomly).
        '''
        self.get_clusters_recursive(np.ones(self.n_songs) == 1, '.')
        sys.stdout.write('\033[F\033[K')
        print('{} clusters found'.format(self.total_clusters))

        index_cluster = 1
        for cluster_info in sorted(self.clusters.values(), key=lambda x:x.get(self.order_by, random.random())):
            if cluster_info['size'] == 1:   
                cluster_info['name'] = 'Singleton Cluster'
                cluster_info['info'] = ''
                cluster_info['number'] = -1
            else:
                cluster_info['name'] = 'Cluster {}'.format(index_cluster)
                cluster_info['info'] = ' (dist={:.2f},size={},iter={})'.format(100*cluster_info['dist'],cluster_info['size'],cluster_info['iter'])
                cluster_info['number'] = index_cluster
                index_cluster += 1

        return index_cluster - 1

    def move_images(self):
        '''
        This function copies the song images into their corresponding clusters.
        It creates the folders corresponding to the first 'n_folders' clusters.
        '''
        if osp.exists(self.clusters_dir):
            shutil.rmtree(self.clusters_dir) #remove previously computed clustering if exists
        os.makedirs(self.clusters_dir)

        for cluster_info in self.clusters.values():
            if cluster_info['number'] <= self.n_folders:
                if self.write_group_info:
                    cluster_path = osp.join(self.clusters_dir, cluster_info['name'] + cluster_info['info'])
                else:
                    cluster_path = osp.join(self.clusters_dir, cluster_info['name'])

                if not osp.exists(cluster_path):
                    os.makedirs(cluster_path)

                for song in cluster_info['songs']:
                    im = song + '.png'
                    shutil.copyfile(osp.join(self.im_dir, im), osp.join(cluster_path, im))

    def run(self):
        '''
        This function runs the clustering algorithm.
        It does so by computing the clusters, moving the images accordingly, and saving and plotting the results.
        '''
        start_time = time()
        print('Song Clustering starting...')
        print('{} clusters done'.format(self.total_clusters))
        non_singletons = self.get_clusters()
        self.move_images()
        self.save_results(self.clusters, 'clusters', 'Properties of the ordered clusters', 'Clusters', True)
        sys.stdout.write('\033[F\033[K')
        print('Song Clustering executed in {}'.format(time_to_string(time() - start_time)))
        print('{} non-trivial clusters found, {} in total'.format(non_singletons, self.total_clusters))
        print('Clusters saved in \'{}\' and images copied to \'{}\''.format(osp.join(self.res_dir, 'clusters.json'), self.clusters_dir))
        print('The cluster properties are represented in \'{}\''.format(osp.join(self.res_dir, 'clusters.png')))



class SongNeighbouring(Grouping):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.neighbours_dir = cmd['neighbours_dir']
        self.n_neighbours = min(cmd['n_neighbours'], self.n_songs - 1)
        self.neighbours = {}

        self.total_neighbours = 0

    def get_neighbours(self):
        '''
        This function computes the nearest neighbours for each song.
        '''
        for index_song, song in self.song_dict.items():
            self.total_neighbours += 1
            neighbours = {}
            neighbours_dists = self.dists[index_song, :]
            indices = np.argsort(neighbours_dists)[1:1+self.n_neighbours]

            for (number_n, index_n) in enumerate(indices):
                neighbours[self.song_dict[index_n]] = {
                    'dist' : neighbours_dists[index_n],
                    'number' : number_n + 1,
                }
            neighbours_dists = neighbours_dists[indices]

            self.neighbours[song] = {
                  'dist' : np.mean(neighbours_dists),
                  'std_dist' : np.std(neighbours_dists),
                  'center' : song,
                  'neighbours' : neighbours
            }

            sys.stdout.write('\033[F\033[K')
            print('{} neighbourhoods done'.format(self.total_neighbours))

        index_song = 1
        for song_info in sorted(self.neighbours.values(), key=lambda x:x.get(self.order_by, random.random())):
            song_info['name'] = 'Song {}: {}'.format(index_song, song_info['center'])
            song_info['info'] = ' (dist={:.2f})'.format(100*song_info['dist'])
            song_info['number'] = index_song
            index_song += 1

    def move_images(self):
        '''
        This function copies the song images into their corresponding neighbourhoods.
        It creates the folders corresponding to the first 'n_folders' song neighbourhoods.
        When copying the images to the folders, it adds the information regarding the role of the song (center or neighbour).
        '''
        if osp.exists(self.neighbours_dir):
            shutil.rmtree(self.neighbours_dir)
        os.makedirs(self.neighbours_dir)

        for song_info in self.neighbours.values():
            if song_info['number'] <= self.n_folders:
                if self.write_group_info:
                    song_path = osp.join(self.neighbours_dir, song_info['name'] + song_info['info'])
                else:
                    song_path = osp.join(self.neighbours_dir, song_info['name'])

                if not osp.exists(song_path):
                    os.makedirs(song_path)

                im = song_info['center'] + '.png'
                im_copy = 'Center: ' + im
                shutil.copyfile(osp.join(self.im_dir, im), osp.join(song_path, im_copy))
                for neigh, neigh_info in song_info['neighbours'].items():
                    im = neigh + '.png'
                    im_copy = 'N{}: '.format(neigh_info['number']) + im
                    shutil.copyfile(osp.join(self.im_dir, im), osp.join(song_path, im_copy))

    def run(self):
        '''
        This function runs the neighbouring algorithm.
        It does so by computing the neighbourhoods, moving the images accordingly, and saving and plotting the results.
        '''
        start_time = time()
        print('Song Neighbouring starting...')
        print('{} neighbourhoods done'.format(self.total_neighbours))
        self.get_neighbours()
        self.move_images()
        self.save_results(self.neighbours, 'neighbours', 'Properties of the songs ordered according to their neighbourhood', 'Songs', False)
        sys.stdout.write('\033[F\033[K')
        print('Song Neighbouring executed in {}'.format(time_to_string(time() - start_time)))
        print('Neighbourhoods saved in \'{}\' and images copied to \'{}\''.format(osp.join(self.res_dir, 'neighbours.json'), self.neighbours_dir))
        print('The neighbourhood properties are represented in \'{}\''.format(osp.join(self.res_dir, 'neighbours.png')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_dir', type=str, default='data/images')
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--order_by', type=str, default='dist')
    parser.add_argument('--n_folders', type=int, default=None)
    parser.add_argument('--write_group_info', type=int, default=1)

    parser.add_argument('--clusters_dir', type=str, default='results/clusters')
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--cluster_size_threshold', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=None)
    parser.add_argument('--clustering_method', type=str, default='AC')

    parser.add_argument('--neighbours_dir', type=str, default='results/neighbours')
    parser.add_argument('--n_neighbours', type=int, default=3)
    cmd = vars(parser.parse_args())
    sc = SongClustering(cmd)
    sc.run()
    print()
    sn = SongNeighbouring(cmd)
    sn.run()
