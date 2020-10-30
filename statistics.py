import os.path as osp
import os
import json
import argparse
import numpy as np
from random import random
import matplotlib.pyplot as plt
from collections import Counter


COLOUR = 'royalblue'
COLOUR_DOT = 'navy'
SCALE = .05
BASIC_STATS = {
      'feat' : [
            ('artist', 'mean', False),
            ('year', 'mean', False),
            ('decade', 'mean', False)
      ],
      'clus' : [
            ('artist', 'mean', True),
            ('artist', 'std', True),
            ('year', 'std', False),
            ('decade', 'std', False)
      ]
}


class Statistics(object):

    def __init__(self, cmd):
        self.res_dir = cmd['res_dir']
        self.stats_dir = cmd['stats_dir']
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        self.write_info = cmd['write_info']

        self.min_feat_size = cmd['min_feat_size']
        self.max_feat_plot = cmd['max_feat_plot']
        if self.min_feat_size is None:
            self.min_feat_size = 1

        self.min_clus_size = cmd['min_clus_size']
        self.max_clus_plot = cmd['max_clus_plot']
        if self.min_clus_size is None:
            self.min_clus_size = 1

        self.__initialize__()

        self.__dist_by_feat__()
        self.__feat_by_cluster__()

    def __initialize__(self):
        '''
        creates dists, clusters, clusters_info, songs_info, song_to_index, feat_is_number
        '''
        with open(osp.join(self.res_dir, 'clusters.json')) as clusters:
            self.clusters = json.load(clusters)
            clusters.close()
        with open(osp.join(self.res_dir, 'clusters_info.json')) as clusters_info:
            self.clusters_info = json.load(clusters_info)
            clusters_info.close()
        with open(osp.join(self.res_dir, 'songs_info.json')) as songs_info:
            self.songs_info = json.load(songs_info)
            songs_info.close()
        with open(osp.join(self.res_dir, 'song_list.txt'), 'r') as songs:
            self.song_to_index = {}
            for line in songs:
                index_mat, mat = line.split('\n')[0].split('\t')
                index_mat = int(index_mat)
                self.song_to_index[mat] = index_mat
        self.feat_is_number = {}
        for _, song_feats in self.songs_info.items():
            for song_feat, feat_value in song_feats.items():
                if self.feat_is_number.get(song_feat, True):
                    self.feat_is_number[song_feat] = isinstance(feat_value, int) or isinstance(feat_value, float)

    def __dist_by_feat__(self):
        if osp.exists(osp.join(self.res_dir, 'dist_by_feat.json')):
            with open(osp.join(self.res_dir, 'dist_by_feat.json'), 'r') as dist_by_feat:
                self.dist_by_feat = json.load(dist_by_feat)
                dist_by_feat.close()

        else:
            dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
            list_by_feat = {}
            for song, info in self.songs_info.items():
                for feat in self.feat_is_number:
                    if feat in list_by_feat:
                        if info.get(feat, 'Null') in list_by_feat[feat]:
                            list_by_feat[feat][info.get(feat, 'Null')].append(self.song_to_index[song])
                        else:
                            list_by_feat[feat][info.get(feat, 'Null')] = [self.song_to_index[song]]
                    else:
                        list_by_feat[feat] = {
                              info.get(feat, 'Null') : [self.song_to_index[song]]
                        }
            self.dist_by_feat = {}
            for feat, feat_values in list_by_feat.items():
                for feat_value in sorted(feat_values):
                    value_list = feat_values[feat_value]
                    sub_dist = dists[value_list,:][:,value_list]
                    if feat not in self.dist_by_feat:
                        self.dist_by_feat[feat] = {}
                    self.dist_by_feat[feat][feat_value] = {
                          'mean' : np.mean(sub_dist),
                          'std' : np.std(sub_dist),
                          'size' : len(value_list)
                    }

            with open(osp.join(self.res_dir, 'dist_by_feat.json'), 'w') as dist_by_feat:
                json.dump(self.dist_by_feat, dist_by_feat, indent=2)
                dist_by_feat.close()

    def __feat_by_cluster__(self):
        if osp.exists(osp.join(self.res_dir, 'feat_by_cluster.json')):
            with open(osp.join(self.res_dir, 'feat_by_cluster.json'), 'r') as feat_by_cluster:
                self.feat_by_cluster = json.load(feat_by_cluster)
                feat_by_cluster.close()

        else:
            self.feat_by_cluster = {}
            for song, info in self.songs_info.items():
                for feat in self.feat_is_number:
                    song_cluster = self.clusters[song[:-4]]
                    if song_cluster not in self.feat_by_cluster:
                        self.feat_by_cluster[song_cluster] = {}
                    if feat not in self.feat_by_cluster[song_cluster]:
                        self.feat_by_cluster[song_cluster][feat] = {
                              'list' : []
                        }
                    self.feat_by_cluster[song_cluster][feat]['list'].append(info.get(feat, 'Null'))

            for _, cluster_info in self.feat_by_cluster.items():
                for feat, feat_info in cluster_info.items():
                    feat_list = feat_info['list']
                    if self.feat_is_number.get(feat, False):
                        x = np.array(feat_list)
                    else:
                        x = dict(Counter(feat_info['list']))
                        x = np.array([count for (_, count) in x.items()])
                    feat_info['dist'] = {
                          'mean' : np.mean(x),
                          'std' : np.std(x)
                    }

            with open(osp.join(self.res_dir, 'feat_by_cluster.json'), 'w') as feat_by_cluster:
                json.dump(self.feat_by_cluster, feat_by_cluster, indent=2)
                feat_by_cluster.close()

    def get_feat_stats(self, feat, ordering, reverse):
        feat_stats = {}
        for feat_value, feat_info in self.dist_by_feat.get(feat, {}).items():
            if feat_info['size'] >= self.min_feat_size:
                feat_stats[feat_value] = feat_info

        if reverse:
            name = '\'{}\' by decreasing {}'.format(feat, ordering)
            mult = -1
            add = 0
        else:
            name = '\'{}\' by increasing {}'.format(feat, ordering)
            mult = 1
            add = 1
        title = name + ' (minimum feature size={}; max number of plots={})'.format(self.min_feat_size, self.max_feat_plot)
        if self.write_info:
            name += ' (minFeat={},maxPlot={})'.format(self.min_feat_size, self.max_feat_plot)

        feat_stats = sorted(feat_stats.items(), key=lambda x:x[1].get(ordering, mult*random() + add), reverse=reverse)
        if self.max_feat_plot is not None:
            if len(feat_stats) >= self.max_feat_plot:
                feat_stats = feat_stats[:self.max_feat_plot]

        if feat_stats:
            labels = []
            y = []
            std = []
            for feat_value, feat_info in feat_stats:
                labels.append(feat_value)
                y.append(feat_info['mean'])
                std.append(feat_info['std'])
            size = len(labels)

            x = np.arange(size)
            y = np.array(y)
            std = np.array(std)

            xmin = -.5
            xmax = size - .5
            ymin = max(0,np.min(y) - .55*np.max(std))
            ymax = min(1,np.max(y) + .55*np.max(std))
            if ymin == ymax:
                ymin = 0
                ymax = 1

            make_smaller = not self.feat_is_number.get(feat, False)
            plot_shade = True

            plot_infos = (name, title, x, y, std, labels, xmin, xmax, ymin, ymax, make_smaller)

        else:
            plot_infos = None

        return feat_stats, plot_infos

    def get_clus_stats(self, feat, ordering, reverse):
        clus_stats = {}
        for cluster_id, cluster_info in self.feat_by_cluster.items():
            if self.clusters_info[cluster_id]['size'] >= self.min_clus_size:
                clus_stats[cluster_id] = cluster_info.get(feat, {}).get('dist', {})

        if reverse:
            name = 'Clusters by decreasing {} of \'{}\''.format(ordering, feat)
            mult = -1
            add = 0
        else:
            name = 'Clusters by increasing {} of \'{}\''.format(ordering, feat)
            mult = 1
            add = 1
        title = name + ' (minimum cluster size={}; max number of plots={})'.format(self.min_clus_size, self.max_clus_plot)
        if self.write_info:
            name += ' (minClus={},maxPlot={})'.format(self.min_clus_size, self.max_clus_plot)

        clus_stats = sorted(clus_stats.items(), key=lambda x:x[1].get(ordering, mult*random() + add), reverse=reverse)
        if self.max_clus_plot is not None:
            if len(clus_stats) >= self.max_clus_plot:
                clus_stats = clus_stats[:self.max_clus_plot]

        if clus_stats:
            labels = []
            y = []
            std = []
            for cluster_id, cluster_info in clus_stats:
                label = self.clusters_info[cluster_id]['name']
                label = label.split(' ')
                label = label[0] + ' ' + label[1]
                labels.append(label)
                y.append(cluster_info['mean'])
                std.append(cluster_info['std'])
            size = len(labels)

            x = np.arange(size)
            y = np.array(y)
            std = np.array(std)

            xmin = -.5
            xmax = size - .5
            height = np.max(std)
            if height == 0:
                height = 2
            ymin = np.min(y) - .55*height
            ymax = np.max(y) + .55*height

            make_smaller = True
            plot_shade = False

            plot_infos = (name, title, x, y, std, labels, xmin, xmax, ymin, ymax, make_smaller)

        else:
            plot_infos = None

        return clus_stats, plot_infos

    def plot_stats(self, plot):
        stats_dict, plot_infos = plot
        if stats_dict:
            name, title, x, y, std, labels, xmin, xmax, ymin, ymax, make_smaller = plot_infos

            plt.figure(figsize=(16,9))
            plt.axis(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
            plt.plot(x, y, color=COLOUR_DOT, marker='.', markersize=20, linewidth=0, label='mean')
            plt.errorbar(x, y, .5*std, color=COLOUR, label='std', linewidth=5, ls='', alpha=.5)
            plt.bar(x, ymin+std, color='red', alpha=.2, width=.7, label='std')
            plt.plot(x, y, color=COLOUR_DOT, marker='.', markersize=20, linewidth=0)
            if make_smaller:
                plt.xticks(x, labels, rotation=75, fontsize=5)
            else:
                plt.xticks(x, labels, rotation=75)
            plt.legend()
            plt.title(title)
            plt.savefig(osp.join(self.stats_dir, name + '.png'), dpi=250)
            plt.close()

    def compute_all(self):
        for feat in self.feat_is_number:
            for ordering in ['mean', 'std']:
                for reverse in [True, False]:
                    self.plot_stats(self.get_feat_stats(feat, ordering, reverse))
                    self.plot_stats(self.get_clus_stats(feat, ordering, reverse))

    def compute_basic(self, basic_stats):
    	for (feat, ordering, reverse) in basic_stats.get('feat', []):
    		self.plot_stats(self.get_feat_stats(feat, ordering, reverse))
    	for (feat, ordering, reverse) in basic_stats.get('clus', []):
    		self.plot_stats(self.get_clus_stats(feat, ordering, reverse))

'''
TO BE STILL DONE:
- Stats on the average distance for clusters (and other representation of the clusters).
- Similar stats on the neighbours.
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--stats_dir', type=str, default='statistics')
    parser.add_argument('--write_info', type=int, default=0)
    parser.add_argument('--min_feat_size', type=int, default=3)
    parser.add_argument('--max_feat_plot', type=int, default=50)
    parser.add_argument('--min_clus_size', type=int, default=5)
    parser.add_argument('--max_clus_plot', type=int, default=50)
    cmd = vars(parser.parse_args())
    st = Statistics(cmd)
    st.compute_basic(BASIC_STATS)
