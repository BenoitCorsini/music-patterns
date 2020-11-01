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
LABEL_LIMIT_SIZE = 20
NULL_CENTER_VALUE = -1e6
BASIC_STATS = {
    'feature' : [
        ('artist', 'mean', False, 'size'),
        ('year', 'mean', False, 'size'),
        ('decade', 'mean', False, 'size')
    ],
    'cluster' : [
        ('artist', 'mean', True, 'size'),
        ('artist', 'std', True, 'size'),
        ('year', 'std', False, 'size'),
        ('decade', 'std', False, 'size')
    ],
    'neighbour' : [
        ('artist', 'mean', True, 'std'),
        ('artist', 'std', True, 'std'),
        ('year', 'std', False, 'std'),
        ('decade', 'std', False, 'std')
    ]
}


class Statistics(object):

    def __init__(self, cmd):
        self.res_dir = cmd['res_dir']
        self.stats_dir = cmd['stats_dir']
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        self.write_info = cmd['write_info']
        self.max_plot = cmd['max_plot']

        self.__initialize__()

    def __initialize__(self):
        with open(osp.join(self.res_dir, 'songs_info.json')) as songs_info:
            self.songs_info = json.load(songs_info)
            songs_info.close()
        self.feat_is_number = {}
        for song_feats in self.songs_info.values():
            for song_feat, feat_value in song_feats.items():
                if self.feat_is_number.get(song_feat, True):
                    self.feat_is_number[song_feat] = isinstance(feat_value, int) or isinstance(feat_value, float)

    def plot_stats(self, plot_infos):
        stats, name, title, bars_name, make_smaller, plot_center = plot_infos
        if stats:
            labels = []
            y = []
            std = []
            bars = []
            if plot_center:
                centers = []
            for count, (label, infos) in enumerate(stats.items()):
                if self.max_plot is not None:
                    if count < self.max_plot:
                        if isinstance(label, str):
                            if len(label) > LABEL_LIMIT_SIZE:
                                label = label[:LABEL_LIMIT_SIZE - 3] + '...'
                        labels.append(label)
                        y.append(infos['y'])
                        std.append(infos['std'])
                        bars.append(infos[bars_name])
                        if plot_center:
                            centers.append(infos['center'])
                else:
                    if isinstance(label, str):
                        if len(label) > LABEL_LIMIT_SIZE:
                            label = label[:LABEL_LIMIT_SIZE - 3] + '...'
                    labels.append(label)
                    y.append(infos['y'])
                    std.append(infos['std'])
                    bars.append(infos[bars_name])
                    if plot_center:
                        centers.append(infos['center'])
            size = len(labels)
            x = np.arange(size)
            y = np.array(y)
            std = np.array(std)
            bars = np.array(bars)
            if plot_center:
                centers = np.array(centers)
                if np.all(centers == NULL_CENTER_VALUE):
                    plot_center = False

            xmin = -.5
            xmax = size - .5

            height = np.max(std)
            if height == 0:
                height = 2
            ymin = np.min(y) - .55*height
            ymax = np.max(y) + .55*height
            if plot_center:
                center_values = centers[centers != NULL_CENTER_VALUE]
                ymin = min(ymin, np.min(center_values) - .05*height)
                ymax = max(ymax, np.max(center_values) + .05*height)
            ymin = max(ymin, 0)
            if ymax <= ymin:
                ymax = ymin + 1.1*height

            max_bars = np.max(bars)
            bars = (height*bars)/(max_bars + (max_bars==0))

            plt.figure(figsize=(16,9))
            plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            plt.plot(x, y, color=COLOUR_DOT, marker='.', markersize=20, linewidth=0, label='mean')
            plt.errorbar(x, y, .5*std, color=COLOUR, label='std', linewidth=5, ls='', alpha=.5)
            plt.bar(x, bars, color='red', alpha=.2, width=.7, label=bars_name, bottom=ymin)
            plt.plot(x, y, color=COLOUR_DOT, marker='.', markersize=20, linewidth=0)
            if make_smaller:
                plt.xticks(x, labels, rotation=75, fontsize=5)
            else:
                plt.xticks(x, labels, rotation=75)
            if plot_center:
                plt.plot(x, centers, color='gold', marker='*', markersize=11, linewidth=0, mec='black', mew=.5, label='center')
            plt.legend()
            plt.title(title)
            plt.savefig(osp.join(self.stats_dir, name + '.png'), dpi=250)
            plt.close()

    def plot(self, basic_stats=None):
        if basic_stats is None:
            for feat, is_number in self.feat_is_number.items():
                for ordering in ['mean', 'std']:
                    for reverse in [True, False]:
                        self.plot_stats(self.get_stats(feat, ordering, reverse, 'size'))
        else:
            for (feat, ordering, reverse, bars) in basic_stats.get(self.type, []):
                self.plot_stats(self.get_stats(feat, ordering, reverse, bars))


class FeatureStatistics(Statistics):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.__initialize_features__()

    def __initialize_features__(self):
        self.type = 'feature'
        self.min_feat_size = cmd['min_feat_size']
        if self.min_feat_size is None:
            self.min_feat_size = 1
        self.min_feat_size = max(self.min_feat_size, 1)

        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')

        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[song] = index_song

        list_by_feat = {}
        for song, info in self.songs_info.items():
            for feat in self.feat_is_number:
                if feat in info:
                    feat_value = info[feat]
                    if feat not in list_by_feat:
                        list_by_feat[feat] = {}
                    if feat_value not in list_by_feat[feat]:
                        list_by_feat[feat][feat_value] = []
                    list_by_feat[feat][info[feat]].append(self.song_dict[song])

        self.dist_by_feat = {}
        for feat, feat_values in list_by_feat.items():
            for feat_value, value_list in sorted(feat_values.items(), key=lambda x:x[0]):
                sub_dist = self.dists[value_list,:][:,value_list]
                if feat not in self.dist_by_feat:
                    self.dist_by_feat[feat] = {}
                self.dist_by_feat[feat][feat_value] = {
                    'mean' : np.mean(sub_dist),
                    'std' : np.std(sub_dist),
                    'size' : len(value_list)
                }

    def get_stats(self, feat, ordering, reverse, bars):
        if reverse:
            name = 'Feature \'{}\' by decreasing distance {}'.format(feat, ordering)
            mult = -1
            add = 0
        else:
            name = 'Feature \'{}\' by increasing distance {}'.format(feat, ordering)
            mult = 1
            add = 1
        title = name + ' (minimum feature size={}; max number of plots={})'.format(self.min_feat_size, self.max_plot)
        if self.write_info:
            name += ' (minFeat={},maxPlot={})'.format(self.min_feat_size, self.max_plot)

        stats = {}
        bars_name = bars
        for feat_value, feat_info in sorted(self.dist_by_feat.get(feat, {}).items(), key=lambda x:x[1].get(ordering, mult*random() + add), reverse=reverse):
            if feat_info['size'] >= self.min_feat_size:
                if bars not in feat_info:
                    bars_name = 'std'
                stats[feat_value] = {
                    'y' : feat_info['mean'],
                    'std' : feat_info['std'],
                    bars : feat_info.get(bars, feat_info['std'])
                }

        return stats, name, title, bars_name, not self.feat_is_number.get(feat, False), False


class ClusterStatistics(Statistics):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.__initialize_clusters__()

    def __initialize_clusters__(self):
        self.type = 'cluster'
        self.min_clus_size = cmd['min_clus_size']
        if self.min_clus_size is None:
            self.min_clus_size = 2
        self.min_clus_size = max(self.min_clus_size, 2)

        with open(osp.join(self.res_dir, 'clusters.json')) as clusters:
            self.clusters = json.load(clusters)
            clusters.close()

        self.feat_by_cluster = {}
        for cluster_id, cluster_info in self.clusters.items():
            self.feat_by_cluster[cluster_id] = {
                'dist' : {
                    'mean' : cluster_info['dist'],
                    'std' : cluster_info['std_dist']
                },
                'size' : cluster_info['size'],
                'name' : cluster_info['name']
            }
            for feat, is_number in self.feat_is_number.items():
                feat_stats = []
                for song in cluster_info['songs']:
                    if feat in self.songs_info[song]:
                        feat_stats.append(self.songs_info[song][feat])
                if feat_stats:
                    if is_number:
                        feat_stats = np.array(feat_stats)
                    else:
                        feat_stats = dict(Counter(feat_stats))
                        feat_stats = np.array(list(feat_stats.values()))
                self.feat_by_cluster[cluster_id][feat] = {
                    'mean' : np.mean(feat_stats),
                    'std' : np.std(feat_stats)
                }

    def get_stats(self, feat, ordering, reverse, bars):
        if reverse:
            name = 'Clusters by decreasing {} of \'{}\''.format(ordering, feat)
            mult = -1
            add = 0
        else:
            name = 'Clusters by increasing {} of \'{}\''.format(ordering, feat)
            mult = 1
            add = 1
        title = name + ' (minimum cluster size={}; max number of plots={})'.format(self.min_clus_size, self.max_plot)
        if self.write_info:
            name += ' (minClus={},maxPlot={})'.format(self.min_clus_size, self.max_plot)

        stats = {}
        bars_name = bars
        for cluster_id, cluster_info in sorted(self.feat_by_cluster.items(), key=lambda x:x[1].get(feat, {}).get(ordering, mult*random() + add), reverse=reverse):
            if (cluster_info['size'] >= self.min_clus_size) & (feat in cluster_info):
                if bars not in cluster_info:
                    bars_name = 'std'
                stats[cluster_info['name']] = {
                    'y' : cluster_info[feat]['mean'],
                    'std' : cluster_info[feat]['std'],
                    bars : cluster_info.get(bars, cluster_info[feat]['std'])
                }

        return stats, name, title, bars_name, True, False


class NeighbourStatistics(Statistics):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.__initialize_neighbours__()

    def __initialize_neighbours__(self):
        self.type = 'neighbour'

        with open(osp.join(self.res_dir, 'neighbours.json')) as neighbours:
            self.neighbours = json.load(neighbours)
            neighbours.close()

        self.feat_by_neighbour = {}
        for song, song_info in self.neighbours.items():
            if ' - ' in song:
                split_song = song.split(' - ')
                artist, title = split_song[0], split_song[1]
                if ' ' in artist:
                    artist = artist.split(' ')[0]
                name = artist + ' - ' + title
            else:
                name = song
            self.feat_by_neighbour[song] = {
                'dist' : {
                    'mean' : song_info['dist'],
                    'std' : song_info['std_dist']
                },
                'name' : name
            }
            for feat, is_number in self.feat_is_number.items():
                feat_stats = []
                for neigh in song_info['neighbours']:
                    if feat in self.songs_info[neigh]:
                        feat_stats.append(self.songs_info[neigh][feat])
                if feat_stats:
                    if is_number:
                        feat_stats = np.array(feat_stats)
                    else:
                        feat_stats = dict(Counter(feat_stats))
                        feat_stats = np.array(list(feat_stats.values()))
                self.feat_by_neighbour[song][feat] = {
                    'mean' : np.mean(feat_stats),
                    'std' : np.std(feat_stats),
                    'center' : self.songs_info[song].get(feat, None)
                }

    def get_stats(self, feat, ordering, reverse, bars):
        if reverse:
            name = 'Neighbourhoods by decreasing {} of \'{}\''.format(ordering, feat)
            mult = -1
            add = 0
        else:
            name = 'Neighbourhoods by increasing {} of \'{}\''.format(ordering, feat)
            mult = 1
            add = 1
        title = name + ' (max number of plots={})'.format(self.max_plot)
        if self.write_info:
            name += ' (maxPlot={})'.format(self.max_plot)

        stats = {}
        bars_name = bars
        for song, song_info in sorted(self.feat_by_neighbour.items(), key=lambda x:x[1].get(feat, {}).get(ordering, mult*random() + add), reverse=reverse):
            if feat in song_info:
                if bars not in song_info:
                    bars_name = 'std'
                if self.feat_is_number.get(feat, False):
                    center = self.songs_info[song].get(feat, NULL_CENTER_VALUE)
                else:
                    center = NULL_CENTER_VALUE
                stats[song_info['name']] = {
                    'y' : song_info[feat]['mean'],
                    'std' : song_info[feat]['std'],
                    'center' : center,
                    bars : song_info.get(bars, song_info[feat]['std'])
                }

        return stats, name, title, bars_name, True, True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--stats_dir', type=str, default='results/statistics')
    parser.add_argument('--write_info', type=int, default=0)
    parser.add_argument('--max_plot', type=int, default=50)
    parser.add_argument('--min_feat_size', type=int, default=None)
    parser.add_argument('--min_clus_size', type=int, default=None)
    cmd = vars(parser.parse_args())
    fs = FeatureStatistics(cmd)
    fs.plot(BASIC_STATS)
    cs = ClusterStatistics(cmd)
    cs.plot(BASIC_STATS)
    ns = NeighbourStatistics(cmd)
    ns.plot(BASIC_STATS)
