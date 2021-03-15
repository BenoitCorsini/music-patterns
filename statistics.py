import argparse
import os
import os.path as osp
import json
import sys
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from utils import time_to_string, basic_stats


MEAN_COLOUR = 'navy'
STD_COLOUR = 'royalblue'
CENTER_COLOUR = 'gold'
LABEL_LIMIT_SIZE = 20
NULL_CENTER_VALUE = -1e6
BASIC_STATS = basic_stats()


class Statistics(object):

    def __init__(self, cmd):
        self.res_dir = cmd['res_dir']
        self.stats_dir = cmd['stats_dir']
        if not osp.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        self.song_file = cmd['song_file']
        self.write_stats_info = cmd['write_stats_info']
        self.max_plot = cmd['max_plot']

        self.min_feat_size = cmd['min_feat_size']
        self.min_clus_size = cmd['min_clus_size']

        self.__initialize__()

    def __initialize__(self):
        '''
        This function initializes some of the objects useful for statistics.
        It creates:
        - songs_info: a dictionary containing the songs and their features.
        - feat_infos: a dictionary containing information on the features (such as the values of the features)
        '''
        self.songs_info = json.load(open(self.song_file, 'r'))
        self.feat_infos = {}
        for song_feats in self.songs_info.values():
            for song_feat, feat_value in song_feats.items():

                if song_feat not in self.feat_infos:
                    self.feat_infos[song_feat] = {
                        'feat_values' : [],
                        'is_list' : isinstance(feat_value, list)
                    }

                if isinstance(feat_value, list):
                    if not self.feat_infos[song_feat].get('is_list', True):
                        raise Exception('Feature {} is and is not a list...'.format(song_feat))
                    for sub_feat_value in feat_value:
                        self.feat_infos[song_feat]['feat_values'].append(sub_feat_value)

                else:
                    if self.feat_infos[song_feat].get('is_list', False):
                        raise Exception('Feature {} is and is not a list...'.format(song_feat))
                    self.feat_infos[song_feat]['feat_values'].append(feat_value)

        for feat_info in self.feat_infos.values():
            feat_info['feat_values'] = sorted(list(set(feat_info['feat_values'])))
            feat_info['is_number'] = all(
                [(isinstance(feat_value, int) or isinstance(feat_value, float)) for feat_value in feat_info['feat_values']]
            )

    def list_to_stats(self, feat_list, is_number):
        '''
        This function plots the mean and std of a list of features.
        If the features are numbers, then it is the regular mean and std.
        Otherwise, it counts the elements of the list, orders them by number and computes the mean and std.
        '''
        if is_number:
            stats = np.array(feat_list)
            mean = np.mean(stats)
            std = np.std(stats)

        else:
            stats = dict(Counter(feat_list))
            stats = np.sort(np.array(list(stats.values())))[::-1]
            stats = stats / np.sum(stats)
            indices = np.arange(np.size(stats))
            mean = np.sum(indices*stats)
            std = np.sqrt(np.sum((mean - indices)**2*stats))

        return mean, std


    def plot_stats(self, plot_infos):
        '''
        This function plots the statistics contained in 'plot_infos'.
        'plot_infos' should contain:
        - stats: a dictionary with the label of the group and statistics on this group.
        - name, title, bars_name: informations on the saved name, title and bars of the plot.
        - make_smaller: reduce the size of the labels on the horizontal axis.
        - plot_center: if the plotted stats have a center. In this case, stats should contain this center.
        '''
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
                            centers.append(infos.get('center', NULL_CENTER_VALUE))
                else:
                    if isinstance(label, str):
                        if len(label) > LABEL_LIMIT_SIZE:
                            label = label[:LABEL_LIMIT_SIZE - 3] + '...'
                    labels.append(label)
                    y.append(infos['y'])
                    std.append(infos['std'])
                    bars.append(infos[bars_name])
                    if plot_center:
                        centers.append(infos.get('center', NULL_CENTER_VALUE))
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
            plt.plot(x, y, color=MEAN_COLOUR, marker='.', markersize=20, linewidth=0, label='mean')
            plt.errorbar(x, y, .5*std, color=STD_COLOUR, label='std', linewidth=5, ls='', alpha=.5)
            plt.bar(x, bars, color='red', alpha=.2, width=.7, label=bars_name, bottom=ymin)
            plt.plot(x, y, color=MEAN_COLOUR, marker='.', markersize=20, linewidth=0)
            if make_smaller:
                plt.xticks(x, labels, rotation=75, fontsize=5)
            else:
                plt.xticks(x, labels, rotation=75)
            if plot_center:
                plt.plot(x, centers, color=CENTER_COLOUR, marker='*', markersize=11, linewidth=0, mec='black', mew=.5, label='center')
            plt.legend()
            plt.title(title)
            plt.savefig(osp.join(self.stats_dir, name + '.png'), dpi=250)
            plt.close()

            return True
        else:
        	return False

    def plot(self, basic_stats=None):
        '''
        This function plots the statistics corresponding to the dictionary 'basic_stats'.
        If not dictionary is given as input, all stats are plotted.
        '''
        start_time = time()
        print('{} Statistics starting...'.format(self.type.capitalize()))
        not_plotted = 0
        if basic_stats is None:
            n_plots = len(self.feat_infos)*4
            index_plot = 1
            for feat in self.feat_infos:
                for ordering in ['mean', 'std']:
                    for reverse in [True, False]:
                        current_time = time_to_string(time() - start_time)
                        print('Plotting stat {} of {} ({})'.format(index_plot, n_plots, current_time))
                        if not self.plot_stats(self.get_stats(feat, ordering, reverse, 'size')):
                        	not_plotted += 1
                        index_plot += 1
                        sys.stdout.write('\033[F\033[K')
        else:
            n_plots = len(basic_stats.get(self.type, []))
            for index_plot, (feat, ordering, reverse, bars) in enumerate(basic_stats.get(self.type, [])):
                current_time = time_to_string(time() - start_time)
                print('Plotting stat {} of {} ({})'.format(index_plot+1, n_plots, current_time))
                if not self.plot_stats(self.get_stats(feat, ordering, reverse, bars)):
                	not_plotted += 1
                sys.stdout.write('\033[F\033[K')
        time_algorithm = time_to_string(time() - start_time)
        print('{} Statistics executed in {}'.format(self.type.capitalize(), time_algorithm))
        print('{} figures available at {}'.format(n_plots - not_plotted, self.stats_dir))


class FeatureStatistics(Statistics):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.__initialize_features__()
        self.__get_dist_by_feat__()

    def __initialize_features__(self):
        '''
        This function adds parameters to the 'Statistics' class specific to stats on the features.
        '''
        self.type = 'feature'
        if self.min_feat_size is None:
            self.min_feat_size = 1
        self.min_feat_size = max(self.min_feat_size, 1)

        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')

        self.song_dict = {}
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_song, song = line.split('\n')[0].split('\t')
            index_song = int(index_song)
            self.song_dict[song] = index_song

    def __get_dist_by_feat__(self):
        '''
        This function adds the parameter 'dist_by_feat' to the class.
        This parameters contains a dictionary of dictionaries organized in the following way:
        - The keys are features.
        - For each feature, it contains a dictionary with all the feature values as keys.
        - For each feature value, it contains properties of this feature value.
        '''
        list_by_feat = {}
        for song, info in self.songs_info.items():
            for feat, feat_info in self.feat_infos.items():
                if feat in info:
                    if feat not in list_by_feat:
                        list_by_feat[feat] = {}

                    if feat_info['is_list']:
                        for feat_value in info[feat]:
                            if feat_value not in list_by_feat[feat]:
                                list_by_feat[feat][feat_value] = []
                            list_by_feat[feat][feat_value].append(self.song_dict[song])

                    else:
                        feat_value = info[feat]
                        if feat_value not in list_by_feat[feat]:
                            list_by_feat[feat][feat_value] = []
                        list_by_feat[feat][feat_value].append(self.song_dict[song])

        # 'list_by_feat' now contains all the features, feature values, and the corresponding index of the songs.
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
        '''
        This function outputs the stats info in the format of the 'plot_infos' used in the 'Statistics' class.
        '''
        if reverse:
            name = 'Feature \'{}\' by decreasing distance {}'.format(feat, ordering)
            mult = -1
            add = 0
        else:
            name = 'Feature \'{}\' by increasing distance {}'.format(feat, ordering)
            mult = 1
            add = 1

        title = name + ' (minimum feature size={}; max number of plots={})'.format(self.min_feat_size, self.max_plot)
        if self.write_stats_info:
            name += ' (minFeat={},maxPlot={})'.format(self.min_feat_size, self.max_plot)

        stats = {}
        bars_name = bars
        for feat_value, feat_info in sorted(
                # If the feat does not exist, do nothing
                self.dist_by_feat.get(feat, {}).items(),
                # If the ordering does not exists, sort randomly
                # 'mult' and 'add' are useful if only a subset of the features have the 'ordering' parameter
                key=lambda x:x[1].get(ordering, mult*random.random() + add),
                reverse=reverse
            ):
            if feat_info.get('size', 0) >= self.min_feat_size:
                if bars not in feat_info:
                    bars_name = 'std'

                stats[feat_value] = {
                    'y' : feat_info['mean'],
                    'std' : feat_info['std'],
                    bars : feat_info.get(bars, feat_info['std'])
                }

        return stats, name, title, bars_name, not self.feat_infos.get(feat, {}).get('is_number', False), False


class ClusterStatistics(Statistics):

    def __init__(self, cmd):
        super().__init__(cmd)
        self.__initialize_clusters__()
        self.__get_feat_by_cluster__()

    def __initialize_clusters__(self):
        '''
        This function adds parameters to the 'Statistics' class specific to stats on the cluster.
        '''
        self.type = 'cluster'
        if self.min_clus_size is None:
            self.min_clus_size = 2
        self.min_clus_size = max(self.min_clus_size, 2)
        with open(osp.join(self.res_dir, 'clusters.json')) as clusters:
            self.clusters = json.load(clusters)
            clusters.close()

    def __get_feat_by_cluster__(self):
        '''
        This function adds the parameter 'feat_by_cluster' to the class.
        This parameters contains a dictionary of dictionaries organized in the following way:
        - The keys are cluster ids.
        - For each cluster, it contains a dictionary with all the features as keys.
        - For each feature, it contains properties of this feature related to the cluster.
        '''
        self.feat_by_cluster = {}
        for cluster_id, cluster_info in self.clusters.items():
            self.feat_by_cluster[cluster_id] = {
                'dist_mean' : cluster_info['dist'],
                'dist_std' : cluster_info['std_dist'],
                'size' : cluster_info['size'],
                'name' : cluster_info['name']
            }

            for feat, feat_info in self.feat_infos.items():
                feat_list = []
                for song in cluster_info['songs']:
                    if feat in self.songs_info[song]:
                        if feat_info['is_list']:
                            feat_list += self.songs_info[song][feat]
                        else:
                            feat_list.append(self.songs_info[song][feat])

                if feat_list:
                    feat_mean, feat_std = self.list_to_stats(feat_list, feat_info['is_number'])
                    self.feat_by_cluster[cluster_id][feat] = {
                        'mean' : feat_mean,
                        'std' : feat_std
                    }

    def get_stats(self, feat, ordering, reverse, bars):
        '''
        This function outputs the stats info in the format of the 'plot_infos' used in the 'Statistics' class.
        '''
        if reverse:
            name = 'Clusters by decreasing {} of \'{}\''.format(ordering, feat)
            mult = -1
            add = 0
        else:
            name = 'Clusters by increasing {} of \'{}\''.format(ordering, feat)
            mult = 1
            add = 1

        title = name + ' (minimum cluster size={}; max number of plots={})'.format(self.min_clus_size, self.max_plot)
        if self.write_stats_info:
            name += ' (minClus={},maxPlot={})'.format(self.min_clus_size, self.max_plot)

        stats = {}
        bars_name = bars
        for cluster_id, cluster_info in sorted(
                self.feat_by_cluster.items(),
                # If the ordering does not exists, sort randomly
                # 'mult' and 'add' are useful if only a subset of the features have the 'ordering' parameter
                key=lambda x:x[1].get(feat, {}).get(ordering, mult*random.random() + add),
                reverse=reverse
            ):
            if (cluster_info['size'] >= self.min_clus_size) & (feat in cluster_info):
                if bars not in cluster_info:
                    bars_name += 'std'

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
        self.__get_feat_by_neighbour__()

    def __initialize_neighbours__(self):
        '''
        This function adds parameters to the 'Statistics' class specific to stats on the neighbours.
        '''
        self.type = 'neighbour'
        with open(osp.join(self.res_dir, 'neighbours.json')) as neighbours:
            self.neighbours = json.load(neighbours)
            neighbours.close()

    def __get_feat_by_neighbour__(self):
        '''
        This function adds the parameter 'feat_by_neighbour' to the class.
        This parameters contains a dictionary of dictionaries organized in the following way:
        - The keys are songs.
        - For each song, it contains a dictionary with all the features as keys.
        - For each feature, it contains properties of this feature related to the neighbours of the song.
        '''
        self.feat_by_neighbour = {}
        for song, song_info in self.neighbours.items():
            if ' - ' in song:
                split_song = song.split(' - ')
                artist, title = split_song[0], split_song[1]
                if ' ' in artist:
                    if artist.upper().startswith('THE'):
                        artist = artist.split(' ')[1]
                    else:
                        artist = artist.split(' ')[0]
                name = artist + ' - ' + title
            else:
                name = song

            self.feat_by_neighbour[song] = {
                'dist_mean' : song_info['dist'],
                'dist_std' : song_info['std_dist'],
                'name' : name
            }

            for feat, feat_info in self.feat_infos.items():
                feat_list = []
                for neigh in song_info['neighbours']:
                    if feat in self.songs_info[neigh]:
                        if feat_info['is_list']:
                            feat_list += self.songs_info[neigh][feat]
                        else:
                            feat_list.append(self.songs_info[neigh][feat])

                if feat_list:
                    feat_mean, feat_std = self.list_to_stats(feat_list, feat_info['is_number'])
                    self.feat_by_neighbour[song][feat] = {
                        'mean' : feat_mean,
                        'std' : feat_std,
                        'center' : self.songs_info[song].get(feat, None)
                    }

    def get_stats(self, feat, ordering, reverse, bars):
        '''
        This function outputs the stats info in the format of the 'plot_infos' used in the 'Statistics' class.
        '''
        if reverse:
            name = 'Neighbourhoods by decreasing {} of \'{}\''.format(ordering, feat)
            mult = -1
            add = 0
        else:
            name = 'Neighbourhoods by increasing {} of \'{}\''.format(ordering, feat)
            mult = 1
            add = 1

        title = name + ' (max number of plots={})'.format(self.max_plot)
        if self.write_stats_info:
            name += ' (maxPlot={})'.format(self.max_plot)

        stats = {}
        bars_name = bars
        for song, song_info in sorted(
                self.feat_by_neighbour.items(),
                # If the ordering does not exists, sort randomly
                # 'mult' and 'add' are useful if only a subset of the features have the 'ordering' parameter
                key=lambda x:x[1].get(feat, {}).get(ordering, mult*random.random() + add),
                reverse=reverse
            ):
            if feat in song_info:
                if bars not in song_info:
                    bars_name = 'std'

                is_number = self.feat_infos.get(feat, {}).get('is_number', False)
                is_not_list = not self.feat_infos.get(feat, {}).get('is_list', True)
                if is_number & is_not_list:
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
    parser.add_argument('--song_file', type=str, default='data/songs.json')
    parser.add_argument('--write_stats_info', type=int, default=0)
    parser.add_argument('--max_plot', type=int, default=50)
    parser.add_argument('--min_feat_size', type=int, default=2)
    parser.add_argument('--min_clus_size', type=int, default=2)
    cmd = vars(parser.parse_args())
    fs = FeatureStatistics(cmd)
    fs.plot(BASIC_STATS)
    print()
    cs = ClusterStatistics(cmd)
    cs.plot(BASIC_STATS)
    print()
    ns = NeighbourStatistics(cmd)
    ns.plot(BASIC_STATS)