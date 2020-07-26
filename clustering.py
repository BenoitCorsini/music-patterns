import os
import os.path as osp
import numpy as np
from time import time
from shutil import copyfile, rmtree
from random import sample
from sklearn.cluster import AgglomerativeClustering as AC


class SongClustering(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']
        self.clusters_dir = cmd['clusters_dir']

        self.clusters_threshold = cmd['clusters_threshold']
        self.AC_division = cmd['AC_division']
        self.write_cluster_info = cmd['write_cluster_info']

        self.__get_measures__()

    def __get_measures__(self):
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
        self.mat_dict = {}
        self.n_mat = 0
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_mat, mat = line.split('\n')[0].split('\t')
            index_mat = int(index_mat)
            self.mat_dict[index_mat] = mat
            self.n_mat += 1

    def get_clusters_recursive(self, labels, cluster_id, clusters, info):
        cluster_dists = self.dists[labels,:][:,labels]
        cluster_size = np.sum(labels)
        if cluster_size <= self.clusters_threshold:
            cluster_dists = np.mean(cluster_dists)
            info.append((cluster_dists, cluster_id))
            for index_mat in np.where(labels)[0]:
                clusters[self.mat_dict[index_mat]] = cluster_id
        else:
            cluster_labels = AC(n_clusters=self.AC_division, affinity='precomputed', linkage='complete').fit_predict(cluster_dists)
            for l in range(self.AC_division):
                new_labels = labels.copy()
                new_labels[labels] = cluster_labels == l
                self.get_clusters_recursive(new_labels, cluster_id + str(l), clusters, info)

    def get_clusters(self):
        clusters = {}
        info = []
        self.get_clusters_recursive(np.ones(self.n_mat) == 1, '', clusters, info)
        return clusters, info

    def get_clusters_id(self, info):
        clusters_id = {}
        index_cluster = 1
        for (cluster_dist, cluster_id) in sorted(info):
            if cluster_dist == 0:
                clusters_id[cluster_id] = 'Singleton Cluster'
            else:
                clusters_id[cluster_id] = 'Cluster {}'.format(index_cluster)
                index_cluster += 1
                if self.write_cluster_info:
                    clusters_id[cluster_id] += ' (dist={};depth={})'.format(int(cluster_dist), len(cluster_id))
        return clusters_id

    def move_mats(self, clusters, clusters_id):
        if osp.exists(self.clusters_dir):
            rmtree(self.clusters_dir)
        os.makedirs(self.clusters_dir)
        for mat in clusters:
            if not osp.exists(osp.join(self.clusters_dir, clusters_id[clusters[mat]])):
                os.makedirs(osp.join(self.clusters_dir, clusters_id[clusters[mat]]))
            im = mat[:-4] + '.png'
            copyfile(osp.join(self.im_dir, im), osp.join(self.clusters_dir, clusters_id[clusters[mat]], im))

    def run(self):
        clusters, info = self.get_clusters()
        clusters_id = self.get_clusters_id(info)
        self.move_mats(clusters, clusters_id)

class SongNeighbouring(object):

    def __init__(self, cmd):
        self.im_dir = cmd['im_dir']
        self.res_dir = cmd['res_dir']
        self.neighbours_dir = cmd['neighbours_dir']

        self.n_neighbours = cmd['n_neighbours']
        self.write_neighbours_info = cmd['write_neighbours_info']
        self.n_neighbour_songs = cmd['n_neighbour_songs']

        self.__get_measures__()

    def __get_measures__(self):
        self.dists = np.loadtxt(osp.join(self.res_dir, 'dists.txt'), delimiter='\t')
        self.mat_dict = {}
        self.n_mat = 0
        for line in open(osp.join(self.res_dir, 'song_list.txt'), 'r'):
            index_mat, mat = line.split('\n')[0].split('\t')
            index_mat = int(index_mat)
            self.mat_dict[index_mat] = mat
            self.n_mat += 1

    def get_neighbours(self):
        neighbours = {}
        for index_mat in self.mat_dict:
            neighbours[self.mat_dict[index_mat]] = []
            neighbours_dists = self.dists[index_mat, :]
            neighs = neighbours_dists <= np.percentile(neighbours_dists, (100.*(self.n_neighbours + 1))/self.n_mat)
            neighbours_dists = np.mean(neighbours_dists[neighs])
            for index_n in np.where(neighs)[0]:
                if index_n != index_mat:
                    neighbours[self.mat_dict[index_mat]].append(self.mat_dict[index_n])
            neighbours[self.mat_dict[index_mat]] = neighbours_dists, neighbours[self.mat_dict[index_mat]]
        return neighbours

    def move_mats(self, neighbours):
        if osp.exists(self.neighbours_dir):
            rmtree(self.neighbours_dir)
        os.makedirs(self.neighbours_dir)
        if self.n_neighbour_songs is None:
            list_neighbours = list(neighbours)
        else:
            list_neighbours = sample(list(neighbours), min(self.n_neighbour_songs, self.n_mat))
        for mat in list_neighbours:
            neighbours_dists, neighbours_list = neighbours[mat]
            dir_name = mat
            if self.write_neighbours_info:
                dir_name += ' (dist={})'.format(int(neighbours_dists))
            if not osp.exists(osp.join(self.neighbours_dir, dir_name)):
                os.makedirs(osp.join(self.neighbours_dir, dir_name))
            im = mat[:-4] + '.png'
            copyfile(osp.join(self.im_dir, im), osp.join(self.neighbours_dir, dir_name, '(CENTER) ' + im))
            for neighbour_mat in neighbours_list:
                im = neighbour_mat[:-4] + '.png'
                copyfile(osp.join(self.im_dir, im), osp.join(self.neighbours_dir, dir_name, im))

    def run(self):
        neighbours = self.get_neighbours()
        self.move_mats(neighbours)




if __name__ == '__main__':
    cmd = {}
    cmd['im_dir'] = 'data/images'
    cmd['res_dir'] = 'data/results'
    cmd['clusters_dir'] = 'data/clusters'
    cmd['clusters_threshold'] = 5
    cmd['AC_division'] = 2
    cmd['write_cluster_info'] = True

    cmd['neighbours_dir'] = 'data/neighbours'
    cmd['n_neighbours'] = 2
    cmd['write_neighbours_info'] = True
    cmd['n_neighbour_songs'] = 2
    sc = SongClustering(cmd)
    sc.run()
    sn = SongNeighbouring(cmd)
    sn.run()
