import argparse
import os
from time import time

from utils import intro, chorus, outro, file_to_songs, basic_stats
from scroller import TabScroller
from patterns import PatternMatrix
from measures import DistanceMatrix
from grouping import SongClustering, SongNeighbouring
from statistics import FeatureStatistics, ClusterStatistics, NeighbourStatistics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--run_tab_scroller', type=int, default=0,
    	help='run the TabScroller algorithm, which downloads the tablatures of the given list of songs')
    parser.add_argument('--run_pattern_matrix', type=int, default=1,
    	help='run the PatternMatrix algorithm, which transforms tablatures into matrices and images')
    parser.add_argument('--run_distance_matrix', type=int, default=1,
    	help='run the DistanceMatrix algorithm, which computes the distance between the songs')
    parser.add_argument('--run_song_clustering', type=int, default=1,
    	help='run the SongClustering algorithm, which cluster songs according to similarity of pattern matrix')
    parser.add_argument('--run_song_neighbouring', type=int, default=1,
    	help='run the SongNeighbouring algorithm, which groups songs according to their neighbourhoods')
    parser.add_argument('--run_feature_statistics', type=int, default=1,
    	help='run the FeatureStatistics algorithm, which plots the statistics of the different given features')
    parser.add_argument('--run_cluster_statistics', type=int, default=1,
    	help='run the ClusterStatistics algorithm, which plots the statistics of the clusters')
    parser.add_argument('--run_neighbour_statistics', type=int, default=1,
    	help='run the NeighbourStatistics algorithm, which plots the statistics of the neighbourhoods')

    parser.add_argument('--res_dir', type=str, default='results',
    	help='the directory where the results will be saved')
    parser.add_argument('--tab_dir', type=str, default='data/tablatures',
    	help='the directory where the tablatures will be saved and/or found')
    parser.add_argument('--mat_dir', type=str, default='data/matrices',
    	help='the directory where the pattern matrices will be computed')
    parser.add_argument('--im_dir', type=str, default='data/images',
    	help='the directory where the images for the songs will be saved')
    parser.add_argument('--clusters_dir', type=str, default='results/clusters',
    	help='the directory where the culster folders will be saved')
    parser.add_argument('--neighbours_dir', type=str, default='results/neighbours',
    	help='the directory where the neighbourhood folders will be saved')
    parser.add_argument('--stats_dir', type=str, default='results/statistics',
    	help='the directory where the plotted statistics will be saved')
    parser.add_argument('--song_file', type=str, default='data/songs.json',
        help='the file containing the songs to be downloaded and their features')

    # TabScroller parameters
    parser.add_argument('--chromedriver', type=str, default='chromedriver',
    	help='the path to the chromedriver')
    parser.add_argument('--time_limit', type=float, default=20,
    	help='the time limit before a download is cancelled and considered failed')

    # PatternMatrix parameters
    parser.add_argument('--save_im', type=int, default=1,
    	help='boolean stating if the song images should be saved or not')
    parser.add_argument('--overwrite_mat', type=int, default=1, 
    	help='boolean stating if the matrices should be recomputed when existing')
    parser.add_argument('--overwrite_im', type=int, default=1,
    	help='boolean stating if the images should be recomputed when existing')
    parser.add_argument('--colour', type=str, default='random',
    	help='the colour of the song images \
    	      choices of: grey, blue, red, yellow, green, orange, purple, cyan, salmon, pink, or random')
    parser.add_argument('--min_song_length', type=int, default=2,
    	help='the minimal length in measures for a tablature to be considered')

    # DistanceMatrix parameters
    parser.add_argument('--initialize_distances', type=int, default=1,
    	help='boolean stating if the distance matrix and the song indices should be re-initialized')
    parser.add_argument('--normalized_size', type=int, default=500,
    	help='the common size used to compare songs, since they all have different length in measures')
    parser.add_argument('--batch_size', type=int, default=5,
    	help='the size of the batch of normalized songs computed together \
    	      this parameter is useful to speed the algorithm up but requires more memory power')
    parser.add_argument('--n_batch', type=int, default=4,
    	help='the number of batch to be computed \
    	      the product of "batch_size" and "n_batch" should ideally be larger than the number of matrices')
    parser.add_argument('--p_norm', type=int, default=2,
    	help='the power used in the norm to compare song matrices')

    # Grouping parameters
    parser.add_argument('--order_by', type=str, default='dist',
    	help='the parameter used to number the clusters and neighbourhoods')
    parser.add_argument('--n_folders', type=int, default=None,
    	help='the number of cluster and neighbourhood folders saved \
    	      clusters and neighbourhoods will be saved in order as given by "order_by"')
    parser.add_argument('--write_group_info', type=int, default=1,
    	help='boolean stating if the folders name should contain some of their inherent information')

    parser.add_argument('--n_clusters', type=int, default=2,
    	help='the number of clusters to be used in the recursive clustering algorithm')
    parser.add_argument('--cluster_size_threshold', type=int, default=3,
    	help='the threshold at which the recursive clustering algorithm will stop trying to divide')
    parser.add_argument('--max_iter', type=int, default=None,
    	help='the maximal number of iteration of the recursive cliustering algorithm \
    	      if set to 1, number of clusters will be "n_clusters"')
    parser.add_argument('--clustering_method', type=str, default='AC',
    	help='the algorithm used in the recursive clustering algorithm \
    	      choices of: agglomerative clustering, spectral clustering, k-medoids')

    parser.add_argument('--n_neighbours', type=int, default=3,
    	help='the number of neighbours computed for each song')

    # Statistics parameters
    parser.add_argument('--write_stats_info', type=int, default=0,
    	help='boolean stating if the parameters used in the statistics should be written on the file name')
    parser.add_argument('--max_plot', type=int, default=50,
    	help='the maximal number of groups plotted on the horizontal axis (may refer to features, clusters, or neighbourhoods)')
    parser.add_argument('--min_feat_size', type=int, default=2,
    	help='the minimal number of songs with a specific feature for it to be considered in the statistics')
    parser.add_argument('--min_clus_size', type=int, default=2,
    	help='the minimual number of songs in a cluster for it to be considered in the statistics')

    # Collecting the parameters
    cmd = vars(parser.parse_args())


    # Running the algorithm
    start_time = time()
    intro()

    # TabScroller
    if cmd['run_tab_scroller']:
        songs, ids = file_to_songs(cmd['song_file'])
        ts = TabScroller(
            songs=songs,
            tab_dir=cmd['tab_dir'],
            res_dir=cmd['res_dir'],
            chromedriver=cmd['chromedriver'],
            time_limit=cmd['time_limit'],
            ids=ids
        )
        ts.run()
        chorus()

    # PatternMatrix
    if cmd['run_pattern_matrix']:
        pm = PatternMatrix(cmd)
        pm.compute()
        chorus()

    # DistanceMatrix
    if cmd['run_distance_matrix']:
        dm = DistanceMatrix(cmd)
        dm.compute()
        chorus()

    # SongClustering
    if cmd['run_song_clustering']:
        sc = SongClustering(cmd)
        sc.run()
        chorus()

    # SongNeighbouring
    if cmd['run_song_neighbouring']:
        sn = SongNeighbouring(cmd)
        sn.run()
        chorus()

    # FeatureStatistics
    if cmd['run_feature_statistics']:
        fs = FeatureStatistics(cmd)
        fs.plot(basic_stats())
        chorus()

    # ClusterStatistics
    if cmd['run_cluster_statistics']:
        cs = ClusterStatistics(cmd)
        cs.plot(basic_stats())
        chorus()

    # NeighbourStatistics
    if cmd['run_neighbour_statistics']:
        ns = NeighbourStatistics(cmd)
        ns.plot(basic_stats())
        chorus()

    # Algorithm over
    end_time = time()
    outro(end_time - start_time)