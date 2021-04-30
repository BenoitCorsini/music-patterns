import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_song_clustering', type=int, default=1)
    parser.add_argument('--run_song_neighbouring', type=int, default=1)
    parser.add_argument('--run_feature_statistics', type=int, default=1)
    parser.add_argument('--run_cluster_statistics', type=int, default=1)
    parser.add_argument('--run_neighbour_statistics', type=int, default=1)

    parser.add_argument('--order_by', type=str, default='dist')
    parser.add_argument('--n_folders', type=int, default=-1)
    parser.add_argument('--write_group_info', type=int, default=1)

    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--cluster_size_threshold', type=int, default=15)
    parser.add_argument('--max_iter', type=int, default=-1)
    parser.add_argument('--clustering_method', type=str, default='AC')
    parser.add_argument('--n_neighbours', type=int, default=20)

    parser.add_argument('--write_stats_info', type=int, default=0)
    parser.add_argument('--max_plot', type=int, default=50)
    parser.add_argument('--min_feat_size', type=int, default=5)
    parser.add_argument('--min_clus_size', type=int, default=5)

    cmd = vars(parser.parse_args())


    run_cmd = 'python main.py'

    run_cmd += ' --run_tab_scroller 0'
    run_cmd += ' --run_pattern_matrix 0'
    run_cmd += ' --run_distance_matrix 0'

    run_cmd += ' --res_dir precomputed/results'
    run_cmd += ' --im_dir precomputed/images'
    run_cmd += ' --clusters_dir precomputed/clusters'
    run_cmd += ' --neighbours_dir precomputed/neighbours'
    run_cmd += ' --stats_dir precomputed/statistics'
    run_cmd += ' --song_file dataset/songs.json'

    for arg, value in cmd.items():
    	run_cmd += ' --{} {}'.format(arg, value)

    os.system(run_cmd)