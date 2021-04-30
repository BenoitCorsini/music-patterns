import os


if __name__ == '__main__':

    run_cmd = 'python main.py'

    run_cmd += ' --run_tab_scroller 0'
    run_cmd += ' --run_pattern_matrix 1'
    run_cmd += ' --run_distance_matrix 1'
    run_cmd += ' --run_song_clustering 1'
    run_cmd += ' --run_song_neighbouring 1'
    run_cmd += ' --run_feature_statistics 1'
    run_cmd += ' --run_cluster_statistics 1'
    run_cmd += ' --run_neighbour_statistics 1'

    run_cmd += ' --res_dir dataset/results'
    run_cmd += ' --tab_dir dataset/tablatures'
    run_cmd += ' --mat_dir dataset/matrices'
    run_cmd += ' --im_dir dataset/images'
    run_cmd += ' --clusters_dir dataset/clusters'
    run_cmd += ' --neighbours_dir dataset/neighbours'
    run_cmd += ' --stats_dir dataset/statistics'
    run_cmd += ' --song_file dataset/songs.json'

    run_cmd += ' --save_im 1'
    run_cmd += ' --overwrite_mat 0'
    run_cmd += ' --overwrite_im 0'
    run_cmd += ' --colour blue'
    run_cmd += ' --min_song_length 2'

    run_cmd += ' --initialize_distances 0'
    run_cmd += ' --normalized_size 500'
    run_cmd += ' --batch_size 500'
    run_cmd += ' --n_batch 10'
    run_cmd += ' --p_norm 2'

    run_cmd += ' --order_by dist'
    run_cmd += ' --n_folders -1'
    run_cmd += ' --write_group_info 1'

    run_cmd += ' --n_clusters 2'
    run_cmd += ' --cluster_size_threshold 15'
    run_cmd += ' --max_iter -1'
    run_cmd += ' --clustering_method AC'
    run_cmd += ' --n_neighbours 20'

    run_cmd += ' --write_stats_info 1'
    run_cmd += ' --max_plot 50'
    run_cmd += ' --min_feat_size 5'
    run_cmd += ' --min_clus_size 5'

    os.system(run_cmd)