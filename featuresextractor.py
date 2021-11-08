# coding=utf-8
import json
from os import listdir, getenv
from os.path import join, isfile
from dotenv import load_dotenv

import pandas as pd

from matlab_analysis import filter_pdc_by_freq_band, get_pdc_dims
from data_sets_helper import *
import umap
from sklearn.ensemble import RandomForestRegressor

import config as general_config
import head_it.config as head_it_config
import matplotlib.pyplot as plt
import networkx as nx


def reduce_samples_dimensions(method, features, targets):
    """

    :param method:
    :param features:
    :param targets:
    :return:
    """
    if method == 'random_forest':
        model = RandomForestRegressor(random_state=1, max_depth=10)
        model.fit(features,targets)

        importance = model.feature_importances_
        indices = np.argsort(importance)[-6:]
        new_features = features[:, indices]
    elif method == 'umap':
        new_features = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=6).fit_transform(features)
    else:
        new_features = features

    return new_features


def save_numpy_as_csv(np_array, full_file_path):
    """

    :param np_array:
    :param full_file_path:
    """
    np.savetxt(full_file_path, np_array, delimiter=',')


def save_numpy_3d_file(arr, full_file_path):
    """

    :param arr:
    :param full_file_path:
    """
    # reshaping the array from 3D
    # matrice to 2D matrice.
    arr_reshaped = arr.reshape(arr.shape[0], -1)

    # saving reshaped array to file.
    np.savetxt(full_file_path, arr_reshaped)


def load_numpy_from_csv(full_file_path):
    """

    :param full_file_path:
    :return:
    """
    array_data = np.genfromtxt(full_file_path, delimiter=',')
    return array_data


def load_numpy_3d_file(full_file_path, shape):
    """

    :param full_file_path:
    :param shape:
    :return:
    """
    # retrieving data from file.
    loaded_arr = np.loadtxt(full_file_path)

    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original
    # array shape.reshaping to get original
    # matrice with original shape.
    load_original_arr = loaded_arr.reshape(shape)

    # check the shapes:
    '''print("shape of arr: ", shape)
    print("shape of load_original_arr: ", load_original_arr.shape)'''
    return load_original_arr


def build_and_save_features():
    """

    :return:
    """
    features = []
    return features


def get_csv_content(path_file):
    """

    :param path_file:
    :return:
    """
    return pd.read_csv(path_file, index_col=0)


def get_graph_metrics_table(DG):
    """

    :param DG:
    :return:
    """
    metric = {'degree': dict(nx.degree(DG)),
              'in_degree': nx.in_degree_centrality(DG),
              'out_degree': nx.out_degree_centrality(DG),
              'betweenness_centrality': nx.betweenness_centrality(DG)}

    return pd.DataFrame(metric)


if __name__ == '__main__':
    threshold = 0.25
    general_config.extend_sys_path_with_current_dir()

    load_dotenv()
    emotion_files_path = getenv('PDCS_DIRS')
    files_path = listdir(emotion_files_path)
    pdc_all, pdc_significance, graph_metrics_all = [], [], []
    for path in files_path:
        raw_path_item = join(emotion_files_path, path)

        if isfile(raw_path_item):
            # Getting CSV file as numpy array
            try:
                matlab_vars = read_mat_file(raw_path_item)
                pdc_all.append(matlab_vars['c']['pdc'][0, 0])
            except ValueError as e:
                print(raw_path_item)
                print(e)
            else:
                ith_subj, emotion, nth_try = path.split('_')[1], path.split('_')[2], path.split('.')[0].split('-')[1]
                ch_labels = head_it_config.CHANNELS_PER_SUBJECT[ith_subj]

                for freq_band in general_config.FREQ_BANDS:
                    freq_name, freq_range = freq_band[0], freq_band[1:]
                    filtered_freq_band_pdc = filter_pdc_by_freq_band(matlab_vars['c']['pdc_th'][0, 0], freq_name, freq_range)
                    graph_metrics_row = {'subj': ith_subj, 'freq_band': freq_name.lower(), 'emotion': emotion,
                                     'try': nth_try}
                    metric = {'degree': None, 'in_degree': None, 'out_degree': None, 'betweenness_centrality': None,
                              'global_efficiency': None}

                    N, N, m = get_pdc_dims(filtered_freq_band_pdc)
                    DG, weighted_edges, ch_names = nx.DiGraph(), [], tuple(ch_labels.values())
                    for i in range(N):
                        for j in range(N):
                            query_pdc = filtered_freq_band_pdc[i, j] > 0.0
                            mean_value = filtered_freq_band_pdc[i, j][query_pdc].mean()
                            if str(mean_value) != 'nan' and mean_value >= threshold:
                                pair_nodes = (ch_names[j], ch_names[i], mean_value)
                                weighted_edges.append(pair_nodes)
                    DG.add_weighted_edges_from(weighted_edges)

                    metric['degree'] = nx.degree(DG)
                    print('Degree: {}'.format(metric['degree']))

                    for lobe_name, value in metric['degree']:
                        new_key = lobe_name + '_degree'
                        graph_metrics_row[new_key] = value

                    metric['in_degree'] = nx.in_degree_centrality(DG)
                    print('In Degree Centrality: {}'.format(metric['in_degree']))
                    for lobe_name, value in metric['in_degree'].items():
                        new_key = lobe_name + '_in_degree'
                        graph_metrics_row[new_key] = value

                    metric['out_degree'] = nx.out_degree_centrality(DG)
                    print('Out Degree Centrality: {}'.format(metric['out_degree']))
                    for lobe_name, value in metric['out_degree'].items():
                        new_key = lobe_name + '_out_degree'
                        graph_metrics_row[new_key] = value

                    metric['betweenness_centrality'] = nx.betweenness_centrality(DG)
                    print('Betweenness Centrality: {}'.format(metric['betweenness_centrality']))
                    for lobe_name, value in metric['betweenness_centrality'].items():
                        new_key = lobe_name + '_betweenness_centrality'
                        graph_metrics_row[new_key] = value

                    G = DG.to_undirected()
                    metric['global_efficiency'] = nx.global_efficiency(G)
                    print('Global Efficiency: {}'.format(metric['global_efficiency']))
                    graph_metrics_row['global_efficiency'] = metric['global_efficiency']

                    graph_metrics_row['file'] = path

                    graph_metrics_all.append(graph_metrics_row)
                    img_file_name = freq_name.lower() + '/' + path
                    # plt.close()
                    print(img_file_name)
                    print('=============================================')
    graph_metrics_all = pd.DataFrame(graph_metrics_all)
    graph_metrics_all.to_csv(general_config.DATA_DIR + '/graph_metrics_all.csv', index=False)
    print('end of main process')
