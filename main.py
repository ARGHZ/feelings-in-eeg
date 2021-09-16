# coding=utf-8
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt

import config as general_config
import head_it.config as head_it_config
from data_sets_helper import read_mat_file
from digraph import serialize_and_save_graph

from matlab_analysis import filter_pdc_by_freq_band, get_pandas_dt
import networkx as nx


if __name__ == '__main__':
    emotion_files_path = head_it_config.PDCS_DIRS
    files_path = listdir(emotion_files_path)
    pdc_significance_all, pdc_significance = np.array([]), []

    emotion, freq_band = general_config.EMOTIONAL_LABELS[1], general_config.FREQ_BANDS[3]
    for emotion in general_config.EMOTIONAL_LABELS:
        for freq_band in general_config.FREQ_BANDS:
            MDG = nx.MultiDiGraph()

            freq_name, freq_range = freq_band[0], freq_band[1:]
            for path in files_path:

                raw_path_item = join(emotion_files_path, path)
                file_emotion_match = raw_path_item.find(emotion + '_') >= 0
                nth_subject = 36
                file_subject_match = raw_path_item.find('subj_{}_'.format(str(nth_subject))) >= 0

                if file_emotion_match and isfile(raw_path_item):
                    # Getting CSV file as numpy array
                    try:
                        matlab_vars = read_mat_file(raw_path_item)
                    except ValueError as e:
                        print(raw_path_item)
                        print(e)
                    else:
                        img_file_name = '' + path
                        ith_subj, emotion, nth_try = path.split('_')[1], path.split('_')[2], path.split('.')[0].split('-')[1]
                        ch_labels = tuple(head_it_config.CHANNELS_PER_SUBJECT[ith_subj].values())

                        matlab_vars['c']['pdc_th'][0, 0] = filter_pdc_by_freq_band(matlab_vars['c']['pdc_th'][0, 0], freq_name,
                                                                                   freq_range)
                        N, N, m = matlab_vars['c']['pdc_th'][0, 0].shape
                        weighted_edges, ch_names = [], ch_labels
                        for i in range(N):
                            for j in range(N):
                                query_pdc = matlab_vars['c']['pdc_th'][0, 0][i, j] > 0.0
                                mean_value = matlab_vars['c']['pdc_th'][0, 0][i, j][query_pdc].mean()
                                if str(mean_value) != 'nan' and mean_value > 0.25:
                                    pair_nodes = (ch_names[j], ch_names[i], mean_value)
                                    weighted_edges.append(pair_nodes)
                        MDG.add_weighted_edges_from(weighted_edges)
                        # dt = get_pandas_dt(matlab_vars['c']['pdc_th'][0, 0], ch_labels)
                        # sns.heatmap(dt, annot=True, cmap="coolwarm")
                        # pdc_significance.append(dt)
                        print(img_file_name)

            graph_file_path = general_config.DATA_DIR + '/graphs/' + emotion + '_' + freq_name.lower()
            '''graph_file_path = general_config.DATA_DIR + '/graphs/' + emotion + '_' + freq_name.lower() + '_' \
                              + 'subj_' + str(nth_subject)'''
            print(graph_file_path)
            serialize_and_save_graph(MDG, graph_file_path)

    '''G = MDG
    pos = head_it_config.NODES_LOBES_POSITIONS

    node_sizes = [2000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    cmap = plt.cm.coolwarm

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=30,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )

    nx.draw_networkx_labels(G, pos, font_size=22, font_color="whitesmoke")

    ax = plt.gca()
    ax.set_axis_off()
    # ax.figure(figsize=(25, 14))
    plt.show()
    print('Degree: {}'.format(nx.degree(MDG)))
    graph_metrics = {'subj': None, 'freq_band': freq_name.lower(), 'emotion': emotion, 'try': None}
    for lobe_name, value in nx.degree(MDG):
        new_key = lobe_name + '_degree'
        graph_metrics[new_key] = value

    print('In Degree Centrality: {}'.format(nx.in_degree_centrality(MDG)))
    for lobe_name, value in nx.in_degree_centrality(MDG).items():
        new_key = lobe_name + '_in_degree'
        graph_metrics[new_key] = value

    print('Out Degree Centrality: {}'.format(nx.out_degree_centrality(MDG)))
    for lobe_name, value in nx.out_degree_centrality(MDG).items():
        new_key = lobe_name + '_out_degree'
        graph_metrics[new_key] = value

    G = MDG.to_undirected()
    print('Global Efficiency: {}'.format(nx.global_efficiency(G)))
    graph_metrics['global_efficiency'] = nx.global_efficiency(G)'''
    # n, bins, patches = plt.hist(pdc_significance_all)
    print("end of main process")
