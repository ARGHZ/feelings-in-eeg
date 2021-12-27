from os import listdir, getenv
from os.path import join, isfile
from dotenv import load_dotenv

import json
import config as general_config
import head_it.config as head_it_config

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

from data_sets_helper import read_mat_file
from graph_analysis import get_and_construct_graph
from matlab_analysis import filter_pdc_by_freq_band, get_pdc_dims


def draw_graph(G, full_path_file, save_file=False):
    # seed = 13648  # Seed random number generators for reproducibility
    # pos = nx.spring_layout(G, seed=seed)
    # pos = nx.circular_layout(G)

    pos = {'LTI': [-1.00000000e+00, 2.45045699e-08], 'LPD': [0.49999998, 0.86602546],
           'LPI': [-0.50000004, 0.8660254], 'LTD': [9.99999970e-01, -6.29182054e-08],
           'LFD': [0.49999989, -0.86602541], 'LFI': [-0.49999992, -0.86602541]}

    node_sizes = [2000 for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
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
        connectionstyle="arc3,rad=0.5"
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    nx.draw_networkx_labels(G, pos, font_size=22, font_color="whitesmoke")

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    # plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    # ax.figure(figsize=(25, 14))
    plt.show()
    # exit()
    if save_file:
        fig_file_path = full_path_file + '.png'
        plt.savefig(fig_file_path, dpi=100)
        plt.close()


def serialize_and_save_graph(graph, full_path_file):
    data1 = json_graph.node_link_data(graph)

    full_path_file = full_path_file + '.json'
    with open(full_path_file, 'w') as outfile:
        json.dump(data1, outfile)
    return full_path_file


if __name__ == '__main__':
    general_config.extend_sys_path_with_current_dir()

    load_dotenv()
    emotion_files_path = 'C:/Users/Juan/OneDrive - CINVESTAV/head_it/graphs'
    files_path = pd.Series(listdir(emotion_files_path))
    pdc_all, pdc_significance, graph_metrics_all = [], [], []

    for emotion in general_config.EMOTIONAL_LABELS:
        query = files_path.str.contains(emotion)
        files_path_queried = files_path[query].to_numpy()
        for path in files_path_queried:
            json_file_path = join(emotion_files_path, path)

            if isfile(json_file_path):
                # Getting CSV file as numpy array
                try:
                    G = get_and_construct_graph(json_file_path)
                except ValueError as e:
                    print(json_file_path)
                    print(e)
                else:
                    graph_file_path = 'C:/Users/Juan/PycharmProjects/feelings-in-eeg/data/' + path

                    draw_graph(G, graph_file_path, True)
                    print(graph_file_path)
    print('End of main process')
