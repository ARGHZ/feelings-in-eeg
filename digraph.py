from os import listdir, getenv
from os.path import join, isfile
from dotenv import load_dotenv

import json
import config as general_config
import head_it.config as head_it_config

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

from data_sets_helper import read_mat_file
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
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    # ax.figure(figsize=(25, 14))
    plt.show()
    # exit()
    if save_file:
        fig_file_path = general_config.DATA_DIR + '/pdc/' + img_file_name + '.png'
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
    emotion_files_path = getenv('PDCS_DIRS')
    files_path = listdir(emotion_files_path)
    pdc_all, pdc_significance, graph_metrics_all = [], [], []
    for path in files_path:
        raw_path_item = join(emotion_files_path, path)
        file_emotion_match = raw_path_item.find('anger_') >= 0
        file_subject_match = raw_path_item.find('subj_1_') >= 0

        if True and True and isfile(raw_path_item):
            # Getting CSV file as numpy array
            try:
                matlab_vars = read_mat_file(raw_path_item)
                pdc_all.append(matlab_vars['c']['pdc'][0, 0])
            except ValueError as e:
                print(raw_path_item)
                print(e)
            else:
                ch_labels = head_it_config.CHANNELS_PER_SUBJECT[path.split('_')[1]]
                for freq_band in general_config.FREQ_BANDS:
                    freq_name, freq_range = freq_band[0], freq_band[1:]
                    img_file_name = freq_name.lower() + '/' + path
                    graph_file_path = general_config.DATA_DIR + '/graphs/' + img_file_name
                    filtered_freq_band_pdc = filter_pdc_by_freq_band(matlab_vars['c']['pdc_th'][0, 0], freq_name,
                                                                     freq_range)

                    N, N, m = get_pdc_dims(filtered_freq_band_pdc)
                    DG, weighted_edges, ch_names = nx.DiGraph(), [], tuple(ch_labels.values())
                    for i in range(N):
                        for j in range(N):
                            query_pdc = filtered_freq_band_pdc[i, j] > 0.0
                            mean_value = filtered_freq_band_pdc[i, j][query_pdc].mean()
                            if str(mean_value) != 'nan' and mean_value > 0.25:
                                pair_nodes = (ch_names[j], ch_names[i], mean_value)
                                weighted_edges.append(pair_nodes)
                    DG.add_weighted_edges_from(weighted_edges)
