from os import listdir

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config as general_config
import json
from networkx.readwrite import json_graph


def get_and_construct_graph(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return json_graph.node_link_graph(data)


def mann_whitney_u_test(distribution_1, distribution_2):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    :param distribution_1:
    :param distribution_2:
    :return: u_statistic, p_value
    """

    u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
    if p_value <= 0.05:
        print("Null hipothesis can be rejected (p-value: {}) \n "
              "Which states that the distributions of two data sets are identical".format(p_value))
    return u_statistic, p_value


def perform_u_test_in_every_freq_band():
    for freq_band in general_config.FREQ_BANDS:
        freq_name, freq_range = freq_band[0].lower(), freq_band[1:]
        print("\n=======================> Current Frequency Band {}".format(freq_name))

        json_file_path = general_config.DATA_DIR + '/graphs/' + freq_name + '.json'
        G = get_and_construct_graph(json_file_path)

        # pesos = [(u, v) for u,v,e in G.edges(data=True) if e['weight'] > 0.25]
        data = np.array([e['weight'] for u, v, e in G.edges(data=True)])

        steps = np.linspace(50, 1, 100)
        for greater_than_value in steps:
            if greater_than_value != 50:
                current_threshold = greater_than_value / 100
                print("Comparing {} agains {}".format(prev_threshold, current_threshold))
                sample_1, sample_2 = data[data >= prev_threshold], data[data >= current_threshold]
                test_mwu = mann_whitney_u_test(sample_1, sample_2)
                prev_threshold = current_threshold
            else:
                prev_threshold = greater_than_value / 100


if __name__ == '__main__':
    for emotion in general_config.EMOTIONAL_LABELS:
        for freq_band in general_config.FREQ_BANDS:
            base_files_path = 'C:/Users/Juan/OneDrive - CINVESTAV/head_it/graphs'
            files_path = pd.Series(listdir(base_files_path))

            freq_name, freq_range = freq_band[0].lower(), freq_band[1:]
            print("\n=======================> Current Frequency Band {}".format(freq_name))

            json_file_path = general_config.DATA_DIR + '/graphs/' + freq_name + '.json'
            G = get_and_construct_graph(json_file_path)
