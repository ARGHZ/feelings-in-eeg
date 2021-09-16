WORKING_DIR = "/"

import sys

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

import config as general_config
import head_it.config as head_it_config
from graph_analysis import get_and_construct_graph

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from networkx.readwrite import json_graph
from networkx import density, Graph, subgraph_view


def histogram_plot_of_emotions_pairs():
    file_path, emotions = WORKING_DIR + '/data/binary_classification_metrics.csv', head_it_config.EMOTIONAL_LABELS
    data = pd.read_csv(file_path).dropna().query('test_accuracy >= 0.6116')
    data.sort_values(by='test_accuracy', ascending=False, inplace=True)

    clf_samples = data['clf'].to_numpy()

    sns.histplot(data=data, y="clf")
    sns.histplot(data=data, y="emotions")


THRESHOLD = 0.5


def filter_edge(n1, n2, e):
    return G[n1][n2][e]["weight"] >= THRESHOLD


def build_and_save_density_metric():
    n_steps = 100
    samples = {'delta': None, 'theta': None, 'alpha': None, 'beta': None, 'gamma': None}
    for freq_band in general_config.FREQ_BANDS:
        freq_name, freq_range = freq_band[0].lower(), freq_band[1:]
        print("\n=======================> Current Frequency Band {}".format(freq_name))
        json_file_path = general_config.DATA_DIR + '/graphs/' + freq_name + '.json'
        G = get_and_construct_graph(json_file_path)

        steps = np.linspace(100, 1, 100)
        sample = []
        for greater_than_value in steps:
            THRESHOLD = greater_than_value / 100
            view = subgraph_view(G, filter_edge=filter_edge)
            sample.append(density(view))
            print("Threshold {} | Filtered edges: {} | density {}".format(THRESHOLD, len(view.edges), density(view)))
        samples[freq_name] = np.array(sample)
    samples = pd.DataFrame(samples)
    # samples.to_csv(general_config.DATA_DIR + '/density_by_freq_band_200.csv', index=False)
    sns.displot(samples)

    # plt.show()
    return samples


def test_density_significance():
    n_steps = 100
    path_file = general_config.DATA_DIR + '/density_by_freq_band_' + str(n_steps) + '.csv'
    densities = pd.read_csv(path_file)

    print("Original dimensions: {}".format(densities.shape))
    steps = np.linspace(100, 1, n_steps)
    densities_with_pvalues = []
    for ith, limit in enumerate(steps):
        print("Threshold {} | {}".format(limit / 100, ith + 1))
        samples = densities[0:ith + 1]
        print(samples.shape)
        try:
            statistic, p_value = stats.kruskal(samples['delta'], samples['theta'], samples['alpha'], samples['beta'])
        except ValueError as e:
            print(e)
        else:
            densities_with_pvalues.append({'threshold': limit / 100, 'ith_row': ith + 1, 'p_value': p_value,
                                           'statistic': statistic})

            alpha = 0.05
            """if p_value > alpha:
                print('Same distributions (fail to reject H0)')
            else:
                print('Different distributions (reject H0)')"""
            if p_value <= alpha:
                print("Null hypothesis can be rejected (p-value: {}) \n "
                      "Which states the population median of all of the groups are equal".format(p_value))
    densities_with_pvalues = pd.DataFrame(densities_with_pvalues)
    # densities_with_pvalues.to_csv(general_config.DATA_DIR + '/density_p_values_' + str(n_steps) + '.csv', index=False)
    return densities_with_pvalues


if __name__ == '__main__':
    densities = test_density_significance()

    # sns.lineplot(data=densities_with_pvalues['p_value'])
    chart = sns.lineplot(data=densities, x='threshold', y='p_value')
    chart.axhline(0.05, ls='--', c='red')
    # plt.show()

    print('End of main process')
