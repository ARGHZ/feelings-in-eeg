# Load libraries
import sys

import pandas as pd
from sklearn.model_selection import cross_validate, KFold

WORKING_DIR = "/"

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from config import WORKING_DIR, DATA_DIR, FREQ_BANDS
from featuresextractor import get_csv_content
from classification import binarize_targets
import head_it.config as head_it_config

from itertools import combinations


def features_importance_cross_validate(clf, X, y):
    # cross-validation
    kf_total = KFold(n_splits=10)
    scores = cross_validate(clf, X, y, cv=kf_total.split(X), scoring=('accuracy', 'f1_micro', 'f1_weighted',
                                                                      'balanced_accuracy'), return_estimator=True,
                            n_jobs=4)
    scorings = pd.DataFrame(scores)

    gini_importances = []
    for estimator in scorings['estimator'].values:
        gini_importances.append(estimator.feature_importances_)
    gini_importances = pd.DataFrame(np.array(gini_importances), columns=head_it_config.CLASSIFICATION_VARS[1:])

    return gini_importances.mean().to_dict()


def features_importance_positive_vs_negative():
    features_importances = []

    file_path, emotions = WORKING_DIR + '/graph_metrics_mayor_igual_que_25_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna()
    data = data[head_it_config.CLASSIFICATION_VARS]

    y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    # binarize targets
    y = binarize_targets(y, 7)

    # Building the model
    random_forest_clf = RandomForestClassifier(n_estimators=500, random_state=0)

    importances = features_importance_cross_validate(random_forest_clf, X, y)
    features_importances.append(importances)
    features_importances = pd.DataFrame(features_importances).loc[0].to_dict()
    x_labels = ['$c_{' + str(i) + '}$' for i in range(1, 20)]
    print(features_importances)
    features_importance_plot(features_importances)


def features_importance_plot(importances, x_labels=None, use_seaborn=True):
    if x_labels is None:
        x_labels = ['$c_{' + str(i) + '}$' for i in range(1, 20)]

    if use_seaborn:
        sns.barplot(x=x_labels, y=list(importances.values()), palette='Accent')
        # plt.xticks(rotation=21)
    else:
        plt.bar(importances.keys(), importances.values())
        # plt.xticks(rotation=21, horizontalalignment='right')
        # plt.title('Comparison of different Feature Importances')
    plt.xlabel('Características $c$', fontsize=17)
    plt.ylabel('$IMP_{Gini}$', fontsize=17)
    plt.show()


def features_importances_subplots(data, x_labels=None):
    nr_rows = 5
    nr_cols = 1

    cols_review = data['freq_band'].to_list()

    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(19, nr_rows * 9), squeeze=False, sharex=True)

    x_labels = ['$c_{' + str(i) + '}$' for i in range(1, 20)]

    for r in range(0, nr_rows):
        freq_band_name = data['freq_band'][r]
        for c in range(0, nr_cols):
            col = r * nr_cols + c
            if col < len(cols_review):
                sns.barplot(data=data.query('freq_band == "' + freq_band_name + '"').drop(columns=['freq_band']),
                            palette='Accent', label=freq_band_name, ax=axs[r][c])
                axs[r][c].legend(loc='upper left')
                axs[r][c].set_xticklabels(x_labels)
                # plt.xlabel(col, fontsize=12)
    # plt.tight_layout()
    fig.text(0.5, 0.04, 'Características $c$', ha='center', fontsize=17)
    fig.text(0.04, 0.5, '$IMP_{Gini}$', va='center', rotation='vertical', fontsize=17)
    plt.show()


if __name__ == '__main__':
    # opposite_pair_emotions = combinations(head_it_config.EMOTIONAL_LABELS, 2)

    emotion_1, emotion_2 = 'relief', 'grief'

    emotions_query = 'emotion_a == "' + emotion_1 + '" & emotion_b == "' + emotion_2 + '"'
    file_path, emotions = WORKING_DIR + '/feature_importances_pairs_emotions_freq_band.csv', head_it_config.EMOTIONAL_LABELS
    features_importances = pd.read_csv(file_path, index_col=None).dropna().query(emotions_query).reset_index().drop(
        columns=['emotion_a', 'emotion_b', 'index'])
    print(features_importances)

    features_importances_subplots(features_importances)
