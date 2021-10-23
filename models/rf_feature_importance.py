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


def features_importance_plot(importances, use_seaborn=True):
    if use_seaborn:
        sns.barplot(x=list(importances.keys()), y=list(importances.values()))
        plt.xticks(rotation=21)
        plt.show()
    else:
        plt.bar(importances.keys(), importances.values())
        plt.xticks(rotation=21, horizontalalignment='right')
        # plt.xlabel('Feature Labels')
        # plt.ylabel('Feature Importances')
        # plt.title('Comparison of different Feature Importances')
        plt.show()


if __name__ == '__main__':
    opposite_pair_emotions = combinations(head_it_config.EMOTIONAL_LABELS, 2)

    features_importances = []
    for emotion_1, emotion_2 in opposite_pair_emotions:
        for freq_band in FREQ_BANDS:
            freq_name = freq_band[0].lower()
            print('\n__{}__{} vs {}_'.format(freq_name, emotion_1, emotion_2))
            emotions_query = 'emotion in ("' + emotion_1 + '", "' + emotion_2 + '")'
            file_path, emotions = WORKING_DIR + '/graph_metrics_mayor_igual_que_25_all.csv', head_it_config.EMOTIONAL_LABELS
            data = get_csv_content(file_path).dropna().query('freq_band == "' + freq_name + '" & ' + emotions_query)
            print(data['freq_band'].describe())
            data = data[head_it_config.CLASSIFICATION_VARS]

            y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
                   data[head_it_config.CLASSIFICATION_VARS[1:]]

            # Building the model
            random_forest_clf = RandomForestClassifier(n_estimators=500, random_state=0)

            importances = features_importance_cross_validate(random_forest_clf, X, y)
            importances['freq_band'] = freq_name
            features_importances.append(importances)
    features_importances = pd.DataFrame(features_importances)
    features_importances.to_csv(DATA_DIR + '/feature_importances.csv', index=False)
