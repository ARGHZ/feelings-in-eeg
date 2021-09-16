# Load libraries
import sys

WORKING_DIR = "/"


sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from config import WORKING_DIR
from featuresextractor import get_csv_content
import head_it.config as head_it_config


if __name__ == '__main__':
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna().query('freq_band == "gamma"')
    print(data['freq_band'].describe())
    data = data[head_it_config.CLASSIFICATION_VARS]

    y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]

    # Load iris data
    # iris_dataset = load_iris()

    # Create features and target
    # X = iris_dataset.data
    # y = iris_dataset.target

    # Convert to categorical data by converting data to integers
    # X = X.astype(int)

    # Building the model
    random_forest_clf = RandomForestClassifier(n_estimators=500, random_state=0)

    # Training the model
    random_forest_clf.fit(X, y)

    # Computing the importance of each feature
    feature_importance = random_forest_clf.feature_importances_

    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            random_forest_clf.estimators_],
                                           axis=0)

    # Plotting a Bar Graph to compare the models
    plt.bar(head_it_config.CLASSIFICATION_VARS[1:], feature_importance)
    plt.xticks(rotation=21, horizontalalignment='right')
    # plt.xlabel('Feature Labels')
    # plt.ylabel('Feature Importances')
    # plt.title('Comparison of different Feature Importances')
    plt.show()