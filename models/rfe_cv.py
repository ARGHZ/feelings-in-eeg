from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from config import WORKING_DIR
import head_it.config as head_it_config
from featuresextractor import get_csv_content
from classification import binarize_targets


if __name__ == '__main__':
    # Own dataset
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna().query('freq_band == "beta"')

    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    # binarize targets
    y = binarize_targets(y, 7)

    # Build a classification task using 3 informative features
    '''X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                               n_redundant=2, n_repeated=0, n_classes=8,
                               n_clusters_per_class=1, random_state=0)'''

    # Create the RFE object an<d compute a cross-validated score.
    clf = RandomForestClassifier(n_estimators=100)
    # clf = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications

    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xticks(np.arange(1, X.shape[1] + 1))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)
    plt.show()