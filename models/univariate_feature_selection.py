from sklearn.ensemble import ExtraTreesClassifier

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from config import WORKING_DIR
import head_it.config as head_it_config
from featuresextractor import get_csv_content
from classification import binarize_targets


if __name__ == '__main__':
    # #############################################################################
    # Import some data to play with

    # Own dataset
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna().query('freq_band == "beta"')

    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    # binarize targets
    y = binarize_targets(y, 7)

    # The iris dataset
    # X, y = load_iris(return_X_y=True)

    # Some noisy data not correlated
    # E = np.random.RandomState(42).uniform(0, 0.1, size=(X.shape[0], 20))

    # Add the noisy data to the informative features
    # X = np.hstack((X, E))

    # Split dataset to select feature and evaluate the classifier
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=0
    )

    plt.figure(1)
    plt.clf()

    X_indices = np.arange(X.shape[-1])

    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function to select the four
    # most significant features
    selector = SelectKBest(f_classif, k=4)
    selector.fit(X_train, y_train)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)')

    # #############################################################################
    # Compare to the weights of an SVM
    clf = ExtraTreesClassifier(n_estimators=100, max_features=9)
    # clf = make_pipeline(MinMaxScaler(), LinearSVC())
    clf.fit(X_train, y_train)
    print('Classification accuracy without selecting features: {:.3f}'
          .format(clf.score(X_test, y_test)))

    svm_weights = np.abs(clf[-1].feature_importances_).sum(axis=0)
    svm_weights /= svm_weights.sum()

    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight')

    clf_selected = make_pipeline(
            SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after univariate feature selection: {:.3f}'
          .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()