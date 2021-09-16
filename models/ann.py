import sys


WORKING_DIR = "/"


sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])


import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from featuresextractor import build_and_save_features, get_csv_content
import head_it.config as head_it_config
from classification import binarize_targets, zero_center, whiten
from sklearn import preprocessing


CLFS = {'sgd': SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True, max_iter=5000),
        'mlp': MLPClassifier(activation='logistic', solver='sgd', max_iter=5000)}


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == '__main__':
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna().query('freq_band == "beta"')

    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    # binarize targets
    y = binarize_targets(y, 7)

    '''clf = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(clf, X, y, cv=10)'''
    matrix = X.assign(target=y).to_numpy()
    
    X, y = matrix[:, :matrix.shape[1] - 1], matrix[:, matrix.shape[1] - 1].astype(int)

    # Zero-centering
    X = zero_center(X)

    # Whitenning
    X = whiten(X)

    # build a classifier
    clf = CLFS['mlp']


    # specify parameters and distributions to sample from
    '''param_dist = {'average': [True, False],
                  'l1_ratio': stats.uniform(0, 1),
                  'alpha': loguniform(1e-4, 1e0)}'''
    param_dist = {'learning_rate': ['constant', 'adaptive'], 'alpha': 10.0 ** -np.arange(1, 7)}

    # run randomized search
    n_iter_search = 10
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring='accuracy', n_iter=n_iter_search,
                                       verbose=1, cv=10, n_jobs=5)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # use a full grid over all parameters
    '''param_grid = {'average': [True, False],
                  'l1_ratio': np.linspace(0, 1, num=10),
                  'alpha': np.power(10, np.arange(-4, 1, dtype=float))}'''
    param_grid = {'learning_rate': ['constant', 'adaptive'], 'alpha': 10.0 ** -np.arange(1, 7)}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, n_jobs=5)
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)