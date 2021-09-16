import sys


WORKING_DIR = "/"


sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from featuresextractor import get_csv_content
import head_it.config as head_it_config
from classification import binarize_targets


if __name__ == '__main__':
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna()

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
    # X = zero_center(X)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [10, 50, 100, 200, 500], 'max_features': [4, 5, 6, 7, 8, 9, 10, 15, 19]}]

    scores = ['accuracy', ]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring='%s' % score, verbose=1, cv=10, n_jobs=3)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()