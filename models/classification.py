import sys

import numpy as np
import pandas as pd

WORKING_DIR = "/"

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
from data_sets_helper import *
from itertools import combinations

import head_it.config as head_it_config
from featuresextractor import get_csv_content


def zero_center(X):
    return X - np.mean(X, axis=0)


def whiten(X, correct=True):
    Xc = zero_center(X)
    _, L, V = np.linalg.svd(Xc)
    W = np.dot(V.T, np.diag(1.0 / L))
    return np.dot(Xc, W) * np.sqrt(X.shape[0]) if correct else 1.0


def fit_and_predict(ith_subset, clf_name, model, training_x, training_y, testing_x, testing_y):
    clf_metrics = {'clf_name': clf_name, 'accuracy': 0.0, 'f1_score': 0.0,
                   'kfold': ith_subset + 1, 'y_pred': [], 'y_true': []}
    # print("{} in K-Fold {}".format(clf_metrics['clf_name'], clf_metrics['kfold']))

    model.fit(training_x, training_y)

    clf_metrics['y_pred'] = model.predict(testing_x)
    clf_metrics['y_true'] = testing_y
    are_same_type_targets = clf_metrics['y_pred'].dtype == clf_metrics['y_true'].dtype
    if not are_same_type_targets:
        print("{} == {}".format(clf_metrics['y_pred'].dtype, clf_metrics['y_true'].dtype))
        exit()
    clf_metrics['accuracy'] = accuracy_score(clf_metrics['y_true'], clf_metrics['y_pred'])
    clf_metrics['f1_score'] = f1_score(clf_metrics['y_true'], clf_metrics['y_pred'], average='micro')
    clf_metrics['f1_score_weighted'] = f1_score(clf_metrics['y_true'], clf_metrics['y_pred'], average='weighted')
    clf_metrics['balanced_accuracy'] = balanced_accuracy_score(clf_metrics['y_true'], clf_metrics['y_pred'])

    return pd.Series(clf_metrics)


def sample_classifier(clf_name, model, data_set):
    x, y, n_splits = data_set[:, :data_set.shape[1] - 1], data_set[:, data_set.shape[1] - 1].astype(int), 10

    kf_total = KFold(n_splits=n_splits)

    scores = cross_validate(model, x, y, cv=kf_total.split(x), scoring=['accuracy', 'f1_micro', 'f1_weighted',
                                                                        'balanced_accuracy', 'precision_micro', 
                                                                        'precision_weighted', 'recall_micro', 
                                                                        'recall_weighted'])
    metrics = {'clf': clf_name, 'fit_time': scores['fit_time'].mean(), 'score_time': scores['score_time'].mean(),
               'kfolds': n_splits, 'test_accuracy': round(scores['test_accuracy'].mean(), 4),
               'test_f1_micro': round(scores['test_f1_micro'].mean(), 4),
               'test_balanced_accuracy': round(scores['test_balanced_accuracy'].mean(), 4),
               'test_f1_weighted': round(scores['test_f1_weighted'].mean(), 4),
               'test_precision_micro': round(scores['test_precision_micro'].mean(), 4),
               'test_precision_weighted': round(scores['test_precision_weighted'].mean(), 4),
               'test_recall_micro': round(scores['test_recall_micro'].mean(), 4),
               'test_recall_weighted': round(scores['test_recall_weighted'].mean(), 4)}
    
    print("\nWith {} kfolds accuracy: {} | f1_score: {} | balanced_accuracy: {} | f1_score_weighted: {}".
          format(metrics['kfolds'], metrics['test_accuracy'], metrics['test_f1_micro'],
                 metrics['test_balanced_accuracy'], metrics['test_f1_weighted']))

    msg = "Process execution of {} model in {} seconds".format(clf_name, metrics['fit_time'])
    print(msg)
    return metrics


def run_classification_experiment(data):
    c, gamma, cache_size = 0.9, 0.1, 600

    classifiers = {'dummy': DummyClassifier(strategy='uniform', random_state=0),
                   'randomforest-5': RandomForestClassifier(n_estimators=500, random_state=0),
                   'svm': svm.SVC(kernel='linear', C=c, cache_size=cache_size), }
    dt_metrics = []
    for i_clf, (clf_name, model) in enumerate(classifiers.items()):
        row = sample_classifier(clf_name, model, data)
        dt_metrics.append(row)

    return dt_metrics


def binarize_targets(targets, threshold, query='<='):
    if query == '<=':
        # positive emotions
        targets[targets <= threshold] = 0
        # other wise 1 for negative emotions
        targets[targets > threshold] = 1
    elif query == '!=':
        targets[targets != threshold] = 0
    return targets


if __name__ == '__main__':
    '''opposite_pair_emotions = (('anger', 'love'), ('jealousy', 'awe'), ('disgust', 'relief'),
                              ('frustration', 'content'), ('fear', 'excite'),
                              ('sad', 'happy'), ('grief', 'joy'))
    '''
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_mayor_igual_que_25_all.csv', head_it_config.EMOTIONAL_LABELS
    opposite_pair_emotions = combinations(emotions, 2)
    all_experiment_metrics = []
    print("\n---------------------> CONSIDERING ALL FREQUENCY BAND FEATURES")
    for emotion_1, emotion_2 in opposite_pair_emotions:
        print('____________{} vs {}__________'.format(emotion_1, emotion_2))

        data = get_csv_content(file_path).dropna().query('emotion in ("' + emotion_1 + '", "' + emotion_2 +
                                                         '") & freq_band == "gamma"')
        data = data[head_it_config.CLASSIFICATION_VARS]
        y, X = np.array([emotions.index(value) for ith, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
               data[head_it_config.CLASSIFICATION_VARS[1:]]

        data = X.assign(target=y)
        experiment_metrics = run_classification_experiment(data.to_numpy())
        experiment_metrics = pd.DataFrame(experiment_metrics)
        # emotions_col = np.repeat(emotion_1 + '-' + emotion_2, experiment_metrics.shape[0])
        emotion_1_col = np.repeat(emotion_1, experiment_metrics.shape[0])
        emotion_2_col = np.repeat(emotion_2, experiment_metrics.shape[0])
        experiment_metrics.insert(0, 'emotion_a', emotion_2_col, True)
        experiment_metrics.insert(0, 'emotion_b', emotion_1_col, True)

        all_experiment_metrics.append(experiment_metrics)
        print('____________done__________\n'.format(emotion_1, emotion_2))
    all_experiment_metrics = pd.concat(all_experiment_metrics)
    all_experiment_metrics.to_csv(WORKING_DIR + '/data/binary_classification_metrics.csv', index=False)
    print('end of main process')
