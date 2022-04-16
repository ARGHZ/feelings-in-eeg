import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import head_it.config as head_it_config
from featuresextractor import get_csv_content

CLS_VARS = "C:/Users/Juan/OneDrive - CINVESTAV/head_it"

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', CLS_VARS])

if __name__ == '__main__':
    emotion_1, emotion_2 = 'happy', 'sad'
    file_path, emotions = CLS_VARS + '/graph_metrics_mayor_igual_que_25_all.csv', head_it_config.EMOTIONAL_LABELS
    data = get_csv_content(file_path).dropna().query('emotion in ("' + emotion_1 + '", "' + emotion_2 +
                                                     '")')
    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for ith, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]

    '''X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    rng = np.random.RandomState(seed=42)
    X['random_cat'] = rng.randint(3, size=X.shape[0])
    X['random_num'] = rng.randn(X.shape[0])

    categorical_columns = ['pclass', 'sex', 'embarked', 'random_cat']
    numerical_columns = ['age', 'sibsp', 'parch', 'fare', 'random_num']

    X = X[categorical_columns + numerical_columns]'''

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

    '''categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessing = ColumnTransformer(
        [('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])

    rf = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', RandomForestClassifier(n_estimators=500, random_state=42))
    ])'''
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)

    print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
    print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))

    '''ohe = (rf.named_steps['preprocess']
        .named_transformers_['cat']
        .named_steps['onehot'])'''
    feature_names = X.columns

    tree_feature_importances = (
        rf.feature_importances_)
    sorted_idx = tree_feature_importances.argsort()

    y_ticks = np.arange(0, len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, tree_feature_importances[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("Random Forest Feature Importances (MDI)")
    fig.tight_layout()
    plt.show()

    result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()

    result = permutation_importance(rf, X_train, y_train, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_train.columns[sorted_idx])
    ax.set_title("Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()