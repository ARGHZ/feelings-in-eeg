import sys

WORKING_DIR = "/"


sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

from config import WORKING_DIR
import head_it.config as head_it_config
from featuresextractor import get_csv_content
from models.classification import binarize_targets
import numpy as np

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

print(__doc__)

RANDOM_STATE = 0

file_path, emotions = WORKING_DIR + '/data/graph_metrics_mayor_que_25_all.csv', head_it_config.EMOTIONAL_LABELS
data = get_csv_content(file_path).dropna()

data = data[head_it_config.CLASSIFICATION_VARS]
y, X = np.array([emotions.index(value) for index, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
       data[head_it_config.CLASSIFICATION_VARS[1:]]
# binarize targets
y = binarize_targets(y, 7)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               n_jobs=6,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 25
max_estimators = 500

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("Número de árboles de decisión")
plt.ylabel("Error del clasificador")
plt.show()