import sys

from sklearn.metrics import homogeneity_score, completeness_score

WORKING_DIR = "/"

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

import head_it.config as head_it_config

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# Set random seed for reproducibility
from featuresextractor import get_csv_content

np.random.seed(1000)


min_nb_clusters = 2
max_nb_clusters = 20


if __name__ == '__main__':
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS

    emotions_set = "'love', 'joy', 'happy', 'relief', 'compassion', 'content', 'excite', 'awe'"
    # emotions_set = "'anger', 'jealousy', 'disgust', 'frustration', 'fear', 'sad', 'grief'"
    # Load the dataset
    data = get_csv_content(file_path).dropna().query('freq_band == "beta" & emotion in (' + emotions_set + ')')
    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for ith, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    X_train = X.to_numpy()

    # Compute the inertias
    inertias = np.zeros(shape=(max_nb_clusters - min_nb_clusters + 1,))

    for i in range(min_nb_clusters, max_nb_clusters + 1):
        km = KMeans(n_clusters=i, random_state=1000)
        km.fit(X_train)
        inertias[i - min_nb_clusters] = km.inertia_

    # Plot the inertias
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(np.arange(2, max_nb_clusters + 1), inertias)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_xticks(np.arange(2, max_nb_clusters + 1))
    ax.grid()
    plt.show()

    # Perform K-Means with N clusters
    km = KMeans(n_clusters=8, random_state=1000)
    Y = km.fit_predict(X_train)

    print('Homogeneity score: {}'.format(homogeneity_score(y, Y)))
    print('Completeness score: {}'.format(completeness_score(y, Y)))

    # Perform t-SNE on the clustered dataset
    tsne = TSNE(n_components=2, perplexity=20.0, random_state=1000)
    X_tsne = tsne.fit_transform(X_train)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Show the t-SNE clustered dataset
    for i in range(X_tsne.shape[0]):
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='o', color=cm.Pastel1(Y[i]), s=150)
        ax.annotate('%d' % Y[i], xy=(X_tsne[i, 0] - 0.5, X_tsne[i, 1] - 0.5))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()

    plt.show()



