import sys

from sklearn.manifold import TSNE
from sklearn.metrics import homogeneity_score, completeness_score, fowlkes_mallows_score

WORKING_DIR = "/"

sys.path.extend(['C:\\Users\\Juan\\PycharmProjects\\feelings-in-eeg', WORKING_DIR])

import head_it.config as head_it_config

from featuresextractor import get_csv_content

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create the dataset
    file_path, emotions = WORKING_DIR + '/data/graph_metrics_all.csv', head_it_config.EMOTIONAL_LABELS

    emotions_set = "'love', 'joy', 'happy', 'relief', 'compassion', 'content', 'excite', 'awe'"
    # emotions_set = "'anger', 'jealousy', 'disgust', 'frustration', 'fear', 'sad', 'grief'"
    # Load the dataset
    data = get_csv_content(file_path).dropna().query('emotion in (' + emotions_set + ')')
    data = data[head_it_config.CLASSIFICATION_VARS]
    y, X = np.array([emotions.index(value) for ith, value in data[head_it_config.CLASSIFICATION_VARS[0]].items()]), \
           data[head_it_config.CLASSIFICATION_VARS[1:]]
    X_train = X.to_numpy()

    ss = StandardScaler()
    Xs = ss.fit_transform(X_train)

    # Test K-Means
    km = KMeans(n_clusters=8, random_state=1000)
    Y_km = km.fit_predict(Xs)
    print('Homogeneity score: {}'.format(homogeneity_score(y, Y_km)))
    print('Completeness score: {}'.format(completeness_score(y, Y_km)))
    print('Fowlkes-Mallows index: {}'.format(fowlkes_mallows_score(y, Y_km)))

    # Plot the result
    fig, ax = plt.subplots(figsize=(16, 8))

    # Perform t-SNE on the clustered dataset
    tsne = TSNE(n_components=2, perplexity=20.0, random_state=1000)
    X_tsne = tsne.fit_transform(X_train)

    # Show the t-SNE clustered dataset
    for i in range(X_tsne.shape[0]):
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='o', color=cm.Pastel1(Y_km[i]), s=150)
        ax.annotate('%d' % Y_km[i], xy=(X_tsne[i, 0] - 0.5, X_tsne[i, 1] - 0.5))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()

    # Apply Spectral clustering
    sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', n_neighbors=20, random_state=1000)
    Y_sc = sc.fit_predict(Xs)
    print('Homogeneity score: {}'.format(homogeneity_score(y, Y_sc)))
    print('Completeness score: {}'.format(completeness_score(y, Y_sc)))
    print('Fowlkes-Mallows index: {}'.format(fowlkes_mallows_score(y, Y_sc)))

    # Plot the result
    fig, ax = plt.subplots(figsize=(16, 8))

    # Perform t-SNE on the clustered dataset
    tsne = TSNE(n_components=2, perplexity=20.0, random_state=1000)
    X_tsne = tsne.fit_transform(X_train)

    # Show the t-SNE clustered dataset
    for i in range(X_tsne.shape[0]):
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='o', color=cm.Pastel1(Y_sc[i]), s=150)
        ax.annotate('%d' % Y_sc[i], xy=(X_tsne[i, 0] - 0.5, X_tsne[i, 1] - 0.5))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.grid()
    plt.show()
