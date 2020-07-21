import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# k-means on a randomly generated dataset
np.random.seed(0)

"""
Make random clusters of points by using the make_blobs class.

Some inputs available for make_blobs class:

Input

- n_samples: the total number of points equally divided among clusters.
- centers: the number of centers to generate, or the fixed center locations.
- cluster_std: the standard deviation of the clusters.

Output

- X: Array of shape [n_samples, n_features]. (Feature Matrix)
- y: Array of shape [n_samples]. (Response Vector)
"""

X, y = make_blobs(
    n_samples = 5000,
    centers=[[4,4], [-2, -1], [2, -3], [1, 1]],
    cluster_std=0.9)

plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

"""
Setting up k-Means

- init: initialization method of the centroids
    --> k-means++: selects initial cluster centers for k-mean clustering
                   in a smart way to speed up convergence.
- n_clusters: the number of clusters to form as well as the number
              centroids to generate.
- n_init: number of times the k-means algorithm will be run with
          different centroid seeds. The final results will be
          the best out of n_init consecutive runs in terms of intertia.
"""

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
print(k_means_labels)
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)


# Visual Plot

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1,1,1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.

for k, col in zip(range(len([[4,4], [-2,-1], [2,-3], [1,1]])), colors):
    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plot the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')

    # Plot the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col, markeredgecolor='k', markersize=6)


ax.set_title("KMeans")
# Remove x-axis and y-axis ticks
ax.set_xticks(())
ax.set_yticks(())
plt.show()
