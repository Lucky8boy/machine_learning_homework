import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(14, 8))

for i in range(3):
    cluster_pts = X[labels == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], label=f'Cluster {i + 1}')

plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.xlabel('Property 1')
plt.ylabel('Property 2')
plt.title('k-means using Iris')
plt.legend()
plt.show()
