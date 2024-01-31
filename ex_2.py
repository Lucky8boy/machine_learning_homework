import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = data[cols]

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(X))

for cluster in range(6):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Fresh'], cluster_data['Frozen'], label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, cols.index('Fresh')],
            kmeans.cluster_centers_[:, cols.index('Frozen')],
            s=300, c='red', marker='X', label='Centroids')

plt.xlabel('Fresh')
plt.ylabel('Frozen')
plt.legend()
plt.show()
