import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
pca = PCA()

X_pca = pca.fit_transform(StandardScaler().fit_transform(data.data))

exp_var_rat = pca.explained_variance_ratio_

plt.figure(figsize=(14, 9))
plt.plot(range(1, len(exp_var_rat) + 1), np.cumsum(exp_var_rat), marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()
