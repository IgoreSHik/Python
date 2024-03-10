import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=1, random_state=42)
X_ranked = kmeans.fit_transform(X_normalized)

pca = PCA(n_components=1)
X_linearized = pca.fit_transform(X_ranked)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Oryginalne dane Iris (2 cechy)')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')

plt.subplot(1, 2, 2)
plt.scatter(X_linearized, np.zeros_like(X_linearized), c=y, cmap='viridis')
plt.title('Linearyzacja danych Iris (1 cecha)')
plt.xlabel('Cecha zlinearyzowana')

plt.show()
