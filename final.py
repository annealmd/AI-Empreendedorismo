#Anne Almeida & Débora Liliane

#import libraries
import numpy as np # to work with math
import matplotlib.pyplot as plt
import pandas as pd # to import dataset

#importing dataset
dataset = pd.read_csv('quest.csv')

X = dataset.iloc[:, 5:].values

from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)
pca_model.fit(X)
X = pca_model.transform(X)
# 2-Dimensions
X[:]

# using elbow method to dinf out the optimal number of clusters
from sklearn.cluster import KMeans #use 10 clusters
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title ('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state= 0)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

# Silhouette
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_avg = silhouette_score(X, y_kmeans)

print()
print("para Kmeans: 4 " )
print("The average silhouette_score is :", silhouette_avg)

#visualising GRAFICO DE K-MEAN
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=30,c ='magenta', label = 'Grupo 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=30, c ='blue', label = 'Grupo 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=30, c ='cyan', label = 'Grupo 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=30, c ='green', label = 'Grupo 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 50, c = 'yellow', label = 'Centroides')
plt.title('Perfil Empreendendor dos Estudantes do IFNMG - MOC')
plt.legend()
plt.show()