# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 08:23:30 2018

@author: welcome
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans
from pylab import *
from sklearn import preprocessing
X= pd.read_csv("dataset.csv")
print(X)
y = preprocessing.LabelEncoder()
y.fit(X['From'])
X['From'] = y.transform(X['From']) 
y.fit(X['To'])
y.fit(X['To'])
X['To']=y.transform(X['To']) 
y.fit(X['Via'])
X['Via']=y.transform(X['Via'])
M=X.iloc[:,[8,10]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 34)
    kmeans.fit(M)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('To')
plt.ylabel('Via')
plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 34)
y_kmeans = kmeans.fit_predict(M)
plt.scatter(M[y_kmeans == 0, 0], M[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(M[y_kmeans == 1, 0], M[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(M[y_kmeans == 2, 0], M[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(M[y_kmeans == 3, 0], M[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(M[y_kmeans == 4, 0], M[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Vehicles')
plt.xlabel('To')
plt.ylabel('Via')
plt.legend()
plt.show()
 