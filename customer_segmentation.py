KMeans algorithm is used in unsupervised learning

A mall has different types of customers. We have to cluster different categories of customers to help the buisness of mall.

Importing dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

"""Data Collection and analysis"""

#Loading the data from the csv file to a pandas dataframe
customer_data= pd.read_csv('/content/Mall_Customers.csv')

#Print 1st five rows in the DataFrame
customer_data.head()

#finding the no. of rows and columns
customer_data.shape

#getting some info about the dataset
customer_data.info()

#checking for missing values
customer_data.isnull()

"""choosing the anual income column and Spending score column

"""

X = customer_data.iloc[:,[3,4]].values

print(X)

"""Choosing the number of clusters

WCSS --> Within clusters sum of squares
"""

# finding the WCSS value for different no. of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

#Plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('the elbow point graph')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

"""Optimum number of clusters will be 5

Training the Kmeans clustering models...Unsupervised Ml model. K --> Number of clusters
"""

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

#return a label for each data point based on their cluster.
Y = kmeans.fit_predict(X)
print(Y)

"""Visualizing all the cluster

5 Clusters are- 0,1,2,3,4
"""

# Ploting all the clusters and their centroids.
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroid')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

