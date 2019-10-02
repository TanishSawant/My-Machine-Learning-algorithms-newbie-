import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = 'C:\\Users\\milindsawant\\Downloads\\Cust_Segmentation.csv'

cust_df = pd.read_csv(file)

cust_df.head()

df = cust_df.drop('Address' , axis = 1)

df.head()

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

X = df.values[: , 1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

print('Cluster dataset = ',Clus_dataSet)

#plt.scatter(Clus_dataSet , X)

#plt.show()

clusNum = 5

k_means = KMeans(init = 'k-means++' , n_clusters = clusNum , n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)

df["clus_km"] = labels

df.head(5)

df.groupby('clus_km').mean()

area = np.pi * (X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)



#New code

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize= (8, 6))
plt.clf()
ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))


plt.show()