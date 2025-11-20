import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math

from sklearn.decomposition import PCA # type: ignore
from sklearn.cluster import AgglomerativeClustering # type: ignore
from sklearn.cluster import DBSCAN # type: ignore
from sklearn.cluster import OPTICS # type: ignore
from sklearn.cluster import Birch # type: ignore
from sklearn.cluster import BisectingKMeans # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.cluster import SpectralClustering # type: ignore
from sklearn.mixture import GaussianMixture # type: ignore
from sklearn.cluster import HDBSCAN # type: ignore
from sklearn.preprocessing import StandardScaler, normalize # type: ignore
from sklearn.metrics import silhouette_score # type: ignore
import scipy.cluster.hierarchy as shc # type: ignore
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

df1 = pd.read_csv('FOOD-DATA-GROUP1.csv')
numpy_array1 = df1.to_numpy()
df2 = pd.read_csv('FOOD-DATA-GROUP2.csv')
numpy_array2 = df2.to_numpy()
df3 = pd.read_csv('FOOD-DATA-GROUP3.csv')
numpy_array3 = df3.to_numpy()
df4 = pd.read_csv('FOOD-DATA-GROUP4.csv')
numpy_array4 = df4.to_numpy()
df5 = pd.read_csv('FOOD-DATA-GROUP5.csv')
numpy_array5 = df5.to_numpy()

data = np.vstack((numpy_array1[:, 5:],numpy_array2[:, 5:],numpy_array3[:, 5:],numpy_array4[:, 5:],numpy_array5[:, 5:]))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
X_normalized = normalize(X_scaled)
X = pd.DataFrame(X_normalized)
X = X.to_numpy()

''' K-MEANS CLUSTERING '''
ss5_vals = []
ss5_drops = []
for n in range(2,20):
    model5 = KMeans(n_clusters=n)
    model5.fit(X)
    # ss5 = silhouette_score(X, labels5)
    # ss5_vals.append(ss5)
    ss5 = model5.inertia_
    ss5_vals.append(ss5)
    vector = np.array([n, ss5])
    if n > 3 and n < 19:
        diff1 = ss5_vals[-3] - ss5_vals[-2]
        diff2 = ss5_vals[-2] - ss5
        ss5_drops.append(diff1 / diff2)
    # print('ss5: ', ss5)
    # print('vector: ', vector)
    # print('distance: ', np.sqrt(np.sum(vector**2)))
plt.plot(range(2,20),ss5_vals)
plt.xlim([2,19])
plt.title("Inertia for k-Means Clustering with k Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
plt.plot(range(3,18),ss5_drops)
plt.xlim([2,19])
plt.title("EL1(k) for k-Means Clustering with k Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("EL1(k)")
plt.show()
model = KMeans(n_clusters=7)
labels = model.fit_predict(X)
ss = silhouette_score(X, labels)
print('k-Means Clustering Silhouette Score (k=7): ', ss)
average_elem0 = np.mean(X[labels == 0, :], axis=0)
print('elem 0: ', average_elem0)
average_elem1 = np.mean(X[labels == 1, :], axis=0)
print('elem 1: ', average_elem1)
average_elem2 = np.mean(X[labels == 2, :], axis=0)
print('elem 2: ', average_elem2)
average_elem3 = np.mean(X[labels == 3, :], axis=0)
print('elem 3: ', average_elem3)
average_elem4 = np.mean(X[labels == 4, :], axis=0)
print('elem 4: ', average_elem4)
average_elem5 = np.mean(X[labels == 5, :], axis=0)
print('elem 5: ', average_elem5)
average_elem6 = np.mean(X[labels == 6, :], axis=0)
print('elem 6: ', average_elem6)
averages = np.vstack((average_elem0, average_elem1, average_elem2, average_elem3, average_elem4, average_elem5, average_elem6))
df_avg = pd.DataFrame(averages, columns=["Caloric Value","Fat","Saturated Fats","Monounsaturated Fats","Polyunsaturated Fats","Carbohydrates","Sugars","Protein","Dietary Fiber","Cholesterol","Sodium","Water","Vitamin A","Vitamin B1","Vitamin B11","Vitamin B12","Vitamin B2","Vitamin B3","Vitamin B5","Vitamin B6","Vitamin C","Vitamin D","Vitamin E","Vitamin K","Calcium","Copper","Iron","Magnesium","Manganese","Phosphorus","Potassium","Selenium", "Zinc", "Nutrition Density"])
excel_file_path = 'averages.xlsx'
df_avg.to_excel(excel_file_path, index=False)

print('k-Means Cluster 0:', np.count_nonzero(labels == 0))
print('k-Means Cluster 1:', np.count_nonzero(labels == 1))
print('k-Means Cluster 2:', np.count_nonzero(labels == 2))
print('k-Means Cluster 3:', np.count_nonzero(labels == 3))
print('k-Means Cluster 4:', np.count_nonzero(labels == 4))
print('k-Means Cluster 5:', np.count_nonzero(labels == 5))
print('k-Means Cluster 6:', np.count_nonzero(labels == 6))


''' SPECTRAL CLUSTERING '''
model7 = SpectralClustering(n_clusters=7)
labels7 = model7.fit_predict(X)
ss7 = silhouette_score(X, labels7)
print('Spectral Clustering Silhouette Score (k=7): ', ss7)
print('Spectral Cluster 0:', np.count_nonzero(labels7 == 0))
print('Spectral Cluster 1:', np.count_nonzero(labels7 == 1))
print('Spectral Cluster 2:', np.count_nonzero(labels7 == 2))
print('Spectral Cluster 3:', np.count_nonzero(labels7 == 3))
print('Spectral Cluster 4:', np.count_nonzero(labels7 == 4))
print('Spectral Cluster 5:', np.count_nonzero(labels7 == 5))
print('Spectral Cluster 6:', np.count_nonzero(labels7 == 6))


''' AGGLOMERATIVE CLUSTERING CODE: '''
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plt.title("Agglomerative Clustering Dendrogram for Food Nutrition Dataset")
plot_dendrogram(model, truncate_mode="level", p=4) # plot the top p levels of the dendrogram
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.ylabel("Distance Threshold")
plt.show()
model = AgglomerativeClustering(n_clusters=2) #
labels = model.fit_predict(X)
ss = silhouette_score(X, labels)
print('Agglomerative Clustering Silhouette Score (k=2): ', ss)
print('Agglomerative Cluster 0:', np.count_nonzero(labels == 0))
print('Agglomerative Cluster 1:', np.count_nonzero(labels == 1))

# # ------ ELBOW GRAPH (NOT USEFUL) --------
# # for n in range(2,20):
# #     model = AgglomerativeClustering(n_clusters=n)
# #     labels = model.fit_predict(X)
# #     ss = silhouette_score(X, labels)
# #     ss_vals.append(ss)
# #     print([n, ss])
# # plt.plot(range(2,20),ss_vals)
# # plt.xlim([2,19])
# # plt.title("Silhouette Score for Agglomerative Clustering with n Clusters")
# # plt.xlabel("Number of Clusters (n)")
# # plt.ylabel("Silhouette Score")
# # plt.show()



''' BIRCH CLUSTERING '''
distances = []
for i in range(0, 2395):
    rowi = X[i, :]
    for j in range(i+1,2395):
        rowj = X[j, :]
        dist_ij = np.sqrt(np.sum((rowi-rowj)**2))
        print(dist_ij)
        distances.append(dist_ij)
distances=np.array(distances)
print(np.min(distances), np.max(distances), np.median(distances), np.mean(distances), np.std(distances))

bf_values = np.zeros((23, 2))
for bf_val in range(1, 24):
    model4 = Birch(threshold= 0.41, branching_factor=50*bf_val)
    labels4 = model4.fit_predict(X)
    ss4 = silhouette_score(X, labels4)
    bf_values[bf_val - 1, 0] = 50*bf_val
    bf_values[bf_val - 1, 1] = ss4
    # print([50*bf_val, ss4])
index = np.argmax(bf_values[:, 1])
bf = bf_values[index, 0]
ss4 = np.max(bf_values[:, 1])
labels_array = np.array(labels4)
labels_types = np.unique(labels4)
print('BIRCH Branching Factor: ', bf)
print('BIRCH Number of Clusters: ', labels_types.size)
print('BIRCH Silhouette Score: ', ss4)



''' DBSCAN CLUSTERING '''
# distances = []
# for i in range(0, 2395):
#     rowi = X[i, :]
#     for j in range(i+1,2395):
#         rowj = X[j, :]
#         dist_ij = np.sqrt(np.sum((rowi-rowj)**2))
#         print(dist_ij)
#         distances.append(dist_ij)
# distances=np.array(distances)
# sorted_distances = np.sort(distances)
# plt.hist(sorted_distances)
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Distances")
# plt.show()

model2 = DBSCAN(eps=0.42, min_samples=120)
labels2 = model2.fit_predict(X)
ss2 = silhouette_score(X, labels2)
labels_types = np.unique(labels2)
print('DBSCAN Number of Clusters: ', labels_types.size)
print('DBSCAN Silhouette Score: ', ss2)



''' OPTICS CLUSTERING '''
model3 = OPTICS(min_samples=4, max_eps=2)
labels3 = model3.fit_predict(X)
ss3 = silhouette_score(X, labels3)
labels_types = np.unique(labels3)
print('OPTICS Number of Clusters: ', labels_types.size)
print('OPTICS Silhouette Score: ', ss3)



''' HDBSCAN CLUSTERING '''
model7 = HDBSCAN(min_cluster_size = 60)
labels7 = model7.fit_predict(X)
ss7 = silhouette_score(X, labels7)
labels_types = np.unique(labels7)
print('HDBSCAN Number of Clusters: ', labels_types.size)
print('HDBSCAN Silhouette Score: ', ss7)



''' BISECTING K-MEANS CLUSTERING '''
# ss8_vals = []
# for n in range(2,20):
#     # model5 = BisectingKMeans(n_clusters=n)
#     model8 = SpectralClustering(n_clusters=n)
#     labels8 = model8.fit_predict(X)
#     ss8 = silhouette_score(X, labels8)
#     ss8_vals.append(ss8)
#     print([n, ss8])

# plt.plot(range(2,20),ss8_vals)
# plt.xlim([2,19])
# # plt.title("Silhouette Score for Bisecting k-Means Clustering with n Clusters")
# plt.title("Silhouette Score for Spectral Clustering with n Clusters")
# plt.xlabel("Number of Clusters (n)")
# plt.ylabel("Silhouette Score")
# plt.show()

''' GAUSSIAN MIXTURE CLUSTERING '''
# # model6 = GaussianMixture(n_components=5, reg_covar=0.001)
# # labels6 = model6.fit(X)
# # ss6 = silhouette_score(X, labels6)