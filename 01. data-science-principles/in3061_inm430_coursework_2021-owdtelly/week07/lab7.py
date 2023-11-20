import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances, confusion_matrix, mean_squared_error
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import KFold
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import LinearRegression

census_data = pd.read_csv("censusCrime.csv")

numerical_census = census_data.select_dtypes(include="number")

pca = PCA(n_components=2)

components = pca.fit_transform(numerical_census)

print(pca.explained_variance_ratio_)

scaler = StandardScaler()

standardised_data = scaler.fit_transform(numerical_census)

scaled_components = pca.fit_transform(standardised_data)

print(pca.explained_variance_ratio_)

pca_df = pd.DataFrame(data=scaled_components, columns=["PC1", "PC2"])

# plt.scatter(pca_df["PC1"], pca_df["PC2"])
# plt.show()

loadings_df = pd.DataFrame(pca.components_, columns=numerical_census.columns)

loadings_df = loadings_df.transpose()

loadings_df = loadings_df.abs()

loadings_df = loadings_df.sort_values(by=0, ascending=False)

loadings_reduced = loadings_df.drop(loadings_df.index[0])

# plot_color = loadings_reduced.iloc[28, :]

# cmap = plt.cm.get_cmap("viridis", len(plot_color.unique()))

# plt.scatter(
#     loadings_reduced.iloc[:, 0],
#     loadings_reduced.iloc[:, 1],
#     c=plot_color,
#     cmap=cmap,
# )
# plt.show()

london_data = pd.read_excel("london-borough-profilesV2.xlsx")

london_clean = london_data.apply(pd.to_numeric, errors="coerce")
london_clean.dropna(axis=1, inplace=True)

distance_matrix = euclidean_distances(london_clean, london_clean)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

mds_distance = mds.fit_transform(distance_matrix)

borough_names = london_data["Area/INDICATOR"]

# plt.scatter(mds_distance[0:32, 0], mds_distance[0:32, 1])

# for i, borough in enumerate(borough_names):
#     plt.annotate(str(borough), (mds_distance[i, 0], mds_distance[i, 1]))

# plt.show()

wine_data = pd.read_csv("wine.csv")

wine_slim = wine_data.iloc[:, 1:]

kmeans = KMeans(n_clusters=3, random_state=1)

wine_slim["Cluster"] = kmeans.fit_predict(wine_slim.values)

wine_matrix = euclidean_distances(wine_slim, wine_slim)

wine_mds = mds.fit_transform(wine_matrix)

# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# kmeans_scatter = axs[0].scatter(
#     wine_mds[:, 0], wine_mds[:, 1], c=wine_slim["Cluster"], cmap="viridis"
# )
# axs[0].scatter(
#     kmeans.cluster_centers_[:, 0],
#     kmeans.cluster_centers_[:, 1],
#     s=200,
#     marker="X",
#     c="red",
#     label="Centroids",
# )

# original_scatter = axs[1].scatter(
#     wine_mds[:, 0], wine_mds[:, 1], c=wine_data["Class label"], cmap="viridis"
# )

# plt.tight_layout()
# plt.show()

# conf_matirx = confusion_matrix(wine_data["Class label"], wine_slim["Cluster"])

# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     conf_matirx,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     cbar=False,
#     xticklabels=["Cluster 0", "Cluster 1", "Cluster 2"],
#     yticklabels=["Label 1", "Label 2", "Label 3"],
# )
# plt.show()


independent = census_data[["medIncome"]]
dependent = census_data["ViolentCrimesPerPop"]

kf = KFold(n_splits=5)

mse_score = []

model = LinearRegression()

for train_idxs, test_idxs in kf.split(census_data):
    X_train, X_test = independent.iloc[train_idxs], independent.iloc[test_idxs]
    y_train, y_test = dependent.iloc[train_idxs], dependent[test_idxs]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mse_score.append(mse)

    # print("Run: ", fold_count)
    # print(len(census_data), len(train_idxs), len(test_idxs))
    # dependent_subset.append(dependent[train_idxs])
    # independent_subset.append(independent[train_idxs])
    # dependent_subset_unseen.append(dependent[test_idxs])
    # independent_subset_unseen.append(independent[test_idxs])

average_mse = np.mean(mse_score)

residuals = y_test - y_pred

plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red")
plt.legend()
plt.show()

("hello")
