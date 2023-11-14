import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("site_diary_text.csv")

# Assuming your 'text' column contains the free text
X = df["text"]

# Step 3: Text Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X.values.astype("U"))

# Step 4: Dimensionality Reduction (Optional, for visualization purposes)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_vectorized.toarray())

# Step 5: Clustering (Number of clusters is a hyperparameter)
num_clusters = 5  # Adjust this based on the characteristics of your data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_vectorized)

# Step 6: Visualize the Clusters (Optional)
plt.scatter(
    X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.5
)
plt.title("Clustering of Text Data")
plt.show()

# Step 7: Assign Cluster Labels to Data
df["cluster_label"] = kmeans.labels_

# Now you can use the cluster labels as your target variable for training a model
# You may manually inspect some samples from each cluster to understand the topics

# For example, to see some samples from the first cluster:
cluster_0_samples = df[df["cluster_label"] == 0]["text"].head(5)
print("Samples from Cluster 0:\n", cluster_0_samples)
