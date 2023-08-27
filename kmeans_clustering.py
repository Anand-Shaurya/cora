from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load preprocessed data
from preprocess import X_tfidf, y_encoded

# Apply k-means clustering
num_clusters = 7  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_tfidf)

# Evaluate clustering
silhouette_avg = silhouette_score(X_tfidf, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")
