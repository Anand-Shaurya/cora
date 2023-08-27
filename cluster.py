import numpy as np
from sklearn.cluster import KMeans

# Load scaled features and labels from preprocess.py
scaled_features = np.load('scaled_features.npy')  # Adjust the file path as needed

# Initialize the KMeans model with the desired number of clusters
num_clusters = 5  # You can adjust this number
n_init_value = 70  # Explicitly set the n_init parameter
kmeans = KMeans(n_clusters=num_clusters, n_init=n_init_value, random_state=0)

# Fit the KMeans model on scaled features
kmeans.fit(scaled_features)

# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_
