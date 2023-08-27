import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load scaled features and labels from preprocess.py
scaled_features = np.load('scaled_features.npy')  # Adjust the file path as needed


# Apply the elbow method to determine the optimal number of clusters
wcss = []  # Within-cluster sum of squares

# Try different values of k (e.g., from 1 to 10)
n_init_value = 60  # Explicitly set the n_init parameter
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, n_init=n_init_value, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

# Plot the elbow curve
plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal k')
plt.show()





