import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the cora.content file
data = []
with open('cora.content', 'r') as file:
    for line in file:
        values = line.strip().split('\t')
        data.append(values)

# Convert data to a NumPy array or a Pandas DataFrame
data = np.array(data)  # or use pd.DataFrame(data, columns=column_names)

# Split the data into features and labels
features = data[:, 1:-1].astype(float)  # Assuming columns 1 to second last are features
labels = data[:, -1]  # Assuming the last column is labels


# Assuming 'features' is a NumPy array of your numerical features

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on your features and transform them
scaled_features = scaler.fit_transform(features)

# Now 'scaled_features' contains the scaled and preprocessed features

# Use 'scaled_features' for training your machine learning model

# Save scaled_features to a file or return it as needed
np.save('scaled_features.npy', scaled_features)
