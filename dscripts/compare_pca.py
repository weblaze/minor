import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the normalized features
features = np.load("normalized_features.npy", allow_pickle=True)

# Check the shape and type of the loaded data
print("Loaded feature shape:", features.shape)
print("Data type:", type(features))

# Standardize data (PCA works best with standardized data)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run PCA
n_components = min(features.shape[0], features.shape[1])  # Ensuring valid PCA components
pca = PCA(n_components=0.95)  # Retain 95% variance
features_pca = pca.fit_transform(features_scaled)

# Explained variance ratio
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot variance explained
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o", linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance vs. Number of Components")
plt.grid()
plt.show()

# Save PCA-transformed features
with open("features_pca.json", "w") as f:
    json.dump(features_pca.tolist(), f)

# Save original features for comparison
with open("features_no_pca.json", "w") as f:
    json.dump(features.tolist(), f)

print(f"Original shape: {features.shape}")
print(f"PCA-reduced shape: {features_pca.shape}")
print("✅ PCA comparison done. Both versions saved!")
