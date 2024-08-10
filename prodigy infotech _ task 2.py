import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Step 1: Generate synthetic dataset
np.random.seed(42)
num_customers = 100
purchase_history = {
    'CustomerID': range(1, num_customers + 1),
    'Purchase1': np.random.randint(1, 1000, num_customers),
    'Purchase2': np.random.randint(1, 500, num_customers),
    'Purchase3': np.random.randint(1, 2000, num_customers),
    'Purchase4': np.random.randint(1, 1500, num_customers)
}


data = pd.DataFrame(purchase_history)

# Step 2: Preprocess Data
# Drop the CustomerID column for clustering
X = data.drop('CustomerID', axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 3: Implement K-Means Clustering
# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# From the Elbow Curve, let's choose k=4
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


# Step 4: Evaluate and Visualize Clusters
# Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Create a DataFrame with the PCA results and cluster assignments
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = data['Cluster']

# Plot the PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('Customer Clusters')
plt.show()

# Display the first few rows of the clustered data
data.head()




