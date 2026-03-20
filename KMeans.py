from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time

data = pd.read_csv(
    "Data/HIGGS.csv.gz",
    header = None,
    compression = "gzip",
    nrows = 1000000
)

# Need to seperate the labels and the features
y = data.iloc[:,0].astype(int) #Signals
X = data.iloc[:,1:] #Features

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


kmeans = MiniBatchKMeans(
    n_clusters = 2,
    batch_size = 10000,
    n_init = 10,
    random_state = 42
)

#kmeans.fit(scaled_X)

clusters = kmeans.fit_predict(scaled_X)

# Find Algorithm Accuracy
acc1 = accuracy_score(y, clusters)
acc2 = accuracy_score(y, 1 - clusters)

print("Best clustering accuracy:", max(acc1, acc2))


# PCA dimensions (Part 2)
pca_components = [2, 5, 10]
X_pca_dict = {}  # store each reduced dataset

for n in pca_components:
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(scaled_X)
    
    # Keep each PCA-reduced
    X_pca_dict[n] = X_pca
    
    # show how much variance is preserved
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n} components preserves {explained_var:.2%} of variance")


#Part 3 of re running the Kmeans again with subsampling
sample_size = 50000  # 50k rows for metric calculation

for n, X_pca in X_pca_dict.items():
    kmeans = MiniBatchKMeans(
        n_clusters=2,
        batch_size=10000,
        n_init=10,
        random_state=42
    )
    
    # Fit k-Means on the full PCA-reduced data
    start = time.time()
    clusters = kmeans.fit_predict(X_pca)
    train_time = time.time() - start
    
    # Subsample for metric calculation to avoid long wait time
    if X_pca.shape[0] > sample_size:
        idx = np.random.choice(X_pca.shape[0], size=sample_size, replace=False)
        X_sample = X_pca[idx]
        clusters_sample = clusters[idx]
    else:
        X_sample = X_pca
        clusters_sample = clusters
    
    # Calculate metrics on the subsample
    silhouette = silhouette_score(X_sample, clusters_sample)
    db_index = davies_bouldin_score(X_sample, clusters_sample)
    
    print(f"\nPCA {n} components:")
    print("Training time:", train_time)
    print("Silhouette Score (sampled):", silhouette)
    print("Davies-Bouldin Index (sampled):", db_index)