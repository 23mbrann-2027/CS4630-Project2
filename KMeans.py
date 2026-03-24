from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time
import matplotlib.pyplot as plt

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

start = time.time()
clusters_raw = kmeans.fit_predict(scaled_X)
time_raw = time.time() - start

# Subsample for metrics
idx = np.random.choice(scaled_X.shape[0], size=50000, replace=False)
sil_raw = silhouette_score(scaled_X[idx], clusters_raw[idx])
db_raw  = davies_bouldin_score(scaled_X[idx], clusters_raw[idx])

print(f"Raw KMeans runtime: {time_raw:.2f}s")
print(f"Silhouette: {sil_raw:.4f}  |  Davies-Bouldin: {db_raw:.4f}")

# Find Algorithm Accuracy
acc1 = accuracy_score(y, clusters_raw)
acc2 = accuracy_score(y, 1 - clusters_raw)

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
clusters_pca_dict = {}

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
    clusters_pca_dict[n] = clusters
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

    # Compactness: mean distance from each point to its centroid
    centroids = kmeans.cluster_centers_
    compactness = np.mean([
        np.mean(np.linalg.norm(X_sample[clusters_sample == k] - centroids[k], axis=1))
        for k in range(2)
    ])

    # Separation: distance between the two centroids
    separation = np.linalg.norm(centroids[0] - centroids[1])

    print(f"Silhouette Score:    {silhouette:.4f}")
    print(f"Davies-Bouldin:      {db_index:.4f}")
    print(f"Compactness:         {compactness:.4f}")
    print(f"Separation:          {separation:.4f}")


# ── Scatter plots ─────────────────────────────────────────────────────────────
pca_plot = PCA(n_components=2, random_state=42)
X_2d = pca_plot.fit_transform(scaled_X)

PLOT_N = 10000
idx_plot = np.random.choice(len(X_2d), size=PLOT_N, replace=False)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("K-Means Clustering: Raw vs PCA-Reduced", fontsize=14, fontweight="bold")

colors = ["steelblue", "tomato"]
titles = ["Raw 28-dim (proj to 2D)", "PCA-2", "PCA-5", "PCA-10"]
all_clusters = [clusters_raw, clusters_pca_dict[2], clusters_pca_dict[5], clusters_pca_dict[10]]

for ax, title, lbls in zip(axes, titles, all_clusters):
    for cls, col in zip([0, 1], colors):
        mask = lbls[idx_plot] == cls
        ax.scatter(X_2d[idx_plot][mask, 0], X_2d[idx_plot][mask, 1],
                   c=col, s=5, alpha=0.4, label=f"Cluster {cls}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, fontsize=9)

plt.tight_layout()
plt.savefig("cluster_scatter.png", dpi=150)
plt.show()


# ── Metrics bar chart ─────────────────────────────────────────────────────────
# Pull metrics from the dicts already computed above
idx_m = np.random.choice(scaled_X.shape[0], size=50000, replace=False)
sil_scores = [sil_raw]
db_scores  = [db_raw]

for n in [2, 5, 10]:
    X_s = X_pca_dict[n][idx_m]
    c_s = clusters_pca_dict[n][idx_m]
    sil_scores.append(silhouette_score(X_s, c_s))
    db_scores.append(davies_bouldin_score(X_s, c_s))

labels = ["Raw 28-dim", "PCA-2", "PCA-5", "PCA-10"]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Clustering Quality Metrics", fontsize=14, fontweight="bold")

axes[0].bar(x, sil_scores, color="steelblue")
axes[0].set_title("Silhouette Score (higher = better)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Score")

axes[1].bar(x, db_scores, color="tomato")
axes[1].set_title("Davies-Bouldin Index (lower = better)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Index")

plt.tight_layout()
plt.savefig("metrics_bar.png", dpi=150)
plt.show()


# ── Scree plot ────────────────────────────────────────────────────────────────
pca_full = PCA(random_state=42).fit(scaled_X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 29), cumvar * 100, marker="o", color="steelblue")
plt.axhline(90, color="red", linestyle="--", label="90% threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.title("PCA Scree Plot", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=150)
plt.show()