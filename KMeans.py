from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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


# PCA dimensions 
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