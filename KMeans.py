from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

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

