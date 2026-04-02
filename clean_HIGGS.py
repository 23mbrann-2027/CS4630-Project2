import pandas as pd
import numpy as np

chunksize = 1000000

for chunk in pd.read_csv("Data/raw/HIGGS.csv.gz", compression = "gzip", header = None,chunksize = chunksize):
    chunk = chunk.dropna()
    chunk = chunk.drop_duplicates()

    chunk.to_csv("Data/processed/HIGGS_cleaned.csv", mode = 'a', index = False, header = False)
