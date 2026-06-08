"""
05_cluster_size_analysis.py
============================
Analyses the distribution of cluster sizes for the AgglomerativeClustering
split (n_clusters=150, Ward linkage on standardised Mordred descriptors).

Prints: mean, median, std, min, max, IQR of cluster sizes.

REQUIRES: mordred_features_cached.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

MORDRED_FILE = "mordred_features_cached.csv"
N_CLUSTERS   = 150

print(f"Loading Mordred features ...")
X = pd.read_csv(MORDRED_FILE).values.astype(float)
X = np.nan_to_num(X)
print(f"  {X.shape[0]} molecules × {X.shape[1]} descriptors")

print(f"\nRunning AgglomerativeClustering (n_clusters={N_CLUSTERS}, linkage='ward') ...")
Xs     = StandardScaler().fit_transform(X)
labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(Xs)
_, sizes = np.unique(labels, return_counts=True)

print(f"\nCluster size statistics (n_clusters={N_CLUSTERS}):")
print(f"  mean       = {sizes.mean():.2f}")
print(f"  median     = {np.median(sizes):.1f}")
print(f"  std        = {sizes.std():.2f}")
print(f"  min        = {sizes.min()}")
print(f"  max        = {sizes.max()}")
print(f"  IQR        = {np.percentile(sizes,25):.0f}–{np.percentile(sizes,75):.0f}")
print(f"\nNote: well-balanced clusters (~{X.shape[0]//N_CLUSTERS} molecules each) "
      f"are required for GroupKFold to produce equally-sized folds.")
