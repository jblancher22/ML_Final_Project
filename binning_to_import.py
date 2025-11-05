from for_importing import y
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


#this file bins the data (the same as binning.ipynb) so that the binning doesn't need to be done every file


yflat = y.values.reshape(-1, 1)


kmeans = KMeans(n_clusters=2, random_state=22)
bins_kmeans_raw = kmeans.fit_predict(yflat)


centers = kmeans.cluster_centers_.flatten()
sorted_labels = np.argsort(centers)
label_map = {old: new for new, old in enumerate(sorted_labels)}
clusters = pd.Series(bins_kmeans_raw).map(label_map)


cluster_label={'0':'Low Crime Community','1':'High Crime Community'}
y_bins = pd.DataFrame({'y': y, 'cluster': clusters.astype(str)})
y_bins['cluster_label'] = y_bins['cluster'].map(cluster_label)
y=y_bins['cluster']



