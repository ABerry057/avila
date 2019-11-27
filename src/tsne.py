"""
T-distributed stochastic neighbor embeding visualiation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

np.random.seed(19)

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)
y = data['class'].values
X = data.drop('class',axis=1).values

tsne = TSNE(n_components=2, verbose=1, perplexity=75, n_iter=500)
tsne_results = tsne.fit_transform(X)

data['Class'] = y
data['TSNE Axis 1'] = tsne_results[:,0]
data['TSNE Axis 2'] = tsne_results[:,1]

plt.figure(figsize=(16,12))
sns.scatterplot(
    x="TSNE Axis 1", y="TSNE Axis 2",
    hue="Class",
    palette=sns.color_palette("hls", 12),
    data=data,
    legend="full",
    alpha=0.6
)