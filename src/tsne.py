"""
T-distributed stochastic neighbor embeding visualiation
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")