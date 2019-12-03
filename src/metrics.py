"""
Metrics for classification
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)

file = open(parent_dir + '/results/grid.save', 'rb')
grid, X_test, y_true = pickle.load(file)
file.close()

y_pred = grid.predict(X_test)

confusion matrices = multilabel_confusion_matrix(y_true,
                                                 y_pred,
                                                 labels = [i for i in range(0,12)])