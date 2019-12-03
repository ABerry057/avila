"""
Metrics for classification
"""
import pickle
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)

file = open(parent_dir + '/results/grid.save', 'rb')
grid, X_test, y_true = pickle.load(file)
file.close()

y_pred = grid.predict(X_test)

confusion_matrices = multilabel_confusion_matrix(y_true,
                                                 y_pred,
                                                 labels = [i for i in range(0,12)])
cm_0 = confusion_matrices[0]
cm_1 = confusion_matrices[1]