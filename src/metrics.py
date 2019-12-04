"""
Metrics for classification
"""
import pickle
import pandas as pd
import numpy as np
import random
from sklearn.metrics import multilabel_confusion_matrix
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)
test_data = pd.read_csv(parent_dir + "/data/avila_p_test.csv")
test_data.drop(test_data.columns[0], inplace=True, axis=1)
y_true = test_data['class'].values
X_test = test_data.drop('class',axis=1).values

file = open(parent_dir + '/results/grid.save', 'rb')
models = pickle.load(file)
file.close()

#choose a random model for feature importance instead of averaging to save time
np.random.seed = 19
model = random.choice(models)

y_pred = model.predict(X_test)

confusion_matrices = multilabel_confusion_matrix(y_true,
                                                 y_pred,
                                                 labels = [i for i in range(0,12)])
cm_0 = confusion_matrices[0]
cm_1 = confusion_matrices[1]