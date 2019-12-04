"""
Feature importance via permutation tests
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plt
import random
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)
y = data['class'].values
X = data.drop('class',axis=1).values
ftr_names = data.columns[:-1]

file = open(parent_dir + '/results/grid.save', 'rb')
models, X_test, y_test = pickle.load(file)
file.close()

#choose a random model for feature importance instead of averaging to save time
np.random.seed = 19
model = random.choice(models)

nr_runs = 10
scores = np.zeros([len(ftr_names),nr_runs])

test_score = model.score(X_test,y_test)
print('test score = ',test_score)
print('test baseline = ',np.sum(y_test == 0)/len(y_test))
# loop through the features
for i in range(len(ftr_names)):
    print('shuffling '+str(ftr_names[i]))
    acc_scores = []
    for j in range(nr_runs):
        X_test_df = pd.DataFrame(X_test, columns=ftr_names)
        X_test_shuffled = X_test_df.copy()
        X_test_shuffled[ftr_names[i]] = np.random.permutation(X_test_df[ftr_names[i]].values)
        acc_scores.append(model.score(X_test_shuffled,y_test))
    print('   shuffled test score:',np.around(np.mean(acc_scores),3),'+/-',np.around(np.std(acc_scores),3))
    scores[i] = acc_scores
    
sorted_indcs = np.argsort(np.mean(scores,axis=1))[::-1]
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8,6))
plt.boxplot(scores[sorted_indcs].T,labels=ftr_names[sorted_indcs],vert=False)
plt.axvline(test_score,label='Overall Test Score')
plt.title("Permutation Importances (Test Set)")
plt.xlabel('Accuracy Score with Perturbed Feature')
plt.legend()
plt.tight_layout()
# to save the figure, uncomment the line below
#plt.savefig(parent_dir + '/figures/permutation_importance.png',dpi=300)
plt.show()