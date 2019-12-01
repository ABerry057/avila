"""
Feature importance via permutation tests
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)
y = data['class'].values
X = data.drop('class',axis=1).values
ftr_names = data.columns[:-1]

def ML_pipeline_kfold(X,y,random_state,n_folds, model, param_grid):
    # create a test set
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    # splitter for _other
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    pipe = make_pipeline(model)
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=param_grid,cv=kf, return_train_score = True,n_jobs=-1,verbose=10)
    # do kfold CV on _other
    grid.fit(X_other, y_other)
    return grid, X_test, y_test

model = KNeighborsClassifier(weights='distance')
param_grid = {"kneighborsclassifier__n_neighbors": range(1,21)}
grid, X_test, y_test = ML_pipeline_kfold(X = X, y = y, random_state = 19, n_folds = 5, model = model, param_grid = param_grid)
print(grid.best_score_)
print(grid.score(X_test,y_test))
print(grid.best_params_)

# save the output
import pickle
file = open(parent_dir + '/results/grid.save', 'wb')
pickle.dump((grid,X_test,y_test),file)
file.close()

file = open('/results/grid.save', 'rb')
grid, X_test, y_test = pickle.load(file)
file.close()

nr_runs = 10
scores = np.zeros([len(ftr_names),nr_runs])

test_score = grid.score(X_test,y_test)
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
        acc_scores.append(grid.score(X_test_shuffled,y_test))
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
plt.show()