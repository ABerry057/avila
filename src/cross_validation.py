"""
Cross-validation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

data = pd.read_csv(parent_dir + "/data/p_set.csv")
data.drop(data.columns[0], inplace=True, axis=1)
y = data['class'].values
X = data.drop('class',axis=1).values

# balance of data set
#balance = data['class'].value_counts(normalize=True)
#print(f"The balance of this data set is {balance[0]}.")

def ML_pipeline_kfold_GridSearchCV(X,y,random_state,n_folds, model, param_grid):
    # create a test set
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state,stratify=y)
    # splitter for _other
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    pipe = make_pipeline(model)
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                        cv=kf, return_train_score = True,iid=True)
    # do kfold CV on _other
    grid.fit(X_other, y_other)
    return grid, grid.score(X_test, y_test)

def five_fold_CV_rfc(iter=10):
    rfc = RandomForestClassifier(n_estimators = 10, random_state = 20, class_weight="balanced")
    rfc_param_grid = {"randomforestclassifier__max_depth": range(1,31), "randomforestclassifier__min_samples_split": range(2,21)}
    test_scores = []
    for i in range(10):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X = X, y = y, random_state = i*119, n_folds = 5, model = rfc, param_grid = rfc_param_grid)
        print(f'For iteration {i}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))

def five_fold_CV_logit(iter=10):
    logit = LogisticRegression(penalty = "l1", 
                               solver = "saga", 
                               max_iter = 8000, 
                               multi_class = "auto", 
                               random_state = 20)  # fixed random state for model
    logit_param_grid = {'logisticregression__C': np.logspace(-5,4,num=10)}
    test_scores = []
    for i in range(iter):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X = X, y = y, random_state = i*119, n_folds = 5, model = logit, param_grid = logit_param_grid)
        print(f'For iteration {i}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))
           
#random_mean, random_std = five_fold_CV_rfc()
logit_mean, logit_std = five_fold_CV_logit(2)