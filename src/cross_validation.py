"""
Cross-validation
"""
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

train_data = pd.read_csv(parent_dir + "/data/avila_p_train.csv")
train_data.drop(train_data.columns[0], inplace=True, axis=1)
test_data = pd.read_csv(parent_dir + "/data/avila_p_test.csv")
test_data.drop(test_data.columns[0], inplace=True, axis=1)

y_train = train_data['class'].values
X_train = train_data.drop('class',axis=1).values
y_test = test_data['class'].values
X_test = test_data.drop('class',axis=1).values

# balance of data set
#balance = data['class'].value_counts(normalize=True)
#print(f"The balance of this data set is {balance[0]}.")

def ML_pipeline_kfold_GridSearchCV(X_train,y_train,X_test,y_test,random_state,n_folds, model, param_grid):
    # create a test set
#    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state,stratify=y)
    # splitter for _other
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=random_state)
    scaler = StandardScaler()
    pipe = make_pipeline(scaler, model)
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                        cv=kf, return_train_score = True,iid=True, verbose=10)
    # do kfold CV on _other
    grid.fit(X_train, y_train)
    return grid, grid.score(X_test, y_test)


def n_fold_CV_rfc(n_iter=10, n_folds=5):
    rfc = RandomForestClassifier(n_estimators = 10, random_state = 20, class_weight="balanced", n_jobs=-1)
    rfc_param_grid = {"randomforestclassifier__max_depth": range(18,23), "randomforestclassifier__min_samples_split": range(2,16)}
    test_scores = []
    for i in range(n_iter):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X_train = X_train,
                                                          y_train = y_train,
                                                          X_test = X_test,
                                                          y_test = y_test, random_state = i*119, n_folds = n_folds, model = rfc, param_grid = rfc_param_grid)
        print(f'For iteration {i+1}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))


def n_fold_CV_gbc(n_iter=10, n_folds=5):
    gbc = GradientBoostingClassifier(loss="deviance", random_state=20)
    gbc_param_grid = {"gradientboostingclassifier__min_samples_split": range(2,16),
                      "gradientboostingclassifier__max_depth": range(1,21)}
    test_scores = []
    for i in range(n_iter):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X_train = X_train,
                                                          y_train = y_train,
                                                          X_test = X_test,
                                                          y_test = y_test, random_state = i*119, n_folds = n_folds, model = gbc, param_grid = gbc_param_grid)
        print(f'For iteration {i+1}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))


def n_fold_CV_logit(n_iter=10, n_folds=5):
    logit = LogisticRegression(penalty = "l1", 
                               solver = "saga", 
                               max_iter = 8000, 
                               multi_class = "auto",
                               n_jobs = -1,
                               random_state = 20)  # fixed random state for model
    logit_param_grid = {'logisticregression__C': np.logspace(-5,4,num=10)}
    test_scores = []
    for i in range(n_iter):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X_train = X_train,
                                                          y_train = y_train,
                                                          X_test = X_test,
                                                          y_test = y_test, random_state = i*119, n_folds = n_folds, model = logit, param_grid = logit_param_grid)
        print(f'For iteration {i+1}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))


def n_fold_CV_kns(n_iter=10, n_folds=5):
    """
    Notes: 1 iteration over the whole data set takes about 170 seconds with one core, about 60 seconds with all cores.
    Accuracy: 0.951 +/- 0.002
    For weights, distance seems to be the better choice in grid search, so I made it constant.
    """
    kns = KNeighborsClassifier(algorithm = "auto",
                               weights = "distance",
                               n_jobs = -1)
    kns_param_grid = {"kneighborsclassifier__n_neighbors": range(2,16)}
    test_scores = []
    for i in range(n_iter):
        print(f"Iteration {i+1}")
        grid, test_score = ML_pipeline_kfold_GridSearchCV(X_train = X_train,
                                                          y_train = y_train,
                                                          X_test = X_test,
                                                          y_test = y_test, random_state = i*119, n_folds = n_folds, model = kns, param_grid = kns_param_grid)
        print(f'For iteration {i+1}, the best hyperparameters are {grid.best_params_}')
        test_scores.append(test_score)
    print(f'Mean score: {np.around(np.mean(test_scores),3)} +/- {np.around(np.std(test_scores),3)}')
    return (np.mean(test_scores), np.std(test_scores))

#uncomment  below to run CV
#random_mean, random_std = n_fold_CV_rfc(n_iter=10, n_folds=5)
#logit_mean, logit_std = n_fold_CV_logit(n_iter=10, n_folds=5)
#kns_mean, knss_std = n_fold_CV_kns(n_iter=10, n_folds=5)
#gbc_mean, gbc_std = n_fold_CV_gbc(n_iter=1, n_folds=5)

#best model
rfc = RandomForestClassifier(n_estimators = 10,
                             random_state = 20,
                             class_weight="balanced",
                             max_depth = 21,
                             min_samples_split = 7,
                             n_jobs=-1)
rfc.fit(X_train, y_train)
#save best model
import pickle
file = open(parent_dir + '/results/grid.save', 'wb')
pickle.dump((rfc, X_test,y_test),file)
file.close()