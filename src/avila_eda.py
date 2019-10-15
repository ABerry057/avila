
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from matplotlib import pylab as plt
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

#import raw and preprocessed training data with column names
#11 features including target variable
feature_names = ['ic-dist', 'u-margin', 'l-margin', 'exp', 'rows', 'mod-ratio', 'il-space', 'weight', 'peaks', 'mod_over_il-space', 'class']
dtypes = {'ic-dist': "float64", 
          'u-margin': "float64",
          'l-margin': "float64",
          'exp': "float64",
          'rows': "float64",
          'mod-ratio': "float64",
          'il-space': "float64",
          'weight': "float64",
          'peaks': "float64",
          'mod_over_il-space': "float64",
          'class' : "int64"} 
avila_p = pd.read_csv(parent_dir + "/data/avila_p_train.csv",
                      header = 0,
                      names=feature_names,
                      dtype=dtypes)

#balance of target variable classes
balance = avila_p['class'].value_counts(normalize=True)
print(balance)
#
#histograms for features
for feat in avila_p.columns:
    plt.hist(avila_p[feat],
             label = feat, 
             bins = 50)
    plt.xlabel(feat)
    plt.ylabel('Count')
    plt.title('Distribution of ' + feat)
    plt.tight_layout()
    #uncomment the following lines to save generated figures
#    plt.savefig(parent_dir + '/figures/' + feat + '_histogram' +'.png',
#                dpi=300)
    plt.show()

#f-regression and mutual information
X = avila_p.drop('class', axis=1)
y = avila_p['class']
f_test, p_values = f_regression(X, y)
print('f score',f_test)
print('p values',p_values)
mi = mutual_info_regression(X, y)
print('mi',mi)

#PCA
pca = PCA()
pca.fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Nr. components')
plt.ylabel('variance explained')
plt.show()