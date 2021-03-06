
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
#remove broken data point
avila_p.drop(avila_p.index[6619], inplace=True)

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

#histogram for just class balance
plt.hist(avila_p['class'],
         bins = 12,
         density = True,
         color = "#BD2677")
plt.xlabel("Target Variable Class (Copyist Identity)")
plt.ylabel('Percentage of Data Points')
plt.title('Class Balance')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],
           ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10','Class 11'],
           rotation = 30)
plt.tight_layout()
#uncomment the following lines to save generated figures
#plt.savefig(parent_dir + '/figures/classBalance_histogram.png',
#            dpi=300)
plt.show()

#correlation matrix
avila_p_c = avila_p.drop(['class'], axis=1) #remove class
plt.figure(figsize=(10,10))
plt.matshow(avila_p_c.corr(),vmin=-1,vmax=1,cmap='seismic',fignum=0)
plt.colorbar(label='corr. coeff.')
plt.xticks(np.arange(avila_p_c.corr().shape[0]),list(avila_p_c.corr().columns),rotation=90)
plt.yticks(np.arange(avila_p_c.corr().shape[0]),list(avila_p_c.corr().columns))
plt.title("Feature Correlation Matrix", pad = 20)
plt.tight_layout()
#uncomment to save
#plt.savefig(parent_dir + '/figures/corr_coeff.png',dpi=300)
plt.show()

#boxplot for each feature with class
for feat in avila_p_c.columns:
    avila_p[[feat,'class']].boxplot(by='class')
    plt.xlabel('Class')
    plt.ylabel('Normalized Value')
    plt.suptitle('')
    plt.title(f'Distribution of {feat} by class')
    plt.grid(b=False)
    plt.savefig(f'{parent_dir}/figures/{feat}_boxplot.png',dpi=300)
    plt.show()

#scatter matrix
corrmat = avila_p.corr()

all_cols = corrmat.sort_values('class',ascending=False)['class'].index 
cols = all_cols[:10] # positively correlated features
#cols = ['class']+[col for col in all_cols if '_nan' not in col][:-10:-1] # negatively correlated features

pd.plotting.scatter_matrix(avila_p[cols],c = avila_p['class'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8)
#uncomment to save
#plt.savefig(parent_dir + '/figures/scatter_matrix_avila.png',dpi=300)
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
plt.title("Variance Explained by PCA Component")
#uncomment to save
#plt.savefig(parent_dir + '/figures/' + 'pca_explainedVariance' +'.png',
#                dpi=300)
plt.show()