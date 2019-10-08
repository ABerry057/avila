import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

#import raw and preprocessed training data with column names
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


#histograms for features
for feat in avila_p.columns:
    plt.hist(avila_p[feat],
             label = feat, 
             bins = 50)
    plt.xlabel(feat)
    plt.ylabel('Count')
    plt.title('Distribution of ' + feat)
    plt.tight_layout()
    plt.savefig(parent_dir + '/figures/' + feat + '_histogram' +'.png',
                dpi=300)
    plt.show()
    
#correlation matrix
corrmat = avila_p.corr()

all_cols = corrmat.sort_values('class',ascending=False)['class'].index 
cols = all_cols[:10] # positively correlated features

pd.plotting.scatter_matrix(avila_p[cols],
                           c = avila_p['class'],
                           figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20},
                           s=60,
                           alpha=.8)
plt.savefig(parent_dir + '/figures/scattermatrix.png',
            dpi=300)
plt.show()