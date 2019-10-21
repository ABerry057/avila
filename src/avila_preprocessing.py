# Avila dataset Preproccessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# add column names for convenience
feature_names = ['ic-dist', 'u-margin', 'l-margin', 'exp', 'rows', 'mod-ratio', 'il-space', 'weight', 'peaks', 'mod_over_il-space', 'class']
av_train = pd.read_csv('avila-tr.csv', names=feature_names)
av_test = pd.read_csv('avila-ts.csv', names=feature_names)

# define target variables and drop from X data frames
y_train = av_train['class']
y_test = av_test['class']
av_train = av_train.drop(['class'], axis=1)
av_test = av_test.drop(['class'], axis=1)

# fit label encoder to target classes
le = LabelEncoder()
le.fit_transform(y_train)

# redefine target test and train with new labels
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# add target variable as last column
av_train['class'] = y_train
av_test['class'] = y_test
# save preprocessed variables as csv files
#av_train.to_csv('avila_p_train.csv')
#av_test.to_csv('avila_p_test.csv')