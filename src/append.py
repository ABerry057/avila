# append given train and test csvs
import pandas as pd
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

av_train = pd.read_csv(parent_dir + "/data/avila_p_train.csv")
av_train.drop(av_train.columns[0], axis=1, inplace=True)
av_test = pd.read_csv(parent_dir + "/data/avila_p_train.csv")
av_test.drop(av_test.columns[0], axis=1, inplace=True)

whole_ds = av_train.append(av_test, ignore_index=True)
whole_ds.to_csv(parent_dir + '/data/p_set.csv')