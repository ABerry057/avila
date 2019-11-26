# append given train and test csvs
import pandas as pd
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) #parent directory path

av_train = pd.read_csv(parent_dir + "/data/avila_p_train.csv").drop(av_train.columns[0], axis=1)
av_test = pd.read_csv(parent_dir + "/data/avila_p_train.csv").drop(av_test.columns[0], axis=1)

whole_ds = av_train.append(av_test, ignore_index=True)