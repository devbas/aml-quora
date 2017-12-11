import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


train_duplicate = pd.read_csv("data/train_labels.csv", delimiter=',')
test_distances = np.load('train_distances.npy')
train_distances = np.load('train_distances.npy')

X_train, X_test, y_train, y_test = train_test_split(train_distances, train_duplicate['is_duplicate'].values, test_size=0.2, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.4
params['max_depth'] = 6

bst = xgb.train(params, dtrain, 2000, watchlist, early_stopping_rounds=50, verbose_eval=10)
bst.save_model('0001.model')

d_test = xgb.DMatrix(test_distances)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_df['test_id']
sub['is_duplicate'] = np.round(p_test).astype(int)
sub.to_csv('simple_xgb.csv', index=False)