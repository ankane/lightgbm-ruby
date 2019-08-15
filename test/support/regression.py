import pandas as pd
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb

df = pd.read_csv('boston.csv')

X = df.drop(columns=['medv'])
y = df['medv']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
dataset = lgb.Dataset(X, label=y)

param = {}
param['verbosity'] = -1

bst = lgb.train(param, train_data, valid_sets=[train_data, test_data])

eval_dict = lgb.cv(param, dataset, shuffle=False, stratified=False)
# print(eval_dict)
