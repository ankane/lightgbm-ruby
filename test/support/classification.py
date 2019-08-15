import pandas as pd
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb

df = pd.read_csv('test/support/iris.csv')

X = df.drop(columns=['Species'])
y = df['Species']

X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
dataset = lgb.Dataset(X, label=y)

param = {'objective': 'binary'}
# param = {'objective', 'multiclass', 'num_class': 3}
param['verbosity'] = -1

bst = lgb.train(param, train_data, valid_sets=[train_data, test_data])

eval_dict = lgb.cv(param, dataset, shuffle=False, stratified=False)
# print(eval_dict)
