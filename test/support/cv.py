import lightgbm as lgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
y = df['y']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

print('test_regression')

regression_params = {'objective': 'regression', 'verbosity': -1}
regression_train = lgb.Dataset(X_train, label=y_train)
eval_hist = lgb.cv(regression_params, regression_train, shuffle=False, stratified=False)
print(eval_hist['valid l2-mean'][0])
print(eval_hist['valid l2-mean'][-1])
print(eval_hist['valid l2-stdv'][0])
print(eval_hist['valid l2-stdv'][-1])

print()
print('test_binary')

binary_params = {'objective': 'binary', 'verbosity': -1}
binary_train = lgb.Dataset(X_train, label=y_train.replace(2, 1))
eval_hist = lgb.cv(binary_params, binary_train, shuffle=False, stratified=False)
print(eval_hist['valid binary_logloss-mean'][0])
print(eval_hist['valid binary_logloss-mean'][-1])
print(eval_hist['valid binary_logloss-stdv'][0])
print(eval_hist['valid binary_logloss-stdv'][-1])

print()
print('test_multiclass')

multiclass_params = {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1}
multiclass_train = lgb.Dataset(X_train, label=y_train)
eval_hist = lgb.cv(multiclass_params, multiclass_train, shuffle=False, stratified=False)
print(eval_hist['valid multi_logloss-mean'][0])
print(eval_hist['valid multi_logloss-mean'][-1])
print(eval_hist['valid multi_logloss-stdv'][0])
print(eval_hist['valid multi_logloss-stdv'][-1])

print('')
print('test_early_stopping_early')

regression_params = {'objective': 'regression', 'verbosity': 1, 'early_stopping_round': 5}
eval_hist = lgb.cv(regression_params, regression_train, shuffle=False, stratified=False)
print(len(eval_hist['valid l2-mean']))

print('')
print('test_early_stopping_not_early')

regression_params = {'objective': 'regression', 'verbosity': 1, 'early_stopping_round': 500}
eval_hist = lgb.cv(regression_params, regression_train, shuffle=False, stratified=False)
print(len(eval_hist['valid l2-mean']))


