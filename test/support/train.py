import lightgbm as lgb
import pandas as pd
import numpy as np

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
regression_test = lgb.Dataset(X_test, label=y_test)
bst = lgb.train(regression_params, regression_train, valid_sets=[regression_train, regression_test])
y_pred = bst.predict(X_test)
print(np.sqrt(np.mean((y_pred - y_test)**2)))

print('')
print('test_binary')

binary_params = {'objective': 'binary', 'verbosity': -1}
binary_train = lgb.Dataset(X_train, label=y_train.replace(2, 1))
binary_test = lgb.Dataset(X_test, label=y_test.replace(2, 1))
bst = lgb.train(binary_params, binary_train, valid_sets=[binary_train, binary_test])
y_pred = bst.predict(X_test)
print(y_pred[0])

print('')
print('test_multiclass')

multiclass_params = {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1}
multiclass_train = lgb.Dataset(X_train, label=y_train)
multiclass_test = lgb.Dataset(X_test, label=y_test)
bst = lgb.train(multiclass_params, multiclass_train, valid_sets=[multiclass_train, multiclass_test])
y_pred = bst.predict(X_test)
print(y_pred[0].tolist())

print('')
print('test_early_stopping_early')

regression_params['early_stopping_round'] = 5
bst = lgb.train(regression_params, regression_train, valid_sets=[regression_train, regression_test])
print(bst.best_iteration)

print('')
print('test_early_stopping_not_early')

regression_params['early_stopping_round'] = 500
bst = lgb.train(regression_params, regression_train, valid_sets=[regression_train, regression_test])
# appears to be using training set for best iteration instead of validation set
print(bst.best_iteration)

print('')
print('test_early_stopping_early_higher_better')

binary_params['early_stopping_round'] = 5
binary_params['metric'] = 'auc'
bst = lgb.train(binary_params, binary_train, valid_sets=[binary_train, binary_test])
print(bst.best_iteration)

print('')
print('test_categorical_feature')

regression_params = {'objective': 'regression', 'verbosity': -1}
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=[3])
bst = lgb.train(regression_params, train_set)
print(bst.predict(X_test)[0])
