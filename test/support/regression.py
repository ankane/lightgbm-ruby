import pandas as pd
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb

df = pd.read_csv('test/data/boston/boston.csv')

X = df.drop(columns=['medv'])
y = df['medv']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

categorical_feature = 'auto' # [5]
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature)
test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_feature)
dataset = lgb.Dataset(X, label=y, categorical_feature=categorical_feature)

param = {}
param['verbosity'] = -1
param['metric'] = ['l1', 'l2', 'rmse']

# bst = lgb.train(param, train_data, valid_sets=[train_data, test_data])
# print(bst.predict(X_test)[:1])

eval_dict = lgb.cv(param, dataset, shuffle=False, stratified=False, verbose_eval=True, early_stopping_rounds=5)
# print(eval_dict)
