# Run this script to regenerate the test/support/model.txt and test/support/model_categorical.txt files

import lightgbm as lgb
import pandas as pd

params = {'verbosity': -1}

def booster():
    df = pd.read_csv('test/support/data.csv')

    X = df.drop(columns=['y'])
    y = df['y']

    X_train = X[:300]
    y_train = y[:300]
    X_test = X[300:]
    y_test = y[300:]

    train_data = lgb.Dataset(X_train, label=y_train)
    bst = lgb.train(params, train_data)
    bst.save_model('test/support/model.txt')

    bst = lgb.Booster(model_file='test/support/model.txt')
    print('x', X_train[:2].to_numpy().tolist())
    print('predict', bst.predict(X_train)[:2].tolist())
    print('feature_importance', bst.feature_importance().tolist())
    print('feature_name', bst.feature_name())

def booster_categorical():
    df = pd.read_csv('test/support/data.csv', dtype={'x3': 'category'})

    X = df.drop(columns=['y'])
    y = df['y']

    X_train = X[:300]
    y_train = y[:300]
    X_test = X[300:]
    y_test = y[300:]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature='auto')
    bst = lgb.train(params, train_data)
    bst.save_model('test/support/model_categorical.txt')

    bst = lgb.Booster(model_file='test/support/model_categorical.txt')
    print('x', X_train[:2].to_numpy().tolist())
    print('predict', bst.predict(X_train)[:2].tolist())
    print('feature_importance', bst.feature_importance().tolist())
    print('feature_name', bst.feature_name())


print('booster -> model.txt')
booster()

print('')

print('categorical')
booster_categorical()
