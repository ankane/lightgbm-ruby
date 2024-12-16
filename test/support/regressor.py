import lightgbm as lgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
y = df['y']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

print('predict', model.predict(X_test)[0:6].tolist())

print('feature_importances', model.feature_importances_.tolist())

print('early_stopping')
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation()])
