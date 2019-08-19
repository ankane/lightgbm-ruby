import lightgbm as lgb
import pandas as pd

df = pd.read_csv('test/support/iris.csv')

X = df.drop(columns=['Species'])
y = df['Species']
# y = y.replace(2, 1)

X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]

group = [20, 80]

model = lgb.LGBMRanker()
model.fit(X_train, y_train, group=group)
# print(model.predict(X_test))
# print(model.predict_proba(X_test))
print(model.feature_importances_)
