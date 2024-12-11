import lightgbm as lgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')

X = df.drop(columns=['y'])
yb = df['y'].replace(2, 1)
ym = df['y']

X_train = X[:300]
yb_train = yb[:300]
ym_train = ym[:300]
X_test = X[300:]
yb_test = yb[300:]
ym_test = ym[300:]

print('test_binary')

model = lgb.LGBMClassifier()
model.fit(X_train, yb_train)
print(model.predict(X_test)[0:100].tolist())
print(model.predict_proba(X_test)[0].tolist())
print(model.feature_importances_.tolist())

print()
print('test_multiclass')

model = lgb.LGBMClassifier(verbosity=-1)
model.fit(X_train, ym_train)
print(model.predict(X_test)[0:100].tolist())
print(model.predict_proba(X_test)[0].tolist())
print(model.feature_importances_.tolist())

print()
print('test_early_stopping')
model = lgb.LGBMClassifier(early_stopping_round=5, verbosity=1)
model.fit(X_train, ym_train, eval_set=[(X_test, ym_test)])

print()
print('test_missing_numeric')

X_train_miss = X_train.copy()
X_test_miss = X_test.copy()
X_train_miss[X_train_miss == 3.7] = None
X_test_miss[X_test_miss == 3.7] = None
model = lgb.LGBMClassifier()
model.fit(X_train_miss, ym_train)
print(model.predict(X_test_miss)[0:100].tolist())
print(model.feature_importances_.tolist())

print()
print('test_missing_categorical')

X_train_miss2 = X_train.copy()
X_test_miss2 = X_test.copy()
X_train_miss2["x3"][X_train_miss2["x3"] > 7] = None
X_test_miss2["x3"][X_test_miss2["x3"] > 7] = None
model = lgb.LGBMClassifier()
model.fit(X_train_miss2, ym_train, categorical_feature=[3])
print(model.predict(X_test_miss2)[0:100].tolist())
print(model.feature_importances_.tolist())
