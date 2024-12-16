import lightgbm as lgb
import pandas as pd

df = pd.read_csv('test/support/data.csv')
df['x3'] = ('cat' + df['x3'].astype(str)).astype('category')

X = df.drop(columns=['y'])
y = df['y']

X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

train_data = lgb.Dataset(X_train, label=y_train)
bst = lgb.train({}, train_data, num_boost_round=5)
bst.save_model('test/support/categorical.txt')

bst = lgb.Booster(model_file='test/support/categorical.txt')
print('x', X_train[:2].to_numpy().tolist())
print('predict', bst.predict(X_train)[:2].tolist())
