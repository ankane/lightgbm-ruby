# LightGBM

[LightGBM](https://github.com/microsoft/LightGBM) - the high performance machine learning library - for Ruby

:fire: Uses the C API for blazing performance

[![Build Status](https://travis-ci.org/ankane/lightgbm.svg?branch=master)](https://travis-ci.org/ankane/lightgbm)

## Installation

First, [install LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html). On Mac, copy `lib_lightgbm.so` to `/usr/local/lib`.

Add this line to your application’s Gemfile:

```ruby
gem 'lightgbm'
```

## Getting Started

Train a model

```ruby
params = {objective: "regression"}
train_set = LightGBM::Dataset.new(x_train, label: y_train)
booster = LightGBM.train(params, train_set)
```

Predict

```ruby
booster.predict(x_test)
```

Save the model to a file

```ruby
booster.save_model("model.txt")
```

Load the model from a file

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
```

Get feature importance

```ruby
booster.feature_importance
```

## Early Stopping

```ruby
LightGBM.train(params, train_set, valid_set: [train_set, test_set], early_stopping_rounds: 5)
```

## CV

```ruby
LightGBM.cv(params, train_set, nfold: 5, verbose_eval: true)
```

## Scikit-Learn API [master]

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
model = LightGBM::Regressor.new
model.fit(x, y)
```

> For classification, use `LightGBM::Classifier`

Predict

```ruby
model.predict(x)
```

Save the model to a file

```ruby
model.save_model("model.txt")
```

Load the model from a file

```ruby
model.load_model("model.txt")
```

## Reference

This library follows the [Data Structure, Training, and Scikit-Learn APIs](https://lightgbm.readthedocs.io/en/latest/Python-API.html) of the Python library. A few differences are:

- The `_get` prefix is removed from methods
- The default verbosity is `-1`
- With the `cv` method, `stratified` is set to `false`

Some methods and options are also missing at the moment. PRs welcome!

## Helpful Resources

- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

## Related Projects

- [Xgb](https://github.com/ankane/xgb) - XGBoost for Ruby
- [Eps](https://github.com/ankane/eps) - Machine Learning for Ruby

## Credits

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for serving as an initial reference, and Selva Prabhakaran for the [test datasets](https://github.com/selva86/datasets).

## History

View the [changelog](https://github.com/ankane/lightgbm/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/lightgbm/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/lightgbm/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
