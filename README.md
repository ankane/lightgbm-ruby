# LightGBM

[LightGBM](https://github.com/microsoft/LightGBM) for Ruby

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

Save the model

```ruby
booster.save_model("model.txt")
```

Load a model from a file

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
```

Get feature importance

```ruby
booster.feature_importance
```

## Reference

This library follows the [Data Structure and Training APIs](https://lightgbm.readthedocs.io/en/latest/Python-API.html) for the Python library. However, some features may be missing at the moment. PRs welcome!

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
