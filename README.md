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

Train a model [master]

```ruby
params = {objective: "regression"}
train_set = LightGBM::Dataset.new(x_train, label: y_train)
booster = LightGBM.train(params, train_set)
```

Predict

```ruby
booster.predict(x_test)
```

Save [master]

```ruby
booster.save_model("model.txt")
```

Load a model from a file

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
```

Get feature importance [master]

```ruby
booster.feature_importance
```

## Reference [master]

### Booster

```ruby
booster = LightGBM::Booster.new(model_str: "tree...")
booster.to_json
booster.model_to_string
booster.current_iteration
```

### Dataset

```ruby
dataset = LightGBM::Dataset.new(data, label: label, weight: weight, params: params)
dataset.num_data
dataset.num_feature
dataset.save_binary("train.bin")
dataset.dump_text("train.txt")
```

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
