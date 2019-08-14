# LightGBM

LightGBM for Ruby

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
model = LightGBM::Booster.new(params: params, train_set: train_set)
```

Predict

```ruby
booster.predict(x_test)
```

Save [master]

```ruby
booster.save_model("model.txt")
```

Load a model

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
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
