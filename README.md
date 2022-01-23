# LightGBM Ruby

[LightGBM](https://github.com/microsoft/LightGBM) - high performance gradient boosting - for Ruby

[![Build Status](https://github.com/ankane/lightgbm-ruby/workflows/build/badge.svg?branch=master)](https://github.com/ankane/lightgbm-ruby/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "lightgbm"
```

On Mac, also install OpenMP:

```sh
brew install libomp
```

## Training API

Prep your data

```ruby
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]
```

Train a model

```ruby
params = {objective: "regression"}
train_set = LightGBM::Dataset.new(x, label: y)
booster = LightGBM.train(params, train_set)
```

Predict

```ruby
booster.predict(x)
```

Save the model to a file

```ruby
booster.save_model("model.txt")
```

Load the model from a file

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
```

Get the importance of features

```ruby
booster.feature_importance
```

Early stopping

```ruby
LightGBM.train(params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 5)
```

CV

```ruby
LightGBM.cv(params, train_set, nfold: 5, verbose_eval: true)
```

## Scikit-Learn API

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

> For classification, use `predict_proba` for probabilities

Save the model to a file

```ruby
model.save_model("model.txt")
```

Load the model from a file

```ruby
model.load_model("model.txt")
```

Get the importance of features

```ruby
model.feature_importances
```

Early stopping

```ruby
model.fit(x, y, eval_set: [[x_test, y_test]], early_stopping_rounds: 5)
```

## Data

Data can be an array of arrays

```ruby
[[1, 2, 3], [4, 5, 6]]
```

Or a Numo array

```ruby
Numo::NArray.cast([[1, 2, 3], [4, 5, 6]])
```

Or a Rover data frame

```ruby
Rover.read_csv("houses.csv")
```

Or a Daru data frame

```ruby
Daru::DataFrame.from_csv("houses.csv")
```

## Helpful Resources

- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

## Related Projects

- [XGBoost](https://github.com/ankane/xgboost) - XGBoost for Ruby
- [Eps](https://github.com/ankane/eps) - Machine learning for Ruby

## Credits

This library follows the [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html). A few differences are:

- The `get_` and `set_` prefixes are removed from methods
- The default verbosity is `-1`
- With the `cv` method, `stratified` is set to `false`

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for showing how to use FFI.

## History

View the [changelog](https://github.com/ankane/lightgbm-ruby/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/lightgbm-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/lightgbm-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/lightgbm-ruby.git
cd lightgbm-ruby
bundle install
bundle exec rake vendor:all
bundle exec rake test
```
