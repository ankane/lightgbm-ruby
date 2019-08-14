# LightGBM

LightGBM for Ruby

## Installation

First, [install LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html).

Add this line to your application’s Gemfile:

```ruby
gem 'lightgbm'
```

Load a model

```ruby
booster = LightGBM::Booster.new(model_file: "model.txt")
```

Predict

```ruby
booster.predict([[1, 2], [3, 4]])
```

## Credits

Thanks to the [xgboost](https://github.com/PairOnAir/xgboost-ruby) gem for serving as an initial reference.

## History

View the [changelog](https://github.com/ankane/lightgbm/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/lightgbm/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/lightgbm/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
