# dependencies
require "ffi"

# modules
require "lightgbm/booster"
require "lightgbm/dataset"
require "lightgbm/ffi"
require "lightgbm/version"

module LightGBM
  def self.train(params, train_set, num_boost_round: 100)
    booster = Booster.new(params: params, train_set: train_set)
    num_boost_round.times do
      booster.update
    end
    booster
  end
end
