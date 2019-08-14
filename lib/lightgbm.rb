# dependencies
require "ffi"

# modules
require "lightgbm/utils"
require "lightgbm/booster"
require "lightgbm/dataset"
require "lightgbm/ffi"
require "lightgbm/version"

module LightGBM
  class Error < StandardError; end

  def self.train(params, train_set, num_boost_round: 100, valid_sets: [], valid_names: [])
    booster = Booster.new(params: params, train_set: train_set)

    valid_contain_train = false
    valid_sets.zip(valid_names).each_with_index do |(data, name), i|
      if data == train_set
        booster.train_data_name = name || "training"
        valid_contain_train = true
      else
        booster.add_valid(data, name || "valid_#{i}")
      end
    end

    booster.best_iteration = 0

    num_boost_round.times do |i|
      booster.update

      if valid_sets.any?
        # print results
        messages = []

        if valid_contain_train
          res = booster.eval_train
          messages << "%s: %g" % [res[0], res[2]]
        end

        booster.eval_valid.each do |res|
          messages << "%s: %g" % [res[0], res[2]]
        end

        puts "[#{i}]\t#{messages.join("\t")}"
      end
    end

    booster
  end
end
