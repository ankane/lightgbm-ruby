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

  def self.train(params, train_set, num_boost_round: 100, valid_sets: [], valid_names: [], early_stopping_rounds: nil)
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

    if early_stopping_rounds
      best_score = []
      best_iter = []
      best_message = []

      puts "Training until validation scores don't improve for #{early_stopping_rounds.to_i} rounds."
    end

    (1..num_boost_round).each do |iteration|
      booster.update

      if valid_sets.any?
        # print results
        messages = []

        if valid_contain_train
          res = booster.eval_train
          messages << "%s: %g" % [res[0], res[2]]
        end

        eval_valid = booster.eval_valid
        eval_valid.each do |res|
          messages << "%s: %g" % [res[0], res[2]]
        end

        message = "[#{iteration}]\t#{messages.join("\t")}"

        puts message

        if early_stopping_rounds
          stop_early = false
          eval_valid.each_with_index do |(_, _, score, _), i|
            if best_score[i].nil? || score < best_score[i]
              best_score[i] = score
              best_iter[i] = iteration
              best_message[i] = message
            elsif iteration - best_iter[i] >= early_stopping_rounds
              booster.best_iteration = best_iter[i]
              puts "Early stopping, best iteration is:"
              puts best_message[i]
              stop_early = true
              break
            end
          end

          break if stop_early

          if iteration == num_boost_round
            booster.best_iteration = best_iter[0]
            puts "Did not meet early stopping. Best iteration is:"
            puts best_message[0]
          end
        end
      end
    end

    booster
  end
end
