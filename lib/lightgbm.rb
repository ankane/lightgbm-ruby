# dependencies
require "ffi"

# modules
require "lightgbm/utils"
require "lightgbm/booster"
require "lightgbm/dataset"
require "lightgbm/version"

# scikit-learn API
require "lightgbm/classifier"
require "lightgbm/regressor"

module LightGBM
  class Error < StandardError; end

  class << self
    attr_accessor :ffi_lib
  end
  lib_name = "lib_lightgbm.#{::FFI::Platform::LIBSUFFIX}"
  self.ffi_lib = [lib_name, "lib_lightgbm.so"]

  # friendlier error message
  autoload :FFI, "lightgbm/ffi"

  class << self
    def train(params, train_set, num_boost_round: 100, valid_sets: [], valid_names: [], early_stopping_rounds: nil, verbose_eval: true)
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

        puts "Training until validation scores don't improve for #{early_stopping_rounds.to_i} rounds." if verbose_eval
      end

      num_boost_round.times do |iteration|
        booster.update

        if valid_sets.any?
          # print results
          messages = []

          if valid_contain_train
            # not sure why reversed in output
            booster.eval_train.reverse.each do |res|
              messages << "%s's %s: %g" % [res[0], res[1], res[2]]
            end
          end

          eval_valid = booster.eval_valid
          # not sure why reversed in output
          eval_valid.reverse.each do |res|
            messages << "%s's %s: %g" % [res[0], res[1], res[2]]
          end

          message = "[#{iteration + 1}]\t#{messages.join("\t")}"

          puts message if verbose_eval

          if early_stopping_rounds
            stop_early = false
            eval_valid.each_with_index do |(_, _, score, higher_better), i|
              op = higher_better ? :> : :<
              if best_score[i].nil? || score.send(op, best_score[i])
                best_score[i] = score
                best_iter[i] = iteration
                best_message[i] = message
              elsif iteration - best_iter[i] >= early_stopping_rounds
                booster.best_iteration = best_iter[i] + 1
                puts "Early stopping, best iteration is:\n#{best_message[i]}" if verbose_eval
                stop_early = true
                break
              end
            end

            break if stop_early

            if iteration == num_boost_round - 1
              booster.best_iteration = best_iter[0] + 1
              puts "Did not meet early stopping. Best iteration is: #{best_message[0]}" if verbose_eval
            end
          end
        end
      end

      booster
    end

    def cv(params, train_set, num_boost_round: 100, nfold: 5, seed: 0, shuffle: true, early_stopping_rounds: nil, verbose_eval: nil, show_stdv: true)
      rand_idx = (0...train_set.num_data).to_a
      rand_idx.shuffle!(random: Random.new(seed)) if shuffle

      kstep = rand_idx.size / nfold
      test_id = rand_idx.each_slice(kstep).to_a[0...nfold]
      train_id = []
      nfold.times do |i|
        idx = test_id.dup
        idx.delete_at(i)
        train_id << idx.flatten
      end

      boosters = []
      folds = train_id.zip(test_id)
      folds.each do |(train_idx, test_idx)|
        fold_train_set = train_set.subset(train_idx)
        fold_valid_set = train_set.subset(test_idx)
        booster = Booster.new(params: params, train_set: fold_train_set)
        booster.add_valid(fold_valid_set, "valid")
        boosters << booster
      end

      eval_hist = {}

      if early_stopping_rounds
        best_score = {}
        best_iter = {}
      end

      num_boost_round.times do |iteration|
        boosters.each(&:update)

        scores = {}
        boosters.map(&:eval_valid).map(&:reverse).flatten(1).each do |r|
          (scores[r[1]] ||= []) << r[2]
        end

        message_parts = ["[#{iteration + 1}]"]

        means = {}
        scores.each do |eval_name, vals|
          mean = mean(vals)
          stdev = stdev(vals)

          (eval_hist["#{eval_name}-mean"] ||= []) << mean
          (eval_hist["#{eval_name}-stdv"] ||= []) << stdev

          means[eval_name] = mean

          if show_stdv
            message_parts << "cv_agg's %s: %g + %g" % [eval_name, mean, stdev]
          else
            message_parts << "cv_agg's %s: %g" % [eval_name, mean]
          end
        end

        puts message_parts.join("\t") if verbose_eval

        if early_stopping_rounds
          stop_early = false
          means.each do |k, score|
            # TODO fix higher better
            if best_score[k].nil? || score < best_score[k]
              best_score[k] = score
              best_iter[k] = iteration
            elsif iteration - best_iter[k] >= early_stopping_rounds
              stop_early = true
              break
            end
          end
          break if stop_early
        end
      end

      eval_hist
    end

    private

    def mean(arr)
      arr.sum / arr.size.to_f
    end

    # don't subtract one from arr.size
    def stdev(arr)
      m = mean(arr)
      sum = 0
      arr.each do |v|
        sum += (v - m) ** 2
      end
      Math.sqrt(sum / arr.size)
    end
  end
end
