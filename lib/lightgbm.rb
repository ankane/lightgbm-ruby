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

  class << self
    def train(params, train_set,num_boost_round: 100, valid_sets: [], valid_names: [], early_stopping_rounds: nil, verbose_eval: true)
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
            eval_valid.each_with_index do |(_, _, score, _), i|
              if best_score[i].nil? || score < best_score[i]
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
        best_score = nil
        best_iter = nil
      end

      num_boost_round.times do |iteration|
        boosters.each(&:update)

        scores = []
        eval_name = nil
        p boosters.map(&:eval_valid)
        boosters.map(&:eval_valid).each do |r|
          eval_name = r[0][1]
          scores << r[0][2]
        end
        mean = mean(scores)
        stdev = stdev(scores)

        (eval_hist["#{eval_name}-mean"] ||= []) << mean
        (eval_hist["#{eval_name}-stdv"] ||= []) << stdev

        if verbose_eval
          if show_stdv
            puts "[#{iteration + 1}]\tcv_agg's l2: %g + %g" % [mean, stdev]
          else
            puts "[#{iteration + 1}]\tcv_agg's l2: %g" % [mean]
          end
        end

        if early_stopping_rounds
          score = mean
          if best_score.nil? || score < best_score
            best_score = score
            best_iter = iteration
          elsif iteration - best_iter >= early_stopping_rounds
            break
          end
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
