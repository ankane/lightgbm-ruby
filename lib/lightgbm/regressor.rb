module LightGBM
  class Regressor < Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: "regression", **options)
      super
    end

    def fit(x, y, categorical_feature: "auto", eval_set: nil, eval_names: [], early_stopping_rounds: nil, verbose: true)
      train_set = Dataset.new(x, label: y, categorical_feature: categorical_feature)
      valid_sets = Array(eval_set).map { |v| Dataset.new(v[0], label: v[1], reference: train_set) }

      @booster = LightGBM.train(@params, train_set,
        num_boost_round: @n_estimators,
        early_stopping_rounds: early_stopping_rounds,
        verbose_eval: verbose,
        valid_sets: valid_sets,
        valid_names: eval_names
      )
      nil
    end

    def predict(data, num_iteration: nil)
      @booster.predict(data, num_iteration: num_iteration)
    end
  end
end
