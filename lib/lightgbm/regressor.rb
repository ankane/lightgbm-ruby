module LightGBM
  class Regressor < Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil, **options)
      super
    end

    def fit(x, y, categorical_feature: "auto", early_stopping_rounds: nil, verbose: true)
      train_set = Dataset.new(x, label: y, categorical_feature: categorical_feature)
      @booster = LightGBM.train(@params, train_set,
        num_boost_round: @n_estimators,
        early_stopping_rounds: early_stopping_rounds,
        verbose_eval: verbose
      )
      nil
    end

    def predict(data, num_iteration: nil)
      @booster.predict(data, num_iteration: num_iteration)
    end
  end
end
