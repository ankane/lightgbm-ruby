module LightGBM
  class Ranker < Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: "lambdarank", **options)
      super
    end

    def fit(x, y, group:)
      train_set = Dataset.new(x, label: y, group: group)
      @booster = LightGBM.train(@params, train_set, num_boost_round: @n_estimators)
      nil
    end

    def predict(data, num_iteration: nil)
      @booster.predict(data, num_iteration: num_iteration)
    end
  end
end
