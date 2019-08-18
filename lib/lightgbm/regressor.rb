module LightGBM
  class Regressor < Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil)
      super
    end

    def fit(x, y)
      train_set = Dataset.new(x, label: y)
      @booster = LightGBM.train(@params, train_set, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      @booster.predict(data)
    end
  end
end
