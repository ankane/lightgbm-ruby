module LightGBM
  class Regressor
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil)
      @params = {
        num_leaves: num_leaves,
        learning_rate: learning_rate
      }
      @params[:objective] = objective if objective
      @n_estimators = n_estimators
    end

    def fit(x, y)
      train_set = Dataset.new(x, label: y)
      @booster = LightGBM.train(@params, train_set, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      @booster.predict(data)
    end

    def save_model(fname)
      @booster.save_model(fname)
    end

    def load_model(fname)
      @booster = Booster.new(params: @params, model_file: fname)
    end
  end
end
