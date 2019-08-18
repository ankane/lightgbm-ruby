module LightGBM
  class Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil)
      @params = {
        num_leaves: num_leaves,
        learning_rate: learning_rate
      }
      @params[:objective] = objective if objective
      @n_estimators = n_estimators
    end

    def save_model(fname)
      @booster.save_model(fname)
    end

    def load_model(fname)
      @booster = Booster.new(params: @params, model_file: fname)
    end

    def feature_importances
      @booster.feature_importance
    end
  end
end
