module LightGBM
  class Model
    attr_reader :booster

    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil, **options)
      @params = {
        num_leaves: num_leaves,
        learning_rate: learning_rate
      }.merge(options)
      @params[:objective] = objective if objective
      @n_estimators = n_estimators
    end

    def save_model(fname)
      @booster.save_model(fname)
    end

    def load_model(fname)
      @booster = Booster.new(params: @params, model_file: fname)
    end

    def best_iteration
      @booster.best_iteration
    end

    def feature_importances
      @booster.feature_importance
    end
  end
end
