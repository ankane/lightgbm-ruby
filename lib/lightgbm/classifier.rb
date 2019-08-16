module LightGBM
  class Classifier
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil)
      @params = {
        num_leaves: num_leaves,
        objective: objective,
        learning_rate: learning_rate
      }
      @n_estimators = n_estimators
    end

    def fit(x, y)
      n_classes = y.uniq.size

      params = @params.dup
      if n_classes > 2
        params[:objective] ||= "multiclass"
        params[:num_class] = n_classes
      else
        params[:objective] ||= "binary"
      end

      train_set = Dataset.new(x, label: y)
      @booster = LightGBM.train(params, train_set, num_boost_round: @n_estimators)
      nil
    end

    def predict(data)
      y_pred = @booster.predict(data)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred.map do |v|
          v.map.with_index.max_by { |v2, i| v2 }.last
        end
      else
        y_pred.map { |v| v > 0.5 ? 1 : 0 }
      end
    end

    def save_model(fname)
      @booster.save_model(fname)
    end

    def load_model(fname)
      @booster = Booster.new(params: @params, model_file: fname)
    end
  end
end
