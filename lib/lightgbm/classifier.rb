module LightGBM
  class Classifier < Model
    def initialize(num_leaves: 31, learning_rate: 0.1, n_estimators: 100, objective: nil, **options)
      super
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

    def predict(data, num_iteration: nil)
      y_pred = @booster.predict(data, num_iteration: num_iteration)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred.map do |v|
          v.map.with_index.max_by { |v2, i| v2 }.last
        end
      else
        y_pred.map { |v| v > 0.5 ? 1 : 0 }
      end
    end

    def predict_proba(data, num_iteration: nil)
      y_pred = @booster.predict(data, num_iteration: num_iteration)

      if y_pred.first.is_a?(Array)
        # multiple classes
        y_pred
      else
        y_pred.map { |v| [1 - v, v] }
      end
    end
  end
end
