require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_regression
    x_test = boston_test.data
    y_test = boston_test.label

    model = LightGBM.train(regression_params, boston_train, valid_sets: [boston_train, boston_test], verbose_eval: false)
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  def test_binary
    model = LightGBM.train(binary_params, iris_train, valid_sets: [iris_train, iris_test], verbose_eval: false)
    y_pred = model.predict(iris_test.data)
    assert_in_delta 0.99998366, y_pred[0]

    y_pred = model.predict(iris_test.data)
    assert_equal 50, y_pred.size

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(iris_test.data)
    assert_equal y_pred, y_pred2
  end

  def test_multiclass
    model = LightGBM.train(multiclass_params, iris_train, valid_sets: [iris_train, iris_test], verbose_eval: false)
    y_pred = model.predict([6.3, 2.7, 4.9, 1.8])
    expected = [3.91608299e-04, 3.81933551e-01, 6.17674841e-01]
    assert_elements_in_delta expected, y_pred

    y_pred = model.predict(iris_test.data)
    # ensure reshaped
    assert_equal 50, y_pred.size
    assert_equal 3, y_pred.first.size

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(iris_test.data)
    assert_equal y_pred, y_pred2
  end

  def test_early_stopping_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(regression_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 5)
    end
    assert_equal 55, model.best_iteration
    assert_includes stdout, "Early stopping, best iteration is:\n[55]\ttraining's l2: 2.18872\tvalid_1's l2: 35.6151"
  end

  def test_early_stopping_not_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(regression_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 500)
    end
    assert_equal 71, model.best_iteration
    assert_includes stdout, "Best iteration is: [71]\ttraining's l2: 1.69138\tvalid_1's l2: 35.2563"
  end

  def test_early_stopping_early_higher_better
    model = LightGBM.train(binary_params.merge(metric: 'auc'), iris_train, valid_sets: [iris_train, iris_test], early_stopping_rounds: 5, verbose_eval: false)
    assert_equal 6, model.best_iteration
  end

  def test_verbose_eval_false
    stdout, _ = capture_io do
      LightGBM.train(regression_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 5, verbose_eval: false)
    end
    assert_empty stdout
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, boston_train)
    end
  end

  def test_categorical_feature
    train_set = LightGBM::Dataset.new(boston_train.data, label: boston_train.label, categorical_feature: [5])
    model = LightGBM.train(regression_params, train_set)
    assert_in_delta 22.33155937, model.predict(boston_test.data[0])
  end

  def test_multiple_metrics
    params = regression_params.dup
    params[:metric] = ["l1", "l2", "rmse"]
    LightGBM.train(params, boston_train, valid_sets: [boston_train, boston_test], verbose_eval: false, early_stopping_rounds: 5)
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
