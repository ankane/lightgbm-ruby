require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_regression
    model = LightGBM.train(regression_params, regression_train, valid_sets: [regression_train, regression_test], verbose_eval: false)
    y_pred = model.predict(regression_test.data)
    assert_operator rsme(regression_test.label, y_pred), :<=, 0.3

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred = model.predict(regression_test.data)
    assert_operator rsme(regression_test.label, y_pred), :<=, 0.3
  end

  def test_binary
    model = LightGBM.train(binary_params, binary_train, valid_sets: [binary_train, binary_test], verbose_eval: false)
    y_pred = model.predict(binary_test.data)
    assert_in_delta 0.9999896702079722, y_pred.first

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(binary_test.data)
    assert_equal y_pred, y_pred2
  end

  def test_multiclass
    model = LightGBM.train(multiclass_params, multiclass_train, valid_sets: [multiclass_train, multiclass_test], verbose_eval: false)

    y_pred = model.predict(multiclass_test.data)
    expected = [0.00036627031584163575, 0.9456350323547973, 0.053998697329361176]
    assert_elements_in_delta expected, y_pred.first
    # ensure reshaped
    assert_equal 200, y_pred.size
    assert_equal 3, y_pred.first.size

    model.save_model(tempfile)
    model = LightGBM::Booster.new(model_file: tempfile)
    y_pred2 = model.predict(multiclass_test.data)
    assert_equal y_pred, y_pred2
  end

  def test_early_stopping_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(regression_params, regression_train, valid_sets: [regression_train, regression_test], early_stopping_rounds: 5)
    end
    assert_equal 69, model.best_iteration
    assert_includes stdout, "Early stopping, best iteration is:\n[69]\ttraining's l2: 0.0312266\tvalid_1's l2: 0.0843578"
  end

  def test_early_stopping_not_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(regression_params, regression_train, valid_sets: [regression_train, regression_test], early_stopping_rounds: 500)
    end
    assert_equal 100, model.best_iteration
    assert_includes stdout, "Best iteration is: [100]\ttraining's l2: 0.024524\tvalid_1's l2: 0.0841232"
  end

  def test_early_stopping_early_higher_better
    model = LightGBM.train(binary_params.merge(metric: "auc"), binary_train, valid_sets: [binary_train, binary_test], early_stopping_rounds: 5, verbose_eval: false)
    assert_equal 8, model.best_iteration
  end

  def test_verbose_eval_false
    stdout, _ = capture_io do
      LightGBM.train(regression_params, regression_train, valid_sets: [regression_train, regression_test], early_stopping_rounds: 5, verbose_eval: false)
    end
    assert_empty stdout
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, regression_train)
    end
  end

  def test_early_stopping_no_valid_set
    error = assert_raises ArgumentError do
      LightGBM.train(regression_params, regression_train, valid_sets: [], early_stopping_rounds: 5)
    end
    assert_includes error.message, "at least one validation set is required"
  end

  def test_early_stopping_valid_set_training
    error = assert_raises ArgumentError do
      LightGBM.train(regression_params, regression_train, valid_sets: [regression_train], early_stopping_rounds: 5)
    end
    assert_includes error.message, "at least one validation set is required"
  end

  def test_categorical_feature
    train_set = LightGBM::Dataset.new(regression_train.data, label: regression_train.label, categorical_feature: [3])
    model = LightGBM.train(regression_params, train_set)
    assert_in_delta 1.2914367038779377, model.predict(regression_test.data).first
  end

  def test_multiple_metrics
    params = regression_params.merge(metric: ["l1", "l2", "rmse"])
    LightGBM.train(params, regression_train, valid_sets: [regression_train, regression_test], verbose_eval: false, early_stopping_rounds: 5)
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
