require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_train_regression
    x_test = boston_test.data
    y_test = boston_test.label

    params = {objective: "regression"}
    model = LightGBM.train(params, boston_train)
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  def test_train_classification_binary
    params = {objective: "binary"}
    model = LightGBM.train(params, iris_train, valid_sets: [iris_train, iris_test], verbose_eval: false)
    y_pred = model.predict([6.3, 2.7, 4.9, 1.8])
    assert_in_delta 0.99998366, y_pred
  end

  def test_train_classification_multiclass
    params = {objective: "multiclass", num_class: 3}
    model = LightGBM.train(params, iris_train, valid_sets: [iris_train, iris_test], verbose_eval: false)
    y_pred = model.predict([6.3, 2.7, 4.9, 1.8])
    assert_in_delta 3.91608299e-04, y_pred[0]
    assert_in_delta 3.81933551e-01, y_pred[1]
    assert_in_delta 6.17674841e-01, y_pred[2]
  end

  def test_early_stopping_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(default_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 5)
    end
    assert_equal 55, model.best_iteration
    assert_includes stdout, "Early stopping, best iteration is:\n[55]\ttraining's l2: 2.18872\tvalid_1's l2: 35.6151"
  end

  def test_early_stopping_not_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(default_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 500)
    end
    assert_equal 71, model.best_iteration
    assert_includes stdout, "Best iteration is: [71]\ttraining's l2: 1.69138\tvalid_1's l2: 35.2563"
  end

  def test_verbose_eval_false
    stdout, _ = capture_io do
      LightGBM.train(default_params, boston_train, valid_sets: [boston_train, boston_test], early_stopping_rounds: 5, verbose_eval: false)
    end
    assert_empty stdout
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, boston_train)
    end
  end

  def test_cv
    eval_hist = LightGBM.cv(default_params, boston, shuffle: false)
    assert_in_delta 82.33637413467392, eval_hist["l2-mean"].first
    assert_in_delta 22.55870116923647, eval_hist["l2-mean"].last
    assert_in_delta 35.018415375374886, eval_hist["l2-stdv"].first
    assert_in_delta 11.605523321472438, eval_hist["l2-stdv"].last
  end

  def test_cv_early_stopping_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(default_params, boston, shuffle: false, verbose_eval: true, early_stopping_rounds: 5)
    end
    assert_equal 49, eval_hist["l2-mean"].size
    assert_includes stdout, "[49]\tcv_agg's l2: 21.6348 + 12.0872"
    refute_includes stdout, "[50]"
  end

  def test_cv_early_stopping_not_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(default_params, boston, shuffle: false, verbose_eval: true, early_stopping_rounds: 500)
    end
    assert_equal 100, eval_hist["l2-mean"].size
    assert_includes stdout, "[100]\tcv_agg's l2: 22.5587 + 11.6055"
  end

  private

  def default_params
    {objective: "regression", metric: "mse"}
  end

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
