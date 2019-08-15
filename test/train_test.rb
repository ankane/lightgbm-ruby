require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_train
    x_test = test_set.data
    y_test = test_set.label

    params = {objective: "regression", verbosity: -1, metric: "mse"}
    model = LightGBM.train(params, train_set)
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  def test_early_stopping_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(default_params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 5)
    end
    assert_equal 55, model.best_iteration
    assert_includes stdout, "Early stopping, best iteration is:\n[55]\ttraining: 2.18872\tvalid_1: 35.6151"
  end

  def test_early_stopping_not_early
    model = nil
    stdout, _ = capture_io do
      model = LightGBM.train(default_params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 500)
    end
    assert_equal 71, model.best_iteration
    assert_includes stdout, "Best iteration is: [71]\ttraining: 1.69138\tvalid_1: 35.2563"
  end

  def test_verbose_eval_false
    stdout, _ = capture_io do
      LightGBM.train(default_params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 5, verbose_eval: false)
    end
    assert_empty stdout
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, train_set)
    end
  end

  def test_cv
    eval_hist = LightGBM.cv(default_params, dataset, shuffle: false)
    p eval_hist
  end

  private

  def default_params
    {objective: "regression", metric: "mse"}
  end

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
