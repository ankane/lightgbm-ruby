require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_train
    x_test = test_set.data
    y_test = test_set.label

    params = {objective: "regression", verbosity: -1, metric: "mse"}
    model = LightGBM.train(params, train_set, valid_sets: [train_set, test_set]) #, valid_names: ["train", "test"])
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  def test_early_stopping_early
    params = {objective: "regression", metric: "mse"}
    model = LightGBM.train(params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 5)
    assert_equal 55, model.best_iteration
  end

  def test_early_stopping_not_early
    params = {objective: "regression", metric: "mse"}
    model = LightGBM.train(params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 500)
    assert_equal 71, model.best_iteration
  end

  def test_verbose_eval_false
    stdout, _ = capture_io do
      params = {objective: "regression", metric: "mse"}
      LightGBM.train(params, train_set, valid_sets: [train_set, test_set], early_stopping_rounds: 5, verbose_eval: false)
    end
    assert_empty stdout
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, train_set)
    end
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
