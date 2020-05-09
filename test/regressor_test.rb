require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = regression_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1.3687029666659025, 1.7352643821271516, 1.4988839660914637, 0.8784593080455959, 1.209552643550604, 1.4602293932569006]

    assert_elements_in_delta expected, y_pred.first(6)

    expected = [280, 285, 335, 148]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Regressor.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = regression_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 69, model.best_iteration
  end

  def test_daru
    data = Daru::DataFrame.from_csv(data_path)
    y = data["y"]
    x = data.delete_vector("y")

    # daru has bug with 0...300
    x_train = x.row[0..299]
    y_train = y[0..299]
    x_test = x.row[300..-1]

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1.3687029666659025, 1.7352643821271516, 1.4988839660914637, 0.8784593080455959, 1.209552643550604, 1.4602293932569006]
    assert_elements_in_delta expected, y_pred.first(6)
  end

  def test_trivial
    x = [[1], [2], [3], [4]]
    y = [0.1, 0.2, 0.3, 0.4]
    model = LightGBM::Regressor.new(min_data_in_bin: 1, min_data_in_leaf: 1)
    model.fit(x, y)
    assert_elements_in_delta y, model.predict(x)
  end
end
