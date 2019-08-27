require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.29122797, 25.87936514, 24.07423114, 31.56982437, 34.88568656, 30.3112404]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = LightGBM::Regressor.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)

    expected = [98, 16, 66, 0, 40, 201, 109, 108, 24, 77, 74, 100, 162]
    assert_equal expected, model.feature_importances
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 55, model.best_iteration
  end

  def test_daru
    data = Daru::DataFrame.from_csv("test/support/boston.csv")
    y = data["medv"]
    x = data.delete_vector("medv")

    # daru has bug with 0...300
    x_train = x.row[0..299]
    y_train = y[0..299]
    x_test = x.row[300..-1]

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.29122797, 25.87936514, 24.07423114, 31.56982437, 34.88568656, 30.3112404]
    assert_elements_in_delta expected, y_pred[0, 6]
  end
end
