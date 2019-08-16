require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.29122797, 25.87936514, 24.07423114, 31.56982437, 34.88568656, 30.3112404]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model("/tmp/model.txt")

    model = LightGBM::Regressor.new
    model.load_model("/tmp/model.txt")
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, x_test, _ = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)

    expected = [98, 16, 66, 0, 40, 201, 109, 108, 24, 77, 74, 100, 162]
    assert_equal expected, model.feature_importances
  end
end
