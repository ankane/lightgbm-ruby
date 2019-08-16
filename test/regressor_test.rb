require_relative "test_helper"

class RegressorTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = boston_data

    model = LightGBM::Regressor.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [28.29122797, 25.87936514, 24.07423114, 31.56982437, 34.88568656, 30.3112404]
    expected.zip(y_pred) do |exp, act|
      assert_in_delta exp, act
    end

    model.save_model("/tmp/model.txt")

    model = LightGBM::Regressor.new
    model.load_model("/tmp/model.txt")
    assert_equal y_pred, model.predict(x_test)
  end
end
