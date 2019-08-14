require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_model_file
    x_test = [
      [0.04417, 70.0, 2.24, 0, 0.400, 6.871, 47.4, 7.8278, 5, 358, 14.8, 390.86, 6.07],
      [0.03537, 34.0, 6.09, 0, 0.433, 6.590, 40.4, 5.4917, 7, 329, 16.1, 395.75, 9.50]
    ]

    model = LightGBM::Booster.new(model_file: "test/support/model.txt")
    y_pred = model.predict(x_test)
    assert_in_delta 28.29122797, y_pred[0]
    assert_in_delta 25.87936514, y_pred[1]
  end

  def test_model_str
    x_test = [
      [0.04417, 70.0, 2.24, 0, 0.400, 6.871, 47.4, 7.8278, 5, 358, 14.8, 390.86, 6.07],
      [0.03537, 34.0, 6.09, 0, 0.433, 6.590, 40.4, 5.4917, 7, 329, 16.1, 395.75, 9.50]
    ]

    model = LightGBM::Booster.new(model_str: File.read("test/support/model.txt"))
    y_pred = model.predict(x_test)
    assert_in_delta 28.29122797, y_pred[0]
    assert_in_delta 25.87936514, y_pred[1]
  end

  def test_feature_importance
    model = LightGBM::Booster.new(model_file: "test/support/model.txt")
    expected = [98, 16, 66, 0, 40, 201, 109, 108, 24, 77, 74, 100, 162]
    assert_equal expected, model.feature_importance
  end

  def test_feature_importance_bad_importance_type
    model = LightGBM::Booster.new(model_file: "test/support/model.txt")
    error = assert_raises LightGBM::Error do
      model.feature_importance(importance_type: "bad")
    end
    assert_includes error.message, "Unknown importance type"
  end

  def test_to_json
    model = LightGBM::Booster.new(model_file: "test/support/model.txt")
    assert JSON.parse(model.to_json)
  end
end
