require_relative "test_helper"

class LightGBMTest < Minitest::Test
  def test_load_model
    model = LightGBM::Booster.new(model_file: "test/support/model.txt")

    x_test = [
      [0.04417, 70.0, 2.24, 0, 0.400, 6.871, 47.4, 7.8278, 5, 358, 14.8, 390.86, 6.07],
      [0.03537, 34.0, 6.09, 0, 0.433, 6.590, 40.4, 5.4917, 7, 329, 16.1, 395.75, 9.50]
    ]
    preds = model.predict(x_test)
    assert_in_delta 28.29122797, preds[0]
    assert_in_delta 25.87936514, preds[1]

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    preds = model.predict(x_test)
    assert_in_delta 28.29122797, preds[0]
    assert_in_delta 25.87936514, preds[1]
  end
end
