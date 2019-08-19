require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = iris_data
    group = [20, 80]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)
    y_pred = model.predict(x_test)
    expected = [0.84250659, 2.52157059, -2.834404, -0.5862299, -0.51153132, -0.78868645]
    assert_elements_in_delta expected, y_pred[0, 6]

    model.save_model(tempfile)

    model = LightGBM::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_feature_importances
    x_train, y_train, _, _ = iris_data
    group = [20, 80]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)

    expected = [6, 20, 155, 83]
    assert_equal expected, model.feature_importances
  end
end
