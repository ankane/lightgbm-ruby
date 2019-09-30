require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = iris_data
    group = [20, 80]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)
    y_pred = model.predict(x_test)
    expected = [3.63957802, 8.11121958, -8.93906771, -2.03459015, -1.34105828, -1.56297634]
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

    expected = [20, 23, 101, 79]
    assert_equal expected, model.feature_importances
  end
end
