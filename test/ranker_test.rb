require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = ranker_data
    group = [100, 200]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)
    y_pred = model.predict(x_test)
    expected = [7.649710976706889, 3.616680140345855, 6.823578048848519, -0.3963428185130039, 6.0873724665636315, 7.14720155523513]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [210, 164, 213, 87]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end
end
