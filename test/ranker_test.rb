require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = ranker_data
    group = [100, 200]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)
    y_pred = model.predict(x_test)
    expected = [4.32677558843951, 1.5663855381974388, 3.8499830924310703, -2.1940085102547804, 3.3916802314416667, 3.488857015835257]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [72, 114, 141, 17]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end
end
